"""
Handles image upload and preprocessing.

Responsibilities:
- Validate input image format (JPG, PNG)
- Validate image quality (blur, blank, size)
- Detect and validate single object
- Remove background (rembg)
- Crop and center object
- Save processed image to storage
- Return upload ID for session tracking
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from app.core.database import DatabaseManager
from app.core.storage import StorageManager
from app.core.logger import logger
from app.services.image_validator import get_validator
from app.models.validation_models import UploadResponse

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image and prepares it for 3D generation.
    
    Validation Steps:
    1. Format validation (image type, dimensions, aspect ratio)
    2. Quality validation (blur detection, blank image check)
    3. Background removal (rembg)
    4. Object detection (Connected Components)
    5. Single object validation
    6. Crop and center object
    
    Database Steps:
    1. Create project in database
    2. Upload original image to storage
    3. Upload processed image to storage
    4. Save validation metadata
    5. Update project status
    
    Returns:
        project_id: UUID for tracking this generation
        original_url: URL of the original uploaded image
        processed_url: URL of the processed (cropped, centered) image
        validation_metadata: Quality metrics and detection results
        
    Raises:
        HTTPException 400: Validation failed (detailed error message)
        HTTPException 500: Internal server error
    """
    project_id = None
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        logger.info(f"Processing upload: {file.filename}")
        
        # Read image data
        image_bytes = await file.read()
        
        # Step 1-6: Validate and process image
        validator = get_validator()
        validation_result = validator.validate_and_process(image_bytes, file.filename)
        
        if not validation_result["valid"]:
            logger.warning(f"Validation failed for {file.filename}: {validation_result['error']}")
            raise HTTPException(
                status_code=400,
                detail=validation_result["error"]
            )
        
        # Extract results
        processed_image = validation_result["processed_image"]
        validation_metadata = validation_result["validation_metadata"]
        
        logger.info(f"✅ Validation passed for {file.filename}")
        logger.info(f"   - Objects detected: {validation_metadata['num_objects_detected']}")
        logger.info(f"   - Blur score: {validation_metadata['blur_score']:.1f}")
        logger.info(f"   - Object area: {validation_metadata['object_area_pixels']} pixels")
        
        # Create project in database
        project_id = DatabaseManager.create_project()
        logger.info(f"Created project {project_id}")
        
        # Load original image for storage
        original_image = Image.open(io.BytesIO(image_bytes))
        
        # Upload original image
        DatabaseManager.update_project_status(
            project_id, "uploading", "Uploading original image"
        )
        original_url = StorageManager.upload_original_image(project_id, original_image)
        DatabaseManager.update_project_images(project_id, original_url=original_url)
        logger.info(f"Uploaded original image to {original_url}")
        
        # Upload processed image
        DatabaseManager.update_project_status(
            project_id, "preprocessing", "Saving processed image"
        )
        processed_url = StorageManager.upload_processed_image(project_id, processed_image)
        DatabaseManager.update_project_images(project_id, processed_url=processed_url)
        logger.info(f"Uploaded processed image to {processed_url}")
        
        # Save validation metadata to database
        DatabaseManager.save_validation_metadata(project_id, validation_metadata)
        
        # Update status to ready
        DatabaseManager.update_project_status(
            project_id, "preprocessing", "Ready for 3D generation"
        )
        
        logger.info(f"✅ Upload complete for project {project_id}")
        
        return UploadResponse(
            project_id=str(project_id),
            filename=file.filename,
            status="uploaded",
            original_url=original_url,
            processed_url=processed_url,
            validation_metadata=validation_metadata
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
        
    except Exception as e:
        # Catch all other errors
        logger.error(f"Upload failed for {file.filename}: {str(e)}", exc_info=True)
        
        # Update project status if created
        if project_id:
            try:
                DatabaseManager.update_project_status(
                    project_id, "failed", "Upload failed", str(e)
                )
            except Exception as db_error:
                logger.error(f"Failed to update project status: {db_error}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during upload: {str(e)}"
        )
