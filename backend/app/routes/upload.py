"""
Handles image upload and preprocessing.

Responsibilities:
- Validate input image format (JPG, PNG)
- Resize/Crop image if necessary
- Remove background (rembg) if requested
- Save processed image to temp storage
- Return upload ID for session tracking
"""

import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from rembg import remove
from app.core.database import DatabaseManager
from app.core.storage import StorageManager
from app.core.logger import logger

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/")
async def upload_image(file: UploadFile = File(...), remove_bg: bool = True):
    """
    Uploads an image and prepares it for 3D generation.
    
    Steps:
    1. Create project in database
    2. Upload original image to storage
    3. Remove background if requested
    4. Upload processed image to storage
    5. Update project status
    
    Returns:
        project_id: UUID for tracking this generation
        original_url: URL of the original uploaded image
        processed_url: URL of the background-removed image
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create project in database
        project_id = DatabaseManager.create_project()
        logger.info(f"Processing upload for project {project_id}")
        
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if necessary
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        # Upload original image
        DatabaseManager.update_project_status(
            project_id, "uploading", "Uploading original image"
        )
        original_url = StorageManager.upload_original_image(project_id, image)
        DatabaseManager.update_project_images(project_id, original_url=original_url)
        
        # Remove background if requested
        if remove_bg:
            DatabaseManager.update_project_status(
                project_id, "preprocessing", "Removing background"
            )
            
            # Convert PIL image to bytes for rembg
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Remove background
            output_bytes = remove(img_byte_arr.getvalue())
            processed_image = Image.open(io.BytesIO(output_bytes))
        else:
            processed_image = image
        
        # Upload processed image
        processed_url = StorageManager.upload_processed_image(project_id, processed_image)
        DatabaseManager.update_project_images(project_id, processed_url=processed_url)
        
        # Update status
        DatabaseManager.update_project_status(
            project_id, "preprocessing", "Ready for 3D generation"
        )
        
        return {
            "project_id": project_id,
            "filename": file.filename,
            "status": "uploaded",
            "original_url": original_url,
            "processed_url": processed_url
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        if 'project_id' in locals():
            DatabaseManager.update_project_status(
                project_id, "failed", "Upload failed", str(e)
            )
        raise HTTPException(status_code=500, detail=str(e))
