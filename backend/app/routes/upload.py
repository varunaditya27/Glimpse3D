"""
Handles image upload and preprocessing.

Responsibilities:
- Validate input image format (JPG, PNG)
- Resize/Crop image if necessary
- Remove background (rembg) if requested
- Save processed image to temp storage
- Return upload ID for session tracking
"""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from PIL import Image
import io

router = APIRouter(prefix="/upload", tags=["Upload"])

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image and prepares it for 3D generation.
    Enforces Strict Validation:
    - Max Size: 20MB
    - Format: Valid Image (PIL Verify)
    """
    try:
        # 1. Validate File Size
        # Check Content-Length header first (fast fail)
        # Note: Client might not send it, or send fake. We double check real size.
        
        # Read file content
        content = await file.read()
        
        if len(content) > MAX_FILE_SIZE:
             raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")
             
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # 2. Validate Image Format (Magic Bytes / PIL Verify)
        try:
            image = Image.open(io.BytesIO(content))
            image.verify() # Checks for corruption
            
            # Reset pointer after verify? PIL verify can consume the file. 
            # We re-open for saving or just save bytes.
            format = image.format
            if format not in ["JPEG", "PNG", "WEBP", "BMP"]:
                 raise HTTPException(status_code=400, detail=f"Unsupported image format: {format}")
                 
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # 3. Save File
        from ..core.config import settings
        upload_dir = settings.PROJECT_ROOT / "assets" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        safe_filename = f"{hash(file.filename)}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
            
        return {
            "success": True,
            "filename": file.filename, 
            "file_path": str(file_path.absolute()),
            "status": "uploaded",
            "size": len(content)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        return {"success": False, "error": str(e)}
