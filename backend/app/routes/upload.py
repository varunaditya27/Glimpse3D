"""
Handles image upload and preprocessing.

Responsibilities:
- Validate input image format (JPG, PNG)
- Resize/Crop image if necessary
- Remove background (rembg) if requested
- Save processed image to temp storage
- Return upload ID for session tracking
"""

from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image and prepares it for 3D generation.
    """
    # TODO: Implement image processing logic
    return {"filename": file.filename, "status": "uploaded"}
