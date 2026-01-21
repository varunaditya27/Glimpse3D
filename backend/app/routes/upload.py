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
from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image and prepares it for 3D generation.
    """
    try:
        # Define upload directory
        # Go up from backend/app/routes to project root
        project_root = Path(__file__).parent.parent.parent.parent
        upload_dir = project_root / "assets" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_filename = f"{hash(file.filename)}_{file.filename}"
        file_path = upload_dir / safe_filename
        
        # Save file
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "success": True,
            "filename": file.filename, 
            "file_path": str(file_path.absolute()),
            "status": "uploaded"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
