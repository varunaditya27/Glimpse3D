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
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image.")

        # 2a. Decompression-bomb guard: reject images whose DECODED pixel count is
        # huge even if the encoded bytes are small (classic DoS). verify()
        # consumes the stream, so re-open to read dimensions without decoding.
        from ..core.config import settings as _settings
        try:
            probe = Image.open(io.BytesIO(content))
            megapixels = (probe.size[0] * probe.size[1]) / 1_000_000
        except Exception:
            megapixels = 0
        if megapixels > _settings.MAX_IMAGE_MEGAPIXELS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Image is too large to process "
                    f"({megapixels:.1f} MP > {_settings.MAX_IMAGE_MEGAPIXELS:.0f} MP limit)."
                ),
            )

        # 2b. Deep image validation (Phase 6).
        # Defensive: a validator/dependency failure must NOT break upload.
        # If the validator runs and reports the image is invalid, we surface
        # a 400-style JSON to the client. Any internal failure (missing optional
        # deps, unexpected errors) is logged and we fall through to saving.
        try:
            from ..services.image_validator import get_validator
            validation_result = get_validator().validate_and_process(content, file.filename)
            if not validation_result.get("valid", True):
                return {
                    "success": False,
                    "error": validation_result.get("error", "Image failed validation."),
                    "details": validation_result.get("validation_metadata"),
                }
        except Exception as val_err:
            from ..core.logger import get_logger
            get_logger(__name__).warning(
                f"Image validator unavailable or errored; proceeding with upload: {val_err}"
            )

        # 3. Save File
        from ..core.config import settings
        from ..core.paths import unique_upload_name

        upload_dir = settings.PROJECT_ROOT / "assets" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize: strip any directory components / traversal from the
        # client-supplied filename and prefix an unguessable token so two
        # uploads of the same name never collide.
        safe_filename = unique_upload_name(file.filename)
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
