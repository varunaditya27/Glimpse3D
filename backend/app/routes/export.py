"""
Handles final model export.

Responsibilities:
- Convert internal Gaussian Splat format to requested output format (.ply, .splat, .glb, .obj)
- Optimize file size if requested
- Provide download link
"""

from fastapi import APIRouter

router = APIRouter(prefix="/export", tags=["Export"])

@router.get("/{model_id}")
async def export_model(model_id: str, format: str = "ply"):
    """
    Exports the 3D model in the requested format.
    """
    # TODO: Implement conversion logic
    return {"status": "ready", "download_url": f"model.{format}"}
