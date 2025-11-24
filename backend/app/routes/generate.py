"""
Handles the initial coarse 3D generation.

Responsibilities:
- Accept processed image ID
- Trigger TripoSR or LGM model inference
- Generate initial Gaussian Splat (.ply or .splat)
- Return 3D model URL for frontend viewer
"""

from fastapi import APIRouter

router = APIRouter(prefix="/generate", tags=["Generate"])

@router.post("/{upload_id}")
async def generate_3d(upload_id: str):
    """
    Triggers the coarse 3D reconstruction from the uploaded image.
    """
    # TODO: Call PipelineManager to run TripoSR/LGM
    return {"status": "generating", "model_url": "placeholder.ply"}
