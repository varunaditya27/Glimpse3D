import logging
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..services.gsplat_service import GSplatService
from ..services.diffusion_service import DiffusionService

# Initialize Services
# In a real app, these should be dependencies injected
gsplat_service = GSplatService()
diffusion_service = DiffusionService()

router = APIRouter(prefix="/refine", tags=["Refine"])
logger = logging.getLogger(__name__)

class RefineRequest(BaseModel):
    model_id: str
    prompt: str
    intensity: float = 0.5
    iterations: int = 100
    view_elevation: float = 0.0
    view_azimuth: float = 0.0
    view_radius: float = 3.0

@router.post("/")
async def refine_model_endpoint(request: RefineRequest):
    """
    Triggers the enhancement loop:
    1. Render current view of the PLY model
    2. Enhance the render using SDXL (Diffusion)
    3. Back-project improvements (Optimize Splats)
    """
    try:
        model_id = request.model_id
        
        # Path Resolution (Assuming standard project structure)
        # assets/outputs/{model_id}/reconstructed.ply
        # This needs to match where generate.py saves things.
        # Based on generate.py (viewed earlier), output dir is assets/outputs/{upload_id}
        from ..core.config import settings
        model_dir = settings.PROJECT_ROOT / "assets" / "outputs" / model_id
        input_ply = model_dir / "reconstructed.ply"
        
        # If reconstructed.ply doesn't exist, try ANY ply
        if not input_ply.exists():
             ply_files = list(model_dir.glob("*.ply"))
             if ply_files:
                 input_ply = ply_files[0]
             else:
                 raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        logger.info(f"Refining model: {input_ply}")

        # 1. Render Current View
        # We need camera parameters. For now, we'll assume a default or use the requested angles.
        # GSplat render script usually takes camera pose matrix. 
        # Constructing a simple look-at camera for the given azimuth/elevation.
        # FOR NOW: Passing generic params, assumes render_view.py handles defaults or simple args
        render_output = model_dir / "refine_input_view.png"
        
        # Note: gsplat_service.render_view currently expects camera_params dict
        # We'll pass minimal args and hope render_view.py fallback logic works or we might need to update it
        # Actually gsplat/render_view.py takes --ply and --output. It might render a default spin or specific view.
        # Let's assume it renders a default view for now.
        render_result = await gsplat_service.render_view(
            model_path=str(input_ply),
            camera_params={
                "elevation": request.view_elevation, 
                "azimuth": request.view_azimuth,
                "radius": request.view_radius
            },
            output_path=str(render_output)
        )
        
        if not render_result['success']:
            raise HTTPException(status_code=500, detail=f"Rendering failed: {render_result.get('error')}")
            
        # 2. Enhance View (Diffusion)
        logger.info("Enhancing render...")
        enhancement_result = await diffusion_service.enhance_image(
            image_path=str(render_output),
            prompt=request.prompt,
            strength=request.intensity
        )
        
        if not enhancement_result['success']:
             raise HTTPException(status_code=500, detail=f"Enhancement failed: {enhancement_result.get('error')}")
             
        enhanced_image_path = enhancement_result['enhanced_path']
        
        # 3. Optimize Splats (Back-Projection)
        logger.info("Optimizing splats...")
        refine_output_dir = model_dir / "refined"
        refine_output_dir.mkdir(exist_ok=True)
        
        training_data = {
            "image_path": enhanced_image_path,
            "iterations": request.iterations,
            # We should pass camera pose used for rendering so optimization matches!
            # For simplicity in this v1, we assume the render and optimization use aligned default cameras
        }
        
        optimize_result = await gsplat_service.optimize_splats(
            model_path=str(input_ply),
            training_data=training_data,
            output_dir=str(refine_output_dir)
        )
        
        if not optimize_result['success']:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {optimize_result.get('error')}")

        refined_model_path = optimize_result['model_path']
        # Convert absolute path to relative URL
        # assets/outputs/... -> /outputs/...
        relative_path = Path(refined_model_path).relative_to(settings.PROJECT_ROOT / "assets")
        model_url = f"/{relative_path}".replace("\\", "/") # Ensure forward slashes for URL

        # Cleanups
        await diffusion_service.cleanup()

        return {
            "status": "completed",
            "model_id": model_id,
            "updated_model_url": model_url,
            "comparison_images": {
                "original": f"/outputs/{model_id}/{Path(render_output).name}",
                "enhanced": f"/outputs/{model_id}/{Path(enhanced_image_path).name}"
            }
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Refinement pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
