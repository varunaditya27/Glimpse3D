
import asyncio
import os
import gc
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..core.logger import get_logger

try:
    import torch
except ImportError:
    torch = None

class DiffusionService:
    """
    Service for enhancing views using SDXL Lightning (Diffusion).
    Wraps ai_modules/diffusion/enhance_service.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.service_instance = None

    async def enhance_views(
        self, 
        views_data: Dict[str, str], 
        depth_data: Dict[str, str], 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Enhance generated views using SDXL Lightning + ControlNet (Depth).
        
        Args:
            views_data: Dict mapping view IDs to image paths.
            depth_data: Dict mapping view IDs to depth map paths (.npy or .png).
            output_dir: Directory to save enhanced images.
            
        Returns:
            Dict containing:
            - success: bool
            - enhanced_paths: Dict[str, str] mapping view IDs to enhanced image paths
            - error: str (optional)
        """
        try:
            self.logger.info(f"Starting diffusion enhancement for {len(views_data)} views")
            
            os.makedirs(output_dir, exist_ok=True)
            
            loop = asyncio.get_running_loop()
            
            result = await loop.run_in_executor(
                None,
                self._run_inference_sync,
                views_data,
                depth_data,
                output_dir
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Diffusion enhancement failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "enhanced_paths": {}
            }

    def _run_inference_sync(
        self, 
        views_data: Dict[str, str], 
        depth_data: Dict[str, str], 
        output_dir: str
    ) -> Dict[str, Any]:
        """Synchronous wrapper for diffusion inference."""
        try:
            from ai_modules.diffusion.enhance_service import EnhanceService, EnhanceConfig
            from ai_modules.midas_depth.run_depth import load_depth_raw
            import numpy as np
            from PIL import Image

            # Initialize service with T4 config for speed/memory balance
            config = EnhanceConfig.for_t4_gpu()
            self.service_instance = EnhanceService(config=config)
            self.service_instance.load()

            enhanced_paths = {}
            
            # Process each view
            # Note: EnhanceService has a batch mode, but our views_data is a dictionary with IDs.
            # Processing sequentially or rebuilding list for batch is fine. 
            # Given we have matching depth maps by ID, sequential loop is clearer for mapping back.
            
            for view_id, image_path in views_data.items():
                if not os.path.exists(image_path):
                    continue
                    
                # Load depth map if available
                depth_map = None
                depth_path = depth_data.get(view_id)
                if depth_path and os.path.exists(depth_path):
                    if depth_path.endswith('.npy'):
                        depth_map = load_depth_raw(depth_path)
                    else:
                        # If it's an image, load it and maybe conversion is needed inside enhance?
                        # enhance() accepts numpy array for depth.
                        depth_img = Image.open(depth_path).convert('L')
                        depth_map = np.array(depth_img).astype(np.float32) / 255.0
                
                # Enhance
                # Use a specific prompt or default
                enhanced_img = self.service_instance.enhance(
                    image=image_path,
                    depth_map=depth_map,
                    prompt="high quality 3D asset, detailed texture, studio lighting, 4k",
                    # blend_with_original=True # Not directly in enhance(), meant for confidence wrapper
                )
                
                # Save result
                filename = f"{view_id}_enhanced.png"
                save_path = os.path.join(output_dir, filename)
                enhanced_img.save(save_path)
                
                enhanced_paths[view_id] = save_path
                
            self.logger.info(f"Enhanced {len(enhanced_paths)} views")
            
            return {
                "success": True,
                "enhanced_paths": enhanced_paths
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Diffusion module not found: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Diffusion inference error: {e}"}

    async def cleanup(self):
        """Cleanup VRAM."""
        try:
            if self.service_instance:
                self.service_instance.unload()
                self.service_instance = None
                
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.logger.info("Diffusion VRAM cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during Diffusion cleanup: {e}")
