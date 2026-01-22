
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

class DepthService:
    """
    Service for estimating depth maps using MiDaS.
    Wraps ai_modules/midas_depth/run_depth.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.estimator = None

    async def estimate_depth_batch(self, views_data: Dict[str, str], output_dir: str) -> Dict[str, Any]:
        """
        Estimate depth maps for a batch of view images.
        
        Args:
            views_data: Dict mapping view IDs to image paths.
            output_dir: Directory to save depth maps.
            
        Returns:
            Dict containing:
            - success: bool
            - depth_paths: Dict[str, str] mapping view IDs to depth map paths
            - error: str (optional)
        """
        try:
            self.logger.info(f"Starting depth estimation for {len(views_data)} views")
            
            os.makedirs(output_dir, exist_ok=True)
            
            loop = asyncio.get_running_loop()
            
            result = await loop.run_in_executor(
                None,
                self._run_inference_sync,
                views_data,
                output_dir
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Depth estimation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "depth_paths": {}
            }

    def _run_inference_sync(self, views_data: Dict[str, str], output_dir: str) -> Dict[str, Any]:
        """Synchronous wrapper for depth inference."""
        try:
            from ai_modules.midas_depth.run_depth import get_estimator, save_depth_visualization, save_depth_raw
            
            # Get singular estimator instance (MiDaS_small is default/fastest)
            # Use 'DPT_Hybrid' for better quality if GPU memory allows, but stick to small for safety
            self.estimator = get_estimator(model_type="MiDaS_small")
            
            depth_paths = {}
            
            for view_id, image_path in views_data.items():
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}, skipping depth.")
                    continue
                    
                # Estimate depth
                depth_map = self.estimator.estimate(image_path, normalize=True)
                
                # Save outputs
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save visualization (PNG) and raw (NPY)
                viz_path = os.path.join(output_dir, f"{base_name}_depth.png")
                raw_path = os.path.join(output_dir, f"{base_name}_depth.npy")
                
                save_depth_visualization(depth_map, viz_path, colormap="magma")
                save_depth_raw(depth_map, raw_path)
                
                # We typically track the visualization path or raw path depending on downstream needs
                # Usually pipeline uses the image path for simple passing, or raw for precision
                depth_paths[view_id] = raw_path # Passing RAW path for precision in next steps if needed
                # Also handy to keep the visualization path if needed for debugging
                
            self.logger.info(f"Depth estimation completed for {len(depth_paths)} views")
            
            return {
                "success": True,
                "depth_paths": depth_paths
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Depth module not found: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Depth inference error: {e}"}

    async def cleanup(self):
        """Cleanup VRAM."""
        # MiDaS model is small, but good practice to clear if possible
        # The module uses a singleton `_default_estimator`, we can't easily kill it 
        # unless we modify the module or just clear global reference if we wanted to be aggressive.
        # For now, just torch empty cache.
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
