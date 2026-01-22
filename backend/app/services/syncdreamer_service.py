
import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import logging
import gc

from ..core.logger import get_logger

# Try to import torch for cleanup
try:
    import torch
except ImportError:
    torch = None

class SyncDreamerService:
    """
    Service for generating multi-view images using SyncDreamer.
    Wraps ai_modules/sync_dreamer/inference.py
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.service_instance = None

    async def generate_views(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Generate 16 consistent views from a single input image.
        
        Args:
            image_path: Path to the input image (RGBA).
            output_dir: Directory to save generated views.
            
        Returns:
            Dict containing:
            - success: bool
            - view_paths: Dict[str, str] mapping view IDs to file paths
            - error: str (optional)
        """
        try:
            self.logger.info(f"Starting SyncDreamer generation for {image_path}")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Run inference in a separate thread to avoid blocking the async loop
            # since the underlying inference is synchronous and GPU-bound
            loop = asyncio.get_running_loop()
            
            # Use run_in_executor to run the blocking inference code
            result = await loop.run_in_executor(
                None, 
                self._run_inference_sync, 
                image_path, 
                output_dir
            )
            
            return result

        except Exception as e:
            self.logger.error(f"SyncDreamer generation failed: {e}", exc_info=True)
            return {
                "success": False, 
                "error": str(e),
                "view_paths": {}
            }

    def _run_inference_sync(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Synchronous wrapper for the actual inference call."""
        try:
            # Lazy import to avoid loading heavy libraries at startup
            from ai_modules.sync_dreamer.inference import get_service, cleanup as cleanup_module
            
            # Get service instance
            self.service_instance = get_service()
            
            # Generate views
            # We generate 16 views (0-15)
            output_paths = self.service_instance.generate_and_save(
                image_path, 
                output_dir,
                elevation=30.0, # Default elevation
                save_grid=True
            )
            
            # Organize paths into a dictionary
            view_paths = {}
            for path in output_paths:
                filename = os.path.basename(path)
                if "view_" in filename:
                    # extract index from filename "view_00_..."
                    try:
                        idx = int(filename.split('_')[1])
                        view_paths[f"view_{idx}"] = path
                    except (IndexError, ValueError):
                        continue
            
            self.logger.info(f"SyncDreamer generated {len(view_paths)} views")
            
            return {
                "success": True,
                "view_paths": view_paths
            }
            
        except ImportError as e:
            return {"success": False, "error": f"SyncDreamer module not found: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Inference error: {e}"}

    async def cleanup(self):
        """Cleanup VRAM."""
        try:
            from ai_modules.sync_dreamer.inference import cleanup as cleanup_module
            cleanup_module()
            
            if self.service_instance:
                if hasattr(self.service_instance, 'unload_model'):
                    self.service_instance.unload_model()
                self.service_instance = None
                
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self.logger.info("SyncDreamer VRAM cleaned up")
            
        except Exception as e:
            self.logger.warning(f"Error during SyncDreamer cleanup: {e}")
