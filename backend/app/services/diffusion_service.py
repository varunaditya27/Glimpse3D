"""
Service wrapper for Stable Diffusion XL (SDXL) + ControlNet.

Responsibilities:
- Load SDXL and ControlNet checkpoints
- Perform image-to-image enhancement guided by depth/canny edges
- Inpaint missing regions if necessary
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.logger import get_logger

logger = get_logger(__name__)

class DiffusionService:
    """
    Backend service for diffusion-based image enhancement.

    Uses the ai_modules/diffusion/EnhanceService for SDXL + ControlNet enhancement.
    """

    def __init__(self):
        self.logger = logger
        self.enhance_service = None

    async def initialize(self):
        """Initialize the enhancement service."""
        if self.enhance_service is None:
            try:
                from ai_modules.diffusion import EnhanceService, EnhanceConfig

                # Use optimized config for T4 GPU (or CPU fallback)
                config = EnhanceConfig.for_t4_gpu()
                self.enhance_service = EnhanceService(config=config)
                self.enhance_service.load()

                self.logger.info("Diffusion service initialized successfully")
            except ImportError as e:
                self.logger.error(f"Failed to import diffusion modules: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize diffusion service: {e}")
                raise

    async def enhance_image(self, image_path: str, prompt: str,
                          depth_map_path: Optional[str] = None,
                          strength: float = 0.75) -> Dict[str, Any]:
        """
        Enhance a single image using SDXL + ControlNet.

        Args:
            image_path: Path to input image
            prompt: Enhancement prompt
            depth_map_path: Optional path to depth map for ControlNet
            strength: Denoising strength (0.0-1.0)

        Returns:
            Dict with 'success', 'enhanced_path', and optional error
        """
        try:
            await self.initialize()

            # Load depth map if provided
            depth_map = None
            if depth_map_path and Path(depth_map_path).exists():
                try:
                    import numpy as np
                    depth_map = np.load(depth_map_path)
                    self.logger.info(f"Loaded depth map from {depth_map_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to load depth map: {e}")

            # Enhance the image
            enhanced_image = self.enhance_service.enhance(
                image=image_path,
                prompt=prompt,
                depth_map=depth_map,
                strength=strength
            )

            # Save enhanced image
            output_path = Path(image_path).parent / f"enhanced_{Path(image_path).name}"
            enhanced_image.save(output_path)

            return {
                'success': True,
                'enhanced_path': str(output_path),
                'original_path': image_path
            }

        except Exception as e:
            error_msg = f"Image enhancement failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'original_path': image_path
            }

    async def enhance_views(self, view_paths: Dict[str, str],
                          depth_paths: Dict[str, str],
                          output_dir: str,
                          prompt: str = "high quality 3D render, detailed texture, studio lighting") -> Dict[str, Any]:
        """
        Enhance multiple views for 3D reconstruction.

        Args:
            view_paths: Dict mapping view names to image paths
            depth_paths: Dict mapping view names to depth map paths
            output_dir: Directory to save enhanced images
            prompt: Enhancement prompt

        Returns:
            Dict with enhanced paths and status
        """
        try:
            await self.initialize()

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            enhanced_paths = {}
            total_views = len(view_paths)

            for i, (view_name, image_path) in enumerate(view_paths.items()):
                self.logger.info(f"Enhancing view {i+1}/{total_views}: {view_name}")

                # Get corresponding depth map
                depth_path = depth_paths.get(view_name)

                result = await self.enhance_image(
                    image_path=image_path,
                    prompt=prompt,
                    depth_map_path=depth_path
                )

                if result['success']:
                    # Move enhanced image to output directory with consistent naming
                    enhanced_src = Path(result['enhanced_path'])
                    enhanced_dst = output_path / f"enhanced_{view_name}_{enhanced_src.name}"
                    enhanced_src.rename(enhanced_dst)
                    enhanced_paths[view_name] = str(enhanced_dst)
                else:
                    self.logger.warning(f"Failed to enhance {view_name}: {result.get('error')}")
                    # Keep original as fallback
                    enhanced_paths[view_name] = image_path

            return {
                'success': True,
                'enhanced_paths': enhanced_paths,
                'total_processed': len(enhanced_paths)
            }

        except Exception as e:
            error_msg = f"Batch enhancement failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'enhanced_paths': view_paths  # Return originals as fallback
            }

    async def cleanup(self):
        """Clean up resources."""
        if self.enhance_service:
            try:
                self.enhance_service.unload()
                self.enhance_service = None
                self.logger.info("Diffusion service cleaned up")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
