"""
Service wrapper for Gaussian Splatting operations (gsplat library).

Responsibilities:
- Initialize 3D Gaussian Splat model from sparse points or coarse mesh
- Render views from specific camera poses
- Optimize splat parameters (position, covariance, color, opacity)
"""

import asyncio
import logging
import subprocess
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..core.logger import get_logger

logger = get_logger(__name__)

class GSplatService:
    """
    Backend service for Gaussian Splatting operations.

    Uses the ai_modules/gsplat module for reconstruction and rendering.
    Falls back to mock implementations when dependencies are missing.
    """

    def __init__(self):
        self.logger = logger

    async def reconstruct_3d(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Reconstruct 3D model from single image using TripoSR + gsplat.
        Uses cached model instance for performance.

        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs

        Returns:
            Dict with reconstruction results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            output_ply = output_path / "reconstructed.ply"

            # Import directly (paths should be set by backend/app/services/__init__.py or app main)
            # We assume ai_modules is in sys.path. 
            # If not, add it dynamically:
            import sys
            from ..core.config import settings
            if str(settings.PROJECT_ROOT) not in sys.path:
                sys.path.append(str(settings.PROJECT_ROOT))

            from ai_modules.gsplat.reconstruct import TripoInference

            self.logger.info(f"Starting in-process reconstruction for {image_path}")

            # Run in threadpool to avoid blocking async event loop
            loop = asyncio.get_running_loop()
            
            def _run_inference():
                inference = TripoInference()
                return inference.run(image_path, str(output_ply))

            # Execute in thread pool
            success = await loop.run_in_executor(None, _run_inference)

            if success and output_ply.exists():
                self.logger.info(f"Reconstruction successful: {output_ply}")
                return {
                    'success': True,
                    'model_path': str(output_ply),
                    'log': "In-process inference complete"
                }
            else:
                self.logger.error("Reconstruction returned failure status")
                return {
                    'success': False,
                    'error': 'Inference failed internally',
                    'log': "Check backend logs"
                }

        except Exception as e:
            error_msg = f"Reconstruction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}

    async def render_view(self, model_path: str, camera_params: Dict[str, Any],
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Render a view of the Gaussian Splat model.

        Args:
            model_path: Path to PLY model
            camera_params: Camera parameters
            output_path: Path to save rendered image

        Returns:
            Dict with rendering results
        """
        try:
            if not Path(model_path).exists():
                return {'success': False, 'error': f'Model not found: {model_path}'}

            if output_path is None:
                output_path = str(Path(model_path).parent / "rendered_view.png")

            # Use the gsplat render script
            script_path = settings.PROJECT_ROOT / "ai_modules" / "gsplat" / "render_view.py"

            cmd = [
                "python", str(script_path),
                "--ply", model_path,
                "--output", output_path
            ]
            
            # Add camera parameters if present
            if camera_params:
                if 'elevation' in camera_params:
                    cmd.extend(["--elevation", str(camera_params['elevation'])])
                if 'azimuth' in camera_params:
                    cmd.extend(["--azimuth", str(camera_params['azimuth'])])
                if 'radius' in camera_params:
                    cmd.extend(["--radius", str(camera_params['radius'])])

            self.logger.info(f"Running render: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )

            if result.returncode == 0 and Path(output_path).exists():
                self.logger.info(f"Render successful: {output_path}")
                return {
                    'success': True,
                    'image_path': output_path,
                    'log': result.stdout
                }
            else:
                self.logger.error(f"Render failed: {result.stderr}")
                return {
                    'success': False,
                    'error': f'Render failed: {result.stderr}',
                    'log': result.stdout + result.stderr
                }

        except subprocess.TimeoutExpired:
            error_msg = "Render timed out"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        except Exception as e:
            error_msg = f"Render failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}

    async def optimize_splats(self, model_path: str, training_data: Dict[str, Any],
                            output_dir: str) -> Dict[str, Any]:
        """
        Optimize Gaussian Splat parameters.

        Args:
            model_path: Path to initial PLY model
            training_data: Training images and camera poses
            output_dir: Directory to save optimized model

        Returns:
            Dict with optimization results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Use the gsplat train script
            script_path = settings.PROJECT_ROOT / "ai_modules" / "gsplat" / "train.py"

            # Extract image path from training data (required for loss calculation)
            image_path = training_data.get('image_path')
            if not image_path:
                return {'success': False, 'error': 'Missing image_path in training_data'}

            cmd = [
                "python", str(script_path),
                image_path,
                "--out", str(output_path),
                "--ply_path", model_path,
                "--iter", str(training_data.get('iterations', 100))
            ]

            # Add Views Directory if available (for Multi-View Refinement)
            if 'views_dir' in training_data and training_data['views_dir']:
                cmd.extend(["--views_dir", str(training_data['views_dir'])])

            self.logger.info(f"Running optimization: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for training
            )

            if result.returncode == 0:
                # Look for optimized model
                optimized_files = list(output_path.glob("*_optimized.ply")) or list(output_path.glob("*.ply"))
                if optimized_files:
                    optimized_path = str(optimized_files[0])
                    self.logger.info(f"Optimization successful: {optimized_path}")
                    return {
                        'success': True,
                        'model_path': optimized_path,
                        'log': result.stdout
                    }

                return {
                    'success': False,
                    'error': 'No optimized model found',
                    'log': result.stdout + result.stderr
                }
            else:
                self.logger.error(f"Optimization failed: {result.stderr}")
                return {
                    'success': False,
                    'error': f'Optimization failed: {result.stderr}',
                    'log': result.stdout + result.stderr
                }

        except subprocess.TimeoutExpired:
            error_msg = "Optimization timed out"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}

        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}
