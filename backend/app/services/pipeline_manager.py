"""
Orchestrator for the entire 3D generation pipeline.

Responsibilities:
- Manage state transitions (Uploaded -> Coarse -> Refined)
- Coordinate calls between different AI services (Zero123, Depth, Diffusion, GSplat)
- Handle error recovery and logging
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..core.logger import get_logger

logger = get_logger(__name__)

class PipelineStage(Enum):
    UPLOADED = "uploaded"
    COARSE_RECONSTRUCTION = "coarse_reconstruction"
    MULTI_VIEW_GENERATION = "multi_view_generation"
    DEPTH_ESTIMATION = "depth_estimation"
    DIFFUSION_ENHANCEMENT = "diffusion_enhancement"
    REFINEMENT = "refinement"
    EXPORT = "export"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class PipelineState:
    """Current state of the pipeline execution."""
    stage: PipelineStage
    progress: float  # 0.0 to 1.0
    message: str
    current_file: Optional[str] = None
    error: Optional[str] = None

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    final_model_path: Optional[str]
    intermediate_files: Dict[str, str]
    metrics: Dict[str, Any]
    error: Optional[str]

class PipelineManager:
    """
    Orchestrator for the Glimpse3D pipeline.

    Coordinates:
    1. TripoSR/LGM coarse reconstruction
    2. SyncDreamer multi-view generation
    3. MiDaS depth estimation
    4. SDXL diffusion enhancement
    5. MVCRM refinement
    6. Final export
    """

    def __init__(self):
        self.logger = logger
        self.state_callbacks: List[callable] = []

    def add_state_callback(self, callback: callable):
        """Add callback for state updates."""
        self.state_callbacks.append(callback)

    def _update_state(self, stage: PipelineStage, progress: float, message: str,
                     current_file: Optional[str] = None, error: Optional[str] = None):
        """Update pipeline state and notify callbacks."""
        state = PipelineState(
            stage=stage,
            progress=progress,
            message=message,
            current_file=current_file,
            error=error
        )

        for callback in self.state_callbacks:
            try:
                callback(state)
            except Exception as e:
                self.logger.warning(f"State callback failed: {e}")

    async def run_pipeline(self, image_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        """
        Execute the complete Glimpse3D pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory for outputs (temp dir if None)

        Returns:
            PipelineResult with success status and outputs
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="glimpse3d_")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        intermediate_files = {}
        metrics = {}

        try:
            # Stage 1: Coarse Reconstruction
            self._update_state(PipelineStage.COARSE_RECONSTRUCTION, 0.1,
                             "Starting coarse 3D reconstruction...")

            coarse_model_path = await self._run_coarse_reconstruction(image_path, output_path)
            intermediate_files['coarse_model'] = coarse_model_path

            # Stage 2: Multi-view Generation
            self._update_state(PipelineStage.MULTI_VIEW_GENERATION, 0.3,
                             "Generating additional views...")

            views_data = await self._run_multi_view_generation(image_path, output_path)
            intermediate_files.update(views_data)

            # Stage 3: Depth Estimation
            self._update_state(PipelineStage.DEPTH_ESTIMATION, 0.4,
                             "Estimating depth maps...")

            depth_data = await self._run_depth_estimation(views_data, output_path)
            intermediate_files.update(depth_data)

            # Stage 4: Diffusion Enhancement
            self._update_state(PipelineStage.DIFFUSION_ENHANCEMENT, 0.6,
                             "Enhancing views with diffusion...")

            enhanced_views = await self._run_diffusion_enhancement(views_data, depth_data, output_path)
            intermediate_files.update(enhanced_views)

            # Stage 5: Refinement
            self._update_state(PipelineStage.REFINEMENT, 0.8,
                             "Refining 3D model...")

            refined_model_path = await self._run_refinement(
                coarse_model_path, enhanced_views, views_data, depth_data, output_path, image_path
            )
            intermediate_files['refined_model'] = refined_model_path

            # Stage 6: Export
            self._update_state(PipelineStage.EXPORT, 0.95,
                             "Exporting final model...")

            final_model_path = await self._run_export(refined_model_path, output_path)
            intermediate_files['final_model'] = final_model_path

            self._update_state(PipelineStage.COMPLETED, 1.0,
                             "Pipeline completed successfully!")

            return PipelineResult(
                success=True,
                final_model_path=final_model_path,
                intermediate_files=intermediate_files,
                metrics=metrics,
                error=None
            )

        except Exception as e:
            error_msg = f"Pipeline failed at stage {self._current_stage}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            self._update_state(PipelineStage.FAILED, 0.0, error_msg, error=error_msg)

            return PipelineResult(
                success=False,
                final_model_path=None,
                intermediate_files=intermediate_files,
                metrics=metrics,
                error=error_msg
            )

    async def _run_coarse_reconstruction(self, image_path: str, output_path: Path) -> str:
        """Run TripoSR or LGM for initial 3D reconstruction."""
        try:
            # Import here to avoid circular imports
            from .gsplat_service import GSplatService

            gsplat_service = GSplatService()
            result = await gsplat_service.reconstruct_3d(image_path, str(output_path))

            if not result['success']:
                raise RuntimeError(f"Coarse reconstruction failed: {result.get('error', 'Unknown error')}")

            return result['model_path']

        except ImportError:
            # Fallback to mock implementation
            self.logger.warning("GSplatService not available, using mock reconstruction")
            mock_path = output_path / "coarse_model.ply"
            # Create a simple placeholder file
            with open(mock_path, 'w') as f:
                f.write("# Mock PLY file - replace with actual reconstruction\n")
            return str(mock_path)

    async def _run_multi_view_generation(self, image_path: str, output_path: Path) -> Dict[str, str]:
        """Generate additional views using SyncDreamer."""
        try:
            from .syncdreamer_service import SyncDreamerService

            sync_dreamer_service = SyncDreamerService()
            result = await sync_dreamer_service.generate_views(image_path, str(output_path))

            if not result['success']:
                self.logger.warning(f"Multi-view generation failed: {result.get('error')}, continuing with single view")

            return result.get('view_paths', {})

        except ImportError:
            self.logger.warning("SyncDreamerService not available, using single view only")
            return {}

    async def _run_depth_estimation(self, views_data: Dict[str, str], output_path: Path) -> Dict[str, str]:
        """Estimate depth maps for all views."""
        try:
            from .depth_service import DepthService

            depth_service = DepthService()
            result = await depth_service.estimate_depth_batch(views_data, str(output_path))

            if not result['success']:
                self.logger.warning(f"Depth estimation failed: {result.get('error')}")

            return result.get('depth_paths', {})

        except ImportError:
            self.logger.warning("DepthService not available, skipping depth estimation")
            return {}

    async def _run_diffusion_enhancement(self, views_data: Dict[str, str],
                                       depth_data: Dict[str, str], output_path: Path) -> Dict[str, str]:
        """Enhance views using SDXL + ControlNet."""
        try:
            from .diffusion_service import DiffusionService

            diffusion_service = DiffusionService()
            result = await diffusion_service.enhance_views(views_data, depth_data, str(output_path))

            if not result['success']:
                self.logger.warning(f"Diffusion enhancement failed: {result.get('error')}")

            return result.get('enhanced_paths', {})

        except ImportError:
            self.logger.warning("DiffusionService not available, using original views")
            return views_data

    async def _run_refinement(self, coarse_model_path: str, enhanced_views: Dict[str, str],
                            views_data: Dict[str, str], depth_data: Dict[str, str], output_path: Path, image_path: str) -> str:
        """Run refinement using GSplat optimization."""
        try:
            from .gsplat_service import GSplatService
            
            gsplat_service = GSplatService()
            
            # Prepare training data with Multi-View Support
            training_data = {
                'image_path': image_path,
                'iterations': 100
            }
            
            # If we have enhanced views (or original views), pass the directory
            views_source = enhanced_views if enhanced_views else views_data
            if views_source:
                # Get the directory from the first view path
                first_view = list(views_source.values())[0]
                views_dir = str(Path(first_view).parent)
                training_data['views_dir'] = views_dir
                self.logger.info(f"Enabling Multi-View Refinement using views from: {views_dir}")
            
            result = await gsplat_service.optimize_splats(
                coarse_model_path, training_data, str(output_path)
            )

            if not result['success']:
                self.logger.warning(f"Refinement failed: {result.get('error')}, using coarse model")
                return coarse_model_path

            return result['model_path']

        except ImportError:
            self.logger.warning("BackProjectionService not available, using coarse model")
            return coarse_model_path

    async def _run_export(self, model_path: str, output_path: Path) -> str:
        """Export final model in requested formats."""
        # For now, just return the PLY path
        # Could add GLB, SPLAT conversions here
        return model_path

    # Synchronous version for compatibility
    def run_pipeline_sync(self, image_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        """Synchronous wrapper for the async pipeline."""
        try:
            # Try to get or create an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to handle differently
                # For now, just run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.run_pipeline(image_path, output_dir))
                    return future.result()
            else:
                return loop.run_until_complete(self.run_pipeline(image_path, output_dir))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.run_pipeline(image_path, output_dir))
