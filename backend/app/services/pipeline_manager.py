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
from dataclasses import dataclass, field
from enum import Enum
import gc

try:
    import torch
except ImportError:
    torch = None

from ..core.logger import get_logger
from ..core.utils import retry_gpu_operation
from ..core.config import settings

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
    warnings: List[str] = field(default_factory=list)

@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    final_model_path: Optional[str]
    intermediate_files: Dict[str, str]
    metrics: Dict[str, Any]
    error: Optional[str]
    warnings: List[str] = field(default_factory=list)

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
        self._current_stage = PipelineStage.UPLOADED

    def _update_state(self, stage: PipelineStage, progress: float, message: str,
                     current_file: Optional[str] = None, error: Optional[str] = None,
                     warnings: List[str] = None):
        """Update pipeline state and notify callbacks."""
        self._current_stage = stage
        state = PipelineState(
            stage=stage,
            progress=progress,
            message=message,
            current_file=current_file,
            error=error,
            warnings=warnings or []
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
        all_warnings = []

        try:
            # Stage 1: Coarse Reconstruction
            self._update_state(PipelineStage.COARSE_RECONSTRUCTION, 0.1,
                             "Starting coarse 3D reconstruction...", warnings=all_warnings)

            coarse_model_path, w = await self._run_coarse_reconstruction(image_path, output_path)
            all_warnings.extend(w)
            intermediate_files['coarse_model'] = coarse_model_path

            # Stage 2: Multi-view Generation
            self._update_state(PipelineStage.MULTI_VIEW_GENERATION, 0.3,
                             "Generating additional views...", warnings=all_warnings)

            try:
                views_data, w = await self._run_multi_view_generation(image_path, output_path)
                all_warnings.extend(w)
                intermediate_files.update(views_data)
            except Exception as e:
                self.logger.warning(f"Multi-view generation failed completely: {e}")
                all_warnings.append("Multi-view generation failed. Using single-view.")
                views_data = {}

            # Stage 3: Depth Estimation
            self._update_state(PipelineStage.DEPTH_ESTIMATION, 0.4,
                             "Estimating depth maps...", warnings=all_warnings)

            try:
                depth_data, w = await self._run_depth_estimation(views_data, output_path)
                all_warnings.extend(w)
                intermediate_files.update(depth_data)
            except Exception as e:
                self.logger.warning(f"Depth estimation failed completely: {e}")
                all_warnings.append("Depth estimation failed.")
                depth_data = {}

            # Stage 4: Diffusion Enhancement
            self._update_state(PipelineStage.DIFFUSION_ENHANCEMENT, 0.6,
                             "Enhancing views with diffusion...", warnings=all_warnings)

            try:
                enhanced_views, w = await self._run_diffusion_enhancement(views_data, depth_data, output_path)
                all_warnings.extend(w)
                intermediate_files.update(enhanced_views)
            except Exception as e:
                self.logger.warning(f"Diffusion enhancement failed completely: {e}")
                all_warnings.append("AI enhancement failed. Using raw generation.")
                enhanced_views = views_data

            # Stage 5: Refinement
            self._update_state(PipelineStage.REFINEMENT, 0.8,
                             "Refining 3D model...", warnings=all_warnings)

            try:
                refined_model_path, w = await self._run_refinement(
                    coarse_model_path, enhanced_views, views_data, depth_data, output_path, image_path
                )
                all_warnings.extend(w)
            except Exception as e:
                self.logger.warning(f"Refinement failed completely: {e}")
                all_warnings.append("Refinement failed. Using coarse model.")
                refined_model_path = coarse_model_path

            intermediate_files['refined_model'] = refined_model_path

            # Stage 6: Export
            self._update_state(PipelineStage.EXPORT, 0.95,
                             "Exporting final model...", warnings=all_warnings)

            final_model_path = await self._run_export(refined_model_path, output_path)
            intermediate_files['final_model'] = final_model_path

            self._update_state(PipelineStage.COMPLETED, 1.0,
                             "Pipeline completed successfully!", warnings=all_warnings)

            return PipelineResult(
                success=True,
                final_model_path=final_model_path,
                intermediate_files=intermediate_files,
                metrics=metrics,
                error=None,
                warnings=all_warnings
            )

        except Exception as e:
            error_msg = f"Pipeline failed at stage {self._current_stage}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            self._update_state(PipelineStage.FAILED, 0.0, error_msg, error=error_msg)

            # DEMO_MODE may attach a clearly-labeled placeholder model so the demo
            # can render something. It is NEVER reported as success.
            demo_placeholder = getattr(e, 'demo_placeholder_path', None)
            if demo_placeholder is not None:
                intermediate_files['demo_placeholder'] = demo_placeholder
                all_warnings.append("DEMO placeholder - not a real reconstruction")

            return PipelineResult(
                success=False,
                final_model_path=None,
                intermediate_files=intermediate_files,
                metrics=metrics,
                error=error_msg,
                warnings=all_warnings
            )

    @retry_gpu_operation(max_retries=2)
    async def _run_coarse_reconstruction(self, image_path: str, output_path: Path) -> Tuple[str, List[str]]:
        """Run TripoSR or LGM for initial 3D reconstruction."""
        warnings = []
        try:
            # Import here to avoid circular imports
            from .gsplat_service import GSplatService

            gsplat_service = GSplatService()
            result = await gsplat_service.reconstruct_3d(image_path, str(output_path))

            if not result['success']:
                raise RuntimeError(f"Coarse reconstruction failed: {result.get('error', 'Unknown error')}")

            return result['model_path'], warnings

        except Exception as e:
            self.logger.error(f"Coarse reconstruction failed: {e}", exc_info=True)

            if settings.DEMO_MODE:
                # DEMO ONLY: write a clearly-labeled placeholder so the demo can
                # render *something*, but the run must NOT be reported as success.
                placeholder_path = output_path / "coarse_model_DEMO_PLACEHOLDER.ply"
                with open(placeholder_path, 'w') as f:
                    f.write("""ply
format ascii 1.0
comment DEMO placeholder - NOT a real reconstruction
element vertex 4
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 255 0 0
1 0 0 255 0 0
0 1 0 255 0 0
0 0 1 255 0 0
""")
                exc = RuntimeError(f"Coarse reconstruction failed: {e}")
                # Signal the demo placeholder to run_pipeline so it can surface it
                # with success=False instead of silently faking a successful result.
                exc.demo_placeholder_path = str(placeholder_path)
                raise exc

            raise RuntimeError(f"Coarse reconstruction failed: {e}")

    @retry_gpu_operation(max_retries=1)
    async def _run_multi_view_generation(self, image_path: str, output_path: Path) -> Tuple[Dict[str, str], List[str]]:
        """Generate additional views using SyncDreamer."""
        warnings = []
        try:
            from .syncdreamer_service import SyncDreamerService

            sync_dreamer_service = SyncDreamerService()
            result = await sync_dreamer_service.generate_views(image_path, str(output_path))

            if not result['success']:
                # Instead of warning inside loop, we raise to trigger retry if it's transient.
                # But here we want graceful degradation after retries fail.
                # The @retry decorator raises the exception if max_retries exceeded.
                # So we catch that exception in the OUTER try/catch below or let it propagate/handle inside?
                # Actually, the method itself is wrapped. So if it raises, the wrapper catches and retries.
                # Only if wrapper gives up, the exception bubbles out.
                # So we should RAISE inside here if we want retry.
                error = result.get('error', 'Unknown error')
                if "out of memory" in str(error).lower():
                     raise RuntimeError(f"OOM in SyncDreamer: {error}")
                else:
                    # Non-OOM failure might be permanent, but let's try once.
                    if not result.get('view_paths'):
                         raise RuntimeError(f"SyncDreamer failed: {error}")

            # Cleanup to free VRAM for next stages
            if hasattr(sync_dreamer_service, 'cleanup'):
                await sync_dreamer_service.cleanup()
            
            # Force GC and generic VRAM cleanup
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result.get('view_paths', {}), warnings
            
        except ImportError:
             # Do not retry import errors
             msg = "SyncDreamerService not available, using single view only"
             self.logger.warning(msg)
             warnings.append(msg)
             return {}, warnings
        except Exception:
            # Let the exception propagate so @retry_gpu_operation can retry, and so
            # run_pipeline can degrade gracefully if retries are exhausted.
            raise

    @retry_gpu_operation(max_retries=2)
    async def _run_depth_estimation(self, views_data: Dict[str, str], output_path: Path) -> Tuple[Dict[str, str], List[str]]:
        """Estimate depth maps for all views."""
        warnings = []
        try:
            from .depth_service import DepthService
            depth_service = DepthService()
            result = await depth_service.estimate_depth_batch(views_data, str(output_path))
            if not result['success']:
                 raise RuntimeError(f"Depth estimation failed: {result.get('error')}")

            # Cleanup
            if hasattr(depth_service, 'cleanup'):
                await depth_service.cleanup()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return result.get('depth_paths', {}), warnings
        except ImportError:
            msg = "DepthService not available, skipping depth estimation"
            self.logger.warning(msg)
            warnings.append(msg)
            return {}, warnings

    @retry_gpu_operation(max_retries=1)
    async def _run_diffusion_enhancement(self, views_data: Dict[str, str],
                                       depth_data: Dict[str, str], output_path: Path) -> Tuple[Dict[str, str], List[str]]:
        """Enhance views using SDXL + ControlNet."""
        warnings = []
        # No broad try/catch here, let decorator handle retries, and caller handle final fail
        from .diffusion_service import DiffusionService
        diffusion_service = DiffusionService()
        result = await diffusion_service.enhance_views(views_data, depth_data, str(output_path))

        if not result['success']:
             raise RuntimeError(f"Diffusion enhancement failed: {result.get('error')}")

        # Cleanup
        if hasattr(diffusion_service, 'cleanup'):
            await diffusion_service.cleanup()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result.get('enhanced_paths', {}), warnings

    @retry_gpu_operation(max_retries=2)
    async def _run_refinement(self, coarse_model_path: str, enhanced_views: Dict[str, str],
                            views_data: Dict[str, str], depth_data: Dict[str, str], output_path: Path, image_path: str) -> Tuple[str, List[str]]:
        """Run refinement using GSplat optimization."""
        warnings = []
        from .gsplat_service import GSplatService
        gsplat_service = GSplatService()
        
        training_data = {
            'image_path': image_path,
            'iterations': 100
        }
        
        # Select views
        views_source = enhanced_views if enhanced_views else views_data
        if views_source and len(views_source) > 0:
            first_view = list(views_source.values())[0]
            views_dir = str(Path(first_view).parent)
            training_data['views_dir'] = views_dir
            self.logger.info(f"Enabling Multi-View Refinement using views from: {views_dir}")
        else:
            # Soft fail inside? No, refinement is pretty key. 
            # But if no views, we just return coarse model.
            msg = "No views available for refinement. Skipping."
            self.logger.warning(msg)
            warnings.append(msg)
            return coarse_model_path, warnings
        
        result = await gsplat_service.optimize_splats(
            coarse_model_path, training_data, str(output_path)
        )

        if not result['success']:
             raise RuntimeError(f"Refinement failed: {result.get('error')}")

        refined_model_path = result['model_path']

        # ★ MVCRM back-projection refinement (novel contribution).
        # Runs on top of the gsplat-optimized model. On ANY failure we log a
        # warning and fall back to the optimize result; the pipeline never crashes
        # because of MVCRM.
        if settings.MVCRM_ENABLED:
            try:
                from .backprojection import BackProjectionService

                bp_service = BackProjectionService()
                mvcrm_result = await bp_service.refine_model(
                    coarse_model_path=refined_model_path,
                    enhanced_views=enhanced_views or {},
                    depth_maps=depth_data or {},
                    output_dir=str(output_path),
                )

                if mvcrm_result.get('success') and mvcrm_result.get('refined_model_path'):
                    refined_model_path = mvcrm_result['refined_model_path']
                    self.logger.info(
                        f"MVCRM refinement succeeded: {refined_model_path}"
                    )
                else:
                    msg = (
                        "MVCRM refinement did not succeed: "
                        f"{mvcrm_result.get('error', 'unknown error')}. "
                        "Falling back to gsplat optimize result."
                    )
                    self.logger.warning(msg)
                    warnings.append(msg)
            except Exception as e:
                msg = (
                    f"MVCRM refinement failed: {e}. "
                    "Falling back to gsplat optimize result."
                )
                self.logger.warning(msg)
                warnings.append(msg)

        return refined_model_path, warnings

    async def _run_export(self, model_path: str, output_path: Path) -> str:
        """Export final model in requested formats."""
        return model_path

    # Synchronous version for compatibility
    def run_pipeline_sync(self, image_path: str, output_dir: Optional[str] = None) -> PipelineResult:
        """Synchronous wrapper for the async pipeline."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.run_pipeline(image_path, output_dir))
                    return future.result()
            else:
                return loop.run_until_complete(self.run_pipeline(image_path, output_dir))
        except RuntimeError:
            return asyncio.run(self.run_pipeline(image_path, output_dir))
