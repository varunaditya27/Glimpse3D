"""
★ NOVEL CONTRIBUTION: Back-Projection Refinement Module.

Responsibilities:
- Map pixels from the enhanced 2D view back to 3D Gaussian Splats
- Calculate gradients based on the difference between rendered and enhanced views
- Update splat properties to enforce consistency with the enhanced view
- Handle occlusion and depth discrepancies

This service wires the backend pipeline to the MVCRM stack:
- ai_modules.gsplat.render_gsplat.render_with_contributions  (CPU-capable renderer)
- ai_modules.sync_dreamer.utils_syncdreamer.syncdreamer_cameras  (per-view cameras)
- ai_modules.refine_module.FusionController  (iterative refinement loop)
- ai_modules.gsplat.utils_gs  (PLY round-trip in activation space)
"""

import asyncio
import importlib.util
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
import numpy as np

from ..core.logger import get_logger

logger = get_logger(__name__)

# SH degree-0 DC constant (3D Gaussian Splatting convention).
# RGB = clamp(0.5 + C0 * features_dc, 0, 1);  features_dc = (RGB - 0.5) / C0
C0 = 0.28209479177387814

# Repo root: backend/app/services/backprojection.py -> parents[3] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module_from_path(qualified_name: str, relative_path: str):
    """Load a single submodule directly from its file path.

    The ai_modules.gsplat package __init__ eagerly imports the CUDA ``gsplat``
    rasterizer (via optimize.py), so a plain ``from ai_modules.gsplat.X import Y``
    fails in CPU-only / lightweight environments even though ``X`` itself only
    needs torch/numpy/plyfile. Loading the leaf module straight from its file
    bypasses the heavy package __init__. Mirrors the fallback already used in
    ai_modules/sync_dreamer/utils_syncdreamer.syncdreamer_cameras.
    """
    module_path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_gsplat_utils():
    """Return (load_ply, save_ply), preferring the package, with a file fallback."""
    try:
        from ai_modules.gsplat.utils_gs import load_ply, save_ply
        return load_ply, save_ply
    except Exception:
        mod = _load_module_from_path(
            "ai_modules.gsplat.utils_gs", "ai_modules/gsplat/utils_gs.py"
        )
        return mod.load_ply, mod.save_ply


def _import_render_fn():
    """Return render_with_contributions, with a direct-file fallback."""
    try:
        from ai_modules.gsplat.render_gsplat import render_with_contributions
        return render_with_contributions
    except Exception:
        mod = _load_module_from_path(
            "ai_modules.gsplat.render_gsplat", "ai_modules/gsplat/render_gsplat.py"
        )
        return mod.render_with_contributions


def _import_cameras_fn():
    """Return syncdreamer_cameras, with a direct-file fallback."""
    try:
        from ai_modules.sync_dreamer.utils_syncdreamer import syncdreamer_cameras
        return syncdreamer_cameras
    except Exception:
        mod = _load_module_from_path(
            "ai_modules.sync_dreamer.utils_syncdreamer",
            "ai_modules/sync_dreamer/utils_syncdreamer.py",
        )
        return mod.syncdreamer_cameras


class BackProjectionService:
    """
    Backend service for MVCRM refinement using back-projection.

    Uses the ai_modules/refine_module/FusionController for iterative refinement,
    the pure-PyTorch contribution-tracking renderer for the render closure, and
    the SyncDreamer view cameras for per-view geometry.
    """

    def __init__(self):
        self.logger = logger
        self.fusion_controller = None

    async def initialize(self, config: Optional[Any] = None):
        """Initialize the fusion controller.

        Args:
            config: Optional RefinementConfig. If None, defaults are used.
        """
        if self.fusion_controller is None:
            try:
                from ai_modules.refine_module import FusionController, RefinementConfig

                if config is None:
                    config = RefinementConfig()
                self.fusion_controller = FusionController(config=config)

                self.logger.info("Back-projection service initialized successfully")
            except ImportError as e:
                self.logger.error(f"Failed to import refine modules: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize back-projection service: {e}")
                raise

    async def refine_model(self, coarse_model_path: str,
                          enhanced_views: Dict[str, str],
                          depth_maps: Dict[str, str],
                          output_dir: str,
                          config: Optional[Any] = None) -> Dict[str, Any]:
        """
        Refine a 3D model using enhanced 2D views and MVCRM.

        Args:
            coarse_model_path: Path to initial PLY model
            enhanced_views: Dict mapping view names to enhanced image paths.
                            Order maps to the SyncDreamer 16-view schedule; if
                            fewer views are provided, the matching camera subset
                            is used (by enumerated index).
            depth_maps: Dict mapping view names to depth map paths (.npy)
            output_dir: Directory to save refined model
            config: Optional RefinementConfig override

        Returns:
            Dict with keys {success, refined_model_path, metrics} on success, or
            {success: False, error, refined_model_path} on any failure (the
            coarse model path is returned as a safe fallback).
        """
        try:
            await self.initialize(config=config)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            from ai_modules.refine_module import ViewData

            load_ply, save_ply = _import_gsplat_utils()
            render_with_contributions = _import_render_fn()
            syncdreamer_cameras = _import_cameras_fn()

            # --- Load the coarse model (raw fields are in activation space) -----
            try:
                model = load_ply(coarse_model_path)
            except Exception as e:
                self.logger.error(f"Cannot load coarse PLY model: {e}")
                return {
                    'success': False,
                    'error': f'PLY loading failed: {e}',
                    'refined_model_path': coarse_model_path,
                }

            num_splats = int(model._xyz.shape[0])
            self.logger.info(f"Loaded coarse model with {num_splats} splats")

            # --- Extract splat parameters in ACTUAL (post-activation) spaces ----
            # positions: raw _xyz are already world coords.
            positions = model._xyz.detach().float()
            # colors: SH DC -> RGB in [0, 1].
            features_dc = model._features_dc.detach().float()  # (N, 3)
            colors = torch.clamp(0.5 + C0 * features_dc, 0.0, 1.0)
            # opacities: sigmoid(_opacity) -> (N,)
            opacities = torch.sigmoid(model._opacity.detach().float()).reshape(-1)
            # scales: exp(_scaling) -> (N, 3) world-unit std-devs
            scales = torch.exp(model._scaling.detach().float())
            # rotations: normalize the stored quaternion (N, 4)
            rotations = torch.nn.functional.normalize(
                model._rotation.detach().float(), dim=-1
            )

            # --- Per-view cameras (SyncDreamer 16-view schedule) ----------------
            cameras = syncdreamer_cameras()

            # --- Real render closure honoring the renderer contract -------------
            # FusionController calls render_fn(positions, colors, opacities,
            # scales, camera) and expects (rendered, depth, alpha, contributions).
            def render_fn(rf_positions, rf_colors, rf_opacities, rf_scales, camera):
                return render_with_contributions(
                    rf_positions,
                    rf_colors,
                    rf_opacities,
                    rf_scales,
                    camera,
                    rotations=rotations,
                )

            # --- Build ViewData per enhanced view -------------------------------
            views_data = []
            for view_idx, (view_name, enhanced_path) in enumerate(enhanced_views.items()):
                try:
                    if view_idx >= len(cameras):
                        self.logger.warning(
                            f"More enhanced views than cameras "
                            f"({view_idx} >= {len(cameras)}); skipping '{view_name}'"
                        )
                        continue

                    camera = cameras[view_idx]
                    H, W = int(camera.height), int(camera.width)

                    # Load enhanced image -> (H, W, 3) float in [0, 1].
                    from PIL import Image
                    enhanced_img = Image.open(enhanced_path).convert('RGB')
                    if enhanced_img.size != (W, H):
                        enhanced_img = enhanced_img.resize((W, H), Image.BICUBIC)
                    enhanced_tensor = (
                        torch.from_numpy(np.array(enhanced_img)).float() / 255.0
                    )  # (H, W, 3)

                    # Render the current splats from this camera to obtain the
                    # rendered image, depth and alpha used by back-projection.
                    rendered_tensor, render_depth, _alpha, _contrib = render_fn(
                        positions, colors, opacities, scales, camera
                    )

                    # Prefer a provided depth map; fall back to the render depth.
                    depth_tensor = render_depth
                    if view_name in depth_maps and Path(depth_maps[view_name]).exists():
                        try:
                            depth_array = np.load(depth_maps[view_name])
                            depth_tensor = torch.from_numpy(depth_array).float()
                            if depth_tensor.shape != (H, W):
                                # Resize via nearest to keep depth values sane.
                                depth_tensor = torch.nn.functional.interpolate(
                                    depth_tensor.reshape(1, 1, *depth_tensor.shape),
                                    size=(H, W),
                                    mode='nearest',
                                ).reshape(H, W)
                        except Exception as de:
                            self.logger.warning(
                                f"Failed to load depth for '{view_name}': {de}; "
                                f"using render depth"
                            )
                            depth_tensor = render_depth

                    views_data.append(ViewData(
                        enhanced_image=enhanced_tensor,
                        rendered_image=rendered_tensor,
                        depth_map=depth_tensor,
                        camera=camera,
                    ))

                except Exception as e:
                    self.logger.warning(f"Failed to prepare view '{view_name}': {e}")
                    continue

            if not views_data:
                self.logger.warning("No valid views for refinement, returning coarse model")
                return {
                    'success': False,
                    'error': 'No valid views prepared',
                    'refined_model_path': coarse_model_path,
                }

            # --- Run the MVCRM refinement loop ----------------------------------
            self.logger.info(f"Starting refinement with {len(views_data)} views")

            result = self.fusion_controller.refine(
                splat_positions=positions,
                splat_colors=colors,
                splat_opacities=opacities,
                splat_scales=scales,
                views=views_data,
                render_fn=render_fn,
            )

            # --- Map refined params back into a GaussianModel (inverse acts) ----
            refined_positions = result.refined_positions.detach().cpu().float()
            refined_colors = torch.clamp(
                result.refined_colors.detach().cpu().float(), 0.0, 1.0
            )
            refined_opacities = torch.clamp(
                result.refined_opacities.detach().cpu().float().reshape(-1), 0.0, 1.0
            )
            refined_scales = result.refined_scales.detach().cpu().float()

            # Inverse activations back into raw storage space.
            model._xyz = refined_positions
            # features_dc = (RGB - 0.5) / C0
            model._features_dc = ((refined_colors - 0.5) / C0).contiguous()
            # _opacity = logit(opacity)  -> stored as (N, 1)
            model._opacity = torch.logit(
                refined_opacities, eps=1e-4
            ).reshape(-1, 1)
            # _scaling = log(scale); guard against non-positive scales.
            model._scaling = torch.log(
                torch.clamp(refined_scales, min=1e-8)
            )
            # _rotation and _features_rest are preserved as-is from the coarse
            # model (the refiner does not update them).

            refined_path = output_path / "refined_model.ply"
            save_ply(model, str(refined_path))

            self.logger.info(f"Refinement completed. Saved to {refined_path}")

            metrics = dict(result.quality_metrics) if result.quality_metrics else {}
            metrics['iterations_run'] = result.iterations_run
            metrics['converged'] = result.converged
            metrics['num_splats'] = num_splats
            metrics['num_views'] = len(views_data)

            return {
                'success': True,
                'refined_model_path': str(refined_path),
                'metrics': metrics,
            }

        except Exception as e:
            error_msg = f"Model refinement failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'refined_model_path': coarse_model_path,  # safe fallback
            }

    async def cleanup(self):
        """Clean up resources."""
        # Fusion controller doesn't need explicit cleanup.
        pass
