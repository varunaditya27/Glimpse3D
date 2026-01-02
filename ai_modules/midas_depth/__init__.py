"""
ai_modules/midas_depth

Monocular Depth Estimation module using MiDaS.

This module provides depth estimation capabilities for the Glimpse3D pipeline.
Depth maps are used for:
- Guiding 3D reconstruction geometry
- Ensuring consistency during refinement
- Back-projection of enhanced views

Quick Usage:
    from ai_modules.midas_depth import estimate_depth, save_depth_visualization
    
    depth = estimate_depth("image.jpg")
    save_depth_visualization(depth, "depth.png")

Advanced Usage:
    from ai_modules.midas_depth import DepthEstimator
    
    estimator = DepthEstimator(model_type="DPT_Large", device="cuda")
    depth = estimator.estimate("image.jpg")
"""

from .run_depth import (
    DepthEstimator,
    estimate_depth,
    get_estimator,
    save_depth_visualization,
    save_depth_grayscale,
    save_depth_raw,
    load_depth_raw,
)

from .depth_alignment import (
    align_depth_scales,
    align_depth_scales_global,
    compute_overlap_mask,
    estimate_overlap_simple,
    compute_alignment_quality,
)

from .depth_confidence import (
    estimate_depth_confidence,
    apply_confidence_mask,
    weighted_depth_fusion,
    visualize_confidence,
    get_reliable_depth_mask,
)

__all__ = [
    # Core depth estimation
    "DepthEstimator",
    "estimate_depth",
    "get_estimator",
    "save_depth_visualization",
    "save_depth_grayscale",
    "save_depth_raw",
    "load_depth_raw",
    # Novel: Multi-view alignment
    "align_depth_scales",
    "align_depth_scales_global",
    "compute_overlap_mask",
    "estimate_overlap_simple",
    "compute_alignment_quality",
    # Novel: Confidence estimation
    "estimate_depth_confidence",
    "apply_confidence_mask",
    "weighted_depth_fusion",
    "visualize_confidence",
    "get_reliable_depth_mask",
]

__version__ = "1.2.0"
