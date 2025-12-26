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

__all__ = [
    "DepthEstimator",
    "estimate_depth",
    "get_estimator",
    "save_depth_visualization",
    "save_depth_grayscale",
    "save_depth_raw",
    "load_depth_raw",
]

__version__ = "1.0.0"
