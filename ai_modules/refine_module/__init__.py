"""
ai_modules/refine_module

★ MVCRM - Multi-View Consistency Refinement Module ★

The core innovation of Glimpse3D: iteratively refine 3D Gaussian Splat models
by back-projecting AI-enhanced 2D views into 3D space with consistency checks.

Main Components:
- FusionController: Orchestrates the refinement loop
- BackProjector: Maps 2D pixel updates to 3D splat updates
- DepthConsistencyChecker: Prevents geometric violations
- FeatureConsistencyChecker: Ensures semantic consistency
- NormalSmoother: Regularizes geometry to prevent artifacts
- MVCRMEvaluator: Comprehensive quality metrics

Quick Start:
    from ai_modules.refine_module import (
        FusionController,
        RefinementConfig,
        ViewData
    )
    
    # Create controller
    config = RefinementConfig(max_iterations=5, learning_rate=0.05)
    controller = FusionController(config)
    
    # Prepare views
    views = [
        ViewData(enhanced_img, rendered_img, depth, camera)
        for enhanced_img, rendered_img, depth, camera in view_data
    ]
    
    # Refine model
    result = controller.refine(
        splat_positions, splat_colors, splat_opacities, splat_scales,
        views=views,
        render_fn=my_renderer
    )
    
    print(f"Final quality: {result.quality_metrics['final_quality']:.3f}")

Author: Glimpse3D Team
Date: January 2026
"""

__version__ = "1.0.0"

# Core refinement components
from .fusion_controller import (
    FusionController,
    RefinementConfig,
    RefinementResult,
    ViewData,
    create_simple_refinement_config
)

from .back_projector import (
    BackProjector,
    BackProjectionResult,
    CameraParams,
    create_camera_from_pose,
    compute_pixel_to_ray
)

from .depth_consistency import (
    DepthConsistencyChecker,
    DepthConsistencyResult,
    check_consistency,
    compute_depth_confidence
)

from .feature_consistency import (
    FeatureConsistencyChecker,
    FeatureConsistencyResult,
    compute_clip_similarity
)

from .normal_smoothing import (
    NormalSmoother,
    smooth_geometry
)

from .evaluate_mvcrm import (
    MVCRMEvaluator,
    EvaluationMetrics,
    compare_models,
    generate_evaluation_report
)

# Convenience exports
__all__ = [
    # Main controller
    "FusionController",
    "RefinementConfig",
    "RefinementResult",
    "ViewData",
    "create_simple_refinement_config",
    
    # Back-projection
    "BackProjector",
    "BackProjectionResult",
    "CameraParams",
    "create_camera_from_pose",
    "compute_pixel_to_ray",
    
    # Consistency checks
    "DepthConsistencyChecker",
    "DepthConsistencyResult",
    "check_consistency",
    "compute_depth_confidence",
    
    "FeatureConsistencyChecker",
    "FeatureConsistencyResult",
    "compute_clip_similarity",
    
    # Geometry regularization
    "NormalSmoother",
    "smooth_geometry",
    
    # Evaluation
    "MVCRMEvaluator",
    "EvaluationMetrics",
    "compare_models",
    "generate_evaluation_report",
]


def get_version():
    """Get package version."""
    return __version__


def info():
    """Print package information."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  MVCRM - Multi-View Consistency Refinement Module           ║
║  Version: {__version__}                                           ║
║                                                              ║
║  Core Innovation of Glimpse3D                                ║
║                                                              ║
║  Components:                                                 ║
║    • FusionController      - Main orchestrator               ║
║    • BackProjector         - 2D → 3D update mapping          ║
║    • DepthConsistency      - Geometric validation            ║
║    • FeatureConsistency    - Semantic validation             ║
║    • NormalSmoother        - Geometry regularization         ║
║    • MVCRMEvaluator        - Quality metrics                 ║
║                                                              ║
║  Usage:                                                      ║
║    from ai_modules.refine_module import FusionController     ║
║    controller = FusionController()                           ║
║    result = controller.refine(model, views, renderer)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
