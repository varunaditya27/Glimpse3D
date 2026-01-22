"""
Glimpse3D AI Modules
====================

This package contains the AI pipeline components for Glimpse3D:
- 2D Image → 3D Gaussian Splat conversion

Modules:
    - gsplat: Gaussian Splatting reconstruction and rendering
    - sync_dreamer: Multi-view image generation  
    - diffusion: SDXL Lightning + ControlNet enhancement
    - midas_depth: Depth estimation
    - refine_module: MVCRM multi-view consistent refinement
    - colab_utils: Google Colab utilities

For Google Colab usage, see the notebooks/ directory.
"""

# Version
__version__ = "0.2.0"

# Colab utilities (always available)
try:
    from .colab_utils import (
        setup_colab_environment,
        check_dependencies,
        print_dependencies,
        clear_gpu_memory,
        GaussianSplatIO,
        CameraUtils,
        mesh_to_gaussian_ply,
        IN_COLAB,
    )
except ImportError:
    pass

# Lazy imports for heavy modules
def get_gsplat_reconstructor():
    """Get the TripoSR-based reconstructor."""
    from .gsplat.reconstruct import (
        load_triposr_model,
        preprocess_image,
        run_inference,
        sample_and_export,
    )
    return load_triposr_model, preprocess_image, run_inference, sample_and_export


def get_syncdreamer_service():
    """Get the SyncDreamer service."""
    from .sync_dreamer import SyncDreamerService, generate_multiview
    return SyncDreamerService, generate_multiview


def get_diffusion_enhancer():
    """Get the SDXL Lightning enhancer."""
    from .diffusion.enhance_service import EnhanceService
    return EnhanceService


__all__ = [
    '__version__',
    # Colab utilities
    'setup_colab_environment',
    'check_dependencies', 
    'print_dependencies',
    'clear_gpu_memory',
    'GaussianSplatIO',
    'CameraUtils',
    'mesh_to_gaussian_ply',
    'IN_COLAB',
    # Lazy loaders
    'get_gsplat_reconstructor',
    'get_syncdreamer_service',
    'get_diffusion_enhancer',
]
