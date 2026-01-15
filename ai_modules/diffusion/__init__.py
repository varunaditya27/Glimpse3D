"""
ai_modules/diffusion

SDXL Lightning + ControlNet Depth Enhancement Module for Glimpse3D Pipeline.

This module provides diffusion-based image enhancement capabilities optimized
for T4 GPU (15GB VRAM). It integrates with the midas_depth module for
depth-guided enhancement of rendered 3D views.

Pipeline Role:
    SyncDreamer (16 views) → 3DGS → Render → [This Module] → Refined Views

Key Features:
- SDXL Lightning for fast inference (2-4 steps vs 30-50)
- ControlNet depth conditioning for structure preservation
- T4 GPU memory optimization (CPU offloading, VAE slicing, xformers)
- Seamless integration with midas_depth module

Quick Usage:
    from ai_modules.diffusion import enhance_view
    from ai_modules.midas_depth import estimate_depth
    
    # Get depth from midas_depth module
    depth = estimate_depth("rendered_view.png")
    
    # Enhance with SDXL + ControlNet
    enhanced = enhance_view(
        image="rendered_view.png",
        depth_map=depth,
        prompt="high quality, detailed texture"
    )

Advanced Usage:
    from ai_modules.diffusion import EnhanceService
    
    service = EnhanceService(device="cuda", optimize_memory=True)
    enhanced_views = service.enhance_batch(
        images=rendered_views,
        depth_maps=depth_maps,
        prompt="photorealistic, 8k resolution"
    )
"""

from .memory_utils import (
    setup_memory_optimization,
    get_memory_status,
    clear_gpu_memory,
    estimate_vram_usage,
    MemoryConfig,
    print_memory_report,
)

from .image_utils import (
    preprocess_for_diffusion,
    postprocess_from_diffusion,
    resize_with_aspect,
    blend_images,
    match_histogram,
)

from .prompt_builder import (
    PromptBuilder,
    build_prompt,
    load_prompt_templates,
)

from .controlnet_depth import (
    prepare_depth_for_controlnet,
    DepthControlNetAdapter,
    CONTROLNET_MODELS,
)

from .sdxl_lightning import (
    SDXLLightningPipeline,
    load_sdxl_pipeline,
    get_recommended_settings,
)

from .enhance_service import (
    EnhanceService,
    enhance_view,
    enhance_views_batch,
    EnhanceConfig,
)

__all__ = [
    # Memory optimization (critical for T4 GPU)
    "setup_memory_optimization",
    "get_memory_status",
    "clear_gpu_memory",
    "estimate_vram_usage",
    "MemoryConfig",
    "print_memory_report",
    # Image processing
    "preprocess_for_diffusion",
    "postprocess_from_diffusion",
    "resize_with_aspect",
    "blend_images",
    "match_histogram",
    # Prompt building
    "PromptBuilder",
    "build_prompt",
    "load_prompt_templates",
    # ControlNet depth (integrates with midas_depth)
    "prepare_depth_for_controlnet",
    "DepthControlNetAdapter",
    "CONTROLNET_MODELS",
    # SDXL Lightning pipeline
    "SDXLLightningPipeline",
    "load_sdxl_pipeline",
    "get_recommended_settings",
    # Main service (orchestrator)
    "EnhanceService",
    "enhance_view",
    "enhance_views_batch",
    "EnhanceConfig",
]

__version__ = "1.0.0"
