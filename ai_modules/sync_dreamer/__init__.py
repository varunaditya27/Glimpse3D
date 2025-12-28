"""
SyncDreamer Module for Glimpse3D
================================

Multi-view consistent image generation from a single image.

This module replaces Zero123 in the Glimpse3D pipeline, providing:
- 16 consistent multi-view images
- Lower VRAM requirements (~12GB vs ~24GB)
- Better cross-view consistency

Quick Start:
    >>> from ai_modules.sync_dreamer import generate_multiview
    >>> paths = generate_multiview("input.png", "outputs/", elevation=30)

For more control:
    >>> from ai_modules.sync_dreamer import SyncDreamerService
    >>> service = SyncDreamerService()
    >>> images = service.generate(image, elevation=30.0)
"""

from .inference import (
    SyncDreamerService,
    get_service,
    generate_multiview,
    cleanup
)

from .utils_syncdreamer import (
    segment_foreground,
    preprocess_for_syncdreamer,
    get_camera_matrices,
    views_to_video,
    create_comparison_grid,
    estimate_elevation,
    get_view_info,
    SYNCDREAMER_CAMERAS
)

__all__ = [
    # Main inference
    "SyncDreamerService",
    "get_service",
    "generate_multiview",
    "cleanup",
    
    # Utilities
    "segment_foreground",
    "preprocess_for_syncdreamer", 
    "get_camera_matrices",
    "views_to_video",
    "create_comparison_grid",
    "estimate_elevation",
    "get_view_info",
    "SYNCDREAMER_CAMERAS"
]

__version__ = "1.0.0"
