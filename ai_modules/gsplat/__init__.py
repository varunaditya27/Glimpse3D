"""
ai_modules/gsplat

Gaussian Splatting module for Glimpse3D.

This module provides:
- TripoSR-based 3D reconstruction (Image → Mesh → Gaussian Splats)
- Gaussian model representation and PLY I/O
- Optimization loop for multi-view refinement
- Camera utilities for rendering

Quick Usage:
    from ai_modules.gsplat import load_ply, save_ply, GaussianModel
    
    model = load_ply("splat.ply")
    model.to("cuda")

Reconstruction:
    from ai_modules.gsplat import (
        load_triposr_model, preprocess_image, 
        run_inference, sample_and_export
    )
    
    model = load_triposr_model()
    img = preprocess_image("input.png")
    mesh = run_inference(model, img)
    sample_and_export(mesh, "output.ply")

Training/Optimization:
    from ai_modules.gsplat import train_pipeline, refine_model
    
    train_pipeline("input.png", "output_dir/", iterations=100)
"""

# Core PLY I/O and Model (primary)
from .utils_gs import (
    GaussianModel,
    load_ply,
    save_ply,
)

# Reconstruction pipeline
from .reconstruct import (
    load_triposr_model,
    preprocess_image,
    run_inference,
    sample_and_export,
)

# Camera utilities
from .camera_utils import (
    build_intrinsics,
)

# Optimization
from .optimize import (
    refine_model,
)

# Training pipeline
from .train import (
    train_pipeline,
)

__all__ = [
    # Model and I/O
    "GaussianModel",
    "load_ply",
    "save_ply",
    # Reconstruction
    "load_triposr_model",
    "preprocess_image", 
    "run_inference",
    "sample_and_export",
    # Camera
    "build_intrinsics",
    # Optimization
    "refine_model",
    # Training
    "train_pipeline",
]

__version__ = "1.0.0"
