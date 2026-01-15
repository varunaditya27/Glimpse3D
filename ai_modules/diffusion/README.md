# ai_modules/diffusion

**SDXL Lightning + ControlNet Depth Enhancement Module**

This module provides diffusion-based image enhancement for the Glimpse3D pipeline, optimized for T4 GPU (15GB VRAM).

## Pipeline Role

```
Single Image → SyncDreamer (16 views) → 3DGS → Render → [This Module] → Refined Views → Back to 3DGS
```

The diffusion module enhances rendered views from the coarse 3D model using:
- **SDXL Lightning**: Fast inference (2-4 steps vs 30-50 for base SDXL)
- **ControlNet Depth**: Structure preservation using depth maps from `midas_depth` module

## Quick Start

### Simple Enhancement

```python
from ai_modules.diffusion import enhance_view

# One-liner enhancement
enhanced = enhance_view("rendered_view.png")
```

### With Depth from midas_depth

```python
from ai_modules.midas_depth import estimate_depth
from ai_modules.diffusion import EnhanceService

# Get depth
depth = estimate_depth("rendered_view.png")

# Enhance with depth guidance
service = EnhanceService(device="cuda")
enhanced = service.enhance(
    image="rendered_view.png",
    depth_map=depth,
    prompt="photorealistic 3D render, detailed texture"
)
```

### Batch Enhancement

```python
from ai_modules.diffusion import EnhanceService, EnhanceConfig

# Optimized for T4 GPU
config = EnhanceConfig.for_t4_gpu()
service = EnhanceService(config=config)

# Enhance 16 views from SyncDreamer
enhanced_views = service.enhance_batch(
    images=rendered_views,
    prompt="high quality 3D model, studio lighting"
)

# Clean up
service.unload()
```

## Integration with midas_depth

This module seamlessly integrates with `midas_depth`:

```python
from ai_modules.midas_depth import (
    estimate_depth,
    estimate_depth_confidence,
    align_depth_scales,
)
from ai_modules.diffusion import EnhanceService

# Estimate depth
depth = estimate_depth("view.png")

# Get confidence for smart blending
confidence = estimate_depth_confidence(depth, rgb_image)

# Enhance with confidence-aware blending
service = EnhanceService()
enhanced = service.enhance_with_depth_confidence(
    "view.png",
    blend_with_original=True,  # Preserve original in low-confidence regions
    confidence_threshold=0.6
)
```

## Module Structure

```
ai_modules/diffusion/
├── __init__.py           # Module exports
├── enhance_service.py    # Main orchestrator (EnhanceService)
├── sdxl_lightning.py     # SDXL Lightning pipeline wrapper
├── controlnet_depth.py   # ControlNet depth integration
├── prompt_builder.py     # Dynamic prompt construction
├── image_utils.py        # Pre/post processing utilities
├── memory_utils.py       # T4 GPU memory optimization
├── prompt_templates.txt  # Prompt templates
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Configuration

### For T4 GPU (Google Colab)

```python
from ai_modules.diffusion import EnhanceConfig, EnhanceService

config = EnhanceConfig.for_t4_gpu()
# Settings:
# - lightning_steps: 4
# - use_controlnet: True
# - optimize_memory: True
# - depth_model: "MiDaS_small"
```

### For Quality Priority

```python
config = EnhanceConfig.for_quality()
# Settings:
# - strength: 0.65 (preserve more original)
# - depth_model: "DPT_Large"
```

### For Speed Priority

```python
config = EnhanceConfig.for_speed()
# Settings:
# - lightning_steps: 2
# - strength: 0.5
```

## Memory Optimization

Optimized for T4 GPU (15GB VRAM):

| Technique | Memory Savings | Notes |
|-----------|---------------|-------|
| FP16 | ~50% | Enabled by default |
| xFormers | ~20-30% | Requires xformers package |
| VAE Slicing | ~200MB | Enabled by default |
| CPU Offload | Fits in <10GB | Slight speed trade-off |

```python
from ai_modules.diffusion.memory_utils import (
    setup_memory_optimization,
    MemoryConfig,
    get_memory_status,
    print_memory_report,
)

# Check available memory
print_memory_report()

# Custom optimization config
config = MemoryConfig.from_preset("t4_balanced")
```

## Prompt Templates

Built-in templates for common 3D content:

```python
from ai_modules.diffusion import PromptBuilder

builder = PromptBuilder(template="photorealistic")
prompt, negative = builder.build(subject="a ceramic vase")

# Available templates:
# - default, photorealistic, product
# - character, environment, object
# - anime_style
```

## ControlNet Models

Supported depth ControlNet models:

| Model | Description |
|-------|-------------|
| `xinsir_depth` | High quality, recommended for SDXL |
| `diffusers_depth` | Official diffusers model |

## Installation

```bash
# In Google Colab
!pip install -q diffusers transformers accelerate xformers
!pip install -q huggingface_hub safetensors

# Or from requirements
!pip install -r ai_modules/diffusion/requirements.txt
```

## Performance on T4 GPU

| Configuration | Time per Image | VRAM Usage |
|--------------|----------------|------------|
| Lightning 4-step + ControlNet | ~8-10s | ~12GB |
| Lightning 2-step + ControlNet | ~5-6s | ~12GB |
| Lightning 4-step (no ControlNet) | ~6-8s | ~8GB |

## Example: Full Pipeline Integration

```python
# Full Glimpse3D enhancement loop
from ai_modules.midas_depth import estimate_depth, align_depth_scales
from ai_modules.diffusion import EnhanceService, EnhanceConfig

# Setup
config = EnhanceConfig.for_t4_gpu()
service = EnhanceService(config=config)

# Enhancement loop
for iteration in range(3):
    # Render views from current 3DGS
    rendered_views = render_from_3dgs(model)
    
    # Estimate depth for each view
    depth_maps = [estimate_depth(v) for v in rendered_views]
    
    # Align depth scales across views
    aligned_depths = align_depth_scales(depth_maps, poses, intrinsics)
    
    # Enhance views
    enhanced_views = service.enhance_batch(
        images=rendered_views,
        depth_maps=aligned_depths,
        prompt="high quality 3D render, detailed texture"
    )
    
    # Update 3DGS with enhanced views
    update_3dgs(model, enhanced_views)

# Cleanup
service.unload()
```

## API Reference

### EnhanceService

Main orchestrator class.

```python
EnhanceService(config: EnhanceConfig = None, device: str = None)
    .load() -> EnhanceService
    .enhance(image, prompt, depth_map, ...) -> Image
    .enhance_batch(images, prompt, depth_maps, ...) -> List[Image]
    .enhance_with_depth_confidence(image, ...) -> Image
    .unload() -> None
```

### SDXLLightningPipeline

Low-level pipeline wrapper.

```python
SDXLLightningPipeline(device, steps, use_controlnet, ...)
    .load() -> SDXLLightningPipeline
    .enhance(image, prompt, depth_map, ...) -> Image
    .unload() -> None
```

### Convenience Functions

```python
enhance_view(image, prompt, depth_map, device) -> Image
enhance_views_batch(images, prompt, depth_maps, device) -> List[Image]
```

## Troubleshooting

### Out of Memory

```python
# Use minimal memory config
from ai_modules.diffusion import EnhanceConfig

config = EnhanceConfig()
config.optimize_memory = True
config.lightning_steps = 2  # Fewer steps
config.use_controlnet = False  # Disable if needed
```

### Slow Inference

```python
# Ensure xformers is installed
!pip install xformers

# Check it's being used
from ai_modules.diffusion.memory_utils import print_memory_report
print_memory_report()
```

### Model Download Issues

```python
# Set HuggingFace cache
import os
os.environ["HF_HOME"] = "/content/hf_cache"
```

## License

Part of the Glimpse3D project.
