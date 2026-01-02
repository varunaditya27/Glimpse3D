# 🎯 SyncDreamer Module for Glimpse3D

Multi-view consistent image generation from a single input image.

## Overview

SyncDreamer generates **16 consistent multi-view images** from a single input image, which are then used by the Glimpse3D pipeline for 3D reconstruction.

## 📓 Inference Notebooks

| Notebook | Description |
|----------|-------------|
| [Glimpse3D_SyncDreamer_Inference.ipynb](../../notebooks/Glimpse3D_SyncDreamer_Inference.ipynb) | **Recommended** - Uses this module directly. Run via VS Code + Colab Extension |
| [SyncDreamer_Colab_Inference.ipynb](../../notebooks/SyncDreamer_Colab_Inference.ipynb) | Standalone - Clones original SyncDreamer repo |

### Why SyncDreamer over Zero123?

| Aspect | Zero123 | SyncDreamer |
|--------|---------|-------------|
| **VRAM Usage** | ~24GB+ | ~12GB |
| **Multi-view Consistency** | Per-view generation (can be inconsistent) | Synchronized generation (consistent) |
| **Output** | Single view at a time | 16 views simultaneously |
| **Speed** | Multiple passes needed | Single pass for all views |

## Installation

### Prerequisites

```bash
pip install torch torchvision omegaconf pytorch-lightning
pip install rembg  # For background removal (optional but recommended)
```

### Download Checkpoints

You need **two files** for SyncDreamer to work:

#### 1. SyncDreamer Model (~5.2GB)
**Google Drive:** https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view

#### 2. CLIP ViT-L-14 Encoder (~890MB) ⚠️ Required
**Hugging Face:** https://huggingface.co/camenduru/SyncDreamer/resolve/main/ViT-L-14.pt

Download via PowerShell:
```powershell
# Download ViT-L-14.pt
Invoke-WebRequest -Uri "https://huggingface.co/camenduru/SyncDreamer/resolve/main/ViT-L-14.pt" -OutFile "ai_modules/sync_dreamer/ckpt/ViT-L-14.pt"
```

Or via curl:
```bash
curl -L https://huggingface.co/camenduru/SyncDreamer/resolve/main/ViT-L-14.pt -o ai_modules/sync_dreamer/ckpt/ViT-L-14.pt
```

#### Final checkpoint folder:
```
ai_modules/sync_dreamer/ckpt/
├── syncdreamer-pretrain.ckpt  (~5.2GB)
└── ViT-L-14.pt                (~890MB)
```

## Usage

### Quick Start

```python
from ai_modules.sync_dreamer import generate_multiview

# Generate 16 views from a single image
output_paths = generate_multiview(
    image_path="input.png",      # RGBA image with transparent background
    output_dir="outputs/views",
    elevation=30.0,              # Input camera elevation (degrees)
    seed=42
)
print(f"Generated {len(output_paths)} views")
```

### Using the Service Class

```python
from ai_modules.sync_dreamer import SyncDreamerService

# Initialize service
service = SyncDreamerService()

# Load model (done automatically on first generate call)
service.load_model()

# Generate views
images = service.generate(
    image="input.png",
    elevation=30.0,
    cfg_scale=2.0,
    sample_steps=50,
    batch_view_num=8  # Lower this if you have less VRAM
)

# Save with custom naming
for i, img in enumerate(images):
    img.save(f"view_{i:02d}.png")

# Free GPU memory when done
service.unload_model()
```

### Background Removal

For best results, input images should have transparent backgrounds:

```python
from ai_modules.sync_dreamer import segment_foreground
from PIL import Image

# Remove background
image = Image.open("photo.jpg")
rgba_image = segment_foreground(image, method="rembg")
rgba_image.save("input.png")
```

## Output Views

SyncDreamer generates 16 views at fixed camera positions:

| Views 0-7 | Views 8-15 |
|-----------|------------|
| Elevation: 30° | Elevation: -20° |
| Azimuths: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° | Same azimuths |

```
View Layout (4x4 grid):
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │  ← Elevation 30°
├────┼────┼────┼────┤
│ 4  │ 5  │ 6  │ 7  │  ← Elevation 30°
├────┼────┼────┼────┤
│ 8  │ 9  │ 10 │ 11 │  ← Elevation -20°
├────┼────┼────┼────┤
│ 12 │ 13 │ 14 │ 15 │  ← Elevation -20°
└────┴────┴────┴────┘
```

## API Reference

### `generate_multiview(image_path, output_dir, **kwargs)`

Quick function for multi-view generation.

**Parameters:**
- `image_path` (str): Path to input image
- `output_dir` (str): Directory for output images
- `elevation` (float): Input view elevation, default 30.0
- `crop_size` (int): Foreground crop size, default 200
- `cfg_scale` (float): Guidance scale, default 2.0
- `seed` (int): Random seed, default 42

**Returns:** List of output file paths

### `SyncDreamerService`

Main service class for inference.

**Methods:**
- `load_model()`: Load model to GPU
- `unload_model()`: Free GPU memory
- `generate(image, **kwargs)`: Generate 16 views
- `generate_and_save(image, output_dir, **kwargs)`: Generate and save to disk

### Utility Functions

- `segment_foreground(image)`: Remove background
- `preprocess_for_syncdreamer(image)`: Prepare image for inference
- `get_camera_matrices(elevations, azimuths)`: Get camera transforms
- `views_to_video(image_paths, output_path)`: Create turntable video

## Integration with Glimpse3D Pipeline

```
Input Image
    ↓
Background Removal (segment_foreground)
    ↓
SyncDreamer → 16 Consistent Views
    ↓
MiDaS Depth → Depth Maps
    ↓
Gaussian Splatting Reconstruction
    ↓
SDXL Enhancement Loop
    ↓
Refined 3D Model
```

## VRAM Optimization

For GPUs with limited VRAM:

```python
# Use smaller batch size
images = service.generate(
    image,
    batch_view_num=4,  # Default is 8, use 4 for <12GB VRAM
    sample_num=1       # Generate 1 set instead of multiple
)
```

## File Structure

```
ai_modules/sync_dreamer/
├── __init__.py           # Module exports
├── inference.py          # Main inference service
├── utils_syncdreamer.py  # Utility functions
├── README.md             # This file
├── ckpt/                 # Model checkpoints
│   ├── syncdreamer-pretrain.ckpt
│   └── ViT-L-14.pt
├── configs/
│   └── syncdreamer.yaml  # Model configuration
└── ldm/                  # Core model code (from SyncDreamer repo)
    ├── util.py
    ├── base_utils.py
    ├── models/
    │   └── diffusion/
    └── modules/
```

## Troubleshooting

### "Checkpoint not found"
Download both checkpoints:
- `syncdreamer-pretrain.ckpt` from [Google Drive](https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view)
- `ViT-L-14.pt` from [Hugging Face](https://huggingface.co/camenduru/SyncDreamer/resolve/main/ViT-L-14.pt)

### "CUDA out of memory"
- Reduce `batch_view_num` to 4 or 2
- Use `sample_num=1`
- Close other GPU applications

### "No module named 'ldm'"
The ldm module should be in `ai_modules/sync_dreamer/ldm/`. Check that all files were copied correctly.

## License

SyncDreamer is released under the MIT License.
Original repository: https://github.com/liuyuan-pal/SyncDreamer

## Citation

```bibtex
@article{liu2023syncdreamer,
  title={SyncDreamer: Generating Multiview-consistent Images from a Single-view Image},
  author={Liu, Yuan and Lin, Cheng and Zeng, Zijiao and Long, Xiaoxiao and Liu, Lingjie and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2309.03453},
  year={2023}
}
```
