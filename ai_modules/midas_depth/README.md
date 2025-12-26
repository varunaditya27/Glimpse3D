# MiDaS Depth Estimation Module

This module provides monocular depth estimation for the Glimpse3D pipeline using Intel's MiDaS models.

## 📋 Overview

**What it does**: Takes a single RGB image and outputs a depth map showing how far each pixel is from the camera.

**Role in Glimpse3D**: 
- Provides geometric guidance for 3D reconstruction
- Ensures depth consistency during the refinement loop
- Supports back-projection of enhanced views into the 3D Gaussian Splat model

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Glimpse3D/ai_modules/midas_depth
pip install -r requirements.txt
```

**For GPU acceleration (recommended):**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Quick Test

```bash
python test_depth.py
```

This creates a test image and verifies the module works correctly.

### 3. CLI Usage

```bash
# Basic usage
python run_depth.py -i path/to/image.jpg -o output_folder

# With specific model
python run_depth.py -i image.jpg -o output -m DPT_Large

# Save raw numpy file
python run_depth.py -i image.jpg -o output --save-raw

# Use CPU explicitly
python run_depth.py -i image.jpg -o output -d cpu
```

## 📦 Python API

### Simple Usage

```python
from ai_modules.midas_depth import estimate_depth, save_depth_visualization

# Estimate depth
depth = estimate_depth("photo.jpg")

# Save visualization
save_depth_visualization(depth, "depth_colored.png")
```

### Advanced Usage

```python
from ai_modules.midas_depth import DepthEstimator

# Create estimator with specific settings
estimator = DepthEstimator(
    model_type="DPT_Large",  # Higher quality
    device="cuda",           # Use GPU
    optimize=True            # Half precision for speed
)

# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
depths = estimator.estimate_batch(images)

# Process PIL Image or numpy array
from PIL import Image
img = Image.open("photo.jpg")
depth = estimator.estimate(img)
```

## 🔧 Available Models

| Model | Quality | Speed | VRAM | Use Case |
|-------|---------|-------|------|----------|
| `MiDaS_small` | ★★☆ | ★★★ | ~500MB | Development, quick tests |
| `DPT_Hybrid` | ★★★ | ★★☆ | ~1GB | Balanced quality/speed |
| `DPT_Large` | ★★★★ | ★☆☆ | ~1.5GB | Production, best quality |

## 📊 Output Format

The module outputs:
- **Normalized depth map**: NumPy array `(H, W)`, `float32`, values in `[0, 1]`
  - `1.0` = Closest to camera
  - `0.0` = Farthest from camera

### Saving Options

```python
from ai_modules.midas_depth import (
    save_depth_visualization,  # Colored PNG (magma colormap)
    save_depth_grayscale,      # Grayscale PNG
    save_depth_raw,            # NumPy .npy file (full precision)
)
```

## 🧪 Testing

```bash
# Run all tests
pytest test_depth.py -v

# Run with coverage
pytest test_depth.py --cov=. --cov-report=html

# Quick manual test
python test_depth.py
```

## 📁 Files

```
midas_depth/
├── __init__.py       # Module exports
├── run_depth.py      # Core implementation + CLI
├── test_depth.py     # Unit tests
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🔗 Integration with Pipeline

This module is called by `backend/app/services/depth_service.py`:

```python
from ai_modules.midas_depth import estimate_depth

class DepthService:
    def get_depth(self, image_path: str):
        return estimate_depth(image_path, model_type="DPT_Hybrid")
```

## ⚠️ Troubleshooting

### "CUDA out of memory"
- Use `MiDaS_small` model
- Set `device="cpu"`
- Reduce image size before processing

### "torch.hub download fails"
- Check internet connection
- Try: `torch.hub.set_dir("path/to/cache")`

### Depth looks inverted
- MiDaS outputs inverse depth (higher = closer)
- This is already handled; `1.0` = closest after normalization

## 📚 References

- [MiDaS Paper](https://arxiv.org/abs/1907.01341)
- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)
