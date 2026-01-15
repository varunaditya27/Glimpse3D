# 🧠 Glimpse3D AI Modules

This directory contains the core AI components that power the Glimpse3D pipeline. Each module is designed to be modular and independently testable.

## 📦 Modules Overview

### 1. `sync_dreamer/` (Multi-View Synthesis) ⭐ **PRIMARY**
- **Model**: SyncDreamer
- **Purpose**: Generates **16 multi-view consistent images** from a single input image.
- **Output**: 16 views at fixed elevations (30°, -20°) and azimuths (0°-315°)
- **VRAM**: ~12GB (optimized for consumer GPUs)
- **Usage**: Primary multi-view generator for the Glimpse3D pipeline

```python
from ai_modules.sync_dreamer import generate_multiview

paths = generate_multiview("input.png", "outputs/", elevation=30.0)
```

### 2. ~~`zero123/`~~ (Deprecated)
- **Status**: Replaced by SyncDreamer
- **Reason**: High VRAM requirement (~24GB+), less consistent multi-view output
- **Note**: Code kept for reference/fallback

### 3. `midas_depth/` (Depth Estimation)
- **Model**: MiDaS / ZoeDepth
- **Purpose**: Predicts monocular depth maps from RGB images.
- **Usage**: Provides geometric cues to guide the refinement process and ensure structural consistency.

### 4. `gsplat/` (3D Representation)
- **Library**: `gsplat` (Gaussian Splatting)
- **Purpose**: Handles the creation, rendering, and optimization of 3D Gaussian Splats.
- **Usage**: The core 3D format used for real-time rendering and iterative updates.

### 5. `diffusion/` (Texture Enhancement) ⭐ **ENHANCED**
- **Model**: SDXL Lightning + ControlNet Depth
- **Purpose**: Enhances the visual quality of rendered views with depth-guided structure preservation.
- **Features**:
  - SDXL Lightning (4-step inference vs 30-50 for base SDXL)
  - ControlNet depth conditioning using `midas_depth` module output
  - T4 GPU optimized (fits in 15GB VRAM)
  - Confidence-weighted blending with original
- **VRAM**: ~12GB with optimizations

```python
from ai_modules.midas_depth import estimate_depth
from ai_modules.diffusion import EnhanceService

# Get depth from midas_depth module
depth = estimate_depth("rendered_view.png")

# Enhance with SDXL Lightning + ControlNet
service = EnhanceService(device="cuda")
enhanced = service.enhance(
    image="rendered_view.png",
    depth_map=depth,
    prompt="photorealistic 3D render"
)
```


### 6. `refine_module/` (The Core Innovation)
- **Type**: Custom Algorithm (MVCRM)
- **Purpose**: **Back-Projection Refinement**.
- **Logic**:
    1.  Takes an AI-enhanced 2D image.
    2.  Projects pixel differences back into 3D space.
    3.  Updates the color and opacity of specific Gaussian splats.
    4.  Enforces depth and feature consistency to prevent artifacts.

## 🔄 Pipeline Flow

```
Input Image
    ↓
Background Removal (rembg)
    ↓
┌─────────────────────────────────────┐
│  SyncDreamer → 16 Consistent Views  │  ← Multi-View Generation
└─────────────────────────────────────┘
    ↓
MiDaS Depth → Depth Maps for Each View
    ↓
Gaussian Splatting Reconstruction
    ↓
Enhancement Loop (MVCRM):
  1. Render View from 3DGS
  2. Enhance with SDXL Lightning + ControlNet (depth-guided)
  3. Back-Project into Gaussians
    ↓
Refined 3D Model → Export (.ply / .splat / .glb)
```

## 🔗 Module Integration

The modules are designed to work together seamlessly:

```python
# midas_depth → diffusion integration
from ai_modules.midas_depth import estimate_depth, estimate_depth_confidence
from ai_modules.diffusion import EnhanceService

depth = estimate_depth(rendered_view)        # MiDaS depth
confidence = estimate_depth_confidence(...)   # Confidence map
enhanced = service.enhance(view, depth_map=depth)  # SDXL enhancement
```


## 🛠 Integration

These modules are imported and orchestrated by the **Backend Services** (`backend/app/services/`). They are designed to run on GPU and may require significant VRAM (recommended 12GB+).

## 📥 Model Checkpoints

Most models are **auto-downloaded** from HuggingFace on first run. Manual downloads:

```bash
python scripts/download_models.py
```

| Model | Location | Download |
|-------|----------|----------|
| SyncDreamer | `sync_dreamer/ckpt/` | [Google Drive](https://drive.google.com/file/d/1ypyD5WXxAnsWjnHgAfOAGolV0Zd9kpam/view) (manual) |
| MiDaS | Auto-downloaded | PyTorch Hub |
| SDXL Lightning | Auto-downloaded | HuggingFace (`ByteDance/SDXL-Lightning`) |
| ControlNet Depth | Auto-downloaded | HuggingFace (`xinsir/controlnet-depth-sdxl-1.0`) |

**Note**: SDXL + ControlNet downloads ~10GB on first run. Set cache directory:
```python
import os
os.environ["HF_HOME"] = "/path/to/cache"
```


## 📚 Module Documentation

See individual README files in each module folder for detailed usage instructions.
