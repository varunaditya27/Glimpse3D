# 📚 Glimpse3D Colab Notebooks Guide

This directory contains Jupyter notebooks for running the complete Glimpse3D pipeline on **Google Colab**.

## 🎯 Quick Start

**For full pipeline (recommended):** Run `Glimpse3D_Master_Pipeline.ipynb`

**For individual stages:** Run notebooks in order 1-5.

---

## 📓 Notebook Overview

### 1. `Glimpse3D_TripoSR_Reconstruction.ipynb`
**Stage:** Initial 3D Reconstruction  
**Input:** Single 2D image (JPG/PNG)  
**Output:** 3D mesh (OBJ/GLB) + Gaussian PLY  
**Time:** ~30 seconds  
**VRAM:** ~6 GB  

TripoSR converts a single image into a textured 3D mesh in under a second. This notebook then samples points from the mesh to create the initial Gaussian splat representation.

**Key Features:**
- Automatic background removal with rembg
- Mesh export in OBJ, GLB, and PLY formats
- Conversion to Gaussian Splat PLY format
- Optional preview video rendering

---

### 2. `Glimpse3D_SyncDreamer_Inference.ipynb`
**Stage:** Multi-View Generation  
**Input:** Processed image (RGBA with transparent background)  
**Output:** 16 consistent multi-view images  
**Time:** ~2-3 minutes  
**VRAM:** ~12 GB  

SyncDreamer generates 16 photorealistic views of your object from different angles:
- 8 views at elevation 30° (every 45° azimuth)
- 8 views at elevation -20° (every 45° azimuth)

**Key Features:**
- Automatic checkpoint download (~6 GB)
- Background removal preprocessing
- Grid visualization of all views
- Turntable GIF generation

---

### 3. `Glimpse3D_GSplat_Optimization.ipynb`
**Stage:** Gaussian Splat Optimization  
**Input:** Initial Gaussian PLY + Multi-view images  
**Output:** Optimized Gaussian PLY  
**Time:** ~5 minutes (1000 iterations)  
**VRAM:** ~8 GB  

Uses the gsplat library to optimize Gaussian splat parameters using multi-view supervision from SyncDreamer.

**Key Features:**
- Differentiable rendering with gsplat
- Per-parameter learning rates
- Real-time loss visualization
- 360° video generation

---

### 4. `Glimpse3D_Diffusion_Enhancement.ipynb` (existing)
**Stage:** View Enhancement  
**Input:** Multi-view images  
**Output:** Enhanced multi-view images  
**Time:** ~1 minute per image  
**VRAM:** ~12 GB  

Uses SDXL Lightning + ControlNet to enhance rendered views with improved textures and details.

---

### 5. `Glimpse3D_MVCRM_Refinement.ipynb`
**Stage:** Multi-View Consistent Refinement  
**Input:** Optimized Gaussian PLY + Enhanced images  
**Output:** Final refined Gaussian PLY  
**Time:** ~5 minutes  
**VRAM:** ~10 GB  

**★ Novel Contribution ★**  
The MVCRM module back-projects 2D diffusion enhancements into 3D space while maintaining multi-view consistency.

**Key Features:**
- SyncDreamer camera alignment
- Gradient-based back-projection
- Multi-view loss aggregation
- Comparison visualization

---

### 6. `Glimpse3D_Master_Pipeline.ipynb`
**Stage:** Complete End-to-End Pipeline  
**Input:** Single 2D image  
**Output:** Final 3D Gaussian Splat + Mesh + Video  
**Time:** ~30 minutes  
**VRAM:** ~12 GB peak  

Runs all stages sequentially with automatic memory management between stages.

**Key Features:**
- One-click full pipeline
- Automatic GPU detection and configuration
- Memory cleanup between stages
- Progress tracking
- Final ZIP download

---

## 🖥️ GPU Requirements

| Notebook | T4 (15GB) | A100 (40GB) | Notes |
|----------|-----------|-------------|-------|
| TripoSR | ✅ | ✅ | Works on free tier |
| SyncDreamer | ✅ | ✅ | Use batch_view_num=4 on T4 |
| gsplat | ✅ | ✅ | Confirmed by maintainers |
| SDXL Enhancement | ✅ | ✅ | Optional stage |
| MVCRM | ✅ | ✅ | |
| Master Pipeline | ✅ | ✅ | Memory managed per stage |

---

## 📦 Dependencies

All notebooks auto-install dependencies. Key packages:
- `torch`, `torchvision` - PyTorch
- `gsplat` - Gaussian Splatting rendering
- `diffusers` - SDXL models
- `rembg` - Background removal
- `plyfile` - PLY file I/O
- `torchmcubes` - Marching cubes for mesh extraction

---

## 💡 Tips for Best Results

1. **Input Images:**
   - High resolution (512x512 or larger)
   - Clean background or solid color
   - Object centered, filling 80% of frame
   - Avoid reflective/transparent surfaces

2. **GPU Memory:**
   - Use `batch_view_num=4` for T4 GPU
   - Run memory cleanup between stages
   - Close other Colab tabs

3. **Quality:**
   - Increase `num_iterations` in gsplat (2000+)
   - Use SDXL enhancement for textures
   - Run MVCRM for consistency

---

## 🔗 Related Resources

- [TripoSR Paper](https://arxiv.org/abs/2403.02151)
- [SyncDreamer Paper](https://arxiv.org/abs/2309.03453)
- [gsplat Documentation](https://docs.gsplat.studio/)
- [Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

---

## 📝 Citation

If you use Glimpse3D in your research, please cite:

```bibtex
@software{glimpse3d2025,
  title = {Glimpse3D: 2D Image to 3D Gaussian Splat Pipeline},
  year = {2025},
  url = {https://github.com/varunaditya27/Glimpse-3D}
}
```
