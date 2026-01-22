<div align="center">

# **✨ Glimpse3D**

### **AI System for High‑Quality 3D Gaussian Splats From a Single Image**

Transform *one photo* into a **production-ready 3D Gaussian Splat model** using state-of-the-art AI.

**TripoSR × SyncDreamer × SDXL Lightning × gsplat × MiDaS × MVCRM**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varunaditya27/Glimpse3D/blob/main/notebooks/Glimpse3D_Master_Pipeline.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

# **📌 Overview**

Glimpse3D is a **production-ready AI pipeline** that converts a **single 2D image** into a **high-quality 3D Gaussian Splat model** in approximately 30 minutes on a free Google Colab T4 GPU.

### **Complete Pipeline**

```
📷 Input Image
    ↓
🔷 TripoSR (30s) → Initial 3D Mesh → Gaussian Points
    ↓
🎨 SyncDreamer (2-3min) → 16 Consistent Multi-View Images
    ↓  
✨ SDXL Lightning + ControlNet → Enhanced Views (Optional)
    ↓
🔮 gsplat Optimization (5min) → Refined Gaussians
    ↓
🔄 MVCRM → Multi-View Consistent Refinement
    ↓
🏆 Final 3D Gaussian Splat Output (.ply, .glb, .mp4)
```

### **AI Modules Integrated**

| Module | Purpose | Source | Status |
|--------|---------|--------|--------|
| **TripoSR** | Fast single-image 3D reconstruction | [VAST-AI-Research](https://github.com/VAST-AI-Research/TripoSR) | ✅ Verified |
| **SyncDreamer** | Multi-view consistent image generation | [liuyuan-pal](https://github.com/liuyuan-pal/SyncDreamer) | ✅ Verified |
| **SDXL Lightning** | 4-step diffusion enhancement | [ByteDance](https://huggingface.co/ByteDance/SDXL-Lightning) | ✅ Verified |
| **gsplat** | Gaussian splatting optimization | [nerfstudio-project](https://github.com/nerfstudio-project/gsplat) | ✅ Verified |
| **MiDaS** | Monocular depth estimation | [isl-org](https://github.com/isl-org/MiDaS) | ✅ Integrated |
| **MVCRM** | Multi-view consistency refinement | Custom | ✅ Integrated |

This makes Glimpse3D both:

* a **research platform** for multi-view consistency & 3D reconstruction, and
* a **production-ready system** for designers, students, and developers.

---

# **🚀 Quick Start (Google Colab)**

The fastest way to try Glimpse3D is via our **production-ready Colab notebook**:

1. **Open the notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varunaditya27/Glimpse3D/blob/main/notebooks/Glimpse3D_Master_Pipeline.ipynb)
2. **Select GPU runtime**: Runtime → Change runtime type → T4 GPU
3. **Run all cells**: Runtime → Run all
4. **Upload your image** when prompted
5. **Download results** (~30 minutes total)

### **Requirements**
- Google Colab with **T4 GPU** (free tier) or **A100** (faster)
- ~12GB VRAM peak usage
- ~30 minutes total runtime

---

# **🌟 Key Features**

### **✔ Single Image → 3D Gaussian Splats**

Generate a complete 3D Gaussian Splat model from just one photo.

### **✔ Multi‑View Consistency (SyncDreamer)**

Generate 16 geometrically consistent views at 30° elevation with 22.5° azimuth spacing.

### **✔ Lightning-Fast Enhancement (SDXL Lightning)**

4-step diffusion enhancement for sharper textures and improved realism.

### **✔ Production-Ready gsplat Integration**

Optimized Gaussian splatting with correct opacity shapes and quaternion conventions.

### **✔ Depth‑Aware Processing (MiDaS)**

Monocular depth estimation for geometry-aware refinement.

### **✔ Multi-View Consistent Refinement (MVCRM)**

Novel refinement module with depth consistency, normal smoothing, and feature fusion.

### **✔ Multiple Export Formats**

Export as `.ply` (Gaussian Splats), `.glb`/`.obj` (mesh), and `.mp4` (360° video).

---

# **🧠 Architecture**

```
Single Image
     ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: TripoSR                                           │
│  - Background removal (rembg)                               │
│  - 3D mesh reconstruction (~30s)                            │
│  - Mesh → Gaussian point cloud conversion                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: SyncDreamer                                       │
│  - 16 consistent multi-view images                          │
│  - Fixed 30° elevation, 22.5° azimuth spacing               │
│  - Synchronized attention for consistency                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: SDXL Lightning (Optional)                         │
│  - 4-step diffusion enhancement                             │
│  - guidance_scale=0, timestep_spacing="trailing"            │
│  - Sharper textures, improved details                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: gsplat Optimization                               │
│  - Multi-view photometric loss                              │
│  - 1000+ iterations of gradient descent                     │
│  - Opacity shape: [N], quaternions: wxyz format             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: MVCRM Refinement (Optional)                       │
│  - Depth consistency checking                               │
│  - Normal smoothing                                         │
│  - Feature-based fusion                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
Final Output: .ply (Gaussians) + .glb (mesh) + .mp4 (video)
```

---

# **📁 Project Structure**

```
Glimpse3D/
├── notebooks/              # 📓 Colab notebooks (START HERE)
│   ├── Glimpse3D_Master_Pipeline.ipynb    # Complete end-to-end pipeline
│   ├── Glimpse3D_TripoSR_Reconstruction.ipynb
│   ├── Glimpse3D_SyncDreamer_Inference.ipynb
│   ├── Glimpse3D_Diffusion_Enhancement.ipynb
│   ├── Glimpse3D_GSplat_Optimization.ipynb
│   └── Glimpse3D_MVCRM_Refinement.ipynb
│
├── ai_modules/             # 🧠 Core AI modules
│   ├── sync_dreamer/       # SyncDreamer multi-view generation
│   ├── gsplat/             # Gaussian splatting optimization
│   ├── midas_depth/        # MiDaS depth estimation
│   ├── diffusion/          # SDXL Lightning enhancement
│   ├── refine_module/      # MVCRM refinement
│   └── zero123/            # Zero-123 (alternative to SyncDreamer)
│
├── frontend/               # 🖥️ React + Three.js UI
├── backend/                # ⚙️ FastAPI server
├── model_checkpoints/      # 📦 Pretrained model weights
├── assets/                 # 🖼️ Sample inputs/outputs
├── docs/                   # 📚 Documentation
│   ├── CRITICAL_REVIEW_REPORT.md   # Production readiness analysis
│   └── pipeline_flow.md
├── scripts/                # 🔧 Setup & automation
├── research/               # 🔬 Experiments & metrics
└── docker/                 # 🐳 Deployment files
```

---

# **🚀 Local Installation**

## **1. Clone the repo**

```bash
git clone https://github.com/varunaditya27/Glimpse3D.git
cd Glimpse3D
```

## **2. Create environment**

```bash
# Using conda (recommended)
conda create -n glimpse3d python=3.10
conda activate glimpse3d

# Install PyTorch with CUDA
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## **3. Install dependencies**

```bash
# Core dependencies
pip install transformers==4.40.0 diffusers==0.27.2 accelerate==0.28.0
pip install omegaconf==2.3.0 einops==0.7.0 pytorch-lightning==1.9.5
pip install gsplat==1.2.0 trimesh==4.2.0 rembg[gpu]==2.0.55

# SyncDreamer dependencies
pip install git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
```

## **4. Download model checkpoints**

```bash
python scripts/download_models.py
```

## **5. Run the pipeline**

```bash
# Option A: Jupyter notebook
jupyter notebook notebooks/Glimpse3D_Master_Pipeline.ipynb

# Option B: Backend API
cd backend
uvicorn app.main:app --reload
```

---

# **🧩 AI Module Details**

### **TripoSR** — Fast 3D Reconstruction
- **Source**: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **Model**: `stabilityai/TripoSR` on HuggingFace
- **API**: `TSR.from_pretrained()`, `remove_background()`, `resize_foreground()`
- **Output**: 3D mesh with vertex colors

### **SyncDreamer** — Multi-View Generation
- **Source**: [liuyuan-pal/SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) (ICLR 2024 Spotlight)
- **Config**: 16 views, 30° elevation, 22.5° azimuth spacing
- **API**: `prepare_inputs()`, `SyncDDIMSampler`, `model.sample()`
- **Output**: 16 consistent 256×256 images

### **SDXL Lightning** — Fast Enhancement
- **Source**: [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)
- **Config**: 4-step UNet, `guidance_scale=0`, `timestep_spacing="trailing"`
- **API**: `UNet2DConditionModel.from_config()` + `load_state_dict()`
- **Output**: Enhanced images with sharper details

### **gsplat** — Gaussian Splatting
- **Source**: [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **Version**: 1.2.0
- **API**: `rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height)`
- **Critical**: Opacity shape must be `[N]` (1D), quaternions in wxyz format

### **MVCRM** — Refinement Module
- **Location**: `ai_modules/refine_module/`
- **Components**: Depth consistency, normal smoothing, feature fusion
- **Purpose**: Multi-view consistent refinement of Gaussian splats

---

# **� Notebooks**

| Notebook | Purpose | Runtime |
|----------|---------|---------|
| `Glimpse3D_Master_Pipeline.ipynb` | **Complete end-to-end pipeline** | ~30 min |
| `Glimpse3D_TripoSR_Reconstruction.ipynb` | TripoSR mesh generation only | ~2 min |
| `Glimpse3D_SyncDreamer_Inference.ipynb` | Multi-view generation only | ~5 min |
| `Glimpse3D_Diffusion_Enhancement.ipynb` | SDXL enhancement only | ~3 min |
| `Glimpse3D_GSplat_Optimization.ipynb` | Gaussian optimization only | ~10 min |
| `Glimpse3D_MVCRM_Refinement.ipynb` | MVCRM refinement only | ~5 min |

---

# **📊 Research Components**

Located in `/research/`:

* Multi-view consistency analysis
* Depth variance evaluation
* CLIP similarity metrics
* Gaussian Splatting quality comparisons
* Ablation studies on pipeline components

---

# **🛠 Tech Stack**

### **AI / ML**
- PyTorch 2.0.1 + CUDA 11.8
- gsplat 1.2.0
- Diffusers 0.27.2
- PyTorch Lightning 1.9.5

### **Frontend**
- React + Vite
- Three.js + React Three Fiber
- Framer Motion

### **Backend**
- FastAPI
- Uvicorn

---

# **📦 Output Formats**

| Format | Description | Viewer |
|--------|-------------|--------|
| `.ply` | Gaussian Splat model | [SuperSplat](https://playcanvas.com/supersplat/editor), [Luma AI](https://lumalabs.ai/) |
| `.glb` | 3D mesh (GLTF binary) | [glTF Viewer](https://gltf-viewer.donmccurdy.com/), Blender |
| `.obj` | 3D mesh (Wavefront) | Any 3D software |
| `.mp4` | 360° turntable video | Any video player |

---

# **🧪 Testing**

```bash
# Run backend tests
cd backend
pytest tests/

# Verify full pipeline
python scripts/verify_full_stack.py

# Test individual modules
python ai_modules/midas_depth/test_depth.py
python ai_modules/refine_module/test_refine.py
```

---

# **⚠️ Known Issues & Solutions**

| Issue | Solution |
|-------|----------|
| CUDA OOM on T4 | Reduce `BATCH_VIEW_NUM` to 2, `MC_RESOLUTION` to 192 |
| gsplat shape error | Ensure opacity is `[N]` shape, not `[N,1]` |
| SyncDreamer view mismatch | Use 30° elevation for all 16 views, 22.5° azimuth spacing |
| SDXL Lightning artifacts | Set `guidance_scale=0`, use `timestep_spacing="trailing"` |

See [CRITICAL_REVIEW_REPORT.md](docs/CRITICAL_REVIEW_REPORT.md) for detailed production readiness analysis.

---

# **🤝 Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

# **📜 License**

MIT License - see [LICENSE](LICENSE) for details.

---

# **🙏 Acknowledgments**

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR) by VAST-AI-Research & Stability AI
- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) by Yuan Liu et al. (ICLR 2024)
- [SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) by ByteDance
- [gsplat](https://github.com/nerfstudio-project/gsplat) by Nerfstudio Project
- [MiDaS](https://github.com/isl-org/MiDaS) by Intel ISL

---

<div align="center">

# **✨ Glimpse3D**

### **Turning a Single Glimpse Into a Full 3D Reality**

[Open in Colab](https://colab.research.google.com/github/varunaditya27/Glimpse3D/blob/main/notebooks/Glimpse3D_Master_Pipeline.ipynb) · [Report Bug](https://github.com/varunaditya27/Glimpse3D/issues) · [Request Feature](https://github.com/varunaditya27/Glimpse3D/issues)

[Open in Colab](https://colab.research.google.com/github/varunaditya27/Glimpse3D/blob/main/notebooks/Glimpse3D_Master_Pipeline.ipynb) · [Report Bug](https://github.com/varunaditya27/Glimpse3D/issues) · [Request Feature](https://github.com/varunaditya27/Glimpse3D/issues)

</div>