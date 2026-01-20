# Glimpse3D AI Pipeline Architecture

> **Technical Documentation** — Complete pipeline flow from 2D image to 3D Gaussian Splat

---

## Executive Summary

Glimpse3D is an AI-powered pipeline that converts a **single 2D image** into a **high-quality 3D Gaussian Splat**. The system combines five specialized modules that work together in a sequential-iterative process:

```
┌─────────────┐     ┌─────────────┐    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
│   Input     │───▶│   TripoSR   │───▶│ SyncDreamer │───▶│    SDXL     │───▶│    MVCRM    │
│   Image     │     │  (gsplat)   │    │ Multi-View  │     │ Enhancement │    │ Refinement  │
└─────────────┘     └─────────────┘    └─────────────┘     └─────────────┘    └─────────────┘
                         │                                      │                   │
                         │                                      │                   │
                         ▼                                      ▼                   ▼
                   Initial 3D              16 Views        Enhanced Views    Final 3D Splat
                     Mesh                                  + Depth Maps
```

---

## Module Overview

| Module | Location | Purpose | VRAM | Time |
|--------|----------|---------|------|------|
| **gsplat** | `ai_modules/gsplat/` | TripoSR reconstruction + Gaussian optimization | ~6 GB | 0.5s |
| **SyncDreamer** | `ai_modules/sync_dreamer/` | Multi-view consistent image generation | ~12 GB | 30s |
| **SDXL Diffusion** | `ai_modules/diffusion/` | View enhancement with ControlNet | ~8 GB | 2s/view |
| **MiDaS Depth** | `ai_modules/midas_depth/` | Monocular depth estimation | ~2 GB | 50ms |
| **Refinement (MVCRM)** | `ai_modules/refine_module/` | Back-projection and consistency | ~4 GB | 5s/iter |

---

## Detailed Pipeline Flow

### Phase 1: Cold-Start Reconstruction (gsplat module)

**Purpose:** Generate initial 3D geometry from a single image using TripoSR.

**Location:** `ai_modules/gsplat/reconstruct.py`

**Process:**
1. **Background Removal** — Input image is preprocessed using `rembg` to isolate the foreground object
2. **Compositing** — Foreground is composited over neutral gray (0.5) background
3. **TripoSR Inference** — The image is fed to TripoSR (`stabilityai/TripoSR` from HuggingFace)
4. **Mesh Extraction** — TripoSR outputs a 3D mesh with vertex colors
5. **Point Sampling** — 100,000 points are sampled from the mesh surface
6. **Gaussian Initialization** — Each point becomes a Gaussian splat with:
   - Position (x, y, z)
   - Color (SH coefficients, initialized to gray)
   - Opacity (initialized to ~0.9 via inverse sigmoid)
   - Scale (initialized small, log-space)
   - Rotation (identity quaternion)

**Output:** `initial_splat.ply` — A Gaussian Splat PLY file

```python
# Usage
from ai_modules.gsplat import load_triposr_model, preprocess_image, run_inference, sample_and_export

model = load_triposr_model()
img = preprocess_image("input.png")
mesh = run_inference(model, img)
sample_and_export(mesh, "output.ply")
```

**Key Files:**
- `reconstruct.py` — Main reconstruction logic
- `utils_gs.py` — GaussianModel class and PLY I/O
- `model.py` — Simplified Gaussian model wrapper

---

### Phase 2: Multi-View Generation (SyncDreamer module)

**Purpose:** Generate 16 geometrically-consistent views of the object from different angles.

**Location:** `ai_modules/sync_dreamer/inference.py`

**Process:**
1. **Input Preparation** — Image is cropped and centered on the foreground object
2. **Elevation Estimation** — User provides elevation angle of input view (typically 30°)
3. **Diffusion Sampling** — SyncDreamer's synchronized multi-view diffusion generates all 16 views simultaneously
4. **Camera Positions** — Fixed viewpoints at:
   - **Elevations:** 30° (views 0-7) and -20° (views 8-15)
   - **Azimuths:** 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° for each elevation ring

**Output:** 16 PNG images at 256×256 resolution

```python
# Camera configuration
ELEVATIONS = [30, 30, 30, 30, 30, 30, 30, 30, -20, -20, -20, -20, -20, -20, -20, -20]
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315, 0, 45, 90, 135, 180, 225, 270, 315]
RADIUS = 1.5  # Fixed camera distance
```

**Why SyncDreamer?**
- Unlike Zero123 (single-view generation), SyncDreamer generates all views in a **single forward pass**
- This ensures **3D consistency** — all views share the same underlying geometry
- Critical for subsequent refinement steps

```python
# Usage
from ai_modules.sync_dreamer import generate_multiview

paths = generate_multiview(
    image_path="input.png",
    output_dir="views/",
    elevation=30.0,
    seed=42
)
```

**Key Files:**
- `inference.py` — SyncDreamerService class
- `utils_syncdreamer.py` — Camera matrix utilities

---

### Phase 3: View Enhancement (SDXL Diffusion + MiDaS Depth)

**Purpose:** Enhance each SyncDreamer view with high-quality textures while preserving structure.

**Location:** `ai_modules/diffusion/enhance_service.py`

**Process:**

#### 3a. Depth Estimation (MiDaS)
1. **Depth Inference** — MiDaS estimates per-pixel depth for each view
2. **Normalization** — Depth values normalized to [0, 1] range
3. **Confidence Estimation** — Edge regions and uncertain areas flagged

```python
# MiDaS depth estimation
from ai_modules.midas_depth import estimate_depth, get_estimator

estimator = get_estimator(model_type="MiDaS_small")  # Fast
depth_map = estimator.estimate("view_00.png")  # Returns (H, W) float32 array
```

**Model Options:**
| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| `MiDaS_small` | ~50ms | Good | ~1 GB |
| `DPT_Hybrid` | ~150ms | Better | ~2 GB |
| `DPT_Large` | ~300ms | Best | ~3 GB |

#### 3b. SDXL Lightning Enhancement
1. **Depth Conditioning** — Depth map converted to ControlNet input
2. **SDXL Lightning** — 4-step inference (vs 30-50 for standard SDXL)
3. **img2img Mode** — Original view used as initialization (strength=0.75)
4. **Prompt Guidance** — "high quality 3D render, detailed textures, studio lighting"

```python
# Enhancement pipeline
from ai_modules.diffusion import EnhanceService, EnhanceConfig

service = EnhanceService(config=EnhanceConfig.for_t4_gpu())
service.load()

enhanced = service.enhance(
    image="view_00.png",
    prompt="photorealistic 3D model, detailed texture",
    depth_map=depth_map,  # From MiDaS
    strength=0.75,
    controlnet_scale=0.5
)
```

**Why ControlNet Depth?**
- Prevents SDXL from hallucinating incorrect geometry
- Depth map "locks" the 3D structure while allowing texture enhancement
- Without it, enhanced views would be inconsistent with each other

**Key Files:**
- `enhance_service.py` — Main orchestrator (EnhanceService)
- `sdxl_lightning.py` — SDXL Lightning pipeline wrapper
- `controlnet_depth.py` — Depth preprocessing for ControlNet
- `prompt_builder.py` — Prompt template system

---

### Phase 4: Multi-View Consistent Refinement (MVCRM)

**Purpose:** Back-project enhanced 2D views into 3D to update Gaussian splat colors and properties.

**Location:** `ai_modules/refine_module/`

**This is Glimpse3D's novel contribution** — the Multi-View Consistency Refinement Module (MVCRM).

#### 4a. Architecture Overview

```
Enhanced Views (16×)     Depth Maps (16×)      Initial Splats
       │                       │                     │
       ▼                       ▼                     ▼
┌─────────────────────────────────────────────────────────┐
│                    FUSION CONTROLLER                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Render    │  │   Depth     │  │    Back     │     │
│  │   Views     │──▶│ Consistency │──▶│  Projector  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                 │            │
│         ▼                ▼                 ▼            │
│    Rendered        Consistency        Color/Opacity     │
│     Images           Mask             Updates           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                    Updated Splats
```

#### 4b. Refinement Process

**Step 1: Render Current Splats**
```python
# For each camera pose, render the current Gaussian splats
rendered_image = gsplat.rasterization(
    means=splat_positions,
    quats=splat_rotations,
    scales=splat_scales,
    opacities=splat_opacities,
    colors=splat_colors,
    viewmats=camera_w2c,
    Ks=camera_intrinsics,
    width=256, height=256
)
```

**Step 2: Depth Consistency Check**
- Compare rendered depth (from 3D model) with MiDaS depth (from enhanced image)
- Generate a **consistency mask** marking valid regions for update
- Prevents updating splats where geometry disagrees

```python
from ai_modules.refine_module import DepthConsistencyChecker

checker = DepthConsistencyChecker(base_threshold=0.1)
result = checker.check(rendered_depth, midas_depth)
valid_mask = result.consistency_mask  # (H, W) boolean
```

**Step 3: Compute Residuals**
```python
# Pixel-wise difference between rendered and enhanced views
residuals = enhanced_image - rendered_image  # (H, W, 3)
weighted_residuals = residuals * valid_mask * confidence
```

**Step 4: Back-Projection**
- For each pixel, identify which splats contributed to it during rendering
- Distribute the residual error back to those splats
- Update splat colors using scatter-add accumulation

```python
from ai_modules.refine_module import BackProjector

projector = BackProjector(learning_rate=0.05)
result = projector.project(
    rendered_image=rendered,
    enhanced_image=enhanced,
    depth_map=midas_depth,
    splat_means=positions,
    splat_colors=colors,
    splat_opacities=opacities,
    alpha_map=alpha,
    splat_contributions=contrib_map,
    camera=camera_params,
    consistency_mask=valid_mask
)

# Apply updates
new_colors = colors + result.color_updates * 0.5  # damping
```

**Step 5: Iterate**
- Repeat Steps 1-4 for all 16 views
- Run multiple refinement iterations (typically 3-5)
- Each iteration improves multi-view consistency

```python
from ai_modules.refine_module import FusionController

controller = FusionController(config=RefinementConfig(
    num_iterations=3,
    learning_rate=0.05,
    depth_threshold=0.1
))

refined_splats = controller.refine(
    initial_splats=splats,
    enhanced_views=enhanced_images,
    depth_maps=depth_maps,
    cameras=camera_list
)
```

**Key Files:**
- `fusion_controller.py` — Main MVCRM orchestrator
- `back_projector.py` — 2D→3D update mapping
- `depth_consistency.py` — Geometric validation
- `feature_consistency.py` — Feature-space validation
- `normal_smoothing.py` — Surface regularization

---

### Phase 5: Output

**Final Output:** Refined Gaussian Splat PLY file

```python
from ai_modules.gsplat import save_ply

save_ply(refined_model, "final_splat.ply")
```

**PLY Format:**
```
ply
format binary_little_endian 1.0
element vertex 100000
property float x
property float y
property float z
property float f_dc_0    # SH color (R)
property float f_dc_1    # SH color (G)
property float f_dc_2    # SH color (B)
property float f_rest_0  # Higher-order SH (45 coefficients)
...
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0     # Quaternion (w)
property float rot_1     # Quaternion (x)
property float rot_2     # Quaternion (y)
property float rot_3     # Quaternion (z)
end_header
```

---

## Integration Diagram

```
                              ┌──────────────────────────────────────────────────────────────┐
                              │                      GLIMPSE3D PIPELINE                       │
                              └──────────────────────────────────────────────────────────────┘
                                                         │
                                                         ▼
┌──────────────────┐                            ┌─────────────────┐
│   INPUT IMAGE    │                            │  ai_modules/    │
│   (RGB, RGBA)    │                            │  __init__.py    │
└────────┬─────────┘                            └─────────────────┘
         │                                               │
         │  ┌────────────────────────────────────────────┼────────────────────────────────────┐
         │  │                                            │                                    │
         ▼  ▼                                            ▼                                    ▼
┌─────────────────┐                            ┌─────────────────┐                   ┌─────────────────┐
│    gsplat/      │                            │  sync_dreamer/  │                   │   diffusion/    │
│  reconstruct.py │                            │  inference.py   │                   │enhance_service.py
│                 │                            │                 │                   │                 │
│  ┌───────────┐  │                            │  ┌───────────┐  │                   │  ┌───────────┐  │
│  │  TripoSR  │  │                            │  │SyncDreamer│  │                   │  │   SDXL    │  │
│  │  Model    │  │                            │  │  Model    │  │                   │  │ Lightning │  │
│  └───────────┘  │                            │  └───────────┘  │                   │  └───────────┘  │
│        │        │                            │        │        │                   │        │        │
│        ▼        │                            │        ▼        │                   │        ▼        │
│  ┌───────────┐  │                            │  ┌───────────┐  │                   │  ┌───────────┐  │
│  │   Mesh    │  │                            │  │  16 Views │  │                   │  │ Enhanced  │  │
│  │ Sampling  │  │                            │  │  256×256  │  │                   │  │   Views   │  │
│  └───────────┘  │                            │  └───────────┘  │                   │  └───────────┘  │
└────────┬────────┘                            └────────┬────────┘                   └────────┬────────┘
         │                                              │                                     │
         │ Initial                                      │ Multi-View                          │
         │ Splats                                       │ Images                              │ + Depth
         │                                              │                                     │   Maps
         │                                              │                                     │
         │         ┌────────────────────────────────────┼─────────────────────────────────────┤
         │         │                                    │                                     │
         │         │                    ┌───────────────┴───────────────┐                     │
         │         │                    │       midas_depth/            │                     │
         │         │                    │       run_depth.py            │                     │
         │         │                    │                               │                     │
         │         │                    │  ┌─────────────────────────┐  │                     │
         │         │                    │  │  MiDaS / DPT Models     │  │                     │
         │         │                    │  │  Depth Estimation       │  │                     │
         │         │                    │  └─────────────────────────┘  │                     │
         │         │                    │              │                │                     │
         │         │                    │              ▼                │                     │
         │         │                    │  ┌─────────────────────────┐  │                     │
         │         │                    │  │   Depth Maps (16×)      │──┼─────────────────────┤
         │         │                    │  │   Confidence Maps       │  │                     │
         │         │                    │  └─────────────────────────┘  │                     │
         │         │                    └───────────────────────────────┘                     │
         │         │                                                                          │
         ▼         ▼                                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    refine_module/                                                    │
│                              MULTI-VIEW CONSISTENCY REFINEMENT MODULE (MVCRM)                        │
│                                                                                                      │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐                  │
│  │  fusion_controller.py │    │ depth_consistency.py │    │   back_projector.py  │                  │
│  │                       │    │                      │    │                      │                  │
│  │  • Orchestrates       │    │  • Validates depth   │    │  • Maps 2D→3D        │                  │
│  │    refinement loop    │    │    alignment         │    │  • Scatter updates   │                  │
│  │  • Manages views      │◀──▶│  • Generates masks   │◀──▶│  • Color/opacity     │                  │
│  │  • Tracks convergence │    │  • Scale alignment   │    │    adjustments       │                  │
│  └──────────────────────┘    └──────────────────────┘    └──────────────────────┘                  │
│                                           │                                                         │
│                                           ▼                                                         │
│                              ┌──────────────────────┐                                               │
│                              │  Refined Gaussians   │                                               │
│                              │  (Updated colors,    │                                               │
│                              │   opacities, scales) │                                               │
│                              └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────┐
                              │   OUTPUT: PLY FILE   │
                              │   ~100K Gaussians    │
                              │   High-Quality 3D    │
                              └──────────────────────┘
```

---

## Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1 | RGB Image (any size) | TripoSR reconstruction | `initial_splat.ply` (100K points) |
| 2 | RGB Image + Elevation | SyncDreamer diffusion | 16× PNG images (256×256) |
| 3a | 16× PNG images | MiDaS inference | 16× depth maps + confidence |
| 3b | 16× PNG + depth | SDXL + ControlNet | 16× enhanced PNG (1024×1024) |
| 4 | Splats + enhanced + depth | MVCRM refinement | `refined_splat.ply` |

---

## Memory Management

The pipeline is optimized for **Google Colab T4 GPU** (15 GB VRAM):

```python
# Sequential loading pattern to fit in memory
from ai_modules import clear_gpu_memory

# Phase 1: TripoSR
model = load_triposr_model()
mesh = run_inference(model, img)
del model
clear_gpu_memory()  # Free ~6 GB

# Phase 2: SyncDreamer  
service = SyncDreamerService()
views = service.generate(img)
service.unload_model()
clear_gpu_memory()  # Free ~12 GB

# Phase 3: SDXL Enhancement
enhancer = EnhanceService()
enhanced = enhancer.enhance_batch(views)
del enhancer
clear_gpu_memory()  # Free ~8 GB

# Phase 4: Refinement
controller = FusionController()
refined = controller.refine(splats, enhanced, depths)
```

---

## API Quick Reference

```python
# Full pipeline example
from ai_modules.gsplat import load_triposr_model, preprocess_image, run_inference, sample_and_export, load_ply, save_ply
from ai_modules.sync_dreamer import generate_multiview
from ai_modules.diffusion import EnhanceService, EnhanceConfig
from ai_modules.midas_depth import estimate_depth
from ai_modules.refine_module import FusionController, RefinementConfig

# Step 1: Reconstruct
model = load_triposr_model()
img = preprocess_image("input.png")
mesh = run_inference(model, img)
sample_and_export(mesh, "initial.ply")

# Step 2: Multi-view
paths = generate_multiview("input.png", "views/", elevation=30)

# Step 3: Enhance
service = EnhanceService(EnhanceConfig.for_t4_gpu()).load()
enhanced = [service.enhance(p) for p in paths]

# Step 4: Refine
depths = [estimate_depth(p) for p in paths]
splats = load_ply("initial.ply")
controller = FusionController()
refined = controller.refine(splats, enhanced, depths, cameras)

# Output
save_ply(refined, "final.ply")
```

---

## References

- **TripoSR**: [stabilityai/TripoSR](https://huggingface.co/stabilityai/TripoSR)
- **SyncDreamer**: [liuyuan-pal/SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- **SDXL Lightning**: [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)
- **MiDaS**: [intel-isl/MiDaS](https://github.com/isl-org/MiDaS)
- **gsplat**: [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
- **3D Gaussian Splatting**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

---

*Last Updated: January 2026*
*Glimpse3D v0.2.0*
