# 🧠 Glimpse3D AI Modules

This directory contains the core AI components that power the Glimpse3D pipeline. Each module is designed to be modular and independently testable.

## 📦 Modules Overview

### 1. `zero123/` (Multi-View Synthesis)
- **Model**: Zero-123 / Zero123++
- **Purpose**: Generates novel camera views from a single input image.
- **Usage**: Used to "hallucinate" unseen sides of the object before 3D reconstruction.

### 2. `midas_depth/` (Depth Estimation)
- **Model**: MiDaS / ZoeDepth
- **Purpose**: Predicts monocular depth maps from RGB images.
- **Usage**: Provides geometric cues to guide the refinement process and ensure structural consistency.

### 3. `gsplat/` (3D Representation)
- **Library**: `gsplat` (Gaussian Splatting)
- **Purpose**: Handles the creation, rendering, and optimization of 3D Gaussian Splats.
- **Usage**: The core 3D format used for real-time rendering and iterative updates.

### 4. `diffusion/` (Texture Enhancement)
- **Model**: Stable Diffusion XL (SDXL) + ControlNet
- **Purpose**: Enhances the visual quality of rendered views.
- **Usage**: Adds high-frequency details and realistic textures to the coarse 3D model.

### 5. `refine_module/` (The Core Innovation)
- **Type**: Custom Algorithm (MVCRM)
- **Purpose**: **Back-Projection Refinement**.
- **Logic**:
    1.  Takes an AI-enhanced 2D image.
    2.  Projects pixel differences back into 3D space.
    3.  Updates the color and opacity of specific Gaussian splats.
    4.  Enforces depth and feature consistency to prevent artifacts.

## 🛠 Integration

These modules are imported and orchestrated by the **Backend Services** (`backend/app/services/`). They are designed to run on GPU and may require significant VRAM (recommended 16GB+).
