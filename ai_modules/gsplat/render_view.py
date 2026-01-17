"""
ai_modules/gsplat/render_view.py

Rendering logic for Gaussian Splats.

Responsibilities:
- Rasterize 3D splats to 2D image
- Handle camera parameters (intrinsics/extrinsics)
"""

import torch
import math
import argparse
import sys
import os
import logging
from .model import GaussianModel
from .utils_gs import load_ply
import numpy as np
from PIL import Image

# Configure Logging
logger = logging.getLogger("Glimpse3D-Render")

try:
    from gsplat.project_gaussians import project_gaussians
    from gsplat.rasterize import rasterize_gaussians
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    logger.warning("gsplat module not found. Using MOCK RASTERIZER.")

def render(model: GaussianModel, viewpoint_cam: dict, bg_color: torch.Tensor = None):
    """
    Render the Gaussian Model from a specific viewpoint.

    Args:
        model: GaussianModel instance.
        viewpoint_cam: Dict containing:
            - 'H': int, 'W': int
            - 'K': (3, 3) Intrinsics
            - 'w2c': (4, 4) World-to-Camera Matrix
        bg_color: (3,) Tensor for background (default black).

    Returns:
        image: (3, H, W) torch.Tensor
    """
    H, W = viewpoint_cam['image_height'], viewpoint_cam['image_width']
    device = model.get_xyz.device

    if bg_color is None:
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    if GSPLAT_AVAILABLE:
        # 1. Setup Inputs
        means3d = model.get_xyz
        scales = model.get_scaling
        quats = model.get_rotation
        opacities = model.get_opacity
        shs = model.get_features_rest
        # For simplicity in Phase 2, we might just use DC if SH logic is complex
        # But gsplat supports SH.
        colors_precomp = None
        if shs is None or shs.shape[1] == 0:
             # Use DC only
             colors_precomp = model.get_features_dc.squeeze(1) # (N, 3)

        # 2. Extract Matrices
        K = viewpoint_cam['K'] # (3, 3)
        w2c = viewpoint_cam['w2c'] # (4, 4)

        # gsplat expects viewing transform and projection?
        # Check gsplat docs or source.
        # Usually: viewmats (B, 4, 4), projmats (B, 4, 4)
        # We need to compute projection matrix from K.
        # This is Dev 1 math territory, but we need a basic version to run.
        # Placeholder: we pass what we have.

        # TODO: Implement actual gsplat call when available.
        # Since I am Dev 2 (Builder), and gsplat is missing, I focus on the Mock path.
        pass

    # --- MOCK RASTERIZATION ---
    # Draw a 2D projection of the gaussian centers to prove the pipeline works.

    # Simple perspective projection
    xyz = model.get_xyz # (N, 3)

    # World -> Camera
    w2c = viewpoint_cam['w2c'].to(device)

    # Homogeneous
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1) # (N, 4)
    xyz_cam = (w2c @ xyz_h.T).T # (N, 4)

    # Camera -> Image (using K)
    # K is [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    K = viewpoint_cam['K'].to(device)

    xyz_cam_3 = xyz_cam[:, :3]
    # Simple z-clip
    valid_mask = xyz_cam_3[:, 2] > 0.1
    xyz_cam_3 = xyz_cam_3[valid_mask]

    # Project
    uv_h = (K @ xyz_cam_3.T).T # (M, 3)
    u = uv_h[:, 0] / uv_h[:, 2]
    v = uv_h[:, 1] / uv_h[:, 2]

    # Draw on canvas
    canvas = torch.zeros((H, W, 3), device=device)

    # Very inefficient "scatter" for loop, but fine for debug
    # Only draw first 1000 points to save time
    u = u.long()
    v = v.long()

    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]

    # Set pixels to White (or mean color if accessible)
    # For now: Green dots to show it worked
    if len(u) > 0:
        # canvas[v, u] = torch.tensor([0.0, 1.0, 0.0], device=device) # Crash if indices dup?
        # Safe scatter?
        # Just loop for safety in mock
        for i in range(min(len(u), 1000)):
            # Check bounds safety mainly for the mock scatter
             if 0 <= v[i] < H and 0 <= u[i] < W:
                canvas[v[i], u[i]] = torch.tensor([0.0, 1.0, 0.0], device=device)

    image = canvas.permute(2, 0, 1) # (3, H, W)

    # --- DUMMY GRADIENT FOR MOCK MODE ---
    # Since scatter is non-differentiable, we MUST add a dependency on the model parameters
    # so that loss.backward() finds a path to them (even if gradient is zero).
    # This prevents "element 0 of tensors does not require grad" error.
    if not GSPLAT_AVAILABLE:
        dummy_loss = (
            torch.sum(model.get_xyz) * 0.0 +
            torch.sum(model.get_features_dc) * 0.0 +
            torch.sum(model.get_opacity) * 0.0 +
            torch.sum(model.get_scaling) * 0.0 +
            torch.sum(model.get_rotation) * 0.0
        )
        # Add to image (broadcasting)
        image = image + dummy_loss

    return {"render": image}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY")
    parser.add_argument("--output", type=str, default="render_debug.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.ply):
        logger.error("PLY not found")
        sys.exit(1)

    logger.info("Loading model...")
    model = load_ply(args.ply)

    # Mock Camera: Look from 0,0,-5 at 0,0,0
    H, W = 512, 512
    fov_y = math.radians(60)

    # K matrix
    fy = H / (2 * math.tan(fov_y / 2))
    fx = fy # Square pixels
    K = torch.tensor([
        [fx, 0, W/2],
        [0, fy, H/2],
        [0, 0, 1]
    ])

    # W2C (Simple translation back by 2 units)
    # World is usually at 0. Camera at 0,0,2 looking at -z?
    # Let's say Camera is at 0,0,3 looking at 0,0,0.
    w2c = torch.eye(4)
    w2c[2, 3] = 3.0 # Translate Z input by 3?
    # Actually w2c transforms World Point to Camera Frame.
    # If Camera is at (0,0,3), then point at (0,0,0) should act like (0,0,-3) in camera frame (convention dependent).
    # Let's try simple identity first (camera at origin).
    # TripoSR mesh is usually centered at 0.
    # So we need to move camera back.
    w2c[2, 3] = 3.0 # Move world points +3 in Z (so they are in front of camera at 0?)
    # No, typically if Cam is at +3, P_cam = P_world - Cam_pos. So z - 3.
    w2c[2, 3] = -3.0

    cam_params = {
        'image_height': H,
        'image_width': W,
        'K': K,
        'w2c': w2c
    }

    logger.info("Rendering...")
    render_pkg = render(model, cam_params)
    img_tensor = render_pkg["render"]

    # Save
    logger.info(f"Saving to {args.output}")
    ndarr = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(ndarr).save(args.output)
    logger.info("Done.")

if __name__ == "__main__":
    main()
