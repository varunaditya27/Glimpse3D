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
    print("✓ gsplat module loaded successfully (Real Mode)")
except ImportError as e:
    GSPLAT_AVAILABLE = False
    logger.error(f"gsplat module not found: {e}")
    # We want to fail hard if we expect real mode
    raise ImportError("gsplat not found! Please run 'pip install gsplat' or check installation.")

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
        
        # Move to device
        K = K.to(device)
        w2c = w2c.to(device)

        # 3. Compute Projection Matrix (OpenGL style) from intrinsics
        # gsplat expects a full 4x4 projection matrix
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Simple projection matrix construction
        # Mapping: z_near=0.01, z_far=100.0
        z_near, z_far = 0.01, 100.0
        
        proj = torch.zeros((4, 4), device=device)
        proj[0, 0] = 2 * fx / W
        proj[1, 1] = 2 * fy / H
        proj[0, 2] = -(2 * cx / W - 1)
        proj[1, 2] = -(2 * cy / H - 1)
        proj[2, 2] = z_far / (z_far - z_near)
        proj[2, 3] = -(z_far * z_near) / (z_far - z_near)
        proj[3, 2] = 1.0
        
        # gsplat expects 'viewmat' and 'projmat'
        # viewmat: World-to-Camera (4, 4)
        # projmat: Camera-to-Clip (4, 4) => BUT gsplat often takes W2C @ Proj direct?
        # Checking typical usage: project_gaussians(..., viewmat, projmat, ...)
        
        # 4. Project Gaussians
        # glob_scale=1.0 is standard
        glob_scale = 1.0
        
        xys, depths, radii, conics, cum_tiles_hit, num_tiles_hit, cov3d = project_gaussians(
            means3d,
            scales,
            glob_scale,
            quats,
            w2c.unsqueeze(0), # (1, 4, 4)
            proj.unsqueeze(0), # (1, 4, 4)
            H, W,
            fx, fy, cx, cy
        )
        
        # 5. Rasterize
        # Note: gsplat's rasterize_gaussians might act on packed data 
        # or require specific argument order. Using standard signature:
        
        if colors_precomp is None:
             # If using SH, we would use spherical_harmonics() first
             colors_precomp = torch.ones((means3d.shape[0], 3), device=device)

        out_img, _, _ = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors_precomp,
            opacities,
            H, W,
            bg_color
        )
        
        # Output is (H, W, 3). Permute to (3, H, W)
        image = out_img.permute(2, 0, 1)
        
        return {"render": image}
    else:
        raise RuntimeError("GSPLAT_AVAILABLE is False. Cannot run render().")


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
