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


def get_look_at(eye, center, up):
    """
    Compute W2C matrix from LookAt parameters.
    Returns 4x4 W2C matrix (World to Camera).
    """
    z_axis = eye - center
    z_axis = z_axis / torch.norm(z_axis)
    
    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / torch.norm(x_axis)
    
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    
    # Rotation (Row-major in torch constructs, but we need [R|t])
    # W2C = [R^T | -R^T * eye] ? 
    # Standard LookAt Matrix:
    # [ xx xy xz -dot(x,eye) ]
    # [ yx yy yz -dot(y,eye) ]
    # [ zx zy zz -dot(z,eye) ]
    # [ 0  0  0      1       ]
    
    w2c = torch.eye(4)
    w2c[0, :3] = x_axis
    w2c[1, :3] = y_axis
    w2c[2, :3] = z_axis
    
    w2c[0, 3] = -torch.dot(x_axis, eye)
    w2c[1, 3] = -torch.dot(y_axis, eye)
    w2c[2, 3] = -torch.dot(z_axis, eye)
    
    return w2c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY")
    parser.add_argument("--output", type=str, default="render_debug.png")
    parser.add_argument("--elevation", type=float, default=0.0, help="Elevation in degrees")
    parser.add_argument("--azimuth", type=float, default=0.0, help="Azimuth in degrees")
    parser.add_argument("--radius", type=float, default=3.0, help="Camera distance")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.ply):
        logger.error("PLY not found")
        sys.exit(1)

    logger.info("Loading model...")
    model = load_ply(args.ply)

    # Compute Camera Position (Orbit)
    elev_rad = math.radians(args.elevation)
    azi_rad = math.radians(args.azimuth)
    
    # Spherical to Cartesian (Y-up convention)
    # x = r * cos(elev) * sin(azi)
    # y = r * sin(elev)
    # z = r * cos(elev) * cos(azi)
    
    x = args.radius * math.cos(elev_rad) * math.sin(azi_rad)
    y = args.radius * math.sin(elev_rad)
    z = args.radius * math.cos(elev_rad) * math.cos(azi_rad)
    
    eye = torch.tensor([x, y, z], dtype=torch.float32)
    center = torch.zeros(3, dtype=torch.float32)
    up = torch.tensor([0, 1, 0], dtype=torch.float32)
    
    w2c = get_look_at(eye, center, up)

    # Intrinsics (Standard 60 deg FOV)
    H, W = 512, 512
    fov_y = math.radians(60)
    fy = H / (2 * math.tan(fov_y / 2))
    fx = fy
    
    K = torch.tensor([
        [fx, 0, W/2],
        [0, fy, H/2],
        [0, 0, 1]
    ])

    cam_params = {
        'image_height': H,
        'image_width': W,
        'K': K,
        'w2c': w2c
    }

    logger.info(f"Rendering View: Elev={args.elevation}, Azi={args.azimuth}, Rad={args.radius}")
    render_pkg = render(model, cam_params)
    img_tensor = render_pkg["render"]

    # Save
    logger.info(f"Saving to {args.output}")
    ndarr = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(ndarr).save(args.output)
    logger.info("Done.")

if __name__ == "__main__":
    main()
