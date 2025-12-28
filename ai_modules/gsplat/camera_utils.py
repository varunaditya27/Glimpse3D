"""
ai_modules/gsplat/camera_utils.py

Phase: 2 (Support)
Responsibility: Dev 1

Goal:
1. Convert Blender/Zero123 angles -> 4x4 World-to-Camera Matrices.
2. Project 3D points to 2D screen space.
3. Handle Intrinsic matrices.

Coordinate System:
- Right-Handed
- Y-Up
- Camera looks at Origin (0,0,0)
"""

import torch
import numpy as np

def orbit_camera(azimuth_deg: float, elevation_deg: float, radius: float) -> torch.Tensor:
    """
    Creates a 4x4 World-to-Camera matrix (View Matrix) looking at origin.
    Compatible with OpenGL / Gsplat conventions.
    
    Args:
        azimuth_deg: Rotation around Y axis (in degrees).
        elevation_deg: Rotation up from XZ plane (in degrees).
        radius: Distance from origin.
        
    Returns:
        torch.Tensor: (4, 4) W2C matrix.
    """
    # Convert to radians
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    
    # Spherical to Cartesian (Right-handed, Y-up)
    # x = r * cos(el) * sin(az)
    # y = r * sin(el)
    # z = r * cos(el) * cos(az)
    # Note: Azimuth 0 usually maps to +Z or +X depending on convention.
    # Here we assume 0 maps to +Z for standard "Front" view.
    
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    
    eye = torch.tensor([x, y, z], dtype=torch.float32)
    center = torch.zeros(3, dtype=torch.float32) # Look at origin
    up_world = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    
    # Construct Basis Vectors for Camera Frame
    # Forward: Pointing FROM camera TO center (OpenGL convention puts camera at origin looking down -Z usually, 
    # but Gsplat W2C assumes standard view matrix where basis Z points BACKWARDS from scene to eye)
    # Actually, View Matrix turns World Pts into Camera Frame Pts.
    # In Camera Frame: Camera is at (0,0,0), looking down -Z. Right is +X, Up is +Y.
    
    # Z_cam (Back) = normalize(eye - center)
    fwd = eye - center
    z_axis = fwd / torch.norm(fwd)
    
    # X_cam (Right) = cross(up_world, z_cam)
    x_axis = torch.cross(up_world, z_axis)
    x_axis = x_axis / torch.norm(x_axis)
    
    # Y_cam (Up) = cross(z_cam, x_axis)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    
    # Construct 4x4 View Matrix (W2C)
    # [ R   | -R*eye ]
    # [ 0   |    1   ]
    
    # Rotation part (rows are basis vectors because R is orthogonal and we transpose world basis to cam basis)
    R = torch.stack([x_axis, y_axis, z_axis])  # (3, 3)
    
    # Translation t = -R * eye
    t = -torch.matmul(R, eye) # (3,)
    
    w2c = torch.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    
    return w2c.contiguous()

def build_intrinsics(fov_deg: float, width: int, height: int) -> torch.Tensor:
    """
    Creates a 3x3 Intrinsic Matrix (K).
    Assumes fov_deg is VERTICAL Field of View.
    Assumes principal point is at center.
    
    Args:
        fov_deg: Vertical FOV in degrees.
        width: Image width.
        height: Image height.
        
    Returns:
        torch.Tensor: (3, 3) K matrix.
    """
    fov_rad = np.deg2rad(fov_deg)
    
    # focal = (h/2) / tan(fov/2)
    f_y = (height / 2.0) / np.tan(fov_rad / 2.0)
    f_x = f_y # Square pixels implies f_x = f_y usually, unless horizontal/vertical FOV differs.
    # Note: If width != height and square pixels, fov_x will be different.
    # We strictly use vertical FOV to drive focal length for both.
    
    cx = width / 2.0
    cy = height / 2.0
    
    K = torch.tensor([
        [f_x, 0.0, cx],
        [0.0, f_y, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    
    return K.contiguous()
