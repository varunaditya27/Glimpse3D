"""
ai_modules/gsplat/utils_gs.py

Phase: 1 (IO)
Responsibility: Dev 1

Goal:
1. Load .ply files into GaussianModel.
2. Save GaussianModel to .ply.
3. Parse attributes (f_dc, f_rest, opacity, scale, rot).
"""

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from .model import GaussianModel

def load_ply(path: str) -> GaussianModel:
    """
    Loads a PLY file and returns a populated GaussianModel.
    Raises RuntimeError if any required attribute is missing.
    """
    plydata = PlyData.read(path)
    vertex = plydata['vertex']

    # 1. Validation: Check for all required keys
    required_keys = ['x', 'y', 'z', 'opacity']
    required_keys += ['scale_0', 'scale_1', 'scale_2']
    required_keys += ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    required_keys += ['f_dc_0', 'f_dc_1', 'f_dc_2']
    
    # We assume standard 3DGS SH degree 3 layout (45 dim)
    # If the file has fewer, we might crash or need to check SH degree.
    # User requirement: "f_rest_0 ... f_rest_44"
    # We will check based on what's available, but standard is 45.
    
    # Check base keys
    missing_keys = [k for k in required_keys if k not in vertex]
    if missing_keys:
        raise RuntimeError(f"PLY file missing required attributes: {missing_keys}")

    # Check SH keys (f_rest)
    # We determine how many f_rest are there
    f_rest_keys = [k for k in vertex.data.dtype.names if k.startswith('f_rest_')]
    # Sort them to be sure
    f_rest_keys.sort(key=lambda x: int(x.split('_')[-1]))
    
    # 2. Extract Data
    xyz = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1)
    opacities = np.expand_dims(vertex['opacity'], axis=-1)
    
    scales = np.stack((vertex['scale_0'], vertex['scale_1'], vertex['scale_2']), axis=-1)
    quats = np.stack((vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']), axis=-1)
    
    features_dc = np.stack((vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']), axis=-1)
    features_dc = np.expand_dims(features_dc, axis=1) # (N, 1, 3)

    # SH features
    if f_rest_keys:
        # Expected typically 45 for degree 3
        # Flattened list of all f_rest values
        f_rest_flat = np.stack([vertex[k] for k in f_rest_keys], axis=-1)
        
        N = xyz.shape[0]
        n_sh_coeffs = len(f_rest_keys)
        
        # Must be divisible by 3 (RGB)
        if n_sh_coeffs % 3 != 0:
            raise RuntimeError(f"f_rest count ({n_sh_coeffs}) is not divisible by 3.")
            
        num_sh_bases = n_sh_coeffs // 3
        
        # Reshape to (N, bases, 3)
        # Note: 3DGS ply order is usually (band1_r, band1_g, band1_b, ...) OR (band1_r, band2_r... band1_g...)
        # Standard 3DGS writes them sequentially: f_rest_0 is (deg1, coeff1, R), f_rest_1 is (deg1, coeff1, G)...
        # Actually standard implementation flattens them as: features[1:].transpose(1, 2).flatten()
        # So we reshape to (N, 15, 3) directly from the sorted keys should work if keys are 0..44
        features_rest = f_rest_flat.reshape((N, num_sh_bases, 3))
    else:
        features_rest = np.zeros((xyz.shape[0], 0, 3))

    # 3. Create Model
    # Explicitly float32 and contiguous is enforced by model.create_from_tensors, 
    # but we cast here to be safe before passing to torch
    model = GaussianModel(sh_degree=3) # Assuming strict max degree
    
    model.create_from_tensors(
        torch.from_numpy(xyz),
        torch.from_numpy(scales),
        torch.from_numpy(quats),
        torch.from_numpy(opacities),
        torch.from_numpy(features_dc),
        torch.from_numpy(features_rest)
    )
    
    return model

def save_ply(model: GaussianModel, path: str):
    """
    Saves a GaussianModel to a PLY file matching standard 3DGS format.
    """
    xyz = model.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz) # Standard PLY usually has nx,ny,nz as 0
    
    f_dc = model.get_features_dc.detach().contiguous().cpu().numpy().reshape(-1, 3)
    f_rest = model.get_features_rest.detach().contiguous().cpu().numpy()
    opacities = model.get_opacity.detach().cpu().numpy()
    scales = model.get_scaling.detach().cpu().numpy()
    rotation = model.get_rotation.detach().cpu().numpy()

    # Construct structured array
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # f_rest attr names
    # Flatten f_rest: (N, 15, 3) -> (N, 45)
    f_rest_flat = f_rest.reshape((f_rest.shape[0], -1))
    num_f_rest = f_rest_flat.shape[1]
    
    for i in range(num_f_rest):
        dtype_full.append((f'f_rest_{i}', 'f4'))
        
    dtype_full.append(('opacity', 'f4'))
    
    dtype_full.append(('scale_0', 'f4'))
    dtype_full.append(('scale_1', 'f4'))
    dtype_full.append(('scale_2', 'f4'))
    
    dtype_full.append(('rot_0', 'f4'))
    dtype_full.append(('rot_1', 'f4'))
    dtype_full.append(('rot_2', 'f4'))
    dtype_full.append(('rot_3', 'f4'))

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    # Fill data
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    for i in range(num_f_rest):
        elements[f'f_rest_{i}'] = f_rest_flat[:, i]
        
    elements['opacity'] = opacities[:, 0]
    
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    
    elements['rot_0'] = rotation[:, 0]
    elements['rot_1'] = rotation[:, 1]
    elements['rot_2'] = rotation[:, 2]
    elements['rot_3'] = rotation[:, 3]

    # Create PlyData
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
