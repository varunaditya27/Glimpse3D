
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import os

class GaussianModel:
    def __init__(self, sh_degree=3):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = lambda x: x # Not used directly usually
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.logit(x, eps=1e-4) # SAFE LOGIT
        self.rotation_activation = torch.nn.functional.normalize

    def capture(self):
        return (
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict() if self.optimizer else None,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict) = model_args

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                            np.asarray(plydata.elements[0]["f_dc_1"]),
                            np.asarray(plydata.elements[0]["f_dc_2"])), axis=1)

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    # Reshape features to (N, 3, (sh_degree+1)**2 - 1) if needed, but for now flat is fine for loading
    # Standard 3DGS stores as (N, 15, 3) or similar.
    # reconstruct.py stores f_rest as 45 floats.
    # We load them as is.

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Create model
    model = GaussianModel()
    model._xyz = torch.tensor(xyz, dtype=torch.float32)
    model._features_dc = torch.tensor(features_dc, dtype=torch.float32).contiguous() # (N, 3)
    model._features_rest = torch.tensor(features_extra, dtype=torch.float32).contiguous() # (N, 45)
    model._opacity = torch.tensor(opacities, dtype=torch.float32)
    model._scaling = torch.tensor(scales, dtype=torch.float32)
    model._rotation = torch.tensor(rots, dtype=torch.float32)

    # In reconstruct.py, we saved raw values. 
    # But usually optimizations work in "activation space" (logit opacity, log scale).
    # reconstruct.py saved:
    # opacities: ~2.2 -> inv_sigmoid(2.0) is not valid? 
    # Wait, reconstruct.py: opacities = ones * 2.0.
    # Sigmoid(x) is in [0,1]. 2.0 is out of range for sigmoid output.
    # This implies reconstruct.py might be saving pre-activation values?
    # reconstruct.py: "Inverse Sigmoid(0.9) -> ~2.2". 
    # So it saves PRE-ACTIVATION values.
    # Scales: log(0.01). PRE-ACTIVATION.
    # Rots: (1,0,0,0). Normalized? Yes.
    
    # So we don't need to transform them here, just load them.
    return model

def save_ply(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    xyz = model._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = model._features_dc.detach().cpu().numpy()
    f_rest = model._features_rest.detach().cpu().numpy()
    opacities = model._opacity.detach().cpu().numpy()
    scale = model._scaling.detach().cpu().numpy()
    rotation = model._rotation.detach().cpu().numpy()

    # dtype_full removed as it was broken and unused

    # Simple manual construction
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    
    for i in range(f_rest.shape[1]):
        dtype.append((f'f_rest_{i}', 'f4'))
    
    dtype.append(('opacity', 'f4'))
    
    for i in range(scale.shape[1]):
        dtype.append((f'scale_{i}', 'f4'))
        
    for i in range(rotation.shape[1]):
        dtype.append((f'rot_{i}', 'f4'))
        
    elements = np.empty(xyz.shape[0], dtype=dtype)
    
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    for i in range(f_rest.shape[1]):
        elements[f'f_rest_{i}'] = f_rest[:, i]
        
    elements['opacity'] = opacities[:, 0]
    
    for i in range(scale.shape[1]):
        elements[f'scale_{i}'] = scale[:, i]
        
    for i in range(rotation.shape[1]):
        elements[f'rot_{i}'] = rotation[:, i]
        
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
