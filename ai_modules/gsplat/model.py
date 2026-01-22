import torch
import torch.nn as nn

class GaussianModel:
    """
    A simple wrapper for Gaussian Splatting parameters.
    Compatible with gsplat rendering logic.
    """
    def __init__(self, positions, colors=None, opacities=None, scales=None, rotations=None, sh_degree=0):
        self.active_sh_degree = sh_degree
        self._xyz = positions
        self._features_dc = colors
        self._features_rest = None # SH features
        self._opacity = opacities
        self._scaling = scales
        self._rotation = rotations # Quaternions (w, x, y, z) or (x, y, z, w)? gsplat expects (w, x, y, z) usually? 
        # gsplat 1.0+ expects quaternions.
        
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
        
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation
        
    @classmethod
    def from_dict(cls, data, device="cuda"):
        """
        Create model from dictionary of tensors.
        """
        # Ensure all are on device
        def to_dev(t):
           return t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.float32)

        xyz = to_dev(data['positions'])
        
        # Colors (DC)
        colors = data.get('colors')
        if colors is None:
             colors = torch.rand_like(xyz)
        else:
             colors = to_dev(colors)
             
        # Opacities
        opacities = data.get('opacities')
        if opacities is None:
             opacities = torch.ones((xyz.shape[0],), device=device)
        else:
             opacities = to_dev(opacities)
             
        # Scales
        scales = data.get('scales')
        if scales is None:
             scales = torch.ones_like(xyz) * 0.1
        else:
             scales = to_dev(scales)
        
        # Rotations
        rotations = data.get('rotations') # Quats
        if rotations is None:
             # Identity quaternions (1, 0, 0, 0)
             rotations = torch.zeros((xyz.shape[0], 4), device=device)
             rotations[:, 0] = 1.0
        else:
             rotations = to_dev(rotations)
             
        return cls(xyz, colors, opacities, scales, rotations)
