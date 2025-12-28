"""
ai_modules/gsplat/model.py

Phase: 1 (Core)
Responsibility: Dev 1

Goal:
1. Define the GaussianModel class holding the tensors.
2. Store:
   - _xyz (Means)
   - _features_dc (Color)
   - _features_rest (SH)
   - _scaling
   - _rotation
   - _opacity
3. Provide activation functions (sigmoid, exp).
"""

import torch
import torch.nn as nn

class GaussianModel:
    def __init__(self, sh_degree: int = 3):
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        
        # Internal storage for parameters (will be torch.nn.Parameter)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # Optimizer setup
        self.optimizer = None
        self.setup_complete = False

    def create_from_tensors(self, xyz, scales, quats, opacities, features_dc, features_rest):
        """
        Initialize the model from existing tensors.
        
        Args:
            xyz: (N, 3)
            scales: (N, 3) log-space
            quats: (N, 4)
            opacities: (N, 1) logit-space
            features_dc: (N, 1, 3)
            features_rest: (N, 15, 3) or (N, 0, 3) if no SH
        """
        self._xyz = nn.Parameter(xyz.float().contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.float().contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(quats.float().contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.float().contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.float().contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.float().contiguous().requires_grad_(True))
        
        self.setup_complete = True
        self.assert_shapes()

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest

    def parameters(self):
        """
        Returns an iterator over the trainable parameters.
        """
        return [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity
        ]

    def to(self, device):
        """
        Moves all parameters to the specified device.
        """
        self._xyz.data = self._xyz.data.to(device)
        self._features_dc.data = self._features_dc.data.to(device)
        self._features_rest.data = self._features_rest.data.to(device)
        self._scaling.data = self._scaling.data.to(device)
        self._rotation.data = self._rotation.data.to(device)
        self._opacity.data = self._opacity.data.to(device)
        return self

    def assert_shapes(self):
        """
        Validates that all tensor shapes align with the number of points N.
        """
        if not self.setup_complete:
            return

        N = self._xyz.shape[0]
        
        assert self._xyz.shape == (N, 3), f"Means shape mismatch: {self._xyz.shape} vs ({N}, 3)"
        assert self._scaling.shape == (N, 3), f"Scales shape mismatch: {self._scaling.shape} vs ({N}, 3)"
        assert self._rotation.shape == (N, 4), f"Quats shape mismatch: {self._rotation.shape} vs ({N}, 4)"
        assert self._opacity.shape == (N, 1), f"Opacities shape mismatch: {self._opacity.shape} vs ({N}, 1)"
        assert self._features_dc.shape == (N, 1, 3), f"Features DC shape mismatch: {self._features_dc.shape} vs ({N}, 1, 3)"
        
        # Features rest might be empty if SH degree is 0, but usually it corresponds to degree
        # (N, (max_sh_degree + 1)**2 - 1, 3) usually
        # But per user req: features_rest: (N,15,3)
        if self._features_rest.shape[1] != 0:
             # Just checking if dimension 0 and 2 match, dimension 1 depends on SH degree implementation
            assert self._features_rest.shape[0] == N, f"Features Rest N mismatch: {self._features_rest.shape[0]}"
            assert self._features_rest.shape[2] == 3, f"Features Rest color channels mismatch: {self._features_rest.shape[2]}"
