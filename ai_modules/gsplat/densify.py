"""
ai_modules/gsplat/densify.py

Phase: 3 (Refinement)
Responsibility: Dev 2

Goal:
1. Monitor gradient magnitude of means.
2. CLONE large gaussians in high-density areas.
3. SPLIT large gaussians in over-reconstructed areas.
4. PRUNE transparent gaussians.
"""

import torch
import numpy as np
from .model import GaussianModel

def densify(
    model: GaussianModel,
    visibility_mask: torch.Tensor,
    grad_threshold: float,
    max_splats: int,
    scale_threshold: float = None
):
    """
    Adaptive Density Control.
    
    1. Splits Gaussians that are large and have high positional gradients (under-reconstructed).
    2. Prunes Gaussians with low opacity if N > max_splats.
    
    Args:
        model: The GaussianModel.
        visibility_mask: (N,) boolean tensor of points seen in current view.
        grad_threshold: Threshold for positional gradients.
        max_splats: Hard limit on number of splats.
        scale_threshold: (Optional) Threshold for scale vector magnitude to consider "large". 
                         If None, uses 0.05.
    """
    
    # --- PHASE 1: SPLITTING ---
    
    has_grads = model._xyz.grad is not None
    did_split = False
    
    # Temporary variables to hold "Current" state of model tensors
    # We will modify these variables, then at the end update the model once.
    
    curr_xyz = model._xyz
    curr_fdc = model._features_dc
    curr_frest = model._features_rest
    curr_opac = model._opacity
    curr_scale = model._scaling
    curr_rot = model._rotation
    
    if has_grads:
        grads = torch.norm(model._xyz.grad, dim=-1) # (N,)
        
        # Compute scale norm (max component)
        scales_exp = torch.exp(model._scaling)
        scales_norm = torch.max(scales_exp, dim=-1).values # (N,)
        
        if scale_threshold is None:
            scale_threshold = 0.05
            
        selected_mask = visibility_mask & (grads > grad_threshold) & (scales_norm > scale_threshold)
        selected_indices = torch.where(selected_mask)[0]
        
        if len(selected_indices) > 0:
            did_split = True
            
            # SPLIT Logic: Append 2 new, Remove 1 old
            
            stds = scales_exp[selected_indices].repeat(2, 1)
            means = torch.zeros((stds.size(0), 3), device=model._xyz.device)
            samples = torch.normal(mean=means, std=stds)
            
            # Create new attrs
            new_xyz = model._xyz[selected_indices].repeat(2, 1) + samples
            new_scaling = model._scaling[selected_indices].repeat(2, 1) - np.log(1.6)
            new_rotation = model._rotation[selected_indices].repeat(2, 1)
            new_opacity = model._opacity[selected_indices].repeat(2, 1)
            new_features_dc = model._features_dc[selected_indices].repeat(2, 1, 1)
            new_features_rest = model._features_rest[selected_indices].repeat(2, 1, 1)
            
            # Keep mask (Remove original split parents)
            keep_mask = torch.ones(model._xyz.shape[0], dtype=torch.bool, device=model._xyz.device)
            keep_mask[selected_indices] = False
            
            # Concatenate
            curr_xyz = torch.cat([model._xyz[keep_mask], new_xyz], dim=0)
            curr_fdc = torch.cat([model._features_dc[keep_mask], new_features_dc], dim=0)
            curr_frest = torch.cat([model._features_rest[keep_mask], new_features_rest], dim=0)
            curr_opac = torch.cat([model._opacity[keep_mask], new_opacity], dim=0)
            curr_scale = torch.cat([model._scaling[keep_mask], new_scaling], dim=0)
            curr_rot = torch.cat([model._rotation[keep_mask], new_rotation], dim=0)
    
    # --- PHASE 2: PRUNING ---
    
    current_count = curr_xyz.shape[0]
    
    if current_count > max_splats:
        num_to_prune = current_count - max_splats
        
        # Sort by opacity (lowest first)
        sorted_indices = torch.argsort(curr_opac.squeeze())
        prune_indices = sorted_indices[:num_to_prune]
        
        keep_mask_prune = torch.ones(current_count, dtype=torch.bool, device=curr_xyz.device)
        keep_mask_prune[prune_indices] = False
        
        curr_xyz = curr_xyz[keep_mask_prune]
        curr_fdc = curr_fdc[keep_mask_prune]
        curr_frest = curr_frest[keep_mask_prune]
        curr_opac = curr_opac[keep_mask_prune]
        curr_scale = curr_scale[keep_mask_prune]
        curr_rot = curr_rot[keep_mask_prune]
        
    # --- PHASE 3: UPDATE MODEL ---
    
    # Update only if something changed
    if did_split or (current_count > max_splats):
        model.create_from_tensors(
            curr_xyz, curr_scale, curr_rot, curr_opac, curr_fdc, curr_frest
        )
