"""
ai_modules/gsplat/optimize.py

Phase: 3 (Refinement)
Responsibility: Dev 1

Goal:
1. Define the Optimizer (Adam).
2. Calculate Loss (L1 + SSIM).
3. Backpropagate gradients to Gaussians.
4. Step the optimizer.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict
from .model import GaussianModel
from .render_view import render

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def refine_model(
    model: GaussianModel,
    target_images: List[torch.Tensor],
    cameras: List[Dict],
    iterations: int
):
    """
    Refines the Gaussian Splat model by fitting it to the target images.
    
    Args:
        model: The GaussianModel to optimize (must be on correct device).
        target_images: List of (H, W, 3) tensors (GT images).
        cameras: List of camera parameters dicts.
        iterations: Number of optimizer steps.
    """
    
    # 1. Setup Optimizer logic
    # Learning rates from paper/standard config
    lrs = {
        'xyz': 0.00016,
        'features_dc': 0.0025,
        'opacity': 0.05,
        'scaling': 0.005,
        'rotation': 0.001
    }
    
    # Only optimize existing parameters
    param_groups = [
        {'params': [model._xyz], 'lr': lrs['xyz'], "name": "xyz"},
        {'params': [model._features_dc], 'lr': lrs['features_dc'], "name": "f_dc"},
        # Handle features_rest if present
        {'params': [model._features_rest], 'lr': lrs['features_dc'] / 20.0, "name": "f_rest"}, 
        {'params': [model._opacity], 'lr': lrs['opacity'], "name": "opacity"},
        {'params': [model._scaling], 'lr': lrs['scaling'], "name": "scaling"},
        {'params': [model._rotation], 'lr': lrs['rotation'], "name": "rotation"}
    ]
    
    optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

    # 2. Optimization Loop
    # We iterate 'iterations' times.
    # In each iteration, we can pick a random view or cycle through them.
    # Since batch size = 1, we pick one per step.
    
    num_views = len(target_images)
    view_indices = list(range(num_views))
    
    progress_bar = tqdm(range(1, iterations + 1), desc="Refining Splats")
    
    for iteration in progress_bar:
        # Pick view (Sequential for stability in small sets)
        view_idx = view_indices[(iteration - 1) % num_views]
        
        viewpoint_cam = cameras[view_idx]
        gt_image = target_images[view_idx]
        
        # Render
        # Expected render_pkg from render_view.py
        render_pkg = render(model, viewpoint_cam)
        image = render_pkg["render"]
        
        # Loss
        loss = l1_loss(image, gt_image)
        
        # Backprop
        loss.backward()
        
        # Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Constraints / Post-processing
        with torch.no_grad():
            # 1. Normalize Quaternions
            model._rotation.data = F.normalize(model._rotation.data)
            
            # 2. Clamp Scales (log-space)
            # Standard exp scaling means big range. 
            # Check user constraint: [-10, 5]
            model._scaling.data = torch.clamp(model._scaling.data, min=-10.0, max=5.0)
            
            # 3. Clamp Opacities (logit-space)
            # Sigmoid(-8) ~ 0, Sigmoid(8) ~ 1
            model._opacity.data = torch.clamp(model._opacity.data, min=-8.0, max=8.0)
            
        progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})

    return model
