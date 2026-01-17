
import torch
import torch.optim as optim
import math
from tqdm import tqdm
import gsplat
import logging

logger = logging.getLogger("Glimpse3D-Optimize")

def refine_model(gs_model, target_images, cameras, iterations=100):
    """
    Refines the Gaussian Model using the target images and cameras.
    
    Args:
        gs_model: GaussianModel instance
        target_images: List of (C, H, W) tensors (GT images)
        cameras: List of camera dicts {'K': (3,3), 'w2c': (4,4), 'image_height': H, 'image_width': W}
        iterations: Number of steps
    """
    device = gs_model.get_xyz.device
    
    # 1. Setup Optimizer
    lrs = {
        'xyz': 0.00016 * 10.0, # Spatial lr
        'features_dc': 0.0025,
        'features_rest': 0.0025 / 20.0,
        'opacity': 0.05,
        'scaling': 0.005,
        'rotation': 0.001
    }
    
    params = [
        {'params': [gs_model._xyz], 'lr': lrs['xyz'], "name": "xyz"},
        {'params': [gs_model._features_dc], 'lr': lrs['features_dc'], "name": "f_dc"},
        {'params': [gs_model._features_rest], 'lr': lrs['features_rest'], "name": "f_rest"},
        {'params': [gs_model._opacity], 'lr': lrs['opacity'], "name": "opacity"},
        {'params': [gs_model._scaling], 'lr': lrs['scaling'], "name": "scaling"},
        {'params': [gs_model._rotation], 'lr': lrs['rotation'], "name": "rotation"}
    ]
    
    # Enable gradients
    gs_model._xyz.requires_grad = True
    gs_model._features_dc.requires_grad = True
    gs_model._features_rest.requires_grad = True
    gs_model._opacity.requires_grad = True
    gs_model._scaling.requires_grad = True
    gs_model._rotation.requires_grad = True

    optimizer = optim.Adam(params, lr=0.0, eps=1e-15)
    
    # 2. Optimization Loop
    progress_bar = tqdm(range(iterations), desc="Optimizing")
    
    # gsplat rasterization usually expects:
    # means, quats, scales, opacities, colors, viewmats, Ks, width, height...
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # We only have 1 view for now mostly
        view_idx = i % len(cameras)
        gt_image = target_images[view_idx]
        cam = cameras[view_idx]
        
        H, W = cam['image_width'], cam['image_height'] # Swapped? Check keys.
        H, W = cam['image_height'], cam['image_width']
        
        K = cam['K'].unsqueeze(0) # (1, 3, 3)
        w2c = cam['w2c'].unsqueeze(0) # (1, 4, 4)
        
        # Prepare inputs
        means3d = gs_model.get_xyz
        scales = gs_model.get_scaling
        quats = gs_model.get_rotation
        opacities = gs_model.get_opacity
        
        # Simple color handling (DC only for simplicity if SH fails, but we try SH)
        # gsplat 1.0+ handles SH via spherical_harmonics typically, or expects precomputed colors.
        # rasterization() signature varies.
        # Let's try to assume it takes 'colors' which we precompute for now to be safe,
        # OR 1.0+ takes SH.
        # We will iterate and find the right call.
        
        # For this version, let's assume we precompute colors from SH (approx) or just use DC.
        colors = gs_model.get_features_dc # (N, 3) - Only DC
        
        # Rasterize
        try:
            # Note: inputs need to be contiguous and on device
            # Check if gsplat.rasterization exists
            if hasattr(gsplat, "rasterization"):
                # Usually: (means, quats, scales, opacities, colors, viewmats, Ks, width, height)
                # Note: gsplat often takes [N, ...] and batches via viewmats [B, 4, 4]
                
                # Check signature roughly by trying
                # We need to reshape for batch if needed. gsplat 1.0 usually supports B=1.
                
                # IMPORTANT: gsplat v1.0.0 uses rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width, height, ...)
                # colors is (B, N, 3) or (N, 3)
                
                # Forward
                rendered_images, alphas, info = gsplat.rasterization(
                    means=means3d,
                    quats=quats,
                    scales=scales,
                    opacities=opacities.flatten(),
                    colors=colors,
                    viewmats=w2c, # (1, 4, 4)
                    Ks=K,         # (1, 3, 3)
                    width=W,
                    height=H,
                    packed=False # Assume non-packed for simplicity
                )
                
                # rendered_images: (B, H, W, 3)
                image = rendered_images[0].permute(2, 0, 1) # (3, H, W)
                
            else:
                # Fallback or error
                logger.error("gsplat.rasterization not found!")
                break
                
        except Exception as e:
            logger.error(f"Rasterization failed: {e}")
            raise e
            
        # Loss
        l1_loss = torch.nn.functional.l1_loss(image, gt_image)
        loss = l1_loss 
        # Add SSIM here if available (from pytorch_msssim import ssim)
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Simple Density Control (Placeholder)
        # if i % 100 == 0:
        #     pass
            
    progress_bar.close()
    return gs_model
