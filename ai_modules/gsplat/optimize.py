
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
    
    Returns:
        Refined GaussianModel
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
    progress_bar = tqdm(range(iterations), desc="Optimizing Gaussians")
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Cycle through views
        view_idx = i % len(cameras)
        gt_image = target_images[view_idx]
        cam = cameras[view_idx]
        
        H = cam['image_height']
        W = cam['image_width']
        
        K = cam['K'].unsqueeze(0).to(device)  # (1, 3, 3)
        w2c = cam['w2c'].unsqueeze(0).to(device)  # (1, 4, 4)
        
        # Prepare Gaussian parameters
        means3d = gs_model.get_xyz                    # (N, 3)
        scales = gs_model.get_scaling                 # (N, 3)
        quats = gs_model.get_rotation                 # (N, 4) - must be wxyz
        opacities = gs_model.get_opacity              # (N, 1) or (N,)
        
        # ✅ FIXED: Ensure correct tensor shapes for gsplat v1.2.0
        # opacities must be (N,) not (N, 1)
        if opacities.dim() == 2:
            opacities = opacities.squeeze(-1)
        
        # ✅ FIXED: Normalize quaternions (gsplat expects normalized wxyz)
        quats_normalized = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-8)
        
        # Get colors (DC only for initial implementation)
        colors = gs_model.get_features_dc  # (N, 3)
        
        # Rasterize using gsplat v1.2.0 API
        try:
            # gsplat.rasterization() signature:
            # (means, quats, scales, opacities, colors, viewmats, Ks, width, height, ...)
            rendered_images, alphas, meta = gsplat.rasterization(
                means=means3d.contiguous(),
                quats=quats_normalized.contiguous(),
                scales=scales.contiguous(),
                opacities=opacities.contiguous(),
                colors=colors.contiguous(),
                viewmats=w2c.contiguous(),
                Ks=K.contiguous(),
                width=W,
                height=H,
                near_plane=0.01,
                far_plane=100.0,
                packed=False,
                render_mode="RGB"
            )
            
            # rendered_images: (B, H, W, 3) -> (3, H, W)
            image = rendered_images[0].permute(2, 0, 1)
                
        except Exception as e:
            logger.error(f"Rasterization failed at iteration {i}: {e}")
            raise e
            
        # Loss: L1 + optional SSIM
        l1_loss = torch.nn.functional.l1_loss(image, gt_image.to(device))
        loss = l1_loss
        
        # Optional: Add SSIM loss for better perceptual quality
        # from pytorch_msssim import ssim
        # ssim_loss = 1 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0).to(device), data_range=1.0)
        # loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "View": f"{view_idx}/{len(cameras)}"
        })
            
    progress_bar.close()
    logger.info(f"Optimization complete. Final loss: {loss.item():.4f}")
    return gs_model
