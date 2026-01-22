import sys
import os
import torch
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from ai_modules.gsplat.utils_gs import GaussianModel
from ai_modules.gsplat.optimize import refine_model

def test_loop():
    print("Initializing Dummy Model...")
    # Create dummy model
    gs_model = GaussianModel()
    N = 100
    gs_model._xyz = torch.rand(N, 3).cuda()
    gs_model._features_dc = torch.rand(N, 3).cuda()
    gs_model._features_rest = torch.zeros(N, 45).cuda()
    gs_model._scaling = torch.rand(N, 3).cuda()
    gs_model._rotation = torch.rand(N, 4).cuda()
    gs_model._rotation /= gs_model._rotation.norm(dim=-1, keepdim=True)
    gs_model._opacity = torch.rand(N, 1).cuda()
    
    # Dummy Cameras
    H, W = 256, 256
    K = torch.eye(3).cuda()
    K[0, 0] = 200
    K[1, 1] = 200
    K[0, 2] = W/2
    K[1, 2] = H/2
    
    w2c = torch.eye(4).cuda()
    w2c[2, 3] = 2.0
    
    cameras = [{
        'image_height': H,
        'image_width': W,
        'K': K,
        'w2c': w2c
    }]
    
    # Dummy Ground Truth
    target_images = [torch.rand(3, H, W).cuda()]
    
    print("Starting Optimization Loop (5 iterations)...")
    try:
        refine_model(gs_model, target_images, cameras, iterations=5)
        print("Optimization Loop Completed Successfully!")
    except Exception as e:
        print(f"Optimization Loop Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        sys.exit(0)
    test_loop()
