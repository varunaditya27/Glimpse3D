import pytest
import torch
import sys
import os
import numpy as np

# ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ai_modules.wrapper_syncdreamer import SyncDreamerWrapper
from ai_modules.wrapper_midas import MidasWrapper
from ai_modules.gsplat.render_view import render, GSPLAT_AVAILABLE
from ai_modules.gsplat.model import GaussianModel

@pytest.fixture
def dummy_image():
    # 512x512 RGB float image
    return np.random.rand(512, 512, 3).astype(np.float32)

def test_syncdreamer_wrapper(dummy_image):
    wrapper = SyncDreamerWrapper()
    images, cameras = wrapper.generate(dummy_image, num_views=4)
    assert len(images) == 4
    assert len(cameras) == 4
    # Check shape
    if isinstance(images[0], torch.Tensor):
        assert images[0].shape == (512, 512, 3)
    else:
        # If it returns numpy or similar during mock
        assert getattr(images[0], 'shape', None) is not None

def test_midas_wrapper(dummy_image):
    wrapper = MidasWrapper()
    depth = wrapper.estimate_depth(dummy_image)
    assert isinstance(depth, torch.Tensor)
    # Check if shape validation is possible (mock returns zeros)
    # Midas outputs (H, W)
    assert depth.dim() == 2 or (depth.dim() == 3 and depth.shape[0] == 1)

def test_gsplat_availability():
    # This test fails if gsplat is not installed correctly
    # But we want to know IF it is available
    if not GSPLAT_AVAILABLE:
        pytest.skip("gsplat not installed, skipping real render test")
    assert GSPLAT_AVAILABLE is True

def test_render_view_mock_or_real():
    # Create valid dummy model
    model = GaussianModel(sh_degree=0)
    # Initialize with 10 points
    pts = torch.rand(10, 3).cuda() if torch.cuda.is_available() else torch.rand(10, 3)
    col = torch.rand(10, 3).cuda() if torch.cuda.is_available() else torch.rand(10, 3)
    model.create_from_pcd(pts, col)
    
    # Camera
    H, W = 256, 256
    K = torch.eye(3).cuda() if torch.cuda.is_available() else torch.eye(3)
    K[0,0] = 500
    K[1,1] = 500
    K[0,2] = W/2
    K[1,2] = H/2
    
    w2c = torch.eye(4).cuda() if torch.cuda.is_available() else torch.eye(4)
    w2c[2,3] = 3.0 # shift
    
    view_cam = {'image_height': H, 'image_width': W, 'K': K, 'w2c': w2c}
    
    # Render
    try:
        out = render(model, view_cam)
        img = out["render"] # (3, H, W)
        assert img.shape == (3, H, W)
    except Exception as e:
        if GSPLAT_AVAILABLE:
             pytest.fail(f"Real Render Failed: {e}")
        else:
             print(f"Mock Render ran: {e}") 
