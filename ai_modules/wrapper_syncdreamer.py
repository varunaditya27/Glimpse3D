import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image

# Add project root to path to ensure ai_modules is importable
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from ai_modules.sync_dreamer.inference import generate_multiview as sd_generate
except ImportError:
    # Fallback/Mock if submodule has issues
    print("Warning: Could not import SyncDreamer. Using Mock.")
    sd_generate = None

class SyncDreamerWrapper:
    """
    Wrapper for SyncDreamer Multi-View Generation.
    
    Generates 16 consistent views from a single input image.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.model_path = os.path.join(os.path.dirname(__file__), "sync_dreamer", "checkpoints", "syncdreamer-pretrain.ckpt")
        # Actual loading could happen here or lazy loaded
        
    def generate(self, input_image: np.ndarray, num_views: int = 16):
        """
        Generate multi-view images.
        
        Args:
            input_image: (H, W, 3) numpy array, RGB, 0-1 or 0-255
            num_views: Number of views to generate (SyncDreamer defaults to 16)
            
        Returns:
            images: List of (H, W, 3) tensors
            cameras: List of camera parameters (azimuth, elevation, distance)
        """
        if sd_generate is None:
            return self._mock_generate(input_image, num_views)

        # TODO: Call actual SyncDreamer inference
        # This requires setting up the config and model loading which is complex 
        # inside the submodule. For this 'integration' phase, we will wrap 
        # the inference call if available, or mock if still missing weights.
        
        # Check if checkpoint exists
        if not os.path.exists(self.model_path):
             print(f"SyncDreamer Checkpoint not found at {self.model_path}")
             return self._mock_generate(input_image, num_views)

        # Real inference logic would go here
        # For now, to allow the pipeline to flow without downloading 2GB weights immediately:
        return self._mock_generate(input_image, num_views)

    def _mock_generate(self, input_image, num_views):
        """Generate dummy views for testing pipeline flow."""
        H, W = 512, 512
        images = []
        import torch
        for i in range(num_views):
             # Create simple color rotation to distinguish views
             img = torch.zeros((H, W, 3))
             img[:, :, 0] = (i / num_views) # Red channel varies
             img[:, :, 1] = 0.5 
             images.append(img)
             
        # Mock cameras: 16 views in a circle?
        # SyncDreamer standard is usually elevation 0, random azimuths or fixed steps?
        # Typically fixed steps of 360/16 = 22.5 degrees.
        cameras = [] 
        # We will return raw data, the pipeline converts to specific CameraParams
        return images, cameras
