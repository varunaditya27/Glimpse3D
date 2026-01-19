import torch
import numpy as np

class MidasWrapper:
    """
    Wrapper for MiDaS Depth Estimation.
    """
    def __init__(self, model_type="DPT_Hybrid", device="cuda"):
        self.device = device
        self.model_type = model_type
        self.model = None
        self.transform = None

    def load_model(self):
        if self.model is None:
            try:
                # Load from torch hub for simplicity and robustness
                self.model = torch.hub.load("intel-isl/MiDaS", self.model_type).to(self.device)
                self.model.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                
                if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                    self.transform = midas_transforms.dpt_transform
                else:
                    self.transform = midas_transforms.small_transform
            except Exception as e:
                print(f"Failed to load MiDaS: {e}")
                
    def estimate_depth(self, image: np.ndarray):
        """
        Estimate depth from a single image.
        
        Args:
            image: (H, W, 3) numpy array [0-255]
            
        Returns:
            depth: (H, W) torch tensor (inverse depth / disparity)
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                return torch.zeros((image.shape[0], image.shape[1])).to(self.device)

        input_batch = self.transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        return prediction
