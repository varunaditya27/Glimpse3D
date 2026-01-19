
import torch
import math

def build_intrinsics(fov_y, image_width, image_height):
    """
    Builds a (3,3) intrinsics matrix K for a pinhole camera.
    """
    # tan(theta/2) = (H/2) / fy
    # fy = (H/2) / tan(theta/2)
    
    fy = (image_height / 2.0) / math.tan(math.radians(fov_y / 2.0))
    fx = fy # Square pixels assumed
    
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    return K
