"""
ai_modules/midas_depth/run_depth.py

Core logic for Monocular Depth Estimation.

Responsibilities:
- Load MiDaS or ZoeDepth model
- Predict depth map from single RGB image
- Normalize depth values for 3D consistency
- Provide utilities for depth visualization and saving
"""

import os
from typing import Union, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class DepthEstimator: #handles the MiDas inference
    """
    A class to handle depth estimation using MiDaS models.
    
    Supports multiple model types:
    - "MiDaS_small": Fastest, lower quality (~50ms on GPU)
    - "DPT_Hybrid": Balanced speed/quality (~150ms on GPU)
    - "DPT_Large": Highest quality, slower (~300ms on GPU)
    
    Usage:
        estimator = DepthEstimator(model_type="MiDaS_small")
        depth_map = estimator.estimate(image_path)
    """
    
    # Supported model types and their corresponding transforms
    MODEL_TRANSFORMS = {
        "MiDaS_small": "small_transform",
        "DPT_Hybrid": "dpt_transform",
        "DPT_Large": "dpt_transform",
        "DPT_BEiT_L_512": "beit512_transform",
        "DPT_BEiT_L_384": "dpt_transform",
    }
    
    def __init__(
        self,
        model_type: str = "MiDaS_small", #fallback option of the model
        device: Optional[str] = None, #cuda if available, else cpu 
        optimize: bool = True
    ):
        """
        Initialize the depth estimator.
        
        Args:
            model_type: One of "MiDaS_small", "DPT_Hybrid", "DPT_Large", etc.
            device: "cuda", "cpu", or None for auto-detection
            optimize: If True, enables optimization for inference (half precision on GPU)
        """
        if model_type not in self.MODEL_TRANSFORMS:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Choose from: {list(self.MODEL_TRANSFORMS.keys())}"
            )
        
        self.model_type = model_type
        self.optimize = optimize
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and transforms
        self._load_model()
    
    def _load_model(self):
        """Load MiDaS model and transforms from torch hub."""
        print(f"Loading MiDaS model: {self.model_type} on {self.device}...")
        
        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # Optimization: use half precision on GPU for faster inference
        if self.optimize and self.device.type == "cuda":
            self.model = self.model.half()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform_name = self.MODEL_TRANSFORMS[self.model_type]
        self.transform = getattr(midas_transforms, transform_name)
        
        print(f"Model loaded successfully!")
    
    def _load_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Load and convert image to numpy array.
        
        Args:
            image: File path, PIL Image, or numpy array
            
        Returns:
            numpy array (H, W, 3) uint8
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            img = Image.open(image).convert("RGB")
            return np.array(img)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale -> RGB
                return np.stack([image] * 3, axis=-1).astype(np.uint8)
            elif image.shape[2] == 4:
                # RGBA -> RGB
                return image[:, :, :3].astype(np.uint8)
            return image.astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def estimate(
        self,
        image: Union[str, Image.Image, np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Estimate depth from an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            normalize: If True, normalize output to [0, 1] range
            
        Returns:
            depth_map: numpy array (H, W), float32
                - If normalize=True: values in [0, 1] where 1 = closest
                - If normalize=False: raw inverse depth values
        """
        # Load and preprocess image
        img_np = self._load_image(image)
        original_size = (img_np.shape[0], img_np.shape[1])
        
        # Apply MiDaS transform
        input_tensor = self.transform(img_np).to(self.device)
        
        # Use half precision if optimized
        if self.optimize and self.device.type == "cuda":
            input_tensor = input_tensor.half()
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
            
            # Resize to original image size
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=original_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth = prediction.cpu().float().numpy()
        
        # Normalize to [0, 1] if requested
        if normalize:
            depth = self._normalize_depth(depth)
        
        return depth.astype(np.float32)
    
    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0, 1] range.
        
        Note: MiDaS outputs inverse depth (higher = closer).
        After normalization: 1 = closest, 0 = farthest
        """
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            return (depth - d_min) / (d_max - d_min)
        return np.zeros_like(depth)
    
    def estimate_batch(
        self,
        images: list,
        normalize: bool = True
    ) -> list:
        """
        Estimate depth for multiple images.
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            normalize: If True, normalize each output to [0, 1]
            
        Returns:
            List of depth maps
        """
        return [self.estimate(img, normalize=normalize) for img in images]


# Singleton instance for convenience (lazy loaded)
_default_estimator: Optional[DepthEstimator] = None


def get_estimator(model_type: str = "MiDaS_small", device: Optional[str] = None) -> DepthEstimator:
    """
    Get or create a shared DepthEstimator instance.
    
    Uses singleton pattern to avoid reloading the model multiple times.
    """
    global _default_estimator
    if _default_estimator is None or _default_estimator.model_type != model_type:
        _default_estimator = DepthEstimator(model_type=model_type, device=device)
    return _default_estimator


def estimate_depth(
    image: Union[str, Image.Image, np.ndarray],
    model_type: str = "MiDaS_small",
    device: Optional[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Convenience function to estimate depth from an image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        model_type: MiDaS model variant
        device: "cuda", "cpu", or None for auto
        normalize: If True, normalize to [0, 1]
        
    Returns:
        depth_map: numpy array (H, W), float32, values in [0, 1] if normalized
        
    Example:
        depth = estimate_depth("photo.jpg")
        save_depth_visualization(depth, "depth.png")
    """
    estimator = get_estimator(model_type=model_type, device=device)
    return estimator.estimate(image, normalize=normalize)


def save_depth_visualization(
    depth: np.ndarray,
    output_path: str,
    colormap: str = "magma"
) -> None:
    """
    Save depth map as a colored visualization.
    
    Args:
        depth: Depth map array (H, W), expected [0, 1] range
        output_path: Output file path (.png, .jpg)
        colormap: Matplotlib colormap name ("magma", "viridis", "plasma", "inferno")
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(np.clip(depth, 0, 1))
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    Image.fromarray(depth_colored).save(output_path)


def save_depth_grayscale(depth: np.ndarray, output_path: str) -> None:
    """
    Save depth map as grayscale image.
    
    Args:
        depth: Depth map array (H, W), expected [0, 1] range
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(depth_uint8).save(output_path)


def save_depth_raw(depth: np.ndarray, output_path: str) -> None:
    """
    Save depth map as numpy file for precise values.
    
    Args:
        depth: Depth map array
        output_path: Output file path (.npy)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, depth)


def load_depth_raw(input_path: str) -> np.ndarray:
    """
    Load depth map from numpy file.
    
    Args:
        input_path: Path to .npy file
        
    Returns:
        depth: Depth map array
    """
    return np.load(input_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description="MiDaS Depth Estimation - Estimate depth from a single image"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output", "-o",
        default="depth_output",
        help="Output directory (default: depth_output)"
    )
    parser.add_argument(
        "--model", "-m",
        default="MiDaS_small",
        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        help="MiDaS model type (default: MiDaS_small)"
    )
    parser.add_argument(
        "--colormap", "-c",
        default="magma",
        choices=["magma", "viridis", "plasma", "inferno", "gray"],
        help="Colormap for visualization (default: magma)"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw .npy file"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get base name for outputs
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    print("=" * 60)
    print("MiDaS Depth Estimation")
    print("=" * 60)
    print(f"Input:  {args.image}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device or 'auto'}")
    print("=" * 60)
    
    # Run estimation
    start_time = time.time()
    depth = estimate_depth(args.image, model_type=args.model, device=args.device)
    elapsed = time.time() - start_time
    
    print(f"\nInference time: {elapsed:.2f}s")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    
    # Save outputs
    color_path = os.path.join(args.output, f"{base_name}_depth_color.png")
    gray_path = os.path.join(args.output, f"{base_name}_depth_gray.png")
    
    if args.colormap == "gray":
        save_depth_grayscale(depth, gray_path)
        print(f"\nSaved: {gray_path}")
    else:
        save_depth_visualization(depth, color_path, colormap=args.colormap)
        save_depth_grayscale(depth, gray_path)
        print(f"\nSaved: {color_path}")
        print(f"Saved: {gray_path}")
    
    if args.save_raw:
        raw_path = os.path.join(args.output, f"{base_name}_depth.npy")
        save_depth_raw(depth, raw_path)
        print(f"Saved: {raw_path}")
    
    print("\n✅ Done!")
