"""
Service wrapper for MiDaS / ZoeDepth.

Responsibilities:
- Load Depth estimation model
- Generate depth maps from RGB images
- Provide depth cues for the refinement module
- Compute confidence for fusion quality
"""

import os
import sys
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# Add ai_modules to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ai_modules'))

from midas_depth import (
    DepthEstimator,
    estimate_depth,
    estimate_depth_confidence,
    align_depth_scales,
    weighted_depth_fusion,
    save_depth_visualization,
    save_depth_grayscale,
    save_depth_raw,
    load_depth_raw,
)


class DepthService:
    """
    Service for depth estimation and processing.
    
    Provides a high-level API for the backend to:
    - Estimate depth from images
    - Compute confidence maps
    - Align and fuse multi-view depths
    """
    
    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: Optional[str] = None
    ):
        """
        Initialize the depth service.
        
        Args:
            model_type: "MiDaS_small" (fast) or "DPT_Large" (quality)
            device: "cuda", "cpu", or None for auto-detection
        """
        self.model_type = model_type
        self.device = device
        self._estimator = None  # Lazy load
    
    @property
    def estimator(self) -> DepthEstimator:
        """Lazy-load the depth estimator."""
        if self._estimator is None:
            self._estimator = DepthEstimator(
                model_type=self.model_type,
                device=self.device
            )
        return self._estimator
    
    def estimate_depth(self, image_path: str, normalize: bool = True) -> np.ndarray:
        """
        Estimate depth map from an image.
        
        Args:
            image_path: Path to input RGB image
            normalize: If True, normalize output to [0, 1]
        
        Returns:
            depth: Depth map (H, W), float32
        """
        return self.estimator.estimate(image_path, normalize=normalize)
    
    def estimate_depth_with_confidence(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth and confidence from an image.
        
        Args:
            image_path: Path to input RGB image
        
        Returns:
            depth: Depth map (H, W), float32, [0, 1]
            confidence: Confidence map (H, W), float32, [0, 1]
        """
        # Estimate depth
        depth = self.estimator.estimate(image_path, normalize=True)
        
        # Load RGB for texture-based confidence
        rgb = np.array(Image.open(image_path).convert("RGB"))
        
        # Compute confidence
        confidence = estimate_depth_confidence(depth, rgb)
        
        return depth, confidence

    async def estimate_depth_batch(self, views_data: dict, output_dir: str) -> dict:
        """
        Estimate depth for a batch of views.
        Required by PipelineManager.
        """
        try:
            depth_paths = {}
            os.makedirs(output_dir, exist_ok=True)
            
            for view_name, image_path in views_data.items():
                try:
                    # Estimate depth
                    depth = self.estimate_depth(image_path)
                    
                    # Save output
                    base_name = f"depth_{view_name}"
                    paths = self.save_outputs(depth, output_dir, base_name=base_name)
                    
                    if 'npy' in paths:
                        depth_paths[view_name] = paths['npy']
                        
                except Exception as e:
                    # Log but continue
                    print(f"Failed to estimate depth for {view_name}: {e}")
                    continue
                    
            return {
                'success': True,
                'depth_paths': depth_paths
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def align_multi_view_depths(
        self,
        depth_maps: list,
        method: str = "median"
    ) -> list:
        """
        Align multiple depth maps to consistent scale.
        
        Args:
            depth_maps: List of depth arrays from different views
            method: "median" (robust) or "mean"
        
        Returns:
            aligned: List of aligned depth maps
        """
        return align_depth_scales(depth_maps, method=method)
    
    def fuse_depths(
        self,
        depths: list,
        confidences: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse multiple depth maps using confidence-weighted averaging.
        
        Args:
            depths: List of aligned depth maps
            confidences: List of confidence maps
        
        Returns:
            fused_depth: Combined depth map
            fused_confidence: Combined confidence
        """
        return weighted_depth_fusion(depths, confidences)
    
    def save_outputs(
        self,
        depth: np.ndarray,
        output_dir: str,
        base_name: str = "depth",
        save_raw: bool = True,
        save_visual: bool = True
    ) -> dict:
        """
        Save depth outputs to files.
        
        Args:
            depth: Depth map to save
            output_dir: Output directory
            base_name: Base filename (without extension)
            save_raw: Save .npy file
            save_visual: Save .png visualizations
        
        Returns:
            paths: Dictionary of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        
        if save_raw:
            npy_path = os.path.join(output_dir, f"{base_name}.npy")
            save_depth_raw(depth, npy_path)
            paths["npy"] = npy_path
        
        if save_visual:
            gray_path = os.path.join(output_dir, f"{base_name}_gray.png")
            color_path = os.path.join(output_dir, f"{base_name}_color.png")
            save_depth_grayscale(depth, gray_path)
            save_depth_visualization(depth, color_path)
            paths["gray"] = gray_path
            paths["color"] = color_path
        
        return paths
    
    def load_depth(self, npy_path: str) -> np.ndarray:
        """
        Load depth map from .npy file.
        
        Args:
            npy_path: Path to .npy file
        
        Returns:
            depth: Depth map array
        """
        return load_depth_raw(npy_path)


# Singleton instance for convenience
_depth_service: Optional[DepthService] = None


def get_depth_service(
    model_type: str = "MiDaS_small",
    device: Optional[str] = None
) -> DepthService:
    """
    Get or create a shared DepthService instance.
    
    Args:
        model_type: Model type for depth estimation
        device: Device to use
    
    Returns:
        Shared DepthService instance
    """
    global _depth_service
    if _depth_service is None:
        _depth_service = DepthService(model_type=model_type, device=device)
    return _depth_service
