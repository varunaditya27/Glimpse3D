"""
ai_modules/refine_module/depth_consistency.py

Depth Consistency Validation for MVCRM.

★ CRITICAL COMPONENT ★
Prevents geometric violations during back-projection by validating that
depth estimates from MiDaS align with rendered depth from the 3D model.

Responsibilities:
- Compare projected depth of 3D model with estimated depth of enhanced view
- Generate consistency masks for valid update regions
- Compute adaptive thresholds based on depth confidence
- Prevent "floating artifacts" and geometric contradictions

Author: Glimpse3D Team
Date: January 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DepthConsistencyResult:
    """Results from depth consistency check."""
    consistency_mask: torch.Tensor  # (H, W) binary mask of valid regions
    depth_error: torch.Tensor  # (H, W) absolute depth differences
    consistency_score: float  # Overall consistency metric [0, 1]
    valid_pixel_ratio: float  # Fraction of pixels passing consistency check


class DepthConsistencyChecker:
    """
    Validates geometric consistency between rendered and estimated depth.
    
    This prevents the refinement module from introducing impossible geometry
    by ensuring that enhanced views are consistent with the underlying 3D structure.
    
    Usage:
        checker = DepthConsistencyChecker(threshold=0.1)
        result = checker.check(rendered_depth, midas_depth, confidence_map)
        valid_mask = result.consistency_mask
    """
    
    def __init__(
        self,
        base_threshold: float = 0.1,
        adaptive_scaling: bool = True,
        confidence_weighted: bool = True,
        min_valid_ratio: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize depth consistency checker.
        
        Args:
            base_threshold: Base depth error threshold (0.05-0.15 recommended)
            adaptive_scaling: Use adaptive thresholds based on depth range
            confidence_weighted: Weight thresholds by depth confidence
            min_valid_ratio: Minimum fraction of valid pixels required
            device: Computing device
        """
        self.base_threshold = base_threshold
        self.adaptive_scaling = adaptive_scaling
        self.confidence_weighted = confidence_weighted
        self.min_valid_ratio = min_valid_ratio
        self.device = torch.device(device)
    
    def check(
        self,
        rendered_depth: torch.Tensor,
        estimated_depth: torch.Tensor,
        confidence_map: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> DepthConsistencyResult:
        """
        Check depth consistency between rendered and estimated depth maps.
        
        Args:
            rendered_depth: (H, W) depth from 3D model rendering
            estimated_depth: (H, W) depth from MiDaS/ZoeDepth
            confidence_map: (H, W) confidence weights [0, 1]
            mask: (H, W) optional region mask to check
        
        Returns:
            DepthConsistencyResult with validation masks and metrics
        """
        # Ensure tensors on correct device
        rendered_depth = self._to_device(rendered_depth)
        estimated_depth = self._to_device(estimated_depth)
        
        H, W = rendered_depth.shape
        
        # Apply region mask if provided
        if mask is not None:
            mask = self._to_device(mask)
        else:
            mask = torch.ones_like(rendered_depth, dtype=torch.bool)
        
        # Initialize confidence map if not provided
        if confidence_map is None:
            confidence_map = torch.ones_like(rendered_depth)
        else:
            confidence_map = self._to_device(confidence_map)
        
        # 1. Align depth scales (MiDaS outputs relative depth)
        aligned_estimated = self._align_depth_scales(
            estimated_depth, rendered_depth, mask
        )
        
        # 2. Compute depth error
        depth_error = torch.abs(rendered_depth - aligned_estimated)
        
        # 3. Compute adaptive threshold
        threshold = self._compute_adaptive_threshold(
            rendered_depth, confidence_map, mask
        )
        
        # 4. Generate consistency mask
        consistency_mask = (depth_error < threshold) & mask
        
        # 5. Apply morphological operations to clean up mask
        consistency_mask = self._clean_mask(consistency_mask)
        
        # 6. Compute metrics
        valid_pixels = consistency_mask.sum().item()
        total_pixels = mask.sum().item()
        valid_ratio = valid_pixels / max(total_pixels, 1)
        
        # Average error in valid regions
        if valid_pixels > 0:
            consistency_score = 1.0 - torch.mean(
                depth_error[consistency_mask]
            ).item()
        else:
            consistency_score = 0.0
        
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return DepthConsistencyResult(
            consistency_mask=consistency_mask,
            depth_error=depth_error,
            consistency_score=consistency_score,
            valid_pixel_ratio=valid_ratio
        )
    
    def _align_depth_scales(
        self,
        source_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Align depth scales using least-squares fitting.
        
        MiDaS produces relative depth, so we align it to rendered depth scale.
        Uses: aligned = scale * source + shift
        """
        # Get valid pixels for alignment
        valid = mask & (source_depth > 0) & (target_depth > 0)
        
        if valid.sum() < 100:  # Need sufficient points
            # Fallback: simple normalization
            s_min, s_max = source_depth[valid].min(), source_depth[valid].max()
            t_min, t_max = target_depth[valid].min(), target_depth[valid].max()
            
            normalized = (source_depth - s_min) / (s_max - s_min + 1e-6)
            aligned = normalized * (t_max - t_min) + t_min
            return aligned
        
        # Least-squares alignment: solve for scale and shift
        s = source_depth[valid].unsqueeze(1)  # (N, 1)
        t = target_depth[valid].unsqueeze(1)  # (N, 1)
        
        # Build design matrix [source, ones]
        A = torch.cat([s, torch.ones_like(s)], dim=1)  # (N, 2)
        
        # Solve: [scale, shift] = (A^T A)^-1 A^T t
        try:
            params = torch.linalg.lstsq(A, t).solution  # (2, 1)
            scale, shift = params[0, 0], params[1, 0]
            
            # Sanity check on scale
            if scale < 0.1 or scale > 10.0:
                scale = 1.0
                shift = 0.0
        except:
            scale = 1.0
            shift = 0.0
        
        aligned = scale * source_depth + shift
        return aligned
    
    def _compute_adaptive_threshold(
        self,
        depth: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive threshold based on depth range and confidence.
        
        Returns threshold map: (H, W) with per-pixel thresholds
        """
        if not self.adaptive_scaling and not self.confidence_weighted:
            # Fixed threshold
            return torch.full_like(depth, self.base_threshold)
        
        threshold = torch.full_like(depth, self.base_threshold)
        
        # Adaptive scaling: larger errors allowed for distant points
        if self.adaptive_scaling:
            depth_range = depth[mask].max() - depth[mask].min() + 1e-6
            # Scale threshold by local depth (0.5% - 2% of depth range)
            relative_scale = 0.01 * depth / depth_range
            threshold = threshold + relative_scale
        
        # Confidence weighting: stricter in high-confidence regions
        if self.confidence_weighted:
            # Low confidence (0.0) -> 2x threshold, High confidence (1.0) -> 1x threshold
            confidence_scale = 2.0 - confidence
            threshold = threshold * confidence_scale
        
        return threshold
    
    def _clean_mask(
        self,
        mask: torch.Tensor,
        erosion_kernel: int = 3,
        dilation_kernel: int = 5
    ) -> torch.Tensor:
        """
        Apply morphological operations to remove noise from mask.
        
        Erosion removes small isolated valid regions.
        Dilation fills small holes in valid regions.
        """
        # Convert to float for conv operations
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Erosion (minimum filter)
        if erosion_kernel > 1:
            kernel = torch.ones(
                1, 1, erosion_kernel, erosion_kernel,
                device=self.device
            ) / (erosion_kernel ** 2)
            mask_float = F.conv2d(
                mask_float, kernel,
                padding=erosion_kernel // 2
            )
            mask_float = (mask_float > 0.99).float()
        
        # Dilation (maximum filter)
        if dilation_kernel > 1:
            kernel = torch.ones(
                1, 1, dilation_kernel, dilation_kernel,
                device=self.device
            ) / (dilation_kernel ** 2)
            mask_float = F.conv2d(
                mask_float, kernel,
                padding=dilation_kernel // 2
            )
            mask_float = (mask_float > 0.01).float()
        
        cleaned_mask = mask_float.squeeze().bool()
        return cleaned_mask
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)
    
    def is_valid_update(self, result: DepthConsistencyResult) -> bool:
        """
        Check if consistency result meets minimum quality requirements.
        
        Args:
            result: DepthConsistencyResult from check()
        
        Returns:
            True if update should proceed, False to reject
        """
        return (
            result.valid_pixel_ratio >= self.min_valid_ratio and
            result.consistency_score >= 0.3
        )


def compute_depth_confidence(
    depth: torch.Tensor,
    method: str = "gradient"
) -> torch.Tensor:
    """
    Compute confidence map for depth estimates.
    
    Args:
        depth: (H, W) depth map
        method: "gradient" (texture-based) or "uniform"
    
    Returns:
        confidence: (H, W) confidence values [0, 1]
    """
    if method == "uniform":
        return torch.ones_like(depth)
    
    # Gradient-based confidence (low gradient = low confidence)
    grad_x = torch.abs(depth[:, 1:] - depth[:, :-1])
    grad_y = torch.abs(depth[1:, :] - depth[:-1, :])
    
    # Pad to match original size
    grad_x = F.pad(grad_x, (0, 1, 0, 0))
    grad_y = F.pad(grad_y, (0, 0, 0, 1))
    
    # Combine gradients
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Normalize to [0, 1]
    confidence = grad_mag / (grad_mag.max() + 1e-6)
    
    # Invert if needed: high gradient = high confidence
    # (depends on interpretation)
    
    return confidence


# Quick utility function for standalone usage
def check_consistency(
    projected_depth: np.ndarray,
    estimated_depth: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Simple depth consistency check (numpy interface).
    
    Args:
        projected_depth: (H, W) rendered depth from 3D model
        estimated_depth: (H, W) MiDaS depth estimate
        threshold: Maximum allowed relative error
    
    Returns:
        mask: (H, W) binary mask of consistent regions
    """
    checker = DepthConsistencyChecker(base_threshold=threshold)
    
    # Convert to torch
    p_depth = torch.from_numpy(projected_depth).float()
    e_depth = torch.from_numpy(estimated_depth).float()
    
    result = checker.check(p_depth, e_depth)
    
    return result.consistency_mask.cpu().numpy()
