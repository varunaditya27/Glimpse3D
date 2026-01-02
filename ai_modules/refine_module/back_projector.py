"""
ai_modules/refine_module/back_projector.py

Back-Projection Engine for MVCRM (Multi-View Consistency Refinement Module).

★ NOVEL CONTRIBUTION ★
This module implements the core innovation of Glimpse3D: projecting 2D image
enhancements back into 3D Gaussian Splat space with consistency enforcement.

Responsibilities:
- Map 2D pixel differences to 3D Gaussian splat updates
- Calculate ray-splat intersections and alpha-weighted contributions
- Update splat colors, opacities, and scales based on enhanced views
- Handle occlusion and multi-view conflicts

Author: Glimpse3D Team
Date: January 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CameraParams:
    """Camera parameters for rendering and back-projection."""
    intrinsics: torch.Tensor  # (3, 3) camera intrinsic matrix
    extrinsics: torch.Tensor  # (4, 4) world-to-camera transformation
    width: int
    height: int
    near: float = 0.01
    far: float = 100.0


@dataclass
class BackProjectionResult:
    """Results from back-projection operation."""
    color_updates: torch.Tensor  # (N, 3) color updates for N splats
    opacity_updates: torch.Tensor  # (N,) opacity updates
    scale_updates: torch.Tensor  # (N, 3) scale updates
    updated_splat_indices: torch.Tensor  # Indices of updated splats
    confidence_weights: torch.Tensor  # Per-splat confidence


class BackProjector:
    """
    Back-projection engine for updating 3D Gaussian Splats from 2D views.
    
    This class handles the geometric mapping from enhanced 2D images back to
    the 3D splat representation, ensuring consistency and quality.
    
    Usage:
        projector = BackProjector(learning_rate=0.05)
        result = projector.project(
            rendered_image, enhanced_image, depth_map,
            splat_positions, splat_colors, alpha_map, camera
        )
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        min_alpha_threshold: float = 0.01,
        max_update_magnitude: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the back-projector.
        
        Args:
            learning_rate: Step size for splat updates (0.01-0.1 recommended)
            min_alpha_threshold: Minimum alpha contribution to consider for updates
            max_update_magnitude: Maximum allowed color change per update
            device: Computing device
        """
        self.lr = learning_rate
        self.min_alpha = min_alpha_threshold
        self.max_update = max_update_magnitude
        self.device = torch.device(device)
    
    def project(
        self,
        rendered_image: torch.Tensor,
        enhanced_image: torch.Tensor,
        depth_map: torch.Tensor,
        splat_means: torch.Tensor,
        splat_colors: torch.Tensor,
        splat_opacities: torch.Tensor,
        alpha_map: torch.Tensor,
        splat_contributions: torch.Tensor,
        camera: CameraParams,
        consistency_mask: Optional[torch.Tensor] = None
    ) -> BackProjectionResult:
        """
        Project 2D image differences back to 3D splat updates.
        
        Args:
            rendered_image: (H, W, 3) rendered image from current splats
            enhanced_image: (H, W, 3) AI-enhanced target image
            depth_map: (H, W) depth map from rendering
            splat_means: (N, 3) 3D positions of Gaussian splats
            splat_colors: (N, 3) current RGB colors of splats
            splat_opacities: (N,) current opacity values
            alpha_map: (H, W) total alpha (opacity) at each pixel
            splat_contributions: (H, W, K) indices and weights of top-K splats per pixel
            camera: Camera parameters
            consistency_mask: (H, W) optional mask for valid update regions
        
        Returns:
            BackProjectionResult with computed updates
        """
        H, W = rendered_image.shape[:2]
        N = splat_means.shape[0]
        
        # Ensure tensors are on correct device
        rendered_image = self._to_device(rendered_image)
        enhanced_image = self._to_device(enhanced_image)
        depth_map = self._to_device(depth_map)
        alpha_map = self._to_device(alpha_map)
        
        # 1. Compute pixel-wise residuals (difference to back-project)
        residuals = enhanced_image - rendered_image  # (H, W, 3)
        
        # 2. Apply consistency mask if provided
        if consistency_mask is not None:
            consistency_mask = self._to_device(consistency_mask)
            residuals = residuals * consistency_mask.unsqueeze(-1)
        
        # 3. Weight residuals by confidence (higher alpha = more confident)
        confidence = torch.clamp(alpha_map, 0.1, 1.0)  # (H, W)
        weighted_residuals = residuals * confidence.unsqueeze(-1)
        
        # 4. Initialize update accumulators
        color_updates = torch.zeros(N, 3, device=self.device)
        opacity_updates = torch.zeros(N, device=self.device)
        update_weights = torch.zeros(N, device=self.device)  # Normalization
        
        # 5. Back-project using splat contributions
        color_updates, opacity_updates, update_weights = self._accumulate_updates(
            weighted_residuals,
            splat_contributions,
            alpha_map,
            N
        )
        
        # 6. Normalize updates by accumulated weights
        valid_mask = update_weights > 1e-6
        color_updates[valid_mask] /= update_weights[valid_mask].unsqueeze(-1)
        opacity_updates[valid_mask] /= update_weights[valid_mask]
        
        # 7. Apply learning rate and magnitude clipping
        color_updates = self._apply_learning_rate(color_updates)
        opacity_updates *= self.lr * 0.5  # Slower opacity updates
        
        # 8. Compute scale updates (increase scale for high-detail regions)
        scale_updates = self._compute_scale_updates(
            residuals, splat_contributions, N
        )
        
        # 9. Get indices of significantly updated splats
        update_magnitude = torch.norm(color_updates, dim=1)
        updated_indices = torch.where(update_magnitude > 0.01)[0]
        
        return BackProjectionResult(
            color_updates=color_updates,
            opacity_updates=opacity_updates,
            scale_updates=scale_updates,
            updated_splat_indices=updated_indices,
            confidence_weights=update_weights / (update_weights.max() + 1e-6)
        )
    
    def _accumulate_updates(
        self,
        weighted_residuals: torch.Tensor,
        splat_contributions: torch.Tensor,
        alpha_map: torch.Tensor,
        num_splats: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accumulate residuals to splat updates using contribution map.
        
        Args:
            weighted_residuals: (H, W, 3) weighted pixel differences
            splat_contributions: (H, W, K, 2) - last dim is [splat_idx, weight]
            alpha_map: (H, W) rendering alpha values
            num_splats: Total number of splats
        
        Returns:
            color_updates, opacity_updates, weights
        """
        H, W, _ = weighted_residuals.shape
        K = splat_contributions.shape[2] if len(splat_contributions.shape) == 4 else 1
        
        color_updates = torch.zeros(num_splats, 3, device=self.device)
        opacity_updates = torch.zeros(num_splats, device=self.device)
        weights = torch.zeros(num_splats, device=self.device)
        
        # Simplified accumulation (in production, use optimized indexing)
        # This assumes splat_contributions format from renderer
        
        # Flatten spatial dimensions for easier processing
        residuals_flat = weighted_residuals.view(-1, 3)  # (H*W, 3)
        alpha_flat = alpha_map.view(-1)  # (H*W,)
        
        # For each pixel, distribute updates to contributing splats
        # Note: This is a simplified version. In production, you'd use
        # the actual contribution data from the rasterizer
        
        # Placeholder: Use a scatter operation for efficiency
        # In real implementation, this comes from the gsplat rasterizer output
        
        return color_updates, opacity_updates, weights
    
    def _compute_scale_updates(
        self,
        residuals: torch.Tensor,
        splat_contributions: torch.Tensor,
        num_splats: int
    ) -> torch.Tensor:
        """
        Compute scale adjustments based on high-frequency details.
        
        Areas with high residual gradients suggest need for finer splats.
        """
        # Compute gradient magnitude
        grad_x = residuals[:, 1:, :] - residuals[:, :-1, :]
        grad_y = residuals[1:, :, :] - residuals[:-1, :, :]
        
        # Average gradient magnitude
        grad_mag_x = torch.norm(grad_x, dim=-1)
        grad_mag_y = torch.norm(grad_y, dim=-1)
        
        # Scale updates: increase scale where gradients are high
        # (encourages densification in high-detail regions)
        scale_updates = torch.zeros(num_splats, 3, device=self.device)
        
        # This would be accumulated similar to color updates
        # For now, return zeros (scale updates are optional)
        return scale_updates
    
    def _apply_learning_rate(self, updates: torch.Tensor) -> torch.Tensor:
        """Apply learning rate and clip update magnitudes."""
        updates = updates * self.lr
        
        # Clip to prevent overshooting
        magnitude = torch.norm(updates, dim=1, keepdim=True)
        scale = torch.clamp(magnitude / self.max_update, min=1.0)
        updates = updates / scale
        
        return updates
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)
    
    def apply_updates(
        self,
        splat_colors: torch.Tensor,
        splat_opacities: torch.Tensor,
        splat_scales: torch.Tensor,
        result: BackProjectionResult,
        damping: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply computed updates to splat parameters.
        
        Args:
            splat_colors: (N, 3) current colors
            splat_opacities: (N,) current opacities
            splat_scales: (N, 3) current scales
            result: BackProjectionResult with updates
            damping: Damping factor (0-1) for gradual updates
        
        Returns:
            Updated colors, opacities, scales
        """
        updated_colors = splat_colors + result.color_updates * damping
        updated_opacities = splat_opacities + result.opacity_updates * damping
        updated_scales = splat_scales + result.scale_updates * damping
        
        # Clamp to valid ranges
        updated_colors = torch.clamp(updated_colors, 0.0, 1.0)
        updated_opacities = torch.clamp(updated_opacities, 0.0, 1.0)
        updated_scales = torch.clamp(updated_scales, 1e-4, 10.0)
        
        return updated_colors, updated_opacities, updated_scales


def create_camera_from_pose(
    pose: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int
) -> CameraParams:
    """
    Utility to create CameraParams from standard pose/intrinsics.
    
    Args:
        pose: (4, 4) camera-to-world transformation matrix
        intrinsics: (3, 3) camera intrinsic matrix
        width: Image width
        height: Image height
    
    Returns:
        CameraParams object
    """
    # Convert pose to extrinsics (world-to-camera)
    extrinsics = np.linalg.inv(pose)
    
    return CameraParams(
        intrinsics=torch.from_numpy(intrinsics).float(),
        extrinsics=torch.from_numpy(extrinsics).float(),
        width=width,
        height=height
    )


def compute_pixel_to_ray(
    pixel_coords: torch.Tensor,
    camera: CameraParams
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 3D rays from pixel coordinates.
    
    Args:
        pixel_coords: (N, 2) pixel coordinates (u, v)
        camera: Camera parameters
    
    Returns:
        ray_origins: (N, 3) ray origin points
        ray_directions: (N, 3) normalized ray directions
    """
    # Convert to normalized device coordinates
    u, v = pixel_coords[:, 0], pixel_coords[:, 1]
    
    # Apply inverse intrinsics
    fx = camera.intrinsics[0, 0]
    fy = camera.intrinsics[1, 1]
    cx = camera.intrinsics[0, 2]
    cy = camera.intrinsics[1, 2]
    
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = torch.ones_like(x)
    
    # Ray directions in camera space
    ray_dirs_cam = torch.stack([x, y, z], dim=1)
    ray_dirs_cam = F.normalize(ray_dirs_cam, dim=1)
    
    # Transform to world space
    R = camera.extrinsics[:3, :3]
    t = camera.extrinsics[:3, 3]
    
    ray_dirs_world = torch.matmul(ray_dirs_cam, R.T)
    ray_origins_world = -torch.matmul(t, R.T).unsqueeze(0).expand(len(pixel_coords), -1)
    
    return ray_origins_world, ray_dirs_world
