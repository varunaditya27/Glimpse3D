"""
ai_modules/refine_module/normal_smoothing.py

Normal Smoothing and Geometry Regularization for MVCRM.

★ ARTIFACT PREVENTION ★
Prevents high-frequency noise and geometric artifacts by regularizing
surface normals and splat positions after back-projection updates.

Responsibilities:
- Compute surface normals from Gaussian splat positions
- Apply Laplacian smoothing with edge preservation
- Bilateral filtering for feature-aware smoothing
- Regularize splat scales to prevent spiky artifacts

Author: Glimpse3D Team
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from scipy.spatial import KDTree


class NormalSmoother:
    """
    Geometry regularization for Gaussian Splats.
    
    Applies smoothing to splat positions and scales while preserving sharp
    features (edges, corners) to maintain detail without introducing noise.
    
    Uses:
    - k-NN based normal estimation
    - Laplacian smoothing for flat regions
    - Bilateral filtering for edge-aware smoothing
    
    Usage:
        smoother = NormalSmoother(k=8, smoothing_strength=0.3)
        smoothed_positions = smoother.smooth_positions(positions, colors)
    """
    
    def __init__(
        self,
        k_neighbors: int = 8,
        smoothing_strength: float = 0.3,
        edge_threshold: float = 0.2,
        bilateral_sigma_spatial: float = 0.1,
        bilateral_sigma_color: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize normal smoother.
        
        Args:
            k_neighbors: Number of neighbors for normal estimation (8-16)
            smoothing_strength: Smoothing factor [0, 1] (0=no smoothing, 1=max)
            edge_threshold: Color difference threshold for edge detection
            bilateral_sigma_spatial: Spatial bandwidth for bilateral filter
            bilateral_sigma_color: Color bandwidth for bilateral filter
            device: Computing device
        """
        self.k = k_neighbors
        self.strength = smoothing_strength
        self.edge_threshold = edge_threshold
        self.sigma_spatial = bilateral_sigma_spatial
        self.sigma_color = bilateral_sigma_color
        self.device = torch.device(device)
    
    def smooth_positions(
        self,
        positions: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply smoothing to splat positions.
        
        Args:
            positions: (N, 3) 3D positions of Gaussian splats
            colors: (N, 3) RGB colors for edge detection
            normals: (N, 3) surface normals (computed if not provided)
        
        Returns:
            smoothed_positions: (N, 3) regularized positions
        """
        positions = self._to_device(positions)
        
        if colors is not None:
            colors = self._to_device(colors)
        
        # Compute normals if not provided
        if normals is None:
            normals = self.estimate_normals(positions)
        else:
            normals = self._to_device(normals)
        
        # Find k-nearest neighbors
        neighbors_idx = self._find_knn(positions)
        
        # Compute edge weights (preserve edges)
        if colors is not None:
            edge_weights = self._compute_edge_weights(colors, neighbors_idx)
        else:
            edge_weights = torch.ones(positions.shape[0], self.k, device=self.device)
        
        # Apply weighted Laplacian smoothing
        smoothed = self._laplacian_smooth(
            positions, neighbors_idx, edge_weights
        )
        
        return smoothed
    
    def smooth_normals(
        self,
        normals: torch.Tensor,
        positions: torch.Tensor,
        colors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Smooth surface normals while preserving features.
        
        Args:
            normals: (N, 3) surface normal vectors
            positions: (N, 3) 3D positions for spatial weighting
            colors: (N, 3) optional colors for edge detection
        
        Returns:
            smoothed_normals: (N, 3) regularized normals
        """
        normals = self._to_device(normals)
        positions = self._to_device(positions)
        
        # Find neighbors
        neighbors_idx = self._find_knn(positions)
        
        # Compute weights
        if colors is not None:
            colors = self._to_device(colors)
            edge_weights = self._compute_edge_weights(colors, neighbors_idx)
        else:
            edge_weights = torch.ones(normals.shape[0], self.k, device=self.device)
        
        # Smooth normals
        smoothed = torch.zeros_like(normals)
        
        for i in range(normals.shape[0]):
            neighbor_normals = normals[neighbors_idx[i]]
            weights = edge_weights[i].unsqueeze(-1)
            
            # Weighted average
            avg_normal = (neighbor_normals * weights).sum(dim=0)
            
            # Blend with original
            smoothed[i] = (
                (1 - self.strength) * normals[i] +
                self.strength * avg_normal
            )
        
        # Re-normalize
        smoothed = F.normalize(smoothed, dim=1)
        
        return smoothed
    
    def estimate_normals(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Estimate surface normals from point positions using PCA.
        
        Args:
            positions: (N, 3) 3D positions
            k: Number of neighbors (uses self.k if None)
        
        Returns:
            normals: (N, 3) estimated normal vectors
        """
        k = k or self.k
        positions = self._to_device(positions)
        
        N = positions.shape[0]
        normals = torch.zeros_like(positions)
        
        # Find neighbors
        neighbors_idx = self._find_knn(positions, k)
        
        for i in range(N):
            # Get neighbor positions
            neighbor_pos = positions[neighbors_idx[i]]
            
            # Center points
            centered = neighbor_pos - neighbor_pos.mean(dim=0)
            
            # Compute covariance matrix
            cov = centered.T @ centered
            
            # Normal is eigenvector with smallest eigenvalue
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # Smallest eigenvalue
                normals[i] = normal
            except:
                # Fallback: use average direction
                normals[i] = torch.tensor([0, 0, 1], device=self.device)
        
        # Normalize
        normals = F.normalize(normals, dim=1)
        
        return normals
    
    def _find_knn(
        self,
        positions: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Find k-nearest neighbors for each point.
        
        Args:
            positions: (N, 3) point positions
            k: Number of neighbors
        
        Returns:
            indices: (N, k) neighbor indices
        """
        k = k or self.k
        
        # Use KDTree for efficient nearest neighbor search
        positions_np = positions.cpu().numpy()
        tree = KDTree(positions_np)
        
        # Query k+1 neighbors (includes self)
        distances, indices = tree.query(positions_np, k=k+1)
        
        # Remove self (first neighbor)
        indices = indices[:, 1:]
        
        return torch.from_numpy(indices).long().to(self.device)
    
    def _compute_edge_weights(
        self,
        colors: torch.Tensor,
        neighbors_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-preserving weights based on color similarity.
        
        Args:
            colors: (N, 3) RGB colors
            neighbors_idx: (N, k) neighbor indices
        
        Returns:
            weights: (N, k) edge weights [0, 1]
        """
        N, k = neighbors_idx.shape
        weights = torch.zeros(N, k, device=self.device)
        
        for i in range(N):
            neighbor_colors = colors[neighbors_idx[i]]
            color_diff = torch.norm(neighbor_colors - colors[i], dim=1)
            
            # Convert to weights: similar colors -> high weight
            weights[i] = torch.exp(-(color_diff ** 2) / (2 * self.sigma_color ** 2))
        
        return weights
    
    def _laplacian_smooth(
        self,
        positions: torch.Tensor,
        neighbors_idx: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply weighted Laplacian smoothing.
        
        Args:
            positions: (N, 3) current positions
            neighbors_idx: (N, k) neighbor indices
            weights: (N, k) smoothing weights
        
        Returns:
            smoothed: (N, 3) smoothed positions
        """
        N = positions.shape[0]
        smoothed = torch.zeros_like(positions)
        
        for i in range(N):
            neighbor_pos = positions[neighbors_idx[i]]
            neighbor_weights = weights[i].unsqueeze(-1)
            
            # Weighted average of neighbors
            weighted_avg = (neighbor_pos * neighbor_weights).sum(dim=0) / (
                neighbor_weights.sum() + 1e-6
            )
            
            # Blend with original position
            smoothed[i] = (
                (1 - self.strength) * positions[i] +
                self.strength * weighted_avg
            )
        
        return smoothed
    
    def regularize_scales(
        self,
        scales: torch.Tensor,
        positions: torch.Tensor,
        max_scale_ratio: float = 3.0
    ) -> torch.Tensor:
        """
        Regularize splat scales to prevent extreme values.
        
        Args:
            scales: (N, 3) current scales
            positions: (N, 3) splat positions
            max_scale_ratio: Maximum ratio between largest/smallest scale
        
        Returns:
            regularized_scales: (N, 3) clamped scales
        """
        scales = self._to_device(scales)
        
        # Find neighbors
        neighbors_idx = self._find_knn(positions)
        
        # Compute median neighbor scale
        regularized = torch.zeros_like(scales)
        
        for i in range(scales.shape[0]):
            neighbor_scales = scales[neighbors_idx[i]]
            median_scale = neighbor_scales.median(dim=0).values
            
            # Clamp scale relative to neighbors
            min_scale = median_scale / max_scale_ratio
            max_scale = median_scale * max_scale_ratio
            
            regularized[i] = torch.clamp(scales[i], min_scale, max_scale)
        
        return regularized
    
    def bilateral_filter_positions(
        self,
        positions: torch.Tensor,
        colors: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply bilateral filtering to positions (edge-aware smoothing).
        
        Args:
            positions: (N, 3) 3D positions
            colors: (N, 3) RGB colors for similarity
        
        Returns:
            filtered: (N, 3) bilaterally filtered positions
        """
        positions = self._to_device(positions)
        colors = self._to_device(colors)
        
        # Find neighbors
        neighbors_idx = self._find_knn(positions)
        
        filtered = torch.zeros_like(positions)
        
        for i in range(positions.shape[0]):
            neighbor_pos = positions[neighbors_idx[i]]
            neighbor_colors = colors[neighbors_idx[i]]
            
            # Spatial weight
            spatial_dist = torch.norm(neighbor_pos - positions[i], dim=1)
            spatial_weight = torch.exp(
                -(spatial_dist ** 2) / (2 * self.sigma_spatial ** 2)
            )
            
            # Color weight
            color_dist = torch.norm(neighbor_colors - colors[i], dim=1)
            color_weight = torch.exp(
                -(color_dist ** 2) / (2 * self.sigma_color ** 2)
            )
            
            # Combined weight
            weight = (spatial_weight * color_weight).unsqueeze(-1)
            
            # Weighted average
            filtered[i] = (neighbor_pos * weight).sum(dim=0) / (weight.sum() + 1e-6)
        
        return filtered
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on correct device."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor).float()
        return tensor.to(self.device)


def smooth_geometry(
    positions: np.ndarray,
    colors: Optional[np.ndarray] = None,
    strength: float = 0.3,
    k: int = 8
) -> np.ndarray:
    """
    Quick utility function for geometry smoothing (numpy interface).
    
    Args:
        positions: (N, 3) point positions
        colors: (N, 3) optional colors for edge preservation
        strength: Smoothing strength [0, 1]
        k: Number of neighbors
    
    Returns:
        smoothed: (N, 3) smoothed positions
    """
    smoother = NormalSmoother(
        k_neighbors=k,
        smoothing_strength=strength
    )
    
    pos_tensor = torch.from_numpy(positions).float()
    col_tensor = torch.from_numpy(colors).float() if colors is not None else None
    
    smoothed = smoother.smooth_positions(pos_tensor, col_tensor)
    
    return smoothed.cpu().numpy()
