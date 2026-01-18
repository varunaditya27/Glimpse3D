"""
ai_modules/midas_depth/edge_refinement.py

Novel: Depth-Aware Edge Refinement.

Problem: MiDaS blurs depth at object boundaries causing "flying pixels" in 3D.
Solution: Use RGB-guided bilateral filtering to preserve sharp edges while smoothing flat regions.

This module:
- Sharpens depth edges at RGB boundaries
- Smooths depth in textureless regions
- Reduces 3D reconstruction artifacts
"""

import numpy as np
from typing import Optional, Tuple
import cv2
from scipy.ndimage import gaussian_filter, sobel, median_filter


def refine_depth_edges(
    depth: np.ndarray,
    rgb: np.ndarray,
    sigma_space: float = 15.0,
    sigma_color: float = 0.1,
    edge_threshold: float = 0.05
) -> np.ndarray:
    """
    Refine depth map edges using RGB-guided joint bilateral filtering.
    
    Uses RGB edges to guide depth smoothing — preserves sharp depth
    discontinuities at object boundaries while smoothing flat regions.
    
    Args:
        depth: Normalized depth map (H, W), float32, values in [0, 1]
        rgb: RGB image (H, W, 3), uint8
        sigma_space: Spatial smoothing radius (pixels)
        sigma_color: Color similarity threshold (0-1, lower = more edge preservation)
        edge_threshold: Depth gradient threshold to identify edges
    
    Returns:
        refined_depth: Depth map with sharper edges (H, W), float32
    
    Example:
        >>> depth = estimate_depth("image.jpg")
        >>> rgb = np.array(Image.open("image.jpg"))
        >>> refined = refine_depth_edges(depth, rgb)
    """
    # Input validation
    if depth.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Depth shape {depth.shape[:2]} must match RGB shape {rgb.shape[:2]}")
    
    # Convert depth to 8-bit for OpenCV
    depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    
    # Ensure RGB is uint8
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)
    
    # Apply joint bilateral filter: smooth depth guided by RGB edges
    refined = cv2.ximgproc.jointBilateralFilter(
        joint=rgb,
        src=depth_uint8,
        d=-1,  # Compute diameter from sigma
        sigmaColor=sigma_color * 255,
        sigmaSpace=sigma_space
    )
    
    # Convert back to float [0, 1]
    refined = refined.astype(np.float32) / 255.0
    
    # Identify strong depth edges in original
    depth_edges = detect_depth_edges(depth, threshold=edge_threshold)
    
    # Preserve original depth at strong edges to avoid over-smoothing
    alpha = 0.7  # Blend factor (higher = more original at edges)
    refined = np.where(depth_edges, 
                       alpha * depth + (1 - alpha) * refined,
                       refined)
    
    return refined.astype(np.float32)


def detect_depth_edges(
    depth: np.ndarray,
    threshold: float = 0.05,
    dilation_size: int = 3
) -> np.ndarray:
    """
    Detect depth discontinuities (edges) in a depth map.
    
    Args:
        depth: Depth map (H, W), float32
        threshold: Gradient magnitude threshold
        dilation_size: Dilate edges to widen preservation zone
    
    Returns:
        edges: Binary mask (H, W) where True = edge pixel
    """
    # Compute depth gradients
    grad_x = sobel(depth, axis=1, mode='reflect')
    grad_y = sobel(depth, axis=0, mode='reflect')
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to get edges
    edges = grad_mag > threshold
    
    # Dilate to widen edge region
    if dilation_size > 0:
        kernel = np.ones((dilation_size, dilation_size), dtype=np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)
    
    return edges


def edge_preserving_smooth(
    depth: np.ndarray,
    rgb: np.ndarray,
    num_iterations: int = 3,
    lambda_param: float = 0.5
) -> np.ndarray:
    """
    Iterative edge-preserving smoothing using anisotropic diffusion.
    
    Smooths depth while preserving edges aligned with RGB boundaries.
    Better than single-pass bilateral filter for strong noise.
    
    Args:
        depth: Depth map (H, W), float32
        rgb: RGB image (H, W, 3), uint8
        num_iterations: Number of diffusion iterations
        lambda_param: Diffusion rate (0-1, higher = more smoothing)
    
    Returns:
        smoothed: Edge-preserved smooth depth
    """
    # Convert RGB to grayscale for edge detection
    if rgb.ndim == 3:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = rgb.astype(np.float32) / 255.0
    
    # Detect RGB edges
    rgb_grad_x = sobel(gray, axis=1, mode='reflect')
    rgb_grad_y = sobel(gray, axis=0, mode='reflect')
    rgb_edge_strength = np.sqrt(rgb_grad_x**2 + rgb_grad_y**2)
    
    # Diffusion coefficient (lower at edges)
    k = 0.1  # Edge sensitivity parameter
    diffusion_coeff = np.exp(-(rgb_edge_strength / k)**2)
    
    # Iterative smoothing
    smoothed = depth.copy()
    
    for _ in range(num_iterations):
        # Compute depth gradients
        grad_n = np.roll(smoothed, -1, axis=0) - smoothed  # North
        grad_s = np.roll(smoothed, 1, axis=0) - smoothed   # South
        grad_e = np.roll(smoothed, -1, axis=1) - smoothed  # East
        grad_w = np.roll(smoothed, 1, axis=1) - smoothed   # West
        
        # Apply diffusion coefficient
        c_n = np.roll(diffusion_coeff, -1, axis=0)
        c_s = np.roll(diffusion_coeff, 1, axis=0)
        c_e = np.roll(diffusion_coeff, -1, axis=1)
        c_w = np.roll(diffusion_coeff, 1, axis=1)
        
        # Update
        smoothed += lambda_param * (
            c_n * grad_n + c_s * grad_s +
            c_e * grad_e + c_w * grad_w
        )
        
        # Clamp to valid range
        smoothed = np.clip(smoothed, 0, 1)
    
    return smoothed.astype(np.float32)


def guided_filter_depth(
    depth: np.ndarray,
    rgb: np.ndarray,
    radius: int = 8,
    epsilon: float = 0.01
) -> np.ndarray:
    """
    Apply guided filter using RGB as guidance image.
    
    Fast edge-preserving filter that's better than bilateral for large radii.
    
    Args:
        depth: Depth map (H, W), float32, [0, 1]
        rgb: RGB guidance image (H, W, 3), uint8
        radius: Filter radius (larger = more smoothing)
        epsilon: Regularization (smaller = more edge preservation)
    
    Returns:
        filtered: Guided-filtered depth
    """
    # Convert to correct format for cv2.ximgproc.guidedFilter
    depth_uint8 = (depth * 255).astype(np.uint8)
    
    # Apply guided filter
    filtered = cv2.ximgproc.guidedFilter(
        guide=rgb,
        src=depth_uint8,
        radius=radius,
        eps=epsilon
    )
    
    # Convert back to float
    return (filtered.astype(np.float32) / 255.0)


def adaptive_edge_refinement(
    depth: np.ndarray,
    rgb: np.ndarray,
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Adaptive refinement that adjusts filtering based on local confidence.
    
    Applies stronger smoothing in low-confidence regions and lighter
    smoothing in high-confidence regions.
    
    Args:
        depth: Depth map (H, W), float32
        rgb: RGB image (H, W, 3), uint8
        confidence: Optional confidence map (H, W), float32, [0, 1]
                   If None, computed automatically
    
    Returns:
        refined: Adaptively refined depth
    """
    # Compute confidence if not provided
    if confidence is None:
        from .depth_confidence import estimate_depth_confidence
        confidence = estimate_depth_confidence(depth, rgb)
    
    # Apply different levels of refinement based on confidence
    # High confidence: light refinement (sigma_color = 0.05)
    # Low confidence: strong refinement (sigma_color = 0.2)
    
    refined_light = refine_depth_edges(depth, rgb, sigma_space=10, sigma_color=0.05)
    refined_strong = refine_depth_edges(depth, rgb, sigma_space=20, sigma_color=0.2)
    
    # Blend based on confidence
    refined = confidence * refined_light + (1 - confidence) * refined_strong
    
    return refined.astype(np.float32)


def remove_flying_pixels(
    depth: np.ndarray,
    rgb: np.ndarray,
    threshold: float = 0.15,
    min_region_size: int = 50
) -> np.ndarray:
    """
    Remove "flying pixels" - isolated depth outliers that don't match RGB structure.
    
    Flying pixels appear as small disconnected regions in 3D that don't
    correspond to actual object boundaries.
    
    Args:
        depth: Depth map (H, W), float32
        rgb: RGB image (H, W, 3), uint8
        threshold: Depth discontinuity threshold to identify candidates
        min_region_size: Remove regions smaller than this (pixels)
    
    Returns:
        cleaned: Depth with flying pixels removed (filled via inpainting)
    """
    # Detect large depth discontinuities
    grad_x = sobel(depth, axis=1, mode='reflect')
    grad_y = sobel(depth, axis=0, mode='reflect')
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Get RGB edges for comparison
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_edges = cv2.Canny(gray, 50, 150) > 0
    
    # Depth edges that don't align with RGB edges = flying pixels
    depth_edges = grad_mag > threshold
    flying_candidates = depth_edges & ~rgb_edges
    
    # Dilate RGB edges slightly to account for alignment tolerance
    kernel = np.ones((5, 5), dtype=np.uint8)
    rgb_edges_dilated = cv2.dilate(rgb_edges.astype(np.uint8), kernel).astype(bool)
    
    # Refined flying pixel detection
    flying_pixels = depth_edges & ~rgb_edges_dilated
    
    # Remove small regions using connected components
    from scipy.ndimage import label
    labeled, n_features = label(flying_pixels)
    
    mask_to_remove = np.zeros_like(flying_pixels)
    for i in range(1, n_features + 1):
        region = labeled == i
        if np.sum(region) < min_region_size:
            mask_to_remove |= region
    
    # Inpaint removed regions
    if np.any(mask_to_remove):
        depth_uint8 = (depth * 255).astype(np.uint8)
        mask_uint8 = mask_to_remove.astype(np.uint8) * 255
        inpainted = cv2.inpaint(depth_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
        cleaned = inpainted.astype(np.float32) / 255.0
    else:
        cleaned = depth.copy()
    
    return cleaned


def multi_scale_refinement(
    depth: np.ndarray,
    rgb: np.ndarray,
    scales: list = [1.0, 0.5, 0.25]
) -> np.ndarray:
    """
    Multi-scale edge-preserving refinement.
    
    Applies refinement at multiple scales and combines for better
    preservation of both fine and coarse structures.
    
    Args:
        depth: Depth map (H, W), float32
        rgb: RGB image (H, W, 3), uint8
        scales: List of scale factors (1.0 = original size)
    
    Returns:
        refined: Multi-scale refined depth
    """
    H, W = depth.shape
    refined_scales = []
    
    for scale in scales:
        if scale != 1.0:
            # Downscale
            new_h, new_w = int(H * scale), int(W * scale)
            depth_scaled = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb_scaled = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            depth_scaled = depth
            rgb_scaled = rgb
        
        # Refine at this scale
        refined_scaled = refine_depth_edges(
            depth_scaled, rgb_scaled,
            sigma_space=15 * scale,
            sigma_color=0.1
        )
        
        # Upscale back if needed
        if scale != 1.0:
            refined_scaled = cv2.resize(refined_scaled, (W, H), interpolation=cv2.INTER_LINEAR)
        
        refined_scales.append(refined_scaled)
    
    # Combine scales (weighted average, favoring finer scales)
    weights = np.array([1.0, 0.5, 0.25])[:len(scales)]
    weights = weights / weights.sum()
    
    refined = sum(w * r for w, r in zip(weights, refined_scales))
    
    return refined.astype(np.float32)


# =============================================================================
# Visualization and utilities
# =============================================================================

def compare_refinement(
    original_depth: np.ndarray,
    refined_depth: np.ndarray,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create side-by-side comparison visualization.
    
    Args:
        original_depth: Original depth map
        refined_depth: Refined depth map
        output_path: Optional path to save comparison
    
    Returns:
        comparison: Side-by-side comparison image
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_depth, cmap='magma')
    axes[0].set_title('Original Depth')
    axes[0].axis('off')
    
    axes[1].imshow(refined_depth, cmap='magma')
    axes[1].set_title('Edge-Refined Depth')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Convert to numpy array
        fig.canvas.draw()
        comparison = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        comparison = comparison.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return comparison


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="Depth edge refinement")
    parser.add_argument("--depth", "-d", required=True, help="Path to depth .npy file")
    parser.add_argument("--image", "-i", required=True, help="Path to RGB image")
    parser.add_argument("--output", "-o", default="refined_output", help="Output directory")
    parser.add_argument("--method", "-m", default="bilateral",
                       choices=["bilateral", "guided", "anisotropic", "adaptive", "multiscale"],
                       help="Refinement method")
    parser.add_argument("--sigma-space", type=float, default=15.0, help="Spatial sigma")
    parser.add_argument("--sigma-color", type=float, default=0.1, help="Color sigma")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Depth Edge Refinement")
    print("=" * 60)
    
    # Load depth
    depth = np.load(args.depth)
    print(f"Loaded depth: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Load RGB
    rgb = np.array(Image.open(args.image).convert("RGB"))
    print(f"Loaded RGB: {rgb.shape}")
    
    # Apply refinement
    print(f"\nApplying {args.method} refinement...")
    
    if args.method == "bilateral":
        refined = refine_depth_edges(depth, rgb, args.sigma_space, args.sigma_color)
    elif args.method == "guided":
        refined = guided_filter_depth(depth, rgb)
    elif args.method == "anisotropic":
        refined = edge_preserving_smooth(depth, rgb)
    elif args.method == "adaptive":
        refined = adaptive_edge_refinement(depth, rgb)
    elif args.method == "multiscale":
        refined = multi_scale_refinement(depth, rgb)
    
    print(f"Refined depth range: [{refined.min():.3f}, {refined.max():.3f}]")
    
    # Compute improvement metrics
    edges_original = detect_depth_edges(depth)
    edges_refined = detect_depth_edges(refined)
    print(f"\nEdge pixels: {np.sum(edges_original)} → {np.sum(edges_refined)}")
    
    # Save outputs
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    np.save(Path(args.output) / "refined_depth.npy", refined)
    
    # Save comparison
    compare_refinement(depth, refined, str(Path(args.output) / "comparison.png"))
    
    print(f"\nSaved to: {args.output}/")
    print("  - refined_depth.npy")
    print("  - comparison.png")
    print("\n✅ Done!")
