"""
ai_modules/midas_depth/depth_confidence.py

Novel: Confidence-Weighted Depth Estimation.

Problem: Depth predictions are unreliable at edges, reflections, and textureless regions.
Solution: Estimate per-pixel confidence to guide downstream fusion.

This allows the pipeline to:
- Weight reliable depth pixels higher during fusion
- Mask out uncertain regions before backprojection
- Improve 3D reconstruction quality by avoiding "flying pixels"
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import sobel, gaussian_filter, uniform_filter


def estimate_depth_confidence(
    depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    edge_weight: float = 0.4,
    texture_weight: float = 0.3,
    range_weight: float = 0.3
) -> np.ndarray:
    """
    Compute per-pixel confidence map for depth predictions.
    
    Low confidence regions:
    - High depth gradient (edges, discontinuities)
    - Low texture in RGB (textureless surfaces are hard to match)
    - Extreme depth values (very near/far objects)
    
    Args:
        depth: Normalized depth map (H, W), values in [0, 1]
        rgb: Optional RGB image (H, W, 3), uint8. If provided, texture is used.
        edge_weight: Weight for edge-based confidence (0-1)
        texture_weight: Weight for texture-based confidence (0-1)
        range_weight: Weight for depth range confidence (0-1)
    
    Returns:
        confidence: Per-pixel confidence map (H, W), float32, values in [0, 1]
                   1 = high confidence, 0 = low confidence
    
    Example:
        >>> depth = estimate_depth("image.jpg")
        >>> rgb = np.array(Image.open("image.jpg"))
        >>> confidence = estimate_depth_confidence(depth, rgb)
        >>> # Use confidence to weight fusion
        >>> fused_depth = depth1 * conf1 + depth2 * conf2
    """
    # Normalize weights
    total_weight = edge_weight + texture_weight + range_weight
    if total_weight > 0:
        edge_weight /= total_weight
        texture_weight /= total_weight
        range_weight /= total_weight
    
    # 1. Edge confidence: Low confidence at depth discontinuities
    edge_conf = _compute_edge_confidence(depth)
    
    # 2. Texture confidence: Low confidence in textureless regions
    if rgb is not None:
        texture_conf = _compute_texture_confidence(rgb)
    else:
        texture_conf = np.ones_like(depth)
        texture_weight = 0
        # Redistribute weight
        if edge_weight + range_weight > 0:
            edge_weight = edge_weight / (edge_weight + range_weight)
            range_weight = 1 - edge_weight
    
    # 3. Range confidence: Low confidence at extreme depths
    range_conf = _compute_range_confidence(depth)
    
    # Combine confidences (weighted product for stricter masking)
    confidence = (
        edge_conf ** edge_weight *
        texture_conf ** texture_weight *
        range_conf ** range_weight
    )
    
    # Smooth to reduce noise
    confidence = gaussian_filter(confidence, sigma=2)
    
    # Ensure [0, 1] range
    confidence = np.clip(confidence, 0, 1)
    
    return confidence.astype(np.float32)


def _compute_edge_confidence(depth: np.ndarray) -> np.ndarray:
    """
    Compute confidence based on depth edges.
    
    High gradient = depth discontinuity = unreliable
    """
    # Compute gradient magnitude
    grad_x = sobel(depth, axis=1, mode='reflect')
    grad_y = sobel(depth, axis=0, mode='reflect')
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient
    grad_max = np.percentile(grad_mag, 99) + 1e-8  # Use 99th percentile to avoid outliers
    grad_norm = np.clip(grad_mag / grad_max, 0, 1)
    
    # Invert: high gradient = low confidence
    edge_confidence = 1.0 - grad_norm
    
    # Apply soft threshold to preserve some mid-range values
    edge_confidence = np.power(edge_confidence, 0.5)
    
    return edge_confidence


def _compute_texture_confidence(rgb: np.ndarray) -> np.ndarray:
    """
    Compute confidence based on image texture.
    
    Low texture = ambiguous matching = unreliable depth
    """
    # Convert to grayscale if needed
    if rgb.ndim == 3:
        gray = np.mean(rgb.astype(np.float32), axis=2)
    else:
        gray = rgb.astype(np.float32)
    
    # Normalize to [0, 1]
    gray = gray / 255.0 if gray.max() > 1 else gray
    
    # Compute local variance as texture measure
    local_mean = uniform_filter(gray, size=7)
    local_sq_mean = uniform_filter(gray**2, size=7)
    local_var = local_sq_mean - local_mean**2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    
    # Also compute gradient-based texture
    grad_x = sobel(gray, axis=1, mode='reflect')
    grad_y = sobel(gray, axis=0, mode='reflect')
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Combine variance and gradient
    texture = np.sqrt(local_var) + grad_mag * 0.5
    
    # Normalize
    texture_max = np.percentile(texture, 99) + 1e-8
    texture_norm = np.clip(texture / texture_max, 0, 1)
    
    # Higher texture = higher confidence
    return texture_norm


def _compute_range_confidence(depth: np.ndarray) -> np.ndarray:
    """
    Compute confidence based on depth range.
    
    Extreme values (very near 0 or 1) are less reliable.
    """
    # Parabolic confidence: peaks at 0.5, drops at 0 and 1
    # f(x) = 1 - 4*(x - 0.5)^2 = 4*x*(1-x)
    range_confidence = 4 * depth * (1 - depth)
    
    # But also penalize very low variance (flat depth = suspicious)
    # This is already somewhat handled by edge confidence
    
    # Soften the penalty at extremes
    range_confidence = np.power(range_confidence, 0.3)
    
    return range_confidence


def apply_confidence_mask(
    depth: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.3,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Apply confidence threshold to mask unreliable depth pixels.
    
    Args:
        depth: Depth map (H, W)
        confidence: Confidence map (H, W)
        threshold: Pixels below this confidence are masked
        fill_value: Value to use for masked pixels (0 = far/invalid)
    
    Returns:
        masked_depth: Depth with low-confidence pixels replaced
    """
    masked = depth.copy()
    masked[confidence < threshold] = fill_value
    return masked


def weighted_depth_fusion(
    depths: list,
    confidences: list,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse multiple depth maps using confidence-weighted averaging.
    
    Args:
        depths: List of aligned depth maps (each H×W)
        confidences: List of confidence maps (each H×W)
        normalize: If True, normalize output to [0, 1]
    
    Returns:
        fused_depth: Confidence-weighted average depth (H, W)
        fused_confidence: Combined confidence (H, W)
    
    Example:
        >>> # Get depths and confidences from multiple views
        >>> depths = [depth1, depth2, depth3]
        >>> confs = [conf1, conf2, conf3]
        >>> fused, fused_conf = weighted_depth_fusion(depths, confs)
    """
    if len(depths) == 0:
        raise ValueError("At least one depth map required")
    
    if len(depths) != len(confidences):
        raise ValueError("Number of depths and confidences must match")
    
    # Stack for vectorized operations
    depth_stack = np.stack(depths, axis=0)  # (N, H, W)
    conf_stack = np.stack(confidences, axis=0)  # (N, H, W)
    
    # Weighted sum
    weighted_sum = np.sum(depth_stack * conf_stack, axis=0)
    conf_sum = np.sum(conf_stack, axis=0) + 1e-8  # Avoid division by zero
    
    # Weighted average
    fused_depth = weighted_sum / conf_sum
    
    # Combined confidence: max confidence at each pixel
    # (alternative: mean confidence, or sum clamped to 1)
    fused_confidence = np.max(conf_stack, axis=0)
    
    if normalize:
        d_min, d_max = fused_depth.min(), fused_depth.max()
        if d_max - d_min > 1e-8:
            fused_depth = (fused_depth - d_min) / (d_max - d_min)
    
    return fused_depth.astype(np.float32), fused_confidence.astype(np.float32)


def visualize_confidence(
    confidence: np.ndarray,
    output_path: Optional[str] = None,
    colormap: str = "viridis"
) -> np.ndarray:
    """
    Create a visualization of the confidence map.
    
    Args:
        confidence: Confidence map (H, W), values in [0, 1]
        output_path: Optional path to save visualization
        colormap: Matplotlib colormap name
    
    Returns:
        vis: RGB visualization (H, W, 3), uint8
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    
    cmap = plt.get_cmap(colormap)
    vis = cmap(np.clip(confidence, 0, 1))
    vis = (vis[:, :, :3] * 255).astype(np.uint8)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        Image.fromarray(vis).save(output_path)
    
    return vis


def get_reliable_depth_mask(
    confidence: np.ndarray,
    threshold: float = 0.5,
    min_region_size: int = 100
) -> np.ndarray:
    """
    Get binary mask of reliable depth regions.
    
    Args:
        confidence: Confidence map (H, W)
        threshold: Confidence threshold
        min_region_size: Remove small regions below this pixel count
    
    Returns:
        mask: Binary mask (H, W), True = reliable
    """
    from scipy.ndimage import binary_opening, label
    
    # Threshold
    mask = confidence >= threshold
    
    # Remove small isolated regions
    mask = binary_opening(mask, iterations=2)
    
    # Remove small connected components
    if min_region_size > 0:
        labeled, n_features = label(mask)
        for i in range(1, n_features + 1):
            region = labeled == i
            if np.sum(region) < min_region_size:
                mask[region] = False
    
    return mask


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from PIL import Image
    
    parser = argparse.ArgumentParser(description="Compute depth confidence")
    parser.add_argument("--depth", "-d", required=True, help="Path to depth .npy file")
    parser.add_argument("--image", "-i", default=None, help="Path to RGB image (optional)")
    parser.add_argument("--output", "-o", default="confidence_output", help="Output directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.3, help="Confidence threshold")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Depth Confidence Estimation")
    print("=" * 60)
    
    # Load depth
    depth = np.load(args.depth)
    print(f"Loaded depth: {depth.shape}, range [{depth.min():.3f}, {depth.max():.3f}]")
    
    # Load RGB if provided
    rgb = None
    if args.image and Path(args.image).exists():
        rgb = np.array(Image.open(args.image).convert("RGB"))
        print(f"Loaded RGB: {rgb.shape}")
    
    # Compute confidence
    print("\nComputing confidence...")
    confidence = estimate_depth_confidence(depth, rgb)
    print(f"Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"Mean confidence: {confidence.mean():.3f}")
    
    # Get reliable mask
    mask = get_reliable_depth_mask(confidence, threshold=args.threshold)
    reliable_pct = 100 * np.sum(mask) / mask.size
    print(f"Reliable pixels (>{args.threshold}): {reliable_pct:.1f}%")
    
    # Save outputs
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    np.save(Path(args.output) / "confidence.npy", confidence)
    visualize_confidence(confidence, str(Path(args.output) / "confidence_vis.png"))
    
    # Save masked depth
    masked_depth = apply_confidence_mask(depth, confidence, threshold=args.threshold)
    np.save(Path(args.output) / "masked_depth.npy", masked_depth)
    
    print(f"\nSaved to: {args.output}/")
    print("  - confidence.npy")
    print("  - confidence_vis.png")
    print("  - masked_depth.npy")
    print("\n✅ Done!")
