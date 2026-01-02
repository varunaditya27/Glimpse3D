"""
ai_modules/midas_depth/depth_alignment.py

Novel: Cross-view depth scale alignment using overlapping regions.

Problem: MiDaS produces independent depth maps per view — scales don't match.
Solution: Find overlapping regions across views and compute consistent scale factors.

This is a key innovation for multi-view 3D reconstruction pipelines.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import binary_erosion, binary_dilation


def compute_overlap_mask(
    depth1: np.ndarray,
    depth2: np.ndarray,
    pose1: np.ndarray,
    pose2: np.ndarray,
    intrinsics: np.ndarray,
    depth_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute overlap masks between two views using depth reprojection.
    
    Projects pixels from view1 into view2's coordinate frame and checks
    which pixels are visible in both views.
    
    Args:
        depth1: Depth map for view 1 (H, W), normalized [0, 1]
        depth2: Depth map for view 2 (H, W), normalized [0, 1]
        pose1: Camera pose for view 1 (4x4 transformation matrix)
        pose2: Camera pose for view 2 (4x4 transformation matrix)
        intrinsics: Camera intrinsic matrix (3x3) with fx, fy, cx, cy
        depth_threshold: Threshold for depth consistency check
    
    Returns:
        mask1: Binary mask for view 1 indicating overlapping pixels
        mask2: Binary mask for view 2 indicating overlapping pixels
    """
    H, W = depth1.shape
    
    # Create pixel coordinates grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # Extract intrinsics
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Unproject view1 pixels to 3D (camera coordinates)
    z1 = depth1 * 10.0  # Scale normalized depth to approximate metric (adjust as needed)
    x1 = (u - cx) * z1 / fx
    y1 = (v - cy) * z1 / fy
    
    # Stack to 3D points (H, W, 3)
    points1_cam = np.stack([x1, y1, z1], axis=-1)
    
    # Transform to world coordinates using pose1
    points1_flat = points1_cam.reshape(-1, 3)
    points1_homo = np.hstack([points1_flat, np.ones((points1_flat.shape[0], 1))])
    points1_world = (pose1 @ points1_homo.T).T[:, :3]
    
    # Transform world points to view2's camera coordinates
    pose2_inv = np.linalg.inv(pose2)
    points1_in_cam2 = (pose2_inv @ np.hstack([points1_world, np.ones((points1_world.shape[0], 1))]).T).T[:, :3]
    
    # Project to view2's image plane
    x2 = points1_in_cam2[:, 0]
    y2 = points1_in_cam2[:, 1]
    z2 = points1_in_cam2[:, 2]
    
    # Avoid division by zero
    z2 = np.maximum(z2, 1e-8)
    
    u2 = (x2 * fx / z2 + cx).reshape(H, W)
    v2 = (y2 * fy / z2 + cy).reshape(H, W)
    z2 = z2.reshape(H, W)
    
    # Check which points project inside view2's image
    valid_proj = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H) & (z2 > 0)
    
    # Check depth consistency
    u2_int = np.clip(u2.astype(int), 0, W - 1)
    v2_int = np.clip(v2.astype(int), 0, H - 1)
    
    depth2_sampled = depth2[v2_int, u2_int] * 10.0  # Same scale as z1
    depth_consistent = np.abs(z2 - depth2_sampled) < depth_threshold * 10.0
    
    # Final overlap mask for view1
    mask1 = valid_proj & depth_consistent
    
    # Create corresponding mask for view2
    mask2 = np.zeros_like(depth2, dtype=bool)
    mask2[v2_int[mask1], u2_int[mask1]] = True
    
    return mask1, mask2


def estimate_overlap_simple(
    depth1: np.ndarray,
    depth2: np.ndarray,
    view_angle_diff: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified overlap estimation without full pose information.
    
    Assumes views are taken from similar positions with known angular difference.
    Uses heuristic: center region is more likely to overlap.
    
    Args:
        depth1: Depth map for view 1 (H, W)
        depth2: Depth map for view 2 (H, W)
        view_angle_diff: Approximate angle between views in degrees
    
    Returns:
        mask1: Estimated overlap mask for view 1
        mask2: Estimated overlap mask for view 2
    """
    H, W = depth1.shape
    
    # Create center-weighted mask (center more likely to overlap)
    y, x = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    
    # Gaussian-like center weighting
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Narrower overlap region for larger angle differences
    overlap_ratio = max(0.3, 1.0 - view_angle_diff / 180.0)
    threshold = max_dist * overlap_ratio
    
    center_mask = dist_from_center < threshold
    
    # Also exclude regions with very different depth characteristics
    # (extreme near/far regions less likely to have consistent overlap)
    depth1_valid = (depth1 > 0.05) & (depth1 < 0.95)
    depth2_valid = (depth2 > 0.05) & (depth2 < 0.95)
    
    mask1 = center_mask & depth1_valid
    mask2 = center_mask & depth2_valid
    
    # Erode to avoid edge artifacts
    mask1 = binary_erosion(mask1, iterations=5)
    mask2 = binary_erosion(mask2, iterations=5)
    
    return mask1, mask2


def align_depth_scales(
    depth_maps: List[np.ndarray],
    overlap_masks: Optional[List[np.ndarray]] = None,
    method: str = "median"
) -> List[np.ndarray]:
    """
    Align multiple depth maps to a consistent scale using overlapping regions.
    
    Instead of treating each depth map independently, find overlapping 3D points
    across views and compute a global scale factor.
    
    Args:
        depth_maps: List of normalized depth arrays (each H×W, float32)
        overlap_masks: Binary masks indicating shared regions between consecutive views.
                      If None, uses simplified center-based estimation.
        method: Scale estimation method - "median" (robust) or "mean" (sensitive to outliers)
    
    Returns:
        aligned: List of depth maps with consistent scale (first map is reference)
    
    Example:
        >>> depths = [depth1, depth2, depth3]  # From different views
        >>> aligned = align_depth_scales(depths)
        >>> # Now aligned[0], aligned[1], aligned[2] have consistent scale
    """
    if len(depth_maps) == 0:
        return []
    
    if len(depth_maps) == 1:
        return [depth_maps[0].copy()]
    
    # Use first depth map as reference
    reference = depth_maps[0]
    aligned = [reference.copy()]
    
    for i in range(1, len(depth_maps)):
        current = depth_maps[i]
        
        # Get or compute overlap mask
        if overlap_masks is not None and i - 1 < len(overlap_masks):
            overlap = overlap_masks[i - 1]
        else:
            # Estimate overlap using simplified method
            _, overlap = estimate_overlap_simple(reference, current)
        
        # Ensure overlap has enough pixels for reliable estimation
        min_overlap_pixels = 100
        if np.sum(overlap) < min_overlap_pixels:
            print(f"Warning: View {i} has insufficient overlap ({np.sum(overlap)} pixels). "
                  f"Using global statistics instead.")
            # Fallback to global statistics
            if method == "median":
                scale = np.median(reference) / (np.median(current) + 1e-8)
            else:
                scale = np.mean(reference) / (np.mean(current) + 1e-8)
        else:
            # Get values in overlapping region
            ref_vals = reference[overlap]
            cur_vals = current[overlap]
            
            # Filter out invalid values
            valid = (cur_vals > 1e-6) & (ref_vals > 1e-6)
            ref_vals = ref_vals[valid]
            cur_vals = cur_vals[valid]
            
            if len(ref_vals) < min_overlap_pixels:
                # Fallback
                if method == "median":
                    scale = np.median(reference) / (np.median(current) + 1e-8)
                else:
                    scale = np.mean(reference) / (np.mean(current) + 1e-8)
            else:
                # Compute scale factor
                ratios = ref_vals / (cur_vals + 1e-8)
                
                if method == "median":
                    scale = np.median(ratios)
                else:
                    # Trim outliers for mean
                    ratios_sorted = np.sort(ratios)
                    trim = int(len(ratios) * 0.1)
                    if trim > 0:
                        ratios_trimmed = ratios_sorted[trim:-trim]
                    else:
                        ratios_trimmed = ratios_sorted
                    scale = np.mean(ratios_trimmed)
        
        # Clamp scale to reasonable range
        scale = np.clip(scale, 0.1, 10.0)
        
        # Apply scale correction
        aligned_depth = current * scale
        
        # Re-normalize to [0, 1] if needed
        if aligned_depth.max() > 1.0:
            aligned_depth = aligned_depth / aligned_depth.max()
        
        aligned.append(aligned_depth.astype(np.float32))
        
        # Use this aligned depth as reference for next view (chain alignment)
        reference = aligned_depth
    
    return aligned


def align_depth_scales_global(
    depth_maps: List[np.ndarray],
    overlap_masks: Optional[List[List[np.ndarray]]] = None
) -> List[np.ndarray]:
    """
    Global optimization for depth scale alignment across all views.
    
    Unlike sequential alignment, this considers all pairwise overlaps
    and finds globally consistent scales using least squares.
    
    Args:
        depth_maps: List of N depth maps
        overlap_masks: NxN list of overlap masks (overlap_masks[i][j] for views i,j)
    
    Returns:
        Globally aligned depth maps
    """
    n_views = len(depth_maps)
    
    if n_views <= 1:
        return [d.copy() for d in depth_maps]
    
    # Build system of equations for scale factors
    # s_i * d_i[overlap] ≈ s_j * d_j[overlap]
    # We fix s_0 = 1 (reference)
    
    # Collect pairwise scale estimates
    pairwise_scales = {}
    
    for i in range(n_views):
        for j in range(i + 1, n_views):
            # Estimate overlap
            mask_i, mask_j = estimate_overlap_simple(
                depth_maps[i], depth_maps[j],
                view_angle_diff=abs(j - i) * 30  # Approximate angle based on view index
            )
            
            # Use intersection of masks
            if overlap_masks is not None:
                mask_i = mask_i & overlap_masks[i][j]
                mask_j = mask_j & overlap_masks[j][i]
            
            overlap = mask_i & mask_j
            
            if np.sum(overlap) > 100:
                vals_i = depth_maps[i][overlap]
                vals_j = depth_maps[j][overlap]
                
                valid = (vals_i > 1e-6) & (vals_j > 1e-6)
                if np.sum(valid) > 50:
                    ratio = np.median(vals_i[valid] / (vals_j[valid] + 1e-8))
                    pairwise_scales[(i, j)] = ratio
    
    # Solve for global scales using least squares
    # log(s_i) - log(s_j) = log(ratio_ij)
    
    if len(pairwise_scales) == 0:
        # No reliable overlaps, return original
        return [d.copy() for d in depth_maps]
    
    # Build linear system
    n_constraints = len(pairwise_scales)
    A = np.zeros((n_constraints, n_views - 1))  # s_0 = 1, solve for s_1...s_{n-1}
    b = np.zeros(n_constraints)
    
    for k, ((i, j), ratio) in enumerate(pairwise_scales.items()):
        if i > 0:
            A[k, i - 1] = 1
        if j > 0:
            A[k, j - 1] = -1
        b[k] = np.log(ratio + 1e-8)
    
    # Solve least squares
    if n_views > 2:
        log_scales, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        scales = np.exp(np.concatenate([[0], log_scales]))  # s_0 = exp(0) = 1
    else:
        # Only two views
        scales = np.array([1.0, 1.0 / list(pairwise_scales.values())[0]])
    
    # Apply scales
    aligned = []
    for i, (depth, scale) in enumerate(zip(depth_maps, scales)):
        scaled = depth * scale
        # Normalize to [0, 1]
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)
        aligned.append(scaled.astype(np.float32))
    
    return aligned


def compute_alignment_quality(
    aligned_depths: List[np.ndarray],
    overlap_masks: Optional[List[np.ndarray]] = None
) -> dict:
    """
    Compute quality metrics for depth alignment.
    
    Args:
        aligned_depths: List of aligned depth maps
        overlap_masks: Optional overlap masks
    
    Returns:
        Dictionary with quality metrics
    """
    if len(aligned_depths) < 2:
        return {"status": "insufficient_views", "n_views": len(aligned_depths)}
    
    metrics = {
        "n_views": len(aligned_depths),
        "pairwise_errors": [],
        "mean_error": 0.0,
        "max_error": 0.0,
    }
    
    errors = []
    
    for i in range(len(aligned_depths) - 1):
        j = i + 1
        
        # Get overlap
        if overlap_masks is not None and i < len(overlap_masks):
            overlap = overlap_masks[i]
        else:
            overlap, _ = estimate_overlap_simple(aligned_depths[i], aligned_depths[j])
        
        if np.sum(overlap) > 50:
            vals_i = aligned_depths[i][overlap]
            vals_j = aligned_depths[j][overlap]
            
            # Compute relative error in overlap
            rel_error = np.abs(vals_i - vals_j) / (np.maximum(vals_i, vals_j) + 1e-8)
            mean_rel_error = float(np.mean(rel_error))
            
            errors.append(mean_rel_error)
            metrics["pairwise_errors"].append({
                "views": (i, j),
                "mean_relative_error": mean_rel_error,
                "overlap_pixels": int(np.sum(overlap))
            })
    
    if errors:
        metrics["mean_error"] = float(np.mean(errors))
        metrics["max_error"] = float(np.max(errors))
    
    return metrics


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Test depth alignment")
    parser.add_argument("--depths", "-d", nargs="+", required=True,
                       help="Paths to depth .npy files")
    parser.add_argument("--output", "-o", default="aligned_depths",
                       help="Output directory")
    parser.add_argument("--method", "-m", default="median",
                       choices=["median", "mean"],
                       help="Scale estimation method")
    args = parser.parse_args()
    
    # Load depth maps
    depth_maps = [np.load(p) for p in args.depths]
    print(f"Loaded {len(depth_maps)} depth maps")
    
    # Align
    print("Aligning depth scales...")
    aligned = align_depth_scales(depth_maps, method=args.method)
    
    # Compute quality
    quality = compute_alignment_quality(aligned)
    print(f"Alignment quality: mean_error={quality['mean_error']:.4f}")
    
    # Save
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for i, depth in enumerate(aligned):
        out_path = Path(args.output) / f"aligned_depth_{i:03d}.npy"
        np.save(out_path, depth)
        print(f"Saved: {out_path}")
    
    print("✅ Done!")
