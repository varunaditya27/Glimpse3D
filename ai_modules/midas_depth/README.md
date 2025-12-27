# MiDaS Depth Module

Monocular depth estimation for the Glimpse3D pipeline.

---

## What is .npy format?

`.npy` is NumPy's binary file format for storing arrays efficiently:
- **Binary** — not human-readable, but fast to load/save
- **Preserves dtype** — keeps float32 precision exactly
- **Compact** — smaller than text formats like CSV
- **Universal** — any Python script can load it with `np.load('file.npy')`

```python
import numpy as np

# Save
np.save('depth.npy', depth_array)

# Load
depth = np.load('depth.npy')
```

Other modules (Zero-123, gsplat, backprojection) consume these `.npy` files directly.

---

## Primary Outputs

| Output | Format | Purpose |
|--------|--------|----------|
| Raw depth array | `.npy` (float32 H×W, 0..1) | **Pipeline data** — used by other AI modules |
| Grayscale image | `.png` | Quick visual debug (white = near, black = far) |
| Colored heatmap | `.png` | Human-friendly visualization (magma/viridis) |

Immediate next-step use (pipeline integration)

- Backend receives:
    - RGB image (the same view)
    - Depth (.npy)
    - Camera intrinsics (fx, fy, cx, cy) and extrinsics/pose for that rendered view

- Backprojection (pixel → 3D point)
    - Convert normalized depth → metric depth (if a scale is available), then:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        z = depth_value (in metric units)
    - Produce point cloud with colors from RGB.

- Point cloud → 3D representation
    - Compute normals / confidence per point.
    - Convert points into Gaussian splats or merge into existing gsplat model (update splat positions, colors, weights).

- Fusion & refinement
    - If multiple views: align scales (median/ICP), fuse by confidence-weighted blending.
    - After SDXL-enhanced views, re-run depth and fuse improved geometry/texture back into model.
    - Optionally run smoothing, outlier removal, and splat optimization passes.

- Export / render
    - Re-render the gsplat model for preview, iterate enhancements, or export (.ply/.glb/.splat).

---

## Novel Feature: Multi-View Depth Alignment

**Problem:** MiDaS produces independent depth maps per view — scales don't match across views.

**Solution:** `depth_alignment.py` aligns scales using overlapping regions.

### Usage

```python
from ai_modules.midas_depth import align_depth_scales, compute_alignment_quality

# Depth maps from different views (each normalized 0..1)
depths = [depth_view1, depth_view2, depth_view3]

# Align scales
aligned = align_depth_scales(depths)

# Check quality
quality = compute_alignment_quality(aligned)
print(f"Mean error: {quality['mean_error']:.4f}")
```

### CLI

```bash
python depth_alignment.py -d view1.npy view2.npy view3.npy -o aligned_output/
```

### Functions

| Function | Purpose |
|----------|----------|
| `align_depth_scales()` | Sequential alignment (chains views) |
| `align_depth_scales_global()` | Global optimization (least-squares) |
| `compute_overlap_mask()` | Full reprojection-based overlap |
| `estimate_overlap_simple()` | Heuristic overlap (no poses needed) |
| `compute_alignment_quality()` | Metrics to verify alignment |

---

## Practical Notes

- The `.npy` depth is the important programmatic output — always save and pass it to the backprojection module.
- MiDaS depth is relative per-image; use `align_depth_scales()` before metric fusion.
- Use confidence masks and filtering (median/bilateral) before fusing to reduce edge/noise artifacts.
- Keep camera intrinsics and view pose with each depth map — without them backprojection is impossible.

---

## File Structure

```
midas_depth/
├── __init__.py           # Module exports
├── run_depth.py          # Core depth estimation
├── depth_alignment.py    # Novel: Multi-view scale alignment
├── test_depth.py         # Unit tests
├── requirements.txt      # Dependencies
└── README.md             # This file
```
