### MiDaS Depth module (integration with the rest of the project):

---


Final outcome — what the midas_depth module produces and how that output is used downstream

Primary outputs

- Raw depth array (.npy) — float32 H×W, normalized (0..1) per image. This is the canonical data used by the pipeline.
- Grayscale image (.png) — quick visual debug (white = near, black = far).
- Colored heatmap (.png) — human-friendly visualization (magma/viridis).

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

Practical notes / gotchas

- The .npy depth is the important programmatic output — always save and pass it to the backprojection module.
- MiDaS depth is relative per-image; you need scale alignment across views before metric fusion.
- Use confidence masks and filtering (median/bilateral) before fusing to reduce edge/noise artifacts.
- Keep camera intrinsics and view pose with each depth map — without them backprojection is impossible.
