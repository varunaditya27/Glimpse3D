"""
ai_modules/gsplat/render_gsplat.py

Self-contained, CPU-capable Gaussian-Splat renderer with per-pixel contribution
tracking.

Unlike ``render_view.py`` (which wraps the CUDA ``gsplat`` rasterizer), this
module is a pure-PyTorch reference implementation that runs on CPU. It is used by
the MVCRM back-projection pipeline (``ai_modules/refine_module``), which needs to
know *which* splats contributed to each pixel and with what alpha-composite
weight -- information the fused CUDA kernel does not expose.

The renderer projects every Gaussian center to screen space, approximates each
splat as an isotropic screen-space disc whose radius follows from its world
scale and depth, gathers the candidate splats covering each pixel, sorts them
front-to-back, alpha-composites them, and records the ``top_k`` largest
contributions per pixel.

Camera convention (matches ``refine_module.back_projector.CameraParams``):
    OpenCV / "+X right, +Y down, +Z forward". A world point ``p`` projects via
        p_cam = E @ [p, 1]            # E = extrinsics, world->camera (4,4)
        u = fx * x_cam / z_cam + cx   # K = intrinsics (3,3)
        v = fy * y_cam / z_cam + cy
    Splats with ``z_cam <= near`` (behind / on the camera) are culled.

Cost:
    Memory and time are roughly O(N * H * W) in the worst case (every splat is
    tested against every pixel inside its bounding box), tempered by per-splat
    screen-space culling and the optional ``max_splats`` subsample. For the small
    splat counts used during refinement on CPU this is acceptable; for large
    scenes prefer the CUDA path in ``render_view.py``.
"""

from __future__ import annotations

import math

import torch


# Numerical floor used to avoid divide-by-zero and to clamp the largest opacity
# so ``prod(1 - alpha_j)`` never collapses to exactly zero (keeps gradients /
# weights finite and avoids NaNs from 0 * inf style expressions).
_EPS = 1e-8
_MAX_ALPHA = 1.0 - 1e-4

# Gaussian footprint: how many standard deviations of the splat we treat as the
# visible disc radius, and the lower bound (in pixels) so a splat is never
# smaller than a single pixel.
_RADIUS_SIGMAS = 3.0
_MIN_RADIUS_PX = 0.5


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert (N,4) wxyz quaternions to (N,3,3) rotation matrices.

    Currently used only to fold anisotropic scale into a single effective world
    radius; the isotropic disc approximation does not need the full orientation,
    but we keep the helper so callers passing rotations get a stable result.
    """
    quats = quats / (quats.norm(dim=-1, keepdim=True) + _EPS)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = quats.shape[0]
    rot = torch.empty((n, 3, 3), dtype=quats.dtype, device=quats.device)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _empty_outputs(height, width, top_k, dtype, device):
    """Return correctly-shaped zero tensors for the empty-input case."""
    rendered = torch.zeros((height, width, 3), dtype=dtype, device=device)
    depth = torch.zeros((height, width), dtype=dtype, device=device)
    alpha = torch.zeros((height, width), dtype=dtype, device=device)
    contributions = torch.zeros((height, width, top_k, 2), dtype=dtype, device=device)
    contributions[..., 0] = -1.0  # index padding
    return rendered, depth, alpha, contributions


def render_with_contributions(
    positions: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    scales: torch.Tensor,
    camera,
    rotations: torch.Tensor = None,
    top_k: int = 8,
    max_splats: int = None,
):
    """Render Gaussians and record per-pixel alpha-composite contributions.

    Args:
        positions: (N, 3) float, splat centers in world coordinates.
        colors: (N, 3) float RGB in [0, 1].
        opacities: (N,) float in [0, 1].
        scales: (N, 3) float, per-axis world-unit standard deviations.
        camera: object with ``.intrinsics`` (3,3), ``.extrinsics`` (4,4 world->
            camera, OpenCV convention), ``.width``, ``.height``.
        rotations: optional (N, 4) wxyz quaternions. Used only to fold the
            anisotropic scale into a single effective radius.
        top_k: number of strongest contributions recorded per pixel.
        max_splats: if set and N > max_splats, a uniform random subset of this
            many splats is rendered (speed knob). ``None`` renders all.

    Returns:
        rendered: (H, W, 3) float, composited RGB.
        depth: (H, W) float, alpha-weighted expected depth (camera-space z).
        alpha: (H, W) float, total coverage in [0, 1].
        contributions: (H, W, top_k, 2) float. ``[..., 0]`` is the splat index
            (float; -1 where fewer than top_k splats hit the pixel) and
            ``[..., 1]`` is that splat's alpha-composite weight (0 padding).
            Indices refer to the *original* ``positions`` rows even when
            ``max_splats`` subsampling is active.

    The cost is roughly O(N * H * W); see module docstring.
    """
    height = int(camera.height)
    width = int(camera.width)
    top_k = int(top_k)

    # Resolve dtype/device from inputs, falling back to a sane default so the
    # empty-input guard still returns something usable.
    if positions is not None and positions.numel() > 0:
        dtype = positions.dtype if positions.is_floating_point() else torch.float32
        device = positions.device
    else:
        dtype = torch.float32
        device = torch.device("cpu")

    if positions is None or positions.numel() == 0:
        return _empty_outputs(height, width, top_k, dtype, device)

    positions = positions.to(dtype=dtype)
    colors = colors.to(dtype=dtype, device=device)
    opacities = opacities.to(dtype=dtype, device=device).reshape(-1)
    scales = scales.to(dtype=dtype, device=device)
    if scales.dim() == 1:
        scales = scales.reshape(-1, 1).expand(-1, 3)

    n_total = positions.shape[0]
    # Original indices, so subsampling does not corrupt the reported indices.
    orig_idx = torch.arange(n_total, device=device)

    if max_splats is not None and n_total > int(max_splats):
        perm = torch.randperm(n_total, device=device)[: int(max_splats)]
        positions = positions[perm]
        colors = colors[perm]
        opacities = opacities[perm]
        scales = scales[perm]
        orig_idx = orig_idx[perm]
        if rotations is not None:
            rotations = rotations[perm]

    intrinsics = camera.intrinsics.to(dtype=dtype, device=device)
    extrinsics = camera.extrinsics.to(dtype=dtype, device=device)
    near = float(getattr(camera, "near", 0.01))

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # --- Project centers to camera space: p_cam = E @ [x, y, z, 1] -----------
    rot = extrinsics[:3, :3]
    trans = extrinsics[:3, 3]
    p_cam = positions @ rot.T + trans  # (N, 3)
    z_cam = p_cam[:, 2]

    # --- Effective world radius (fold anisotropic scale into one number) ------
    # Largest std-dev axis is a safe over-estimate of the disc extent.
    world_radius = scales.abs().max(dim=1).values  # (N,)
    if rotations is not None:
        # Rotation does not change the max axis length; helper is invoked only to
        # validate / normalize quaternions and keep behavior stable.
        _ = _quat_to_rotmat(rotations)

    # --- Frustum / behind-camera cull ----------------------------------------
    safe_z = z_cam.clamp(min=near)
    u = fx * (p_cam[:, 0] / safe_z) + cx
    v = fy * (p_cam[:, 1] / safe_z) + cy

    # Screen-space radius: project a world-space std-dev at this depth, scaled to
    # the requested sigma footprint, with a per-pixel floor.
    focal = 0.5 * (fx + fy)
    radius_px = _RADIUS_SIGMAS * focal * (world_radius / safe_z)
    radius_px = radius_px.clamp(min=_MIN_RADIUS_PX)

    in_front = z_cam > near
    # Keep splats whose bounding box overlaps the image at all.
    overlaps = (
        (u + radius_px >= 0)
        & (u - radius_px < width)
        & (v + radius_px >= 0)
        & (v - radius_px < height)
    )
    keep = in_front & overlaps & torch.isfinite(u) & torch.isfinite(v)

    if not bool(keep.any()):
        return _empty_outputs(height, width, top_k, dtype, device)

    u = u[keep]
    v = v[keep]
    z_cam = z_cam[keep]
    radius_px = radius_px[keep]
    opacities = opacities[keep].clamp(0.0, 1.0)
    colors = colors[keep]
    kept_idx = orig_idx[keep]
    sigma_px = (radius_px / _RADIUS_SIGMAS).clamp(min=_EPS)  # std-dev in pixels
    n = u.shape[0]

    # --- Front-to-back ordering (nearest first) ------------------------------
    order = torch.argsort(z_cam)
    u = u[order]
    v = v[order]
    z_cam = z_cam[order]
    radius_px = radius_px[order]
    sigma_px = sigma_px[order]
    opacities = opacities[order]
    colors = colors[order]
    kept_idx = kept_idx[order]

    # --- Accumulators --------------------------------------------------------
    rendered = torch.zeros((height, width, 3), dtype=dtype, device=device)
    depth_acc = torch.zeros((height, width), dtype=dtype, device=device)
    trans_remaining = torch.ones((height, width), dtype=dtype, device=device)

    # top_k contribution buffers; track the minimum kept weight per pixel so we
    # can cheaply decide whether a new contribution displaces an existing one.
    contrib_idx = torch.full((height, width, top_k), -1.0, dtype=dtype, device=device)
    contrib_w = torch.zeros((height, width, top_k), dtype=dtype, device=device)

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)

    # Iterate splats front-to-back. For each, splat only into its bounding box.
    for i in range(n):
        cu = u[i]
        cv = v[i]
        r = radius_px[i]
        s = sigma_px[i]

        x0 = int(max(0, math.floor(float(cu - r))))
        x1 = int(min(width - 1, math.ceil(float(cu + r))))
        y0 = int(max(0, math.floor(float(cv - r))))
        y1 = int(min(height - 1, math.ceil(float(cv + r))))
        if x1 < x0 or y1 < y0:
            continue

        bx = xs[x0 : x1 + 1]
        by = ys[y0 : y1 + 1]
        dx = bx[None, :] - cu  # (bh, bw) via broadcast below
        dy = by[:, None] - cv
        dist2 = dx * dx + dy * dy  # (bh, bw)

        # Isotropic Gaussian falloff of opacity across the disc.
        gauss = torch.exp(-0.5 * dist2 / (s * s))
        alpha_i = (opacities[i] * gauss).clamp(0.0, _MAX_ALPHA)  # (bh, bw)

        # Restrict to the disc; ignore negligible contributions.
        mask = (dist2 <= (r * r)) & (alpha_i > _EPS)
        if not bool(mask.any()):
            continue

        t_box = trans_remaining[y0 : y1 + 1, x0 : x1 + 1]  # (bh, bw)
        weight = alpha_i * t_box  # alpha-composite weight at each pixel
        weight = torch.where(mask, weight, torch.zeros_like(weight))

        # Accumulate color and expected depth.
        col = colors[i]  # (3,)
        rendered[y0 : y1 + 1, x0 : x1 + 1, :] += weight[..., None] * col[None, None, :]
        depth_acc[y0 : y1 + 1, x0 : x1 + 1] += weight * z_cam[i]

        # Update remaining transmittance (only where this splat contributed).
        new_t = t_box * (1.0 - torch.where(mask, alpha_i, torch.zeros_like(alpha_i)))
        trans_remaining[y0 : y1 + 1, x0 : x1 + 1] = new_t

        # --- Record top_k contributions in this box --------------------------
        ci_box = contrib_idx[y0 : y1 + 1, x0 : x1 + 1, :]  # (bh, bw, top_k)
        cw_box = contrib_w[y0 : y1 + 1, x0 : x1 + 1, :]

        # The smallest currently-stored weight (and its slot) per pixel.
        min_w, min_slot = cw_box.min(dim=-1)  # (bh, bw)
        # A pixel accepts this contribution if its new weight beats the weakest
        # stored slot (the zero-padded slots have weight 0 and lose ties via >).
        accept = mask & (weight > min_w)
        if bool(accept.any()):
            bh, bw = weight.shape
            flat_accept = accept.reshape(-1)
            flat_slot = min_slot.reshape(-1)
            flat_w = weight.reshape(-1)
            sel = flat_accept.nonzero(as_tuple=False).reshape(-1)
            if sel.numel() > 0:
                slots = flat_slot[sel]
                ci_flat = ci_box.reshape(-1, top_k)
                cw_flat = cw_box.reshape(-1, top_k)
                ci_flat[sel, slots] = kept_idx[i].to(dtype)
                cw_flat[sel, slots] = flat_w[sel]
                contrib_idx[y0 : y1 + 1, x0 : x1 + 1, :] = ci_flat.reshape(bh, bw, top_k)
                contrib_w[y0 : y1 + 1, x0 : x1 + 1, :] = cw_flat.reshape(bh, bw, top_k)

        # Early-out: if the whole scene is opaque everywhere, nothing left to do.
        if float(trans_remaining.max()) <= _EPS:
            break

    alpha = (1.0 - trans_remaining).clamp(0.0, 1.0)

    # Normalize expected depth by accumulated coverage where covered.
    covered = alpha > _EPS
    depth = torch.zeros_like(depth_acc)
    depth[covered] = depth_acc[covered] / alpha[covered].clamp(min=_EPS)

    # --- Sort each pixel's top_k by descending weight ------------------------
    sort_w, sort_order = torch.sort(contrib_w, dim=-1, descending=True)
    contrib_idx = torch.gather(contrib_idx, -1, sort_order)
    # Slots that ended up with zero weight must carry the -1 index padding.
    contrib_idx = torch.where(
        sort_w > 0, contrib_idx, torch.full_like(contrib_idx, -1.0)
    )
    contributions = torch.stack([contrib_idx, sort_w], dim=-1)  # (H, W, top_k, 2)

    # --- Final NaN/Inf scrub (defensive) -------------------------------------
    rendered = torch.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    contributions = torch.nan_to_num(contributions, nan=0.0, posinf=0.0, neginf=0.0)

    return rendered, depth, alpha, contributions
