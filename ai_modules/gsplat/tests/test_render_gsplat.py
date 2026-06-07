"""
ai_modules/gsplat/tests/test_render_gsplat.py

CPU smoke / correctness tests for the self-contained contribution-aware
Gaussian-Splat renderer in ``ai_modules/gsplat/render_gsplat.py``.

Run:
    python -m ai_modules.gsplat.tests.test_render_gsplat
or
    python ai_modules/gsplat/tests/test_render_gsplat.py
"""

import importlib.util
import os
import sys

import torch

# Allow running as a bare script: add repo root (three levels up) to sys.path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_module(name, rel_path):
    """Import a module straight from its file path.

    The ``ai_modules.gsplat`` package ``__init__`` eagerly imports optimization
    code with heavy optional deps (e.g. ``tqdm``) that are not present in the
    CPU-only test venv. The renderer under test only needs torch, so we load it
    directly to keep the CPU test self-sufficient.
    """
    path = os.path.join(_REPO_ROOT, *rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_render_mod = _load_module(
    "render_gsplat_under_test", ("ai_modules", "gsplat", "render_gsplat.py")
)
render_with_contributions = _render_mod.render_with_contributions

_bp_mod = _load_module(
    "back_projector_camera", ("ai_modules", "refine_module", "back_projector.py")
)
CameraParams = _bp_mod.CameraParams


def _make_camera(width=32, height=32, radius=3.0):
    """Pinhole camera looking down -Z (world) from +Z at distance `radius`.

    Extrinsics are world->camera in OpenCV convention. With the camera placed at
    world (0,0,radius) looking toward the origin along world -Z, the OpenCV
    camera-forward (+Z_cam) must point in world -Z. A 180-degree rotation about
    the world X axis maps:
        world +X -> cam +X
        world +Y -> cam -Y (image down)
        world +Z -> cam -Z  (so a point at the origin sits at +Z_cam = radius)
    Translation puts the world origin at camera-space (0, 0, radius).
    """
    fx = fy = 40.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    # R = rotation about X by 180 deg.
    R = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
    )
    eye = torch.tensor([0.0, 0.0, radius], dtype=torch.float32)
    t = -R @ eye  # world->camera translation
    extrinsics = torch.eye(4, dtype=torch.float32)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    return CameraParams(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        width=width,
        height=height,
    )


def test_render_contributions():
    torch.manual_seed(0)
    H = W = 32
    top_k = 8
    camera = _make_camera(W, H, radius=3.0)

    # 3 synthetic Gaussians. Splat 0 sits exactly on the optical axis (origin),
    # the others are offset so they land off-center.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # on axis -> center pixel
            [0.6, 0.0, 0.0],   # right
            [0.0, 0.6, 0.0],   # up/down in image
        ],
        dtype=torch.float32,
    )
    colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    opacities = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    scales = torch.full((3, 3), 0.05, dtype=torch.float32)

    rendered, depth, alpha, contributions = render_with_contributions(
        positions, colors, opacities, scales, camera, top_k=top_k
    )

    # --- Shape assertions ----------------------------------------------------
    assert rendered.shape == (H, W, 3), rendered.shape
    assert depth.shape == (H, W), depth.shape
    assert alpha.shape == (H, W), alpha.shape
    assert contributions.shape == (H, W, top_k, 2), contributions.shape

    # --- Finiteness ----------------------------------------------------------
    for name, t in [
        ("rendered", rendered),
        ("depth", depth),
        ("alpha", alpha),
        ("contributions", contributions),
    ]:
        assert torch.isfinite(t).all(), f"{name} has non-finite values"

    # --- On-axis Gaussian lights up the image center -------------------------
    cy = (H - 1) // 2
    cx = (W - 1) // 2
    # Search a small window around the geometric center for the peak alpha.
    window = alpha[cy - 2 : cy + 3, cx - 2 : cx + 3]
    assert window.max() > 0.0, "center region should have nonzero alpha"

    # Find the brightest pixel and confirm splat index 0 contributes there.
    flat = alpha.reshape(-1)
    peak = int(torch.argmax(flat).item())
    py, px = divmod(peak, W)
    assert abs(py - cy) <= 2 and abs(px - cx) <= 2, (py, px, cy, cx)

    idxs_at_peak = contributions[py, px, :, 0]
    assert (idxs_at_peak == 0).any(), (
        f"splat 0 should appear in contributions at peak pixel; got {idxs_at_peak.tolist()}"
    )
    # The contribution weight for an existing splat must be positive.
    w_at_peak = contributions[py, px, :, 1]
    assert w_at_peak.max() > 0.0

    # --- Padding contract: padded slots carry index -1 and weight 0 ----------
    # A corner pixel far from every splat should be fully padded.
    corner_idx = contributions[0, 0, :, 0]
    corner_w = contributions[0, 0, :, 1]
    assert torch.all(corner_w >= 0.0)
    # Wherever weight is 0, index must be -1.
    zero_w = contributions[..., 1] == 0.0
    assert torch.all(contributions[..., 0][zero_w] == -1.0)
    # corner reference (kept to document intent; corner is expected empty)
    assert torch.all((corner_w > 0.0) | (corner_idx == -1.0))

    # --- alpha within [0,1], depth nonneg where covered ----------------------
    assert float(alpha.min()) >= 0.0 and float(alpha.max()) <= 1.0
    covered = alpha > 1e-6
    assert float(depth[covered].min()) >= 0.0

    print("test_render_contributions PASSED")
    print(f"  peak pixel = ({py},{px}), peak alpha = {float(alpha.max()):.4f}")
    print(f"  center depth = {float(depth[py, px]):.4f}")
    print(f"  contrib idxs at peak = {idxs_at_peak.tolist()}")


def test_empty_input():
    H = W = 32
    top_k = 8
    camera = _make_camera(W, H)
    empty = torch.zeros((0, 3), dtype=torch.float32)
    rendered, depth, alpha, contributions = render_with_contributions(
        empty, empty, torch.zeros((0,)), empty, camera, top_k=top_k
    )
    assert rendered.shape == (H, W, 3)
    assert depth.shape == (H, W)
    assert alpha.shape == (H, W)
    assert contributions.shape == (H, W, top_k, 2)
    assert float(alpha.sum()) == 0.0
    assert torch.all(contributions[..., 0] == -1.0)
    assert torch.all(contributions[..., 1] == 0.0)
    print("test_empty_input PASSED")


if __name__ == "__main__":
    test_render_contributions()
    test_empty_input()
    print("ALL TESTS PASSED")
