"""
CPU unit tests for syncdreamer_cameras() in utils_syncdreamer.

Validates that the 16 SyncDreamer view cameras are well-formed:
- exactly 16 cameras
- intrinsics (3, 3) and extrinsics (4, 4)
- rotation block is orthonormal with det ~= +1
- camera center magnitude ~= radius
- camera forward (+Z) axis points roughly toward the origin
"""

import math
import sys
import importlib.util
from pathlib import Path

import torch

# Make the repo root importable when run directly (CPU, no install).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load utils_syncdreamer directly by file path. Importing it via the package
# (ai_modules.sync_dreamer) would execute the package __init__, which eagerly
# imports the diffusion inference stack (omegaconf, ldm, ...) that is absent in
# a CPU-only environment. Loading the module file directly keeps the test light.
_UTILS_PATH = Path(__file__).resolve().parents[1] / "utils_syncdreamer.py"
_spec = importlib.util.spec_from_file_location(
    "ai_modules.sync_dreamer.utils_syncdreamer", _UTILS_PATH
)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
syncdreamer_cameras = _utils.syncdreamer_cameras


def test_count_is_16():
    cameras = syncdreamer_cameras()
    assert len(cameras) == 16, f"expected 16 cameras, got {len(cameras)}"


def test_matrix_shapes():
    cameras = syncdreamer_cameras(image_size=256)
    for i, cam in enumerate(cameras):
        assert tuple(cam.intrinsics.shape) == (3, 3), \
            f"camera {i} intrinsics shape {tuple(cam.intrinsics.shape)}"
        assert tuple(cam.extrinsics.shape) == (4, 4), \
            f"camera {i} extrinsics shape {tuple(cam.extrinsics.shape)}"
        assert cam.width == 256 and cam.height == 256


def test_intrinsics_values():
    image_size = 256
    fov_deg = 51.98
    cameras = syncdreamer_cameras(image_size=image_size, fov_deg=fov_deg)
    expected_f = 0.5 * image_size / math.tan(0.5 * math.radians(fov_deg))
    expected_c = image_size / 2.0
    K = cameras[0].intrinsics
    assert abs(float(K[0, 0]) - expected_f) < 1e-3
    assert abs(float(K[1, 1]) - expected_f) < 1e-3
    assert abs(float(K[0, 2]) - expected_c) < 1e-6
    assert abs(float(K[1, 2]) - expected_c) < 1e-6
    assert abs(float(K[2, 2]) - 1.0) < 1e-6


def test_rotation_orthonormal_and_det_one():
    cameras = syncdreamer_cameras()
    eye = torch.eye(3, dtype=torch.float32)
    for i, cam in enumerate(cameras):
        R = cam.extrinsics[:3, :3]
        # R @ R.T ~= I
        ortho_err = torch.max(torch.abs(R @ R.T - eye)).item()
        assert ortho_err < 1e-4, f"camera {i} not orthonormal (err={ortho_err})"
        # det(R) ~= +1 (proper rotation, not a reflection)
        det = torch.det(R).item()
        assert abs(det - 1.0) < 1e-4, f"camera {i} det(R)={det}"


def test_camera_center_radius():
    radius = 1.5
    cameras = syncdreamer_cameras(radius=radius)
    for i, cam in enumerate(cameras):
        R = cam.extrinsics[:3, :3]
        t = cam.extrinsics[:3, 3]
        center = -R.T @ t  # world-space camera center
        mag = torch.norm(center).item()
        assert abs(mag - radius) < 1e-4, \
            f"camera {i} center magnitude {mag} != radius {radius}"


def test_camera_looks_at_origin():
    radius = 1.5
    cameras = syncdreamer_cameras(radius=radius)
    for i, cam in enumerate(cameras):
        R = cam.extrinsics[:3, :3]
        t = cam.extrinsics[:3, 3]
        center = -R.T @ t
        # Forward (+Z) axis in world coords is the third row of R (world->camera).
        forward_world = R[2, :]
        direction_to_origin = -center / torch.norm(center)
        # Forward should point toward the origin: cosine ~= +1.
        cos_sim = torch.dot(forward_world, direction_to_origin).item()
        assert cos_sim > 0.999, \
            f"camera {i} forward not aimed at origin (cos={cos_sim})"


if __name__ == "__main__":
    test_count_is_16()
    test_matrix_shapes()
    test_intrinsics_values()
    test_rotation_orthonormal_and_det_one()
    test_camera_center_radius()
    test_camera_looks_at_origin()
    print("All syncdreamer_cameras tests passed (16 cameras validated).")
