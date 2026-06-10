"""
CPU smoke test for the MVCRM back-projection refinement service.

Builds a tiny synthetic GaussianModel (~50 splats), saves it to a temp PLY,
then runs BackProjectionService.refine_model with a couple of small synthetic
enhanced views and a single refinement iteration. Asserts the refined PLY is
written and preserves the input vertex count, without crashing.

Sizes are kept tiny so this runs in a few seconds on CPU.
"""

import asyncio
import importlib.util
import os
import sys
import tempfile

import pytest

# Heavy deps live only in the "backend-heavy" CI job (and locally with torch
# installed). importorskip here so the light/no-torch job skips this module at
# COLLECTION time instead of erroring before @pytest.mark.heavy can deselect it.
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("plyfile")
from PIL import Image

# Ensure the repo root is importable (mirrors test_core_modules.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(_REPO_ROOT)


def _load_utils_gs():
    """Load utils_gs directly by path.

    ai_modules.gsplat.__init__ imports the CUDA ``gsplat`` rasterizer, which is
    not installable on CPU, so a plain package import fails. Loading the leaf
    module from its file bypasses the heavy package __init__.
    """
    try:
        from ai_modules.gsplat.utils_gs import GaussianModel, save_ply, load_ply
        return GaussianModel, save_ply, load_ply
    except Exception:
        path = os.path.join(_REPO_ROOT, "ai_modules", "gsplat", "utils_gs.py")
        spec = importlib.util.spec_from_file_location(
            "ai_modules.gsplat.utils_gs", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.GaussianModel, mod.save_ply, mod.load_ply


GaussianModel, save_ply, load_ply = _load_utils_gs()

from ai_modules.refine_module import RefinementConfig
from backend.app.services.backprojection import BackProjectionService


def _make_synthetic_model(num_splats: int = 50) -> GaussianModel:
    """Build a small GaussianModel with raw fields in activation space."""
    torch.manual_seed(0)
    model = GaussianModel()

    # Positions clustered near the origin so they fall inside the view frustum.
    model._xyz = (torch.rand(num_splats, 3) - 0.5) * 0.5
    # SH DC features (3 channels) -> mid-gray-ish colors.
    model._features_dc = torch.zeros(num_splats, 3)
    # No higher-order SH terms for this smoke test.
    model._features_rest = torch.zeros(num_splats, 0)
    # Opacity stored in logit space (logit(0.9) ~ 2.2).
    model._opacity = torch.full((num_splats, 1), 2.0)
    # Scaling stored in log space (log(0.05)).
    model._scaling = torch.full((num_splats, 3), float(np.log(0.05)))
    # Identity quaternions (w, x, y, z).
    rot = torch.zeros(num_splats, 4)
    rot[:, 0] = 1.0
    model._rotation = rot
    return model


@pytest.mark.heavy
def test_mvcrm_cpu_smoke():
    num_splats = 50

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = tmp

        # 1. Build + save the synthetic coarse model.
        model = _make_synthetic_model(num_splats)
        coarse_path = os.path.join(tmp_path, "coarse_model.ply")
        save_ply(model, coarse_path)
        assert os.path.exists(coarse_path)

        input_count = int(load_ply(coarse_path)._xyz.shape[0])
        assert input_count == num_splats

        # 2. Two small synthetic enhanced views (32x32 RGB).
        enhanced_views = {}
        for i in range(2):
            img_arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
            img_path = os.path.join(tmp_path, f"enhanced_{i}.png")
            Image.fromarray(img_arr).save(img_path)
            enhanced_views[f"view_{i}"] = img_path

        # 3. Run refinement with a single iteration (fast on CPU).
        # Disable the feature-consistency check: it pulls a DINOv2 backbone from
        # torch.hub (network) and requires image dims divisible by 14, neither of
        # which suits a fast, offline CPU smoke test. The render -> back-project
        # -> smooth -> PLY round-trip path is still fully exercised.
        service = BackProjectionService()
        config = RefinementConfig(max_iterations=1, enable_feature_check=False)
        result = asyncio.run(service.refine_model(
            coarse_model_path=coarse_path,
            enhanced_views=enhanced_views,
            depth_maps={},
            output_dir=tmp_path,
            config=config,
        ))

        # 4. Assertions: success, file exists, same vertex count.
        assert result['success'], f"refine_model failed: {result.get('error')}"
        refined_path = result['refined_model_path']
        assert os.path.exists(refined_path), "refined PLY was not written"

        refined_count = int(load_ply(refined_path)._xyz.shape[0])
        assert refined_count == input_count, (
            f"vertex count changed: {refined_count} != {input_count}"
        )


if __name__ == "__main__":
    test_mvcrm_cpu_smoke()
    print("MVCRM CPU smoke test passed.")
