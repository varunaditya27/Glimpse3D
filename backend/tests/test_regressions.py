"""
Source-level regression contracts.

These tests read project files as text / AST (no torch, no app import side
effects beyond locating files) and lock in specific bug fixes so they cannot
silently regress:

  (a) routes/refine.py calls enhance_views with the keyword views_data= (the
      param is views_data, not views).
  (b) services/pipeline_manager.py has no UNCONDITIONAL dummy-PLY writer: no
      `if "mock" in` guard, and any 'element vertex 4' placeholder is confined
      to the DEMO_MODE branch and never reported as success.
  (c) services/gsplat_service.py imports settings at module level.
  (d) backend/app/main.py has an `if __name__ == "__main__"` block.
  (e) ai_modules/gsplat/reconstruct.py has no np.random fabrication / 'Dummy
      Mesh' in the non-demo path.
"""

import ast
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"


def _read(rel_path: str) -> str:
    return (_REPO_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (a) refine.py uses views_data= keyword
# ---------------------------------------------------------------------------
def test_refine_calls_enhance_views_with_views_data_kwarg():
    src = _read("backend/app/routes/refine.py")
    tree = ast.parse(src)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "enhance_views"
    ]
    assert calls, "no enhance_views() call found in refine.py"

    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "views_data" in kwargs, (
            "enhance_views must be called with views_data= (not views=)"
        )
        assert "views" not in kwargs, (
            "enhance_views called with the wrong keyword 'views='"
        )


# ---------------------------------------------------------------------------
# (b) pipeline_manager.py: no unconditional dummy-PLY writer / no mock guard
# ---------------------------------------------------------------------------
def test_pipeline_manager_has_no_mock_guard():
    src = _read("backend/app/services/pipeline_manager.py")
    assert 'if "mock" in' not in src
    assert "if 'mock' in" not in src


def test_pipeline_manager_dummy_ply_is_demo_only_and_not_success():
    src = _read("backend/app/services/pipeline_manager.py")

    if "element vertex 4" not in src:
        # Strongest form of the fix: the placeholder writer is gone entirely.
        return

    # Otherwise it may only survive as the explicitly-labelled DEMO_MODE rescue
    # path, which must (1) be gated on settings.DEMO_MODE and (2) never be
    # reported as a successful pipeline result.
    assert "settings.DEMO_MODE" in src, (
        "'element vertex 4' placeholder present without a DEMO_MODE gate"
    )
    assert "DEMO placeholder" in src, (
        "placeholder PLY is not clearly labelled as a DEMO placeholder"
    )
    # The placeholder must be attached to a failure (success=False), proving it
    # is never passed off as a real reconstruction.
    assert "demo_placeholder_path" in src
    assert "success=False" in src


# ---------------------------------------------------------------------------
# (c) gsplat_service.py imports settings at module level
# ---------------------------------------------------------------------------
def test_gsplat_service_imports_settings_at_module_level():
    src = _read("backend/app/services/gsplat_service.py")
    tree = ast.parse(src)

    module_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    found = False
    for node in module_level_imports:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("config"):
            if any(alias.name == "settings" for alias in node.names):
                found = True
                break
    assert found, "gsplat_service.py must import settings at module level"


# ---------------------------------------------------------------------------
# (d) main.py has an if __name__ == "__main__" block
# ---------------------------------------------------------------------------
def test_main_has_dunder_main_block():
    src = _read("backend/app/main.py")
    tree = ast.parse(src)

    def _is_dunder_main(node):
        if not isinstance(node, ast.If):
            return False
        test = node.test
        return (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(
                isinstance(c, ast.Constant) and c.value == "__main__"
                for c in test.comparators
            )
        )

    assert any(_is_dunder_main(node) for node in tree.body), (
        "main.py is missing an `if __name__ == '__main__'` block"
    )


# ---------------------------------------------------------------------------
# (e) reconstruct.py has no np.random fabrication / 'Dummy Mesh'
# ---------------------------------------------------------------------------
def test_reconstruct_has_no_random_fabrication_or_dummy_mesh():
    src = _read("ai_modules/gsplat/reconstruct.py")

    assert "np.random" not in src, (
        "reconstruct.py must not fabricate geometry with np.random"
    )
    assert "Dummy Mesh" not in src
    assert "dummy mesh" not in src.lower()
