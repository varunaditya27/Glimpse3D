"""
Tests for GET /export/{model_id}.

Heavy conversion deps (trimesh/open3d) are only needed for glb/obj. For the
valid-format case we use 'ply', which is a direct file pass-through, and we mock
the model file on disk under assets/outputs/{model_id}/.
"""

import pytest


def test_unknown_model_returns_404(client):
    resp = client.get("/export/no_such_model_id_xyz", params={"format": "ply"})
    assert resp.status_code == 404


def test_invalid_format_returns_400(client):
    resp = client.get("/export/anything", params={"format": "stl"})
    assert resp.status_code == 400
    assert "format" in resp.json()["detail"].lower()


def test_valid_ply_format_serves_file(client, project_root):
    model_id = "export_test_model"
    model_dir = project_root / "assets" / "outputs" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    ply_path = model_dir / "reconstructed.ply"
    ply_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    )

    try:
        resp = client.get(f"/export/{model_id}", params={"format": "ply"})
        assert resp.status_code == 200, resp.text
        assert resp.content.startswith(b"ply")
    finally:
        # Best-effort cleanup so we don't leave artifacts in assets/outputs.
        try:
            ply_path.unlink()
            model_dir.rmdir()
        except OSError:
            pass
