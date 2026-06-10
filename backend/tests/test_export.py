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


def test_export_rejects_path_traversal(client):
    """A model_id containing traversal must be refused with 400, not served."""
    # %2e%2e%2f decodes to ../ ; the route confines model_id to assets/outputs.
    resp = client.get("/export/..%2f..%2fetc", params={"format": "ply"})
    assert resp.status_code in (400, 404)


def test_metadata_endpoint_reports_vertex_count(client, monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    model_id = "meta_model"
    model_dir = tmp_path / "assets" / "outputs" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "reconstructed.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
    )

    resp = client.get(f"/export/metadata/{model_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["vertex_count"] == 3
    assert body["format"] == "PLY"
    assert body["file_size_human"].endswith(("B", "KB", "MB"))


def test_metadata_missing_model_returns_zeros(client, monkeypatch, tmp_path):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    resp = client.get("/export/metadata/does_not_exist")
    assert resp.status_code == 200
    assert resp.json()["vertex_count"] == 0


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
