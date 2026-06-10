"""
Tests for POST /generate/ and GET /generate/status/{job_id}.

The real pipeline (torch + ML) is never run: we monkeypatch
generate.pipeline_manager.run_pipeline with a fake coroutine returning a
PipelineResult. TestClient executes BackgroundTasks synchronously after the
response is returned, so by the time client.post() returns, the (fake) pipeline
has already finished and the job row reflects the terminal state.

The input image MUST live inside assets/uploads (the route confines image_path
to that tree to block arbitrary-file access). We point settings.PROJECT_ROOT at
a temp dir so the whole upload/output tree is isolated per test.
"""

import pytest


class _FakePipelineResult:
    """Minimal stand-in for services.pipeline_manager.PipelineResult."""

    def __init__(self, success, final_model_path=None, error=None, warnings=None):
        self.success = success
        self.final_model_path = final_model_path
        self.error = error
        self.warnings = warnings or []
        self.intermediate_files = {}
        self.metrics = {}


@pytest.fixture()
def isolated_root(monkeypatch, tmp_path):
    """Point settings.PROJECT_ROOT at a temp dir and return its uploads dir."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "PROJECT_ROOT", tmp_path)
    uploads = tmp_path / "assets" / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads


@pytest.fixture()
def image_file(isolated_root):
    """A real image file inside the (temp) uploads tree so confinement passes."""
    path = isolated_root / "input.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return str(path)


def _patch_pipeline(monkeypatch, result):
    """Replace the module-level pipeline_manager.run_pipeline with a fake."""
    from app.routes import generate

    async def _fake_run_pipeline(image_path, output_dir=None, progress_callback=None):
        # Exercise the per-job progress callback path if one was supplied.
        return result

    monkeypatch.setattr(generate.pipeline_manager, "run_pipeline", _fake_run_pipeline)
    return generate


def test_generate_creates_job_and_status_returns_it(client, monkeypatch, image_file, tmp_path):
    fake_model = str(tmp_path / "model.ply")
    result = _FakePipelineResult(success=True, final_model_path=fake_model)
    _patch_pipeline(monkeypatch, result)

    resp = client.post("/generate/", json={"image_path": image_file})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    job_id = body["job_id"]
    assert job_id

    # Background task has run synchronously under TestClient.
    status_resp = client.get(f"/generate/status/{job_id}")
    assert status_resp.status_code == 200, status_resp.text
    status = status_resp.json()
    assert status["job_id"] == job_id
    assert status["status"] == "completed"
    assert status["result"]["model_url"]


def test_generate_marks_failed_when_pipeline_fails(client, monkeypatch, image_file):
    result = _FakePipelineResult(success=False, error="synthetic pipeline failure")
    _patch_pipeline(monkeypatch, result)

    resp = client.post("/generate/", json={"image_path": image_file})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    status = client.get(f"/generate/status/{job_id}").json()
    assert status["status"] == "failed"
    assert "synthetic pipeline failure" in (status["error"] or "")


def test_generate_missing_image_returns_error(client, monkeypatch, isolated_root):
    # Patch so even if the check passed we would not run the real pipeline.
    _patch_pipeline(monkeypatch, _FakePipelineResult(success=True, final_model_path="x"))

    resp = client.post(
        "/generate/",
        json={"image_path": str(isolated_root / "no_such_image.png")},
    )
    # Missing-but-in-tree file -> 404; the route never reaches the pipeline.
    assert resp.status_code in (400, 404)


def test_generate_rejects_path_traversal(client, monkeypatch, isolated_root, tmp_path):
    """An absolute image_path to a real file OUTSIDE uploads must be refused.

    This is the exact arbitrary-file-access vector: a secret that exists on disk
    but lives outside the uploads tree must never be accepted as an input image.
    """
    _patch_pipeline(monkeypatch, _FakePipelineResult(success=True, final_model_path="x"))

    secret = tmp_path / "secret.txt"  # sibling of assets/uploads, not inside it
    secret.write_text("top secret")

    resp = client.post("/generate/", json={"image_path": str(secret)})
    assert resp.status_code == 400


def test_status_unknown_job_returns_404(client):
    resp = client.get("/generate/status/nonexistent-job")
    assert resp.status_code == 404
