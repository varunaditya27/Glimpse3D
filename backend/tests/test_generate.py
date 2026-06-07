"""
Tests for POST /generate/ and GET /generate/status/{job_id}.

The real pipeline (torch + ML) is never run: we monkeypatch
generate.pipeline_manager.run_pipeline with a fake coroutine returning a
PipelineResult. TestClient executes BackgroundTasks synchronously after the
response is returned, so by the time client.post() returns, the (fake) pipeline
has already finished and the job row reflects the terminal state.
"""

import os
import tempfile

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
def image_file():
    """A real file on disk so the route's os.path.exists() check passes."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, b"\x89PNG\r\n\x1a\nfake")
    os.close(fd)
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


def _patch_pipeline(monkeypatch, result):
    """Replace the module-level pipeline_manager.run_pipeline with a fake."""
    from app.routes import generate

    async def _fake_run_pipeline(image_path, output_dir=None):
        return result

    monkeypatch.setattr(generate.pipeline_manager, "run_pipeline", _fake_run_pipeline)
    return generate


def test_generate_creates_job_and_status_returns_it(client, monkeypatch, image_file, tmp_path):
    fake_model = str(tmp_path / "model.ply")
    result = _FakePipelineResult(success=True, final_model_path=fake_model)
    _patch_pipeline(monkeypatch, result)

    resp = client.post(
        "/generate/",
        json={"image_path": image_file, "output_dir": str(tmp_path)},
    )
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


def test_generate_marks_failed_when_pipeline_fails(client, monkeypatch, image_file, tmp_path):
    result = _FakePipelineResult(success=False, error="synthetic pipeline failure")
    _patch_pipeline(monkeypatch, result)

    resp = client.post(
        "/generate/",
        json={"image_path": image_file, "output_dir": str(tmp_path)},
    )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    status = client.get(f"/generate/status/{job_id}").json()
    assert status["status"] == "failed"
    assert "synthetic pipeline failure" in (status["error"] or "")


def test_generate_missing_image_returns_error(client, monkeypatch, tmp_path):
    # Patch so even if the check passed we would not run the real pipeline.
    _patch_pipeline(monkeypatch, _FakePipelineResult(success=True, final_model_path="x"))

    resp = client.post(
        "/generate/",
        json={"image_path": str(tmp_path / "no_such_image.png")},
    )
    # Route raises HTTPException(400) -> wrapped/re-raised as 500 only on
    # unexpected errors; the explicit 400 path bubbles as 400's detail under a
    # 500 envelope per the route's try/except. Accept either client/server error.
    assert resp.status_code in (400, 500)


def test_status_unknown_job_returns_404(client):
    resp = client.get("/generate/status/nonexistent-job")
    assert resp.status_code == 404
