"""
Tests for the SqliteJobRepository (app.core.repository).

Covers basic CRUD, reap_stale on a stale active job, missing-job lookup, and
the key persistence regression: a job created by one repository instance is
visible to a freshly-constructed instance pointed at the SAME DATABASE_URL.
"""

from datetime import datetime, timedelta, timezone


def test_create_and_get_roundtrip(repo):
    repo.create("job_crud", "/tmp/in.png", "/tmp/out")
    row = repo.get("job_crud")

    assert row is not None
    assert row["job_id"] == "job_crud"
    assert row["status"] == "starting"
    assert row["progress"] == 0.0
    assert row["image_path"] == "/tmp/in.png"
    assert row["output_dir"] == "/tmp/out"
    assert row["warnings"] == []


def test_update_fields(repo):
    repo.create("job_upd", "/tmp/in.png", "/tmp/out")
    repo.update(
        "job_upd",
        status="processing",
        progress=0.5,
        message="halfway",
        warnings=["w1"],
    )
    row = repo.get("job_upd")

    assert row["status"] == "processing"
    assert row["progress"] == 0.5
    assert row["message"] == "halfway"
    assert row["warnings"] == ["w1"]


def test_mark_completed_and_list_completed(repo):
    repo.create("job_done", "/tmp/in.png", "/tmp/out")
    repo.mark_completed("job_done", {"model_url": "/outputs/job_done/m.ply"})

    row = repo.get("job_done")
    assert row["status"] == "completed"
    assert row["progress"] == 1.0
    assert row["result"] == {"model_url": "/outputs/job_done/m.ply"}

    completed = repo.list_completed()
    assert any(r["job_id"] == "job_done" for r in completed)


def test_mark_failed(repo):
    repo.create("job_fail", "/tmp/in.png", "/tmp/out")
    repo.mark_failed("job_fail", "boom", warnings=["partial"])

    row = repo.get("job_fail")
    assert row["status"] == "failed"
    assert row["error"] == "boom"
    assert row["warnings"] == ["partial"]


def test_get_missing_returns_none(repo):
    assert repo.get("does-not-exist") is None


def test_persistence_survives_new_instance(repo, db_url):
    """KEY REGRESSION: a new SqliteJobRepository against the SAME DATABASE_URL
    must still see a job created by the original instance (persistence is in the
    DB, not in-process memory)."""
    from app.core.repository import SqliteJobRepository

    repo.create("job_persist", "/tmp/in.png", "/tmp/out")

    # Brand-new instance, same DATABASE_URL (set in conftest before import).
    new_repo = SqliteJobRepository()
    row = new_repo.get("job_persist")

    assert row is not None, "persisted job not visible from a new repo instance"
    assert row["job_id"] == "job_persist"
    assert row["status"] == "starting"


def test_reap_stale_marks_active_job_failed(repo):
    """A starting/processing job whose updated_at is older than the timeout is
    marked failed by reap_stale()."""
    from app.core.db import SessionLocal
    from app.models.job import Job

    repo.create("job_stale", "/tmp/in.png", "/tmp/out")
    repo.update("job_stale", status="processing")

    # Force updated_at far into the past so it is unambiguously stale.
    session = SessionLocal()
    try:
        job = session.get(Job, "job_stale")
        job.updated_at = datetime.now(timezone.utc) - timedelta(hours=2)
        session.commit()
    finally:
        session.close()

    reaped = repo.reap_stale(timeout_seconds=60)
    assert reaped >= 1

    row = repo.get("job_stale")
    assert row["status"] == "failed"
    assert row["error"]  # a timeout message was set


def test_reap_stale_ignores_fresh_and_terminal_jobs(repo):
    repo.create("job_fresh", "/tmp/in.png", "/tmp/out")
    repo.update("job_fresh", status="processing")  # just-now timestamp

    repo.create("job_terminal", "/tmp/in.png", "/tmp/out")
    repo.mark_completed("job_terminal", {"model_url": "x"})

    reaped = repo.reap_stale(timeout_seconds=60)

    assert reaped == 0
    assert repo.get("job_fresh")["status"] == "processing"
    assert repo.get("job_terminal")["status"] == "completed"
