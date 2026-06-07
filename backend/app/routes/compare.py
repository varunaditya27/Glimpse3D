"""
backend/app/routes/compare.py

Compare API — side-by-side selection of two completed 3D reconstruction jobs.

Built on top of the persistent JobRepository (get_job_repo()), so it works in
the default SQLite mode as well as the Supabase backend. model_url is the
backend-relative path stored in the job result dict
(job["result"]["model_url"], e.g. "/outputs/<id>/...").

Contract:
  GET /compare/projects?limit=50
      -> { "projects": [ {id, model_url, created_at, status} ] }
  GET /compare/{id1}/{id2}
      -> { "a": {id, model_url, created_at, status},
           "b": {id, model_url, created_at, status} }
      404 if either id is missing or not completed.
"""

from fastapi import APIRouter, HTTPException

from ..core.repository import get_job_repo

router = APIRouter(prefix="/compare", tags=["compare"])


def _project_shape(row: dict) -> dict:
    """Map a JobRepository row dict to the compare project shape."""
    result = row.get("result") or {}
    model_url = result.get("model_url") if isinstance(result, dict) else None
    return {
        "id": row.get("job_id"),
        "model_url": model_url,
        "created_at": row.get("created_at"),
        "status": row.get("status"),
    }


@router.get("/projects")
async def list_projects(limit: int = 50):
    """List completed projects available for comparison (most recent first)."""
    repo = get_job_repo()
    rows = repo.list_completed()
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return {"projects": [_project_shape(row) for row in rows]}


@router.get("/{id1}/{id2}")
async def compare(id1: str, id2: str):
    """Fetch two completed projects for side-by-side comparison."""
    repo = get_job_repo()
    a = repo.get(id1)
    b = repo.get(id2)
    for job in (a, b):
        if job is None or job.get("status") != "completed":
            raise HTTPException(
                status_code=404,
                detail="One or both projects were not found or are not completed.",
            )
    return {"a": _project_shape(a), "b": _project_shape(b)}
