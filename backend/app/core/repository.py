"""
Job persistence layer.

Exposes get_job_repo() -> JobRepository, a process-wide singleton chosen by
settings.PERSISTENCE_BACKEND ("sqlite" default | "supabase").

All JobRepository methods are synchronous and safe to call from threads /
executors. The SQLite implementation opens a FRESH session per call and
commits+closes it before returning, so there is no shared session state
across threads.

get() / list_completed() return plain dicts with EXACTLY these keys:
    job_id, status, progress, message, result, error, warnings,
    image_path, output_dir, created_at, updated_at
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from .config import settings


# Active (non-terminal) statuses that reap_stale() may fail.
_ACTIVE_STATUSES = ("starting", "processing")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


class JobRepository(Protocol):
    """Structural interface implemented by both backends."""

    def create(self, job_id: str, image_path: str, output_dir: str) -> None: ...

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        warnings: Optional[list] = None,
        result: Optional[dict] = None,
    ) -> None: ...

    def get(self, job_id: str) -> Optional[dict]: ...

    def mark_completed(self, job_id: str, result: dict) -> None: ...

    def mark_failed(self, job_id: str, error: str, warnings: Optional[list] = None) -> None: ...

    def list_completed(self) -> list: ...

    def reap_stale(self, timeout_seconds: int) -> int: ...


class SqliteJobRepository:
    """SQLAlchemy-backed repository. One fresh session per method call."""

    def __init__(self) -> None:
        # Local imports keep db/model import side effects scoped to the
        # sqlite path and avoid circular imports at module load.
        from .db import init_db

        init_db()

    def create(self, job_id: str, image_path: str, output_dir: str) -> None:
        from .db import SessionLocal
        from ..models.job import Job

        session = SessionLocal()
        try:
            job = Job(
                id=job_id,
                status="starting",
                progress=0.0,
                image_path=str(image_path) if image_path is not None else None,
                output_dir=str(output_dir) if output_dir is not None else None,
                warnings="[]",
            )
            session.add(job)
            session.commit()
        finally:
            session.close()

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        warnings: Optional[list] = None,
        result: Optional[dict] = None,
    ) -> None:
        from .db import SessionLocal
        from ..models.job import Job

        session = SessionLocal()
        try:
            job = session.get(Job, job_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = progress
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if warnings is not None:
                job.warnings = json.dumps(warnings)
            if result is not None:
                job.result = json.dumps(result)
            session.commit()
        finally:
            session.close()

    def get(self, job_id: str) -> Optional[dict]:
        from .db import SessionLocal
        from ..models.job import Job

        session = SessionLocal()
        try:
            job = session.get(Job, job_id)
            return job.to_dict() if job is not None else None
        finally:
            session.close()

    def mark_completed(self, job_id: str, result: dict) -> None:
        self.update(job_id, status="completed", progress=1.0, result=result)

    def mark_failed(self, job_id: str, error: str, warnings: Optional[list] = None) -> None:
        self.update(job_id, status="failed", error=error, warnings=warnings)

    def list_completed(self) -> list:
        from .db import SessionLocal
        from ..models.job import Job

        session = SessionLocal()
        try:
            jobs = (
                session.query(Job)
                .filter(Job.status == "completed")
                .order_by(Job.created_at.desc())
                .all()
            )
            return [job.to_dict() for job in jobs]
        finally:
            session.close()

    def reap_stale(self, timeout_seconds: int) -> int:
        from .db import SessionLocal
        from ..models.job import Job

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)
        session = SessionLocal()
        try:
            stale = (
                session.query(Job)
                .filter(Job.status.in_(_ACTIVE_STATUSES))
                .filter(Job.updated_at < cutoff)
                .all()
            )
            count = 0
            for job in stale:
                job.status = "failed"
                if not job.error:
                    job.error = f"Job timed out (no update for >{timeout_seconds}s)"
                count += 1
            session.commit()
            return count
        finally:
            session.close()


class SupabaseJobRepository:
    """Supabase-backed repository operating on a 'jobs' table.

    Lazily imports the supabase client only when constructed. Raises a clear
    RuntimeError if the package or required credentials are missing.
    """

    _TABLE = "jobs"

    def __init__(self) -> None:
        try:
            import supabase  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "PERSISTENCE_BACKEND=supabase but the 'supabase' package is not "
                "installed. Run: pip install 'supabase>=2.0'."
            ) from exc

        url = settings.SUPABASE_URL
        key = settings.SUPABASE_SERVICE_ROLE_KEY
        if not url or not key:
            raise RuntimeError(
                "PERSISTENCE_BACKEND=supabase requires SUPABASE_URL and "
                "SUPABASE_SERVICE_ROLE_KEY to be set in the environment."
            )
        self._client = supabase.create_client(url, key)

    def _table(self):
        return self._client.table(self._TABLE)

    @staticmethod
    def _row_to_dict(row: dict) -> dict:
        return {
            "job_id": row.get("id"),
            "status": row.get("status"),
            "progress": row.get("progress") if row.get("progress") is not None else 0.0,
            "message": row.get("message"),
            "result": row.get("result"),
            "error": row.get("error"),
            "warnings": row.get("warnings") or [],
            "image_path": row.get("image_path"),
            "output_dir": row.get("output_dir"),
            "created_at": _iso(row.get("created_at")),
            "updated_at": _iso(row.get("updated_at")),
        }

    def create(self, job_id: str, image_path: str, output_dir: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._table().insert(
            {
                "id": job_id,
                "status": "starting",
                "progress": 0.0,
                "image_path": str(image_path) if image_path is not None else None,
                "output_dir": str(output_dir) if output_dir is not None else None,
                "warnings": [],
                "created_at": now,
                "updated_at": now,
            }
        ).execute()

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        warnings: Optional[list] = None,
        result: Optional[dict] = None,
    ) -> None:
        patch: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if status is not None:
            patch["status"] = status
        if progress is not None:
            patch["progress"] = progress
        if message is not None:
            patch["message"] = message
        if error is not None:
            patch["error"] = error
        if warnings is not None:
            patch["warnings"] = warnings
        if result is not None:
            patch["result"] = result
        self._table().update(patch).eq("id", job_id).execute()

    def get(self, job_id: str) -> Optional[dict]:
        resp = self._table().select("*").eq("id", job_id).limit(1).execute()
        data = resp.data or []
        if not data:
            return None
        return self._row_to_dict(data[0])

    def mark_completed(self, job_id: str, result: dict) -> None:
        self.update(job_id, status="completed", progress=1.0, result=result)

    def mark_failed(self, job_id: str, error: str, warnings: Optional[list] = None) -> None:
        self.update(job_id, status="failed", error=error, warnings=warnings)

    def list_completed(self) -> list:
        resp = (
            self._table()
            .select("*")
            .eq("status", "completed")
            .order("created_at", desc=True)
            .execute()
        )
        return [self._row_to_dict(row) for row in (resp.data or [])]

    def reap_stale(self, timeout_seconds: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).isoformat()
        resp = (
            self._table()
            .select("id")
            .in_("status", list(_ACTIVE_STATUSES))
            .lt("updated_at", cutoff)
            .execute()
        )
        rows = resp.data or []
        for row in rows:
            self.update(
                row["id"],
                status="failed",
                error=f"Job timed out (no update for >{timeout_seconds}s)",
            )
        return len(rows)


_repo_singleton: Optional[JobRepository] = None


def get_job_repo() -> JobRepository:
    """Return the process-wide JobRepository singleton.

    Backend is selected by settings.PERSISTENCE_BACKEND ("sqlite" default,
    "supabase" otherwise).
    """
    global _repo_singleton
    if _repo_singleton is not None:
        return _repo_singleton

    backend = (settings.PERSISTENCE_BACKEND or "sqlite").strip().lower()
    if backend == "supabase":
        _repo_singleton = SupabaseJobRepository()
    else:
        _repo_singleton = SqliteJobRepository()
    return _repo_singleton
