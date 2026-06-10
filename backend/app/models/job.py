"""
SQLAlchemy ORM model for a reconstruction job.

A Job tracks the lifecycle of a single image->3D pipeline run. JSON-shaped
columns (result, warnings) are stored as TEXT with json.dumps/loads so the
model is portable across sqlite (and the Supabase path mirrors the same
columns as native JSON).
"""

import json
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, String, Text

from ..core.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)  # == job_id
    status = Column(String, nullable=False, default="starting")
    progress = Column(Float, nullable=False, default=0.0)
    message = Column(Text, nullable=True)
    result = Column(Text, nullable=True)          # JSON-encoded dict
    error = Column(Text, nullable=True)
    warnings = Column(Text, nullable=True, default="[]")  # JSON-encoded list
    image_path = Column(String, nullable=True)
    output_dir = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    def to_dict(self) -> dict:
        """Return the contract dict shape (datetimes -> iso8601, JSON parsed)."""
        try:
            result = json.loads(self.result) if self.result else None
        except (TypeError, ValueError):
            result = None
        try:
            warnings = json.loads(self.warnings) if self.warnings else []
        except (TypeError, ValueError):
            warnings = []
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress if self.progress is not None else 0.0,
            "message": self.message,
            "result": result,
            "error": self.error,
            "warnings": warnings,
            "image_path": self.image_path,
            "output_dir": self.output_dir,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
