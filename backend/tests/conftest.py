"""
Shared pytest fixtures for the Glimpse3D backend test suite.

CRITICAL ORDERING: this module sets DATABASE_URL (and PERSISTENCE_BACKEND) to a
unique temp sqlite file in os.environ BEFORE the FastAPI app (and therefore
app.core.config / app.core.db) is ever imported. Both of those modules read the
env vars at import time, so the env MUST be in place first or the tests would
scribble on the developer's real ./glimpse3d.db.

The light suite (everything except @pytest.mark.heavy) only needs
fastapi / sqlalchemy / pillow / httpx / pytest and never imports torch.
"""

import os
import tempfile
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. Point the app at an isolated temp sqlite DB *before* importing anything
#    that reads settings.DATABASE_URL. This file lives for the whole session.
# ---------------------------------------------------------------------------
_TMP_DB_DIR = tempfile.mkdtemp(prefix="glimpse3d_test_db_")
_TMP_DB_PATH = os.path.join(_TMP_DB_DIR, f"test_{uuid.uuid4().hex}.db")

# SQLAlchemy sqlite URL. Use forward slashes so the URL is valid on Windows too.
_DB_URL = "sqlite:///" + _TMP_DB_PATH.replace("\\", "/")

os.environ["DATABASE_URL"] = _DB_URL
os.environ["PERSISTENCE_BACKEND"] = "sqlite"


@pytest.fixture(scope="session")
def db_url() -> str:
    """The isolated sqlite URL the whole test session runs against."""
    return _DB_URL


@pytest.fixture()
def fresh_db():
    """Truncate the jobs table so each test starts from an empty DB.

    Imports are local so this module stays import-light at collection time and
    so the env vars above are guaranteed to be set first.
    """
    from app.core.db import SessionLocal, init_db
    from app.models.job import Job

    init_db()
    session = SessionLocal()
    try:
        session.query(Job).delete()
        session.commit()
    finally:
        session.close()
    yield
    # Clean up after the test as well so ordering between tests is irrelevant.
    session = SessionLocal()
    try:
        session.query(Job).delete()
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def client(fresh_db):
    """A FastAPI TestClient bound to the isolated DB with an empty jobs table."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def repo(fresh_db):
    """A fresh SqliteJobRepository against the isolated DB (empty jobs table)."""
    from app.core.repository import SqliteJobRepository

    return SqliteJobRepository()


@pytest.fixture()
def project_root() -> Path:
    """Project root used by routes that resolve assets/outputs (settings.PROJECT_ROOT)."""
    from app.core.config import settings

    return Path(settings.PROJECT_ROOT)
