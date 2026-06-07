"""
Database engine and session factory (SQLAlchemy 2.x).

Responsibilities:
- Build the SQLAlchemy engine from settings.DATABASE_URL
- Provide a thread-safe sessionmaker (SessionLocal)
- Expose the declarative Base and an init_db() that creates tables

For sqlite we pass check_same_thread=False so sessions can be used across
threads/executors, and use the default (thread-safe) QueuePool replacement
that SQLAlchemy selects for file-backed sqlite URLs.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

Base = declarative_base()

_DATABASE_URL = settings.DATABASE_URL

_connect_args = {}
if _DATABASE_URL.startswith("sqlite"):
    # Allow the connection to be shared across threads (FastAPI runs the
    # pipeline in executor threads). SQLAlchemy's default pool for a file
    # sqlite URL is thread-safe; we only need to relax the per-connection
    # same-thread check.
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    _DATABASE_URL,
    connect_args=_connect_args,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def init_db() -> None:
    """Create all tables registered on Base.metadata (idempotent)."""
    # Import models so they register on Base.metadata before create_all.
    from ..models import job  # noqa: F401

    Base.metadata.create_all(bind=engine)
