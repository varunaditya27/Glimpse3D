"""
Filesystem path-safety helpers.

Centralizes the rules that keep user-supplied identifiers (upload filenames,
``model_id`` / ``job_id`` path segments) from escaping the directories they are
meant to live in. Every route that builds a path from request data should go
through here instead of joining strings by hand.

The two threats handled:

1. Path traversal — a ``model_id`` like ``../../etc`` or an upload filename like
   ``../../../home/user/.ssh/authorized_keys`` must never resolve outside the
   intended base directory.
2. Absolute-path injection — ``generate`` accepts an ``image_path`` from the
   client; it must be confined to the uploads tree so the pipeline cannot be
   pointed at arbitrary files on the host.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

# Conservative allowlist for a single path segment (job_id / model_id).
# Matches what we generate ("gen_<uuid hex>") plus legacy ids, and nothing that
# could traverse or reference a parent directory.
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

# Characters we keep when sanitizing an arbitrary upload filename. Everything
# else collapses to "_". The result is always a bare filename (no separators).
_FILENAME_KEEP_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_within(base: Path, target: Path) -> bool:
    """Return True iff ``target`` resolves to ``base`` or a descendant of it."""
    try:
        base_r = base.resolve()
        target_r = target.resolve()
    except (OSError, RuntimeError):
        return False
    return base_r == target_r or base_r in target_r.parents


def is_safe_segment(segment: str) -> bool:
    """True if ``segment`` is a single, traversal-free path component."""
    return bool(segment) and _SAFE_SEGMENT_RE.match(segment) is not None


def safe_subdir(base: Path, segment: str) -> Path:
    """Join ``segment`` under ``base``, rejecting traversal.

    Raises ValueError if the segment is not a safe single component or if the
    resolved path escapes ``base``.
    """
    if not is_safe_segment(segment):
        raise ValueError(f"Unsafe path segment: {segment!r}")
    candidate = base / segment
    if not is_within(base, candidate):
        raise ValueError(f"Path escapes base directory: {segment!r}")
    return candidate


def sanitize_filename(filename: str | None) -> str:
    """Reduce an arbitrary client filename to a safe bare filename.

    Strips any directory components, normalizes the extension, and guarantees a
    non-empty result. Never returns something containing a path separator or a
    parent reference.
    """
    if not filename:
        return "upload"
    # Drop any directory part the client may have sent (handles both / and \).
    base = Path(filename.replace("\\", "/")).name
    base = _FILENAME_KEEP_RE.sub("_", base).strip("._")
    if not base or base in {".", ".."}:
        return "upload"
    return base[:128]


def unique_upload_name(filename: str | None) -> str:
    """A collision-free, sanitized storage name for an uploaded file.

    Prefixes a short random token so two users uploading "image.png" never clash
    and so the stored name is unguessable. Deterministic hashing of the filename
    (the previous approach) both collided and leaked nothing useful.
    """
    token = uuid.uuid4().hex[:12]
    return f"{token}_{sanitize_filename(filename)}"


def new_job_id() -> str:
    """A process-stable, collision-free job id.

    ``hash(image_path) % 1000000`` (the previous scheme) was non-deterministic
    across restarts (PYTHONHASHSEED) and collided after ~1k jobs by the birthday
    bound. A uuid4 hex avoids both.
    """
    return f"gen_{uuid.uuid4().hex}"
