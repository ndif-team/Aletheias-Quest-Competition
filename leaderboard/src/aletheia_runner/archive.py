"""Archive every submitted zip so submissions can be retrieved later.

Stored at ``<base>/<team>/<UTC-timestamp>-<sha8>.zip`` — the raw uploaded bytes,
which are already a compressed zip. Each object has a unique path (timestamp +
content hash), so writes are plain puts, not read-modify-write, and need no lock.
Backed by a ``bucket://`` uri (persists across Space restarts) or a local dir.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from pathlib import Path

from .results import is_bucket_uri, parse_bucket_uri

DEFAULT_PREFIX = "submissions"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str, fallback: str) -> str:
    return _UNSAFE.sub("_", name or "").strip("_")[:64] or fallback


class SubmissionArchive:
    """Persists each uploaded submission zip for later inspection."""

    def __init__(self, uri: str, token: str | None = None):
        self.uri = uri
        self.token = token

    def _object_path(self, team: str, data: bytes, when: datetime.datetime) -> str:
        ts = when.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        digest = hashlib.sha256(data).hexdigest()[:8]
        return f"{_safe(team, 'team')}/{ts}-{digest}.zip"

    def save(self, team: str, data: bytes, when: datetime.datetime) -> str:
        """Store the zip and return its location. Raising is the caller's to handle
        (archiving must never sink a submission)."""
        rel = self._object_path(team, data, when)
        if is_bucket_uri(self.uri):
            from huggingface_hub import batch_bucket_files
            bucket_id, base = parse_bucket_uri(self.uri, DEFAULT_PREFIX)
            path = f"{base}/{rel}"
            batch_bucket_files(bucket_id, add=[(data, path)], token=self.token)
            return f"bucket://{bucket_id}/{path}"
        out = Path(self.uri) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)
