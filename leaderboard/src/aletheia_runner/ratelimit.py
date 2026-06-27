"""Per-team submission rate limiting (fixed window), stored in the bucket.

Each team gets ``max_submissions`` per ``window`` — a *tumbling* window that opens
on the team's first submission and resets ``window`` seconds later (so e.g. "3 per
4h" means: submit up to 3, then the budget refills 4h after the first of those).
State lives next to the registry/results in the bucket as
``{team: {"window_start": <epoch>, "count": <int>}}``; the raw NDIF key is never
stored (the team name is already the resolved identity).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from .results import is_bucket_uri, parse_bucket_uri

DEFAULT_NAME = "rate_limits.json"


class RateLimiter:
    """Fixed-window submission limiter backed by a ``bucket://`` or local JSON file.

    ``max_submissions <= 0`` or ``window_seconds <= 0`` disables it (unlimited)."""

    def __init__(self, uri: str, max_submissions: int, window_seconds: float,
                 token: str | None = None):
        self.uri = uri
        self.max = int(max_submissions)
        self.window = float(window_seconds)
        self.token = token

    @property
    def enabled(self) -> bool:
        return self.max > 0 and self.window > 0

    def _load(self) -> dict:
        if is_bucket_uri(self.uri):
            from huggingface_hub import download_bucket_files
            bucket_id, path = parse_bucket_uri(self.uri, DEFAULT_NAME)
            with tempfile.TemporaryDirectory(prefix="aletheia-rl-") as tmp:
                local = Path(tmp) / "rl.json"
                download_bucket_files(bucket_id, files=[(path, str(local))],
                                      raise_on_missing_files=False, token=self.token)
                return json.loads(local.read_text()) if local.exists() else {}
        p = Path(self.uri)
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, state: dict) -> None:
        data = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
        if is_bucket_uri(self.uri):
            from huggingface_hub import batch_bucket_files
            bucket_id, path = parse_bucket_uri(self.uri, DEFAULT_NAME)
            batch_bucket_files(bucket_id, add=[(data, path)], token=self.token)
        else:
            p = Path(self.uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def check_and_consume(self, team: str, now: float | None = None) -> tuple[bool, int]:
        """Consume one submission slot for ``team``.

        Returns ``(allowed, retry_after_seconds)`` — ``retry_after`` is 0 when
        allowed, else the seconds until the current window resets. This is a
        read-modify-write of the shared state, so the caller must serialize it
        (the app holds the bucket lock)."""
        if not self.enabled:
            return True, 0
        now = time.time() if now is None else now
        state = self._load()
        rec = state.get(team)
        if not rec or now >= float(rec.get("window_start", 0)) + self.window:
            state[team] = {"window_start": now, "count": 1}   # open a fresh window
            self._save(state)
            return True, 0
        if int(rec.get("count", 0)) < self.max:
            rec["count"] = int(rec.get("count", 0)) + 1
            state[team] = rec
            self._save(state)
            return True, 0
        retry = int(float(rec["window_start"]) + self.window - now) + 1
        return False, max(1, retry)

    def status(self, team: str, now: float | None = None) -> dict:
        """Current usage for ``team`` without consuming a slot. ``resets_at`` is an
        epoch (seconds) or ``None`` when no window is open."""
        window_hours = self.window / 3600 if self.window else 0
        if not self.enabled:
            return {"enabled": False, "max": self.max, "window_hours": window_hours,
                    "used": 0, "remaining": None, "resets_at": None}
        now = time.time() if now is None else now
        rec = self._load().get(team)
        if not rec or now >= float(rec.get("window_start", 0)) + self.window:
            used, resets_at = 0, None
        else:
            used = int(rec.get("count", 0))
            resets_at = float(rec["window_start"]) + self.window
        return {"enabled": True, "max": self.max, "window_hours": window_hours,
                "used": used, "remaining": max(0, self.max - used),
                "resets_at": resets_at}
