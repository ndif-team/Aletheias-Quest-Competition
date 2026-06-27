"""Result records and persistence.

Two backends behind a common interface:
- ``LocalResultStore`` — append-only JSONL on disk (default; used by tests).
- ``BucketResultStore`` — read-modify-write a JSONL object in a HF bucket.

``make_store(uri, token)`` picks one: a ``bucket://org/name/path`` uri selects
the bucket backend, anything else is a local path. The leaderboard logic lives
on the base class, so it's identical for both.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


def is_bucket_uri(uri: str) -> bool:
    return uri.startswith("bucket://")


def parse_bucket_uri(uri: str, default_path: str) -> tuple[str, str]:
    """Split ``bucket://org/name[/path]`` into ``(org/name, path)``.

    ``path`` falls back to ``default_path`` when the uri names only the bucket.
    Raises ``ValueError`` if the org/name pair is missing.
    """
    parts = uri[len("bucket://"):].split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"bucket uri must be bucket://org/name[/path], got {uri!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]) or default_path


@dataclass
class ResultRecord:
    team: str
    notebook: str
    dataset_key: str
    metric: str
    score: float | None        # None when the run failed
    ok: bool
    # Participant-facing message: generic for sandboxed execution failures (the raw
    # error can echo the private inputs), the real message for format/scoring errors
    # (which describe the participant's own submission.csv).
    error: str | None = None
    # Organizer-only: the FULL real error/traceback for a redacted execution failure.
    # Persisted to the bucket (so failures can be diagnosed from S3) but never returned
    # to participants. None when ``error`` already holds the real error (no redaction).
    error_detail: str | None = None
    submitted_at: str | None = None  # ISO timestamp, stamped by the caller


def _parse_jsonl(text: str) -> list[ResultRecord]:
    out = []
    for line in text.splitlines():
        if line.strip():
            out.append(ResultRecord(**json.loads(line)))
    return out


def _to_jsonl(records: list[ResultRecord]) -> str:
    return "".join(json.dumps(asdict(r)) + "\n" for r in records)


class BaseResultStore:
    def all(self) -> list[ResultRecord]:  # pragma: no cover - interface
        raise NotImplementedError

    def append(self, records: list[ResultRecord]) -> None:  # pragma: no cover
        raise NotImplementedError

    def leaderboard(self) -> list[dict]:
        """Best score per (team, notebook), sorted high to low.

        A submission's score is the **mean of its per-dataset scores** (the runner
        scores each notebook against every dataset; the individual dataset scores
        are kept in the records but never surfaced — the dataset identities aren't
        reported). Resubmitting updates the row only if the new submission's mean is
        better; ``submitted_at`` reflects when that best mean was achieved.
        """
        from collections import defaultdict

        # Group each submission's per-dataset scores. A submission's runs all share
        # one ``submitted_at`` stamp, so (team, notebook, stamp) identifies it.
        subs: dict[tuple, list[float]] = defaultdict(list)
        metric: dict[tuple, str] = {}
        for r in self.all():
            if not r.ok or r.score is None:
                continue
            subs[(r.team, r.notebook, r.submitted_at)].append(r.score)
            metric[(r.team, r.notebook)] = r.metric

        best: dict[tuple, tuple[float, str | None]] = {}   # (team, notebook) -> (mean, stamp)
        for (team, notebook, stamp), scores in subs.items():
            mean = sum(scores) / len(scores)
            tn = (team, notebook)
            if tn not in best or mean > best[tn][0]:
                best[tn] = (mean, stamp)

        rows = [{"team": team, "notebook": notebook, "metric": metric[(team, notebook)],
                 "score": mean, "submitted_at": stamp}
                for (team, notebook), (mean, stamp) in best.items()]
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows


class LocalResultStore(BaseResultStore):
    """Append-only JSONL on the local filesystem."""

    def __init__(self, uri: str):
        self.uri = uri

    def append(self, records: list[ResultRecord]) -> None:
        path = Path(self.uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(_to_jsonl(records))

    def all(self) -> list[ResultRecord]:
        path = Path(self.uri)
        return _parse_jsonl(path.read_text()) if path.exists() else []


class BucketResultStore(BaseResultStore):
    """JSONL object inside a HuggingFace bucket (read-modify-write).

    Suitable for low submission volume — each append downloads the current
    object, appends, and re-uploads. ``bucket_id`` is ``org/bucket-name``.
    """

    def __init__(self, bucket_id: str, remote_path: str = "results.jsonl",
                 token: str | None = None):
        self.bucket_id = bucket_id
        self.remote_path = remote_path
        self.token = token

    def _download_text(self) -> str:
        from huggingface_hub import download_bucket_files

        with tempfile.TemporaryDirectory(prefix="aletheia-bucket-") as tmp:
            local = Path(tmp) / "results.jsonl"
            download_bucket_files(
                self.bucket_id, files=[(self.remote_path, str(local))],
                raise_on_missing_files=False, token=self.token)
            return local.read_text() if local.exists() else ""

    def all(self) -> list[ResultRecord]:
        return _parse_jsonl(self._download_text())

    def append(self, records: list[ResultRecord]) -> None:
        from huggingface_hub import batch_bucket_files

        existing = self._download_text()
        payload = (existing + _to_jsonl(records)).encode("utf-8")
        batch_bucket_files(
            self.bucket_id, add=[(payload, self.remote_path)], token=self.token)


def make_store(uri: str, token: str | None = None) -> BaseResultStore:
    """Build a store from a uri. ``bucket://org/name/path`` → bucket; else local."""
    if is_bucket_uri(uri):
        bucket_id, remote_path = parse_bucket_uri(uri, "results.jsonl")
        return BucketResultStore(bucket_id, remote_path, token=token)
    return LocalResultStore(uri)


# Backwards-compatible alias (the local store was the original name).
ResultStore = LocalResultStore
