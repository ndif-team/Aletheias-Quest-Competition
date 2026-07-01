"""Team registry: binds each NDIF API key to a team name (stored in the bucket).

On a key's **first** submission a team name is required; we record
``sha256(key) -> team`` (the raw key is never stored). After that the key alone
identifies the team, so participants don't resubmit it — and a team name can't be
reused by a different key.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from .results import is_bucket_uri, parse_bucket_uri


class TeamRegistry:
    """``sha256(ndif_key) -> team`` map, backed by a bucket:// or local JSON file."""

    def __init__(self, uri: str, token: str | None = None):
        self.uri = uri
        self.token = token

    @staticmethod
    def key_hash(ndif_key: str) -> str:
        return hashlib.sha256(ndif_key.encode("utf-8")).hexdigest()

    def _load(self) -> dict[str, str]:
        if is_bucket_uri(self.uri):
            from huggingface_hub import download_bucket_files
            bucket_id, path = parse_bucket_uri(self.uri, "teams.json")
            with tempfile.TemporaryDirectory(prefix="aletheia-reg-") as tmp:
                local = Path(tmp) / "teams.json"
                download_bucket_files(bucket_id, files=[(path, str(local))],
                                      raise_on_missing_files=False, token=self.token)
                return json.loads(local.read_text()) if local.exists() else {}
        p = Path(self.uri)
        return json.loads(p.read_text()) if p.exists() else {}

    def _save(self, reg: dict[str, str]) -> None:
        data = json.dumps(reg, indent=2, sort_keys=True).encode("utf-8")
        if is_bucket_uri(self.uri):
            from huggingface_hub import batch_bucket_files
            bucket_id, path = parse_bucket_uri(self.uri, "teams.json")
            batch_bucket_files(bucket_id, add=[(data, path)], token=self.token)
        else:
            p = Path(self.uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def resolve(self, ndif_key: str, submitted_team: str) -> tuple[str | None, str | None]:
        """Return ``(team, None)`` for the key, or ``(None, error)``.

        Known key → its registered team (the submitted name is ignored). New key →
        requires a team name that isn't already taken; registers and returns it.
        """
        kh = self.key_hash(ndif_key)
        reg = self._load()
        if kh in reg:
            return reg[kh], None
        team = (submitted_team or "").strip()
        if not team:
            return None, ("this NDIF key is new — include a team name "
                          "(--team / $ALETHEIA_TEAM) on your first submission")
        if team in reg.values():
            return None, f"team name {team!r} is already taken by another NDIF key"
        reg[kh] = team
        self._save(reg)
        return team, None

    def lookup(self, ndif_key: str) -> str | None:
        """Read-only: the team registered to this key, or ``None``. Never writes."""
        return self._load().get(self.key_hash(ndif_key))
