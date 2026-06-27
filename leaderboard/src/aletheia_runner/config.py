"""Runner configuration: which datasets to score on, where labels/results live.

Loadable from a YAML file (production) or constructed directly (tests). Kept free
of HuggingFace/Space concerns so the core can be exercised locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


# Every dataset (inputs and labels) is a single-split repo standardized to this
# split — no per-dataset config subsets, no split selection.
SPLIT = "test"


@dataclass(frozen=True)
class DatasetConfig:
    """One entry in the runner's inner loop: a dataset to predict on plus where
    its (private, held-out) labels live.

    One repo == one dataset (single ``test`` split, no config subsets).
    ``labels_uri`` may be an HF dataset id or a local ``.csv`` path; the labels
    table must have an id column and a binary label column.
    """

    name: str
    labels_uri: str
    id_column: str = "id"
    label_column: str = "deceptive"

    @property
    def key(self) -> str:
        """Stable identifier used in result records / the leaderboard."""
        return self.name

    def env(self) -> dict[str, str]:
        """Environment the notebook reads to know what to predict on."""
        return {"DATASET_NAME": self.name}


@dataclass(frozen=True)
class RunnerConfig:
    datasets: list[DatasetConfig]
    results_uri: str = "results.jsonl"
    teams_uri: str = "teams.json"   # NDIF-key -> team registry (bucket:// or local)
    submissions_uri: str = "submissions"   # archive of every uploaded zip (bucket:// or local dir)
    # Per-team submission rate limit (fixed window). max <= 0 disables it.
    rate_limits_uri: str = "rate_limits.json"
    rate_limit_max: int = 0
    rate_limit_window_hours: float = 0.0
    metric: str = "auroc"
    notebook_timeout: int = 1800  # seconds, per notebook (wall clock)
    # Sandboxed execution (Landlock + seccomp + rlimits + predownloaded RO data).
    sandbox: bool = False
    confine: bool = True            # apply Landlock/seccomp/egress/rlimits (False for --dry)
    # NDIF endpoint nnsight traces hit (injected as NDIF_HOST). None -> nnsight default.
    ndif_host: str | None = None
    # Replace a notebook's real error with a generic message in returned records
    # (the raw error can echo the private inputs). Off for --dry: local rehearsal on
    # the PUBLIC dataset, where the participant should see their actual error.
    redact_errors: bool = True
    cache_dir: str = "data/cache"   # where predownloaded inputs live
    cpu_seconds: int = 900          # RLIMIT_CPU per notebook
    mem_mb: int | None = None       # RLIMIT_AS (off by default; breaks some ML libs)
    # Egress is allowlisted by HOSTNAME SUFFIX via a loopback CONNECT proxy
    # (CDN-proof; the child is forced through it by a loopback-only seccomp gate).
    enforce_egress: bool = True
    # Run-phase hosts the loopback proxy permits. NDIF is CONNECT-tunneled (HTTPS,
    # unfiltered); HF is CONNECT/MITM (GET-only). api.ndif.us covers the hackathon
    # subdomain (aletheias.api.ndif.us) and any central auth call. The S3 host is
    # NDIF's trace-results bucket: a completed remote trace returns a presigned S3
    # URL nnsight streams the result from, so it must be reachable or every
    # completed trace 403s on the download (blind-tunneled; read-only, not team-writable).
    egress_allowlist: list[str] = field(
        default_factory=lambda: ["api.ndif.us", "huggingface.co", "hf.co",
                                 "ndif-hackathon-results.s3.amazonaws.com"])
    # Hosts within egress_allowlist that are MITM-terminated and restricted to
    # read methods (GET/HEAD). HF is reachable for model configs but can't be used
    # to push the private eval inputs out. NDIF is left blind-tunneled (unfiltered).
    egress_get_only_suffixes: list[str] = field(
        default_factory=lambda: ["huggingface.co", "hf.co"])
    install_allowlist: list[str] = field(          # install phase: PyPI
        default_factory=lambda: ["pypi.org", "pythonhosted.org"])
    hf_token: str | None = None

    def base_env(self) -> dict[str, str]:
        """Environment shared across every notebook run."""
        e: dict[str, str] = {}
        if self.hf_token:
            e["HF_TOKEN"] = self.hf_token
        return e

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunnerConfig":
        datasets = [DatasetConfig(**d) for d in data.get("datasets", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "datasets"}
        rest = {k: v for k, v in data.items() if k in known}
        return cls(datasets=datasets, **rest)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunnerConfig":
        import yaml  # local import so tests that build configs directly need no dep

        data = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls.from_dict(data)
        return cfg.with_env_overrides()

    def with_env_overrides(self) -> "RunnerConfig":
        """Pull secrets/overrides from the process environment."""
        return replace(self, hf_token=self.hf_token or os.environ.get("HF_TOKEN"))
