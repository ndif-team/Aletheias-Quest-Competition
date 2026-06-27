"""Local ``--dry`` rehearsal of a submission.

Runs the **real pipeline** (per-job venv → ``pip install -r requirements.txt`` →
notebook execution → offline data load → scoring) against the **same eval
datasets the Space scores on**, so a participant can confirm their submission
actually runs and produces a valid ``submission.csv`` before submitting.

It runs with ``confine=False`` — the server's Landlock/seccomp/egress confinement
is Linux/kernel-specific, so the dry run skips it to stay portable (the real run
on the Space applies it). The submitter's ``NDIF_API_KEY`` is forwarded so
nnsight remote traces authenticate, exactly as on the Space.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .config import DatasetConfig, RunnerConfig
from .pipeline import run_pipeline
from .results import ResultRecord

# Dry-run rehearses on the SAME three liars-bench eval datasets the Space scores
# on (inputs `index, model, lora, messages`; private `-labels` repo `index,
# deceptive`, joined on `index`). Keep in sync with runner.yaml's `datasets`.
DRYRUN_SUBSETS = ("soft-trigger", "harm-pressure-choice", "instructed-deception")
DRYRUN_DATASETS = [
    DatasetConfig(name=f"NDIF/dev-liars-bench-{s}",
                  labels_uri=f"NDIF/dev-liars-bench-{s}-labels",
                  id_column="index", label_column="deceptive")
    for s in DRYRUN_SUBSETS
]


def dry_config(cache_dir: str | None = None) -> RunnerConfig:
    return RunnerConfig(
        datasets=list(DRYRUN_DATASETS),
        sandbox=True,
        confine=False,           # no Landlock/seccomp/egress locally (portable)
        enforce_egress=False,
        redact_errors=False,     # local rehearsal on public data -> show real errors
        ndif_host="https://aletheias.api.ndif.us",   # hackathon NDIF stack
        metric="auroc",
        notebook_timeout=1200,   # 20 min: batched remote tracing over the full eval sets
        cpu_seconds=1200,
        cache_dir=cache_dir or str(Path(tempfile.gettempdir()) / "aletheia-dryrun-cache"),
    )


def dry_run(submission_root: str | Path, ndif_api_key: str | None = None,
            hf_token: str | None = None, cache_dir: str | None = None
            ) -> list[ResultRecord]:
    """Rehearse every notebook in ``submission_root/submissions`` and score it
    against the eval labels. Returns the per-notebook result records.

    ``ndif_api_key`` and ``hf_token`` are forwarded into the run exactly as the
    Space does — the HF token so notebooks can load gated models you can access."""
    extra_env = {}
    if ndif_api_key:
        extra_env["NDIF_API_KEY"] = ndif_api_key
    if hf_token:
        extra_env["HF_TOKEN"] = hf_token
    return run_pipeline(Path(submission_root), "dry-run", dry_config(cache_dir),
                        extra_env=extra_env or None)
