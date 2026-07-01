"""Local ``--dry`` rehearsal of a submission.

Runs the **real pipeline** (per-job venv → ``pip install -r requirements.txt`` →
notebook execution → offline data load → scoring) against the datasets listed in
``dry.yaml`` at the submission root, so a participant can confirm their submission
actually runs and produces a valid ``submission.csv`` before submitting — and can
edit ``dry.yaml`` to rehearse on whatever datasets they like.

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

# Participants point ``--dry`` at whatever datasets they like by editing
# ``dry.yaml`` at the submission root (inputs `index, model, lora, messages`;
# `-labels` repo `index, deceptive`, joined on `index`).
DRY_YAML = "dry.yaml"


def dry_config(cache_dir: str | None = None, root: str | Path = ".") -> RunnerConfig:
    """Build the ``--dry`` config from ``<root>/dry.yaml``.

    The datasets (and a few optional knobs: ``notebook_timeout``, ``ndif_host``)
    come from that file, so a participant can rehearse on different datasets by
    editing it. The dry-run *semantics* —
    sandboxed venv, no Landlock/seccomp/egress confinement, real (un-redacted)
    errors — are fixed here regardless of the file, so the rehearsal keeps
    mirroring the Space.
    """
    cfg_path = Path(root) / DRY_YAML
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found — --dry reads {DRY_YAML} to know which datasets "
            f"to rehearse on")
    import yaml  # local import: callers that build configs directly need no dep
    data = yaml.safe_load(cfg_path.read_text()) or {}

    datasets = [DatasetConfig(**d) for d in data.get("datasets", [])]
    if not datasets:
        raise ValueError(f"{cfg_path} has no `datasets` entries")
    return RunnerConfig(
        datasets=datasets,
        sandbox=True,
        confine=False,           # no Landlock/seccomp/egress locally (portable)
        enforce_egress=False,
        redact_errors=False,     # local rehearsal -> show the real error
        score_partial=True,      # so `--limit N` still scores the N rows it produced
        ndif_host=data.get("ndif_host", "https://aletheias.api.ndif.us"),
        notebook_timeout=int(data.get("notebook_timeout", 2700)),  # 45 min/dataset (matches the Space)
        cache_dir=cache_dir or str(Path(tempfile.gettempdir()) / "aletheia-dryrun-cache"),
    )


def dry_run(submission_root: str | Path, ndif_api_key: str | None = None,
            hf_token: str | None = None, cache_dir: str | None = None,
            limit: int | None = None, on_progress=None) -> list[ResultRecord]:
    """Rehearse every notebook in ``submission_root/submission`` and score it
    against the labels named in ``<submission_root>/dry.yaml`` (or the defaults).
    Returns the per-notebook result records.

    ``ndif_api_key`` and ``hf_token`` are forwarded into the run exactly as the
    Space does — the HF token so notebooks can load gated models you can access.
    ``limit`` is forwarded as ``ALETHEIA_LIMIT`` (the notebook decides how to use
    it, e.g. score only the first N rows). ``on_progress`` is called per (notebook,
    dataset) as it starts/finishes (real dataset names — fine here, --dry is local
    on public data)."""
    extra_env = {}
    if ndif_api_key:
        extra_env["NDIF_API_KEY"] = ndif_api_key
    if hf_token:
        extra_env["HF_TOKEN"] = hf_token
    if limit is not None:
        extra_env["ALETHEIA_LIMIT"] = str(limit)
    return run_pipeline(Path(submission_root), "dry-run",
                        dry_config(cache_dir, root=submission_root),
                        extra_env=extra_env or None, on_progress=on_progress)
