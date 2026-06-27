"""End-to-end pipeline: unpack a submission, run it against every configured
dataset, score each notebook, and return result records.

This is the unit the Space's ``/submit`` handler calls. It has no web/HF
dependencies so it can be driven directly from tests and a local CLI.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

from . import executor, scoring
from .config import RunnerConfig
from .results import ResultRecord

# What a participant sees when their notebook errors while executing. The raw
# error is the notebook's own stdout/traceback, which can echo the private eval
# inputs — so it is logged server-side (organizer-only) but never returned. (Format
# /scoring errors, which describe the participant's own submission.csv, are kept.)
GENERIC_EXEC_ERROR = (
    "your submission failed to run in the sandbox. Rehearse it locally with "
    "`python submit.py --dry` to see the full error; if it works locally but fails "
    "here, contact the maintainers.")


def _exec_failure(team: str, notebook: str, dataset_key: str, metric: str,
                  real_error: str | None, redact: bool = True) -> ResultRecord:
    """Record an execution failure. When ``redact`` (the Space), the raw error can
    echo the private inputs, so the participant gets a generic message while the full
    real error is kept in ``error_detail`` (persisted to the bucket, organizer-only)
    and echoed to the server log; when not (``--dry``, public data), the real error
    goes straight into ``error`` so the participant can debug."""
    if not redact:
        return ResultRecord(team=team, notebook=notebook, dataset_key=dataset_key,
                            metric=metric, score=None, ok=False, error=real_error)
    print(f"[runner] FAILED team={team!r} notebook={notebook!r} "
          f"dataset={dataset_key!r}:\n{real_error}", file=sys.stderr, flush=True)
    return ResultRecord(team=team, notebook=notebook, dataset_key=dataset_key,
                        metric=metric, score=None, ok=False,
                        error=GENERIC_EXEC_ERROR, error_detail=real_error)


def unpack(zip_path: str | Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not (dest / "submissions").is_dir():
        raise FileNotFoundError("submission must contain a `submissions/` directory at its root")
    return dest


def run_pipeline(submission_root: Path, team: str, config: RunnerConfig,
                 extra_env: dict[str, str] | None = None) -> list[ResultRecord]:
    """Run an already-unpacked submission and score it. Never raises per-notebook;
    failures are recorded as ``ok=False`` records.

    ``extra_env`` is merged into each notebook's environment (e.g. the submitter's
    ``NDIF_API_KEY`` so nnsight can authenticate remote traces)."""
    if config.sandbox:
        return _run_sandboxed(submission_root, team, config, extra_env)
    return _run_in_process(submission_root, team, config, extra_env)


def _score_record(team, notebook, ds, config, labels, submission_csv):
    """Score one produced submission.csv against labels; returns a ResultRecord."""
    try:
        preds = scoring.load_predictions(submission_csv)
        value = scoring.score(preds, labels, config.metric)
        return ResultRecord(team=team, notebook=notebook, dataset_key=ds.key,
                            metric=config.metric, score=value, ok=True)
    except scoring.ScoringError as e:
        return ResultRecord(team=team, notebook=notebook, dataset_key=ds.key,
                            metric=config.metric, score=None, ok=False, error=str(e))


def _run_sandboxed(submission_root: Path, team: str, config: RunnerConfig,
                   extra_env: dict[str, str] | None) -> list[ResultRecord]:
    from . import data, executor, sandbox

    layout = data.prepare_inputs(config)
    notebooks = executor.list_notebooks(submission_root)
    if not notebooks:
        raise FileNotFoundError("submissions/ contains no .ipynb files")
    rels = [nb.relative_to(submission_root).as_posix() for nb in notebooks]

    records: list[ResultRecord] = []
    # One scratch per request: venv, requirements install, and dataset-cache copy
    # are built once here and reused across every (dataset, notebook) run.
    with tempfile.TemporaryDirectory(prefix="aletheia-job-") as job:
        ctx, setup_err = sandbox.setup_job(submission_root, layout, Path(job), config)
        if setup_err is not None:
            # venv/pip failure applies to the whole submission — record it for
            # every (dataset, notebook) so each row reflects the same cause.
            real = f"[{setup_err.phase}] {setup_err.error}"
            return [_exec_failure(team, rel, ds.key, config.metric, real,
                                  redact=config.redact_errors)
                    for ds in config.datasets for rel in rels]

        for ds in config.datasets:
            labels = scoring.load_labels(ds, config.hf_token)
            for rel in rels:
                res = sandbox.run_notebook(ctx, rel, ds, config, extra_env=extra_env)
                if not res.ok:
                    records.append(_exec_failure(
                        team, rel, ds.key, config.metric,
                        f"[{res.phase}] {res.error}", redact=config.redact_errors))
                    continue
                records.append(_score_record(team, rel, ds, config, labels,
                                             res.submission_csv))
    return records


def _run_in_process(submission_root: Path, team: str, config: RunnerConfig,
                    extra_env: dict[str, str] | None) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    base_env = config.base_env()

    for ds in config.datasets:
        labels = scoring.load_labels(ds, config.hf_token)
        env = {**base_env, **ds.env(), **(extra_env or {})}
        with tempfile.TemporaryDirectory(prefix="aletheia-snap-") as snap:
            nb_results = executor.run_submission(
                submission_root, env, config.notebook_timeout, Path(snap))
            for nbr in nb_results:
                if not nbr.ok:
                    records.append(_exec_failure(team, nbr.notebook, ds.key,
                                                 config.metric, nbr.error,
                                                 redact=config.redact_errors))
                    continue
                records.append(_score_record(team, nbr.notebook, ds, config,
                                             labels, nbr.submission_csv))
    return records


def run_zip(zip_path: str | Path, team: str, config: RunnerConfig,
            extra_env: dict[str, str] | None = None) -> list[ResultRecord]:
    """Unpack a submission zip into a temp dir and run the pipeline."""
    with tempfile.TemporaryDirectory(prefix="aletheia-sub-") as tmp:
        root = unpack(zip_path, Path(tmp))
        return run_pipeline(root, team, config, extra_env)
