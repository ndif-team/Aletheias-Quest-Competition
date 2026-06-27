"""End-to-end pipeline: unpack a submission, run it against every configured
dataset, score each notebook, and return result records.

This is the unit the Space's ``/submit`` handler calls. It has no web/HF
dependencies so it can be driven directly from tests and a local CLI.
"""

from __future__ import annotations

import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

from . import executor, scoring
from .config import RunnerConfig
from .log import get_logger
from .results import ResultRecord
from opentelemetry import trace as trace_api

from .trace import attach_context, detach_context, get_tracer

log = get_logger(__name__)
tracer = get_tracer(__name__)

# Sink for produced submission.csv bytes: ``(notebook_rel, dataset_key, csv_bytes)``.
CsvSink = Callable[[str, str, bytes], None]

# What a participant sees when their notebook errors while executing. The raw
# error is the notebook's own stdout/traceback, which can echo the private eval
# inputs — so it is logged server-side (organizer-only) but never returned. (Format
# /scoring errors, which describe the participant's own submission.csv, are kept.)
GENERIC_EXEC_ERROR = (
    "your submission failed to run in the sandbox. Rehearse it locally with "
    "`python submit.py --dry` to see the full error; if it works locally but fails "
    "here, contact the maintainers.")


def _exec_failure(team: str, notebook: str, dataset_key: str,
                  real_error: str | None, redact: bool = True) -> ResultRecord:
    """Record an execution failure. When ``redact`` (the Space), the raw error can
    echo the private inputs, so the participant gets a generic message while the full
    real error is kept in ``error_detail`` (persisted to the bucket, organizer-only)
    and echoed to the server log; when not (``--dry``, public data), the real error
    goes straight into ``error`` so the participant can debug."""
    if not redact:
        return ResultRecord(team=team, notebook=notebook, dataset_key=dataset_key,
                            ok=False, error=real_error)
    log.error("notebook.exec_failed", extra={
        "team": team, "notebook": notebook, "dataset": dataset_key,
        "error": real_error,
    })
    return ResultRecord(team=team, notebook=notebook, dataset_key=dataset_key,
                        ok=False, error=GENERIC_EXEC_ERROR, error_detail=real_error)


def _emit_csv(sink: CsvSink | None, notebook: str, dataset_key: str,
              csv_path: Path | None) -> None:
    """Hand a produced submission.csv to the archive sink. Never sinks the run."""
    if sink is None or csv_path is None:
        return
    try:
        sink(notebook, dataset_key, Path(csv_path).read_bytes())
    except Exception as e:  # noqa: BLE001
        log.warning("csv_archive.failed", extra={
            "notebook": notebook, "dataset": dataset_key, "error": str(e),
        })


def unpack(zip_path: str | Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not (dest / "submission").is_dir():
        raise FileNotFoundError("submission must contain a `submission/` directory at its root")
    return dest


def validate_submission(submission_root: Path) -> list[Path]:
    """Structural checks a submission must pass *before* we spend resources — or a
    rate-limit attempt — running it: a ``submission/`` directory holding **exactly
    one** notebook. Returns that one-element notebook list; raises ``FileNotFoundError``
    (missing dir / no notebook) or ``ValueError`` (more than one notebook).

    Called by ``run_pipeline`` (so ``--dry`` and the Space enforce the same rule) and
    by the Space's ``/submit`` up front, so it can reject before charging an attempt."""
    if not (submission_root / "submission").is_dir():
        raise FileNotFoundError(
            "submission must contain a `submission/` directory at its root")
    notebooks = executor.list_notebooks(submission_root)   # ValueError if more than one
    if not notebooks:
        raise FileNotFoundError("submission/ contains no .ipynb files")
    return notebooks


def run_pipeline(submission_root: Path, team: str, config: RunnerConfig,
                 extra_env: dict[str, str] | None = None,
                 on_submission_csv: CsvSink | None = None,
                 on_progress: Callable[[dict], None] | None = None,
                 ctx=None) -> list[ResultRecord]:
    """Run an already-unpacked submission and score it. Never raises per-notebook;
    failures are recorded as ``ok=False`` records.

    ``extra_env`` is merged into each notebook's environment (e.g. the submitter's
    ``NDIF_API_KEY`` so nnsight can authenticate remote traces). ``on_submission_csv``,
    if given, is called with each produced submission.csv so it can be archived.
    ``on_progress``, if given, is called with a dict as each (notebook, dataset) run
    starts and finishes — ``{"phase": "start"|"done", "dataset", "index", "total",
    "ok", "metrics", "error"}``. NOTE: ``dataset`` is the **real** key, so only wire
    this where that's allowed (``--dry`` on public data, never the participant-facing Space).

    ``ctx`` is an OTEL context captured in the async handler before handing off to
    this worker thread — attaching it makes the pipeline's spans nest under the
    request's root submission.handle span.

    The whole-submission wall-clock is measured and stamped on every record."""
    token = attach_context(ctx) if ctx is not None else None
    try:
        # One notebook per submission, fail fast (covers --dry, which calls straight in).
        validate_submission(submission_root)
        start = time.monotonic()
        if config.sandbox:
            records = _run_sandboxed(submission_root, team, config, extra_env,
                                     on_submission_csv, on_progress)
        else:
            records = _run_in_process(submission_root, team, config, extra_env,
                                      on_submission_csv, on_progress)
        elapsed = time.monotonic() - start
        for r in records:
            r.runtime_seconds = elapsed
        return records
    finally:
        if token is not None:
            detach_context(token)


def _score_record(team, notebook, ds, labels, submission_csv, partial=False):
    """Score one produced submission.csv against labels; returns a ResultRecord."""
    with tracer.start_as_current_span("notebook.score") as span:
        span.set_attribute("team", team)
        span.set_attribute("notebook", notebook.split("/")[-1])
        span.set_attribute("dataset", ds.key)
        try:
            preds = scoring.load_predictions(submission_csv)
            metrics = scoring.compute_metrics(preds, labels, partial=partial)
            return ResultRecord(team=team, notebook=notebook, dataset_key=ds.key,
                                metrics=metrics, ok=True)
        except scoring.ScoringError as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e)[:200])
            log.warning("notebook.scoring_failed", extra={
                "team": team, "notebook": notebook, "dataset": ds.key, "error": str(e),
            })
            return ResultRecord(team=team, notebook=notebook, dataset_key=ds.key,
                                ok=False, error=str(e))


def _run_sandboxed(submission_root: Path, team: str, config: RunnerConfig,
                   extra_env: dict[str, str] | None,
                   on_submission_csv: CsvSink | None,
                   on_progress: Callable[[dict], None] | None = None) -> list[ResultRecord]:
    from . import data, executor, sandbox

    with tracer.start_as_current_span("dataset.prepare") as prep_span:
        prep_span.set_attribute("datasets", [d.key for d in config.datasets])
        layout, cache_hit = data.prepare_inputs(config)
        prep_span.set_attribute("cache_hit", cache_hit)
    notebooks = executor.list_notebooks(submission_root)
    if not notebooks:
        raise FileNotFoundError("submission/ contains no .ipynb files")
    rels = [nb.relative_to(submission_root).as_posix() for nb in notebooks]
    total = len(rels) * len(config.datasets)
    done = 0

    records: list[ResultRecord] = []
    with tempfile.TemporaryDirectory(prefix="aletheia-job-") as job:
        req = submission_root / "requirements.txt"
        has_requirements = req.exists() and bool(req.read_text().strip())
        with tracer.start_as_current_span("sandbox.setup") as setup_span:
            setup_span.set_attribute("team", team)
            setup_span.set_attribute("has_requirements", has_requirements)
            sctx, setup_err = sandbox.setup_job(submission_root, layout, Path(job), config)
            if setup_err is not None:
                setup_span.set_attribute("failed_phase", setup_err.phase)
                setup_span.set_status(trace_api.StatusCode.ERROR, setup_err.error[:200])
        if setup_err is not None:
            real = f"[{setup_err.phase}] {setup_err.error}"
            return [_exec_failure(team, rel, ds.key, real,
                                  redact=config.redact_errors)
                    for ds in config.datasets for rel in rels]

        # Notebook-outer / dataset-inner: a notebook is fully scored across every
        # dataset before the next notebook runs. Each (notebook, dataset) run gets
        # its own ``notebook_timeout`` (per execution, never a shared overall cap).
        # Fail fast: the first failed (notebook, dataset) — execution or scoring —
        # aborts the whole submission, no partial scores.
        labels_cache: dict[str, object] = {}
        for rel in rels:
            nb_name = rel.split("/")[-1]
            for ds in config.datasets:
                done += 1
                if on_progress:
                    on_progress({"phase": "start", "dataset": ds.key,
                                 "index": done, "total": total})
                if ds.key not in labels_cache:
                    with tracer.start_as_current_span("labels.load") as lbl_span:
                        lbl_span.set_attribute("dataset", ds.key)
                        labels_cache[ds.key] = scoring.load_labels(ds, config.hf_token)
                labels = labels_cache[ds.key]
                with tracer.start_as_current_span("notebook.run") as nb_span:
                    nb_span.set_attribute("team", team)
                    nb_span.set_attribute("notebook", nb_name)
                    nb_span.set_attribute("dataset", ds.key)
                    res = sandbox.run_notebook(sctx, rel, ds, config, extra_env=extra_env)
                    if not res.ok:
                        nb_span.set_status(
                            trace_api.StatusCode.ERROR,
                            f"[{res.phase}] {res.error[:200]}")
                if not res.ok:
                    records.append(_exec_failure(
                        team, rel, ds.key,
                        f"[{res.phase}] {res.error}", redact=config.redact_errors))
                    if on_progress:
                        on_progress({"phase": "done", "dataset": ds.key, "index": done,
                                     "total": total, "ok": False, "error": res.error})
                    return records
                _emit_csv(on_submission_csv, rel, ds.key, res.submission_csv)
                rec = _score_record(team, rel, ds, labels, res.submission_csv,
                                    partial=config.score_partial)
                records.append(rec)
                if on_progress:
                    on_progress({"phase": "done", "dataset": ds.key, "index": done,
                                 "total": total, "ok": rec.ok,
                                 "metrics": rec.metrics, "error": rec.error})
                if not rec.ok:
                    return records
    return records


def _run_in_process(submission_root: Path, team: str, config: RunnerConfig,
                    extra_env: dict[str, str] | None,
                    on_submission_csv: CsvSink | None,
                    on_progress: Callable[[dict], None] | None = None) -> list[ResultRecord]:
    records: list[ResultRecord] = []
    base_env = config.base_env()
    notebooks = executor.list_notebooks(submission_root)
    if not notebooks:
        raise FileNotFoundError("submission/ contains no .ipynb files")
    total = len(notebooks) * len(config.datasets)
    done = 0

    # Notebook-outer / dataset-inner with fail-fast: fully score one notebook
    # (across all datasets) before the next; each (notebook, dataset) run gets its
    # own notebook_timeout; the first failure aborts the whole submission.
    labels_cache: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="aletheia-snap-") as snap:
        for nb in notebooks:
            for ds in config.datasets:
                done += 1
                if on_progress:
                    on_progress({"phase": "start", "dataset": ds.key,
                                 "index": done, "total": total})
                if ds.key not in labels_cache:
                    with tracer.start_as_current_span("labels.load") as lbl_span:
                        lbl_span.set_attribute("dataset", ds.key)
                        labels_cache[ds.key] = scoring.load_labels(ds, config.hf_token)
                labels = labels_cache[ds.key]
                env = {**base_env, **ds.env(), **(extra_env or {})}
                nbr = executor.run_notebook(nb, submission_root, env,
                                            config.notebook_timeout, Path(snap))
                if not nbr.ok:
                    records.append(_exec_failure(team, nbr.notebook, ds.key,
                                                 nbr.error, redact=config.redact_errors))
                    if on_progress:
                        on_progress({"phase": "done", "dataset": ds.key, "index": done,
                                     "total": total, "ok": False, "error": nbr.error})
                    return records
                _emit_csv(on_submission_csv, nbr.notebook, ds.key, nbr.submission_csv)
                rec = _score_record(team, nbr.notebook, ds, labels, nbr.submission_csv,
                                    partial=config.score_partial)
                records.append(rec)
                if on_progress:
                    on_progress({"phase": "done", "dataset": ds.key, "index": done,
                                 "total": total, "ok": rec.ok,
                                 "metrics": rec.metrics, "error": rec.error})
                if not rec.ok:
                    return records
    return records


def run_zip(zip_path: str | Path, team: str, config: RunnerConfig,
            ctx=None,
            extra_env: dict[str, str] | None = None,
            on_submission_csv: CsvSink | None = None) -> list[ResultRecord]:
    """Unpack a submission zip into a temp dir and run the pipeline.

    ``ctx`` is an OTEL context captured in the async handler before handing off
    to this thread — attaching it here makes all child spans nest under the
    root submission.handle span.
    """
    token = attach_context(ctx) if ctx is not None else None
    try:
        with tempfile.TemporaryDirectory(prefix="aletheia-sub-") as tmp:
            root = unpack(zip_path, Path(tmp))
            return run_pipeline(root, team, config, extra_env, on_submission_csv)
    finally:
        if token is not None:
            detach_context(token)
