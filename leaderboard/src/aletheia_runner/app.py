"""FastAPI app for the leaderboard Space.

Thin web layer over the runner core:
- ``POST /submit``  — accept a zip + team name, run + score it, persist, return scores
- ``GET  /api/leaderboard`` — JSON leaderboard (best score per team/dataset)
- ``GET  /`` — static leaderboard page that reads the JSON API

``create_app(config, store)`` is injectable for tests; the module-level ``app``
is built from the environment for the Space.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from . import ndif, pipeline
from .archive import SubmissionArchive
from .config import PRIMARY_METRIC, SECONDARY_METRIC, RunnerConfig, dataset_label
from .ratelimit import RateLimiter
from .registry import TeamRegistry
from .results import BaseResultStore, make_store, summarize_submission

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "250"))
# How many submissions may execute at once. Each run holds a model graph + a
# notebook kernel locally (~1-2 GB; the heavy compute is remote on NDIF), so this
# bounds RAM/CPU on a small Space. Extra submissions queue on the semaphore.
MAX_CONCURRENT_SUBMISSIONS = int(os.environ.get("MAX_CONCURRENT_SUBMISSIONS", "2"))
# The leaderboard page polls /api/leaderboard every 30s per open tab, and each
# read pulls the whole results object from the bucket. Cache the computed board
# in-process for this long so N viewers cost ~one read per window, not N; a
# submission invalidates the cache so a fresh score shows up immediately.
LEADERBOARD_CACHE_TTL = float(os.environ.get("LEADERBOARD_CACHE_TTL", "20"))


class _LeaderboardCache:
    """Thread-safe TTL cache over ``store.leaderboard()`` (called from threadpool
    workers). ``invalidate()`` forces the next read to recompute."""

    def __init__(self, store: BaseResultStore, ttl: float):
        self._store = store
        self._ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._val: list[dict] | None = None

    def get(self) -> list[dict]:
        with self._lock:
            if self._val is not None and time.monotonic() - self._at < self._ttl:
                return self._val
        val = self._store.leaderboard()        # network read; outside the lock
        with self._lock:
            self._val, self._at = val, time.monotonic()
        return val

    def invalidate(self) -> None:
        with self._lock:
            self._val = None


def _normalize_tag(raw: str | None) -> str | None:
    """Map a submitted method tag to the canonical "white" / "black" (or None).

    Accepts white / whitebox / white-box / wb and black / blackbox / black-box / bb
    (case-insensitive); anything else is ignored (treated as untagged)."""
    if not raw:
        return None
    t = raw.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if t in ("white", "whitebox", "wb"):
        return "white"
    if t in ("black", "blackbox", "bb"):
        return "black"
    return None


def _redact_dataset_names(text: str, config: RunnerConfig) -> str:
    """Replace any real dataset / labels-repo id with its public codename.

    The private split names must never reach the participant; the run path already
    reports failures by codename, but a data/labels load that fails *before* the run
    (e.g. a dataset not yet on the Hub/NDIF) surfaces HF's own message — which names
    the real dataset — through the 400. Scrub it here too, so the up-front rejection
    keeps the same guarantee as the run path."""
    for d in config.datasets:
        label = dataset_label(d.key)
        if d.name:
            text = text.replace(d.name, label)
        if getattr(d, "labels_uri", None):
            text = text.replace(d.labels_uri, label)
    return text


def _team_submissions(store: BaseResultStore, team: str, limit: int = 100) -> list[dict]:
    """A team's submission history (newest first), capped. One entry per submitted
    notebook: the mean of each metric across datasets, the per-dataset breakdown,
    and total runtime; a notebook with no successful dataset is ``ok=False``."""
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)   # (notebook, stamp) -> records
    for r in store.all():
        if r.team == team:
            groups[(r.notebook, r.submitted_at)].append(r)

    entries = []
    for (notebook, stamp), recs in groups.items():
        summ = summarize_submission(recs)
        entries.append({"notebook": notebook.split("/")[-1], "submitted_at": stamp,
                        "ok": summ["ok"], "metrics": summ["metrics"],
                        "datasets": summ["datasets"],
                        "runtime_seconds": summ["runtime_seconds"],
                        "tag": summ.get("tag"),
                        "failed_dataset": summ["failed_dataset"], "error": summ["error"]})
    entries.sort(key=lambda e: e["submitted_at"] or "", reverse=True)
    return entries[:limit]


def _anonymize(entry: dict, label) -> dict:
    """Return a copy of a reported entry (a leaderboard row, /api/me submission, or
    /submit result) with every real dataset name replaced by its public label
    ("Dataset 1", ...). The real names are private — they must never leave the
    server. Copies rather than mutating, so the cached leaderboard rows stay intact.
    ``label`` maps a dataset key to its label (and passes ``None`` through)."""
    out = dict(entry)
    if out.get("datasets"):
        out["datasets"] = [{**d, "dataset": label(d.get("dataset"))}
                           for d in out["datasets"]]
    if "failed_dataset" in out:
        out["failed_dataset"] = label(out["failed_dataset"])
    if "dataset" in out:                 # /submit failure rows carry a bare key
        out["dataset"] = label(out["dataset"])
    return out


def create_app(config: RunnerConfig, store: BaseResultStore,
               registry: TeamRegistry,
               limiter: RateLimiter | None = None,
               archive: SubmissionArchive | None = None) -> FastAPI:
    app = FastAPI(title="Aletheia's Quest — Leaderboard")
    # No limiter passed (e.g. most tests) -> unlimited.
    limiter = limiter or RateLimiter("rate_limits.json", 0, 0)

    # Single worker (see Dockerfile), so in-process primitives suffice:
    #   - submit_slots bounds concurrent heavy runs (the submission queue);
    #   - bucket_lock serializes read-modify-write of the registry + results
    #     objects so concurrent submissions can't clobber each other.
    submit_slots = asyncio.Semaphore(MAX_CONCURRENT_SUBMISSIONS)
    bucket_lock = asyncio.Lock()
    lb_cache = _LeaderboardCache(store, LEADERBOARD_CACHE_TTL)
    # In-flight submissions per team (queued or running). Mutated only on the event
    # loop, so a plain dict is safe; read by /api/me.
    pending: dict[str, int] = {}

    # Public dataset codenames ("Dataset <Greek deity>"). Real names are private and
    # must never appear in a response; ``label`` maps any key to its stable codename
    # (so even stale keys not in the current config get a real name, not a generic
    # placeholder). None passes through.
    def label(key: str | None) -> str | None:
        return dataset_label(key) if key else key

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    @app.get("/api/leaderboard")
    def api_leaderboard() -> dict:
        return {"primary": PRIMARY_METRIC, "secondary": SECONDARY_METRIC,
                "results": [_anonymize(r, label) for r in lb_cache.get()]}

    @app.post("/api/me")
    async def me(ndif_api_key: str | None = Header(default=None,
                                                   alias="X-NDIF-API-Key")) -> dict:
        """A team's own standing: name, rate-limit status, in-flight + past
        submissions. Keyed by the NDIF key (hashed for lookup, never stored). All
        reads, so no bucket lock — the bucket objects are replaced atomically."""
        if not ndif_api_key:
            raise HTTPException(400, "provide your NDIF API key (X-NDIF-API-Key)")
        team = await run_in_threadpool(registry.lookup, ndif_api_key)
        if team is None:
            return {"registered": False, "team": None,
                    "rate_limit": limiter.status(""), "submissions": [], "pending": 0}
        rate_limit, submissions = await asyncio.gather(
            run_in_threadpool(limiter.status, team),
            run_in_threadpool(_team_submissions, store, team))
        return {"registered": True, "team": team, "rate_limit": rate_limit,
                "submissions": [_anonymize(s, label) for s in submissions],
                "pending": pending.get(team, 0)}

    @app.get("/api/health")
    def health() -> dict:
        # Public codenames only — never the real dataset names.
        return {"ok": True, "datasets": list(config.dataset_label_map().values())}

    @app.get("/api/sandbox-probe")
    def sandbox_probe() -> dict:
        """Report which unprivileged sandboxing primitives work in this env."""
        proc = subprocess.run(
            [sys.executable, "-m", "aletheia_runner.sandbox.probe"],
            capture_output=True, text=True, timeout=30)
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise HTTPException(500, f"probe failed: {proc.stderr[:1000]}")

    @app.get("/api/landlock-selftest")
    def landlock_selftest() -> dict:
        """Apply Landlock in a child and report what it allows/denies here."""
        proc = subprocess.run(
            [sys.executable, "-m", "aletheia_runner.sandbox.landlock"],
            capture_output=True, text=True, timeout=30)
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            raise HTTPException(500, f"selftest failed: {proc.stderr[:1000]}")

    @app.post("/submit")
    async def submit(team: str = Form(default=""), file: UploadFile = File(...),
                     ndif_api_key: str | None = Header(default=None,
                                                       alias="X-NDIF-API-Key"),
                     hf_token: str | None = Header(default=None,
                                                   alias="X-HF-Token"),
                     row_limit: str | None = Header(default=None,
                                                    alias="X-Aletheia-Limit"),
                     method_tag: str | None = Header(default=None,
                                                     alias="X-Aletheia-Tag")) -> dict:
        if not config.datasets:
            raise HTTPException(503, "runner has no datasets configured")
        if not ndif_api_key:
            raise HTTPException(400, "an NDIF API key is required (X-NDIF-API-Key)")

        # Validate the key against NDIF and decide which key drives the run, before
        # binding a team or charging a rate-limit attempt. whoami returns the
        # account (or null email for a key NDIF doesn't recognise), or None when
        # NDIF is unreachable. Only keys with the usable tier can actually run
        # traces, so:
        #   - recognised + tier_1   -> the submitter's OWN key
        #   - recognised, no tier_1 -> the shared leaderboard key (HF secret)
        #   - NOT recognised        -> reject immediately (definitive: null email)
        #   - unknown (NDIF down)   -> shared leaderboard key, so a blip doesn't
        #                              block valid users (fail-open on transient)
        info = await run_in_threadpool(ndif.whoami, ndif_api_key, config.ndif_host)
        if info is not None and not ndif.is_recognized(info):
            raise HTTPException(400, "NDIF does not recognize this API key")
        if ndif.has_usable_tier(info):
            run_ndif_key = ndif_api_key
        elif config.leaderboard_ndif_api_key:
            run_ndif_key = config.leaderboard_ndif_api_key
        else:
            # No shared key configured: keep the submitter's key (their run will
            # fail at the first trace if they lack the tier, but we don't block).
            run_ndif_key = ndif_api_key

        limit = MAX_UPLOAD_MB * 1_000_000
        # Reject by declared size before doing anything else (and before pulling the
        # body into memory); re-check the actual bytes after (size may be unset).
        if file.size is not None and file.size > limit:
            raise HTTPException(413, f"submission exceeds {MAX_UPLOAD_MB} MB")

        # The NDIF key identifies the team: bound to a name on first submission,
        # remembered after (the key alone suffices on later submissions). This is a
        # read-modify-write of the registry object, so serialize it under the bucket
        # lock (and run it off the event loop).
        async with bucket_lock:
            team, err = await run_in_threadpool(registry.resolve, ndif_api_key, team)
        if err:
            raise HTTPException(400, err)

        data = await file.read()
        if len(data) > limit:
            raise HTTPException(413, f"submission exceeds {MAX_UPLOAD_MB} MB")

        when = datetime.datetime.now(datetime.timezone.utc)

        with tempfile.TemporaryDirectory(prefix="aletheia-upload-") as tmp:
            zpath = Path(tmp) / "submission.zip"
            zpath.write_bytes(data)
            root = Path(tmp) / "unpacked"

            # Validate the submission's STRUCTURE before charging a rate-limit
            # attempt: it must unpack and contain exactly one notebook. A
            # structurally-invalid submission is rejected here for free — only a
            # real, runnable submission below costs an attempt. (run_pipeline
            # re-checks, so --dry enforces the same one-notebook rule.)
            try:
                await run_in_threadpool(pipeline.unpack, zpath, root)
                await run_in_threadpool(pipeline.validate_submission, root)
            except (FileNotFoundError, ValueError) as e:
                raise HTTPException(400, f"invalid submission: {e}")

            # Consume a submission slot (per-team fixed-window limit) RIGHT BEFORE we
            # run it — so a rejected submission never costs an attempt. Bucket
            # read-modify-write, so under the lock.
            async with bucket_lock:
                allowed, retry = await run_in_threadpool(limiter.check_and_consume, team)
            if not allowed:
                raise HTTPException(
                    429,
                    f"rate limit reached: {limiter.max} submission(s) per "
                    f"{limiter.window / 3600:g}h. Try again in ~{retry}s.",
                    headers={"Retry-After": str(retry)})

            # Count this submission as in-flight (queued or running) for /api/me until
            # it finishes, pass or fail.
            pending[team] = pending.get(team, 0) + 1
            try:
                # Archive the raw zip (every submission we run, pass or fail) so it
                # can be retrieved later. Unique path -> no lock; never sink the run.
                if archive is not None:
                    try:
                        await run_in_threadpool(archive.save, team, data, when)
                    except Exception as e:  # noqa: BLE001
                        print(f"[archive] failed to store submission for {team!r}: {e}",
                              file=sys.stderr, flush=True)

                # Sink that stores each produced submission.csv next to the zip (same
                # timestamp). Runs inside the worker thread; never sinks the run.
                csv_sink = None
                if archive is not None:
                    def csv_sink(notebook: str, dataset_key: str, csv_bytes: bytes,
                                 _team=team, _when=when) -> None:
                        archive.save_csv(_team, _when, notebook, dataset_key, csv_bytes)

                # The submitter's keys are injected into their sandboxed run:
                # NDIF_API_KEY for nnsight remote traces (run_ndif_key was chosen by
                # tier up front: their own key if tier_1, else the shared leaderboard
                # key), HF_TOKEN for loading gated HF models they have access to. (The
                # token can't reach the private eval/labels — that's the organizers'
                # org, not the participant's.)
                extra_env = {"NDIF_API_KEY": run_ndif_key}
                if hf_token:
                    extra_env["HF_TOKEN"] = hf_token
                # Optional row cap, forwarded as ALETHEIA_LIMIT for the notebook to
                # honor (e.g. score only the first N rows). Ignore if not a positive int.
                if row_limit is not None:
                    try:
                        if int(row_limit) > 0:
                            extra_env["ALETHEIA_LIMIT"] = str(int(row_limit))
                    except ValueError:
                        pass

                # The run is heavy (venv, pip, notebook execution, NDIF traces). Run it
                # off the event loop in a worker thread, bounded by the semaphore: a
                # burst of submissions queues here (the loop stays free to serve the
                # leaderboard/health endpoints) and at most MAX_CONCURRENT_SUBMISSIONS
                # run at once. The submission was already unpacked + validated above.
                async with submit_slots:
                    try:
                        records = await run_in_threadpool(
                            pipeline.run_pipeline, root, team, config,
                            extra_env, csv_sink)
                    except (FileNotFoundError, ValueError) as e:
                        raise HTTPException(
                            400, _redact_dataset_names(f"invalid submission: {e}", config))

                stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                tag = _normalize_tag(method_tag)
                for r in records:
                    r.submitted_at = stamp
                    r.tag = tag
                async with bucket_lock:
                    await run_in_threadpool(store.append, records)
                lb_cache.invalidate()

                # Report per notebook: all four metrics (mean across datasets) + the
                # per-dataset breakdown + runtime. ``scores`` keeps the primary metric
                # keyed by notebook for older clients.
                by_nb: dict[str, list] = {}
                for r in records:
                    by_nb.setdefault(r.notebook, []).append(r)
                results_out, scores, failures = [], {}, []
                for nb, recs in by_nb.items():
                    name = nb.split("/")[-1]
                    summ = summarize_submission(recs)
                    if summ["ok"]:
                        scores[name] = summ["metrics"].get(PRIMARY_METRIC)
                        results_out.append(_anonymize({
                            "notebook": name, "ok": True, "metrics": summ["metrics"],
                            "datasets": summ["datasets"],
                            "runtime_seconds": summ["runtime_seconds"]}, label))
                    else:
                        # Anonymize the failing dataset for the participant-facing
                        # response; the real key stays in the persisted records.
                        fail = _anonymize({"notebook": name,
                                           "dataset": summ["failed_dataset"],
                                           "error": summ["error"]}, label)
                        failures.append(fail)
                        results_out.append({**fail, "ok": False})
                return {
                    "team": team,
                    "primary": PRIMARY_METRIC,
                    "results": results_out,
                    "scores": scores,
                    "failures": failures,
                    "message": (f"scored {len(scores)} notebook(s)"
                                + (f", {len(failures)} failed" if failures else "")),
                }
            finally:
                pending[team] = pending.get(team, 1) - 1
                if pending[team] <= 0:
                    pending.pop(team, None)

    return app


def _app_from_env() -> FastAPI:
    cfg_path = os.environ.get("RUNNER_CONFIG", "runner.yaml")
    if Path(cfg_path).exists():
        config = RunnerConfig.from_yaml(cfg_path)
    else:  # allow the app to import/boot even before a config is mounted
        config = RunnerConfig(datasets=[]).with_env_overrides()
    results_uri = os.environ.get("RESULTS_URI", config.results_uri)
    store = make_store(results_uri, token=config.hf_token)
    registry = TeamRegistry(config.teams_uri, token=config.hf_token)
    limiter = RateLimiter(config.rate_limits_uri, config.rate_limit_max,
                          config.rate_limit_window_hours * 3600,
                          token=config.hf_token)
    archive = SubmissionArchive(config.submissions_uri, token=config.hf_token)
    return create_app(config, store, registry, limiter, archive)


app = _app_from_env()
