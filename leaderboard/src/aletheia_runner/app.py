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

from . import pipeline
from .archive import SubmissionArchive
from .config import RunnerConfig
from .ratelimit import RateLimiter
from .registry import TeamRegistry
from .results import BaseResultStore, make_store

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


def _team_submissions(store: BaseResultStore, team: str, limit: int = 100) -> list[dict]:
    """A team's submission history (newest first), capped. One entry per submitted
    notebook, scored as the MEAN across datasets (dataset identities aren't shown);
    a notebook with no successful dataset is ``ok=False``."""
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)   # (notebook, stamp) -> records
    for r in store.all():
        if r.team == team:
            groups[(r.notebook, r.submitted_at)].append(r)

    entries = []
    for (notebook, stamp), recs in groups.items():
        oks = [x.score for x in recs if x.ok and x.score is not None]
        entries.append({"notebook": notebook.split("/")[-1], "ok": bool(oks),
                        "score": (sum(oks) / len(oks)) if oks else None,
                        "submitted_at": stamp})
    entries.sort(key=lambda e: e["submitted_at"] or "", reverse=True)
    return entries[:limit]


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

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text()

    @app.get("/api/leaderboard")
    def api_leaderboard() -> dict:
        return {"metric": config.metric, "results": lb_cache.get()}

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
                "submissions": submissions, "pending": pending.get(team, 0)}

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "datasets": [d.key for d in config.datasets]}

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
                                                   alias="X-HF-Token")) -> dict:
        if not config.datasets:
            raise HTTPException(503, "runner has no datasets configured")
        if not ndif_api_key:
            raise HTTPException(400, "an NDIF API key is required (X-NDIF-API-Key)")

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

        # Consume a submission slot (per-team fixed-window limit). Same bucket
        # read-modify-write, so under the lock too; reject early before the heavy run.
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
            data = await file.read()
            if len(data) > limit:
                raise HTTPException(413, f"submission exceeds {MAX_UPLOAD_MB} MB")

            # Archive the raw zip (every submission, pass or fail) so it can be
            # retrieved later. Unique path -> no lock; never let it sink the run.
            if archive is not None:
                try:
                    when = datetime.datetime.now(datetime.timezone.utc)
                    await run_in_threadpool(archive.save, team, data, when)
                except Exception as e:  # noqa: BLE001
                    print(f"[archive] failed to store submission for {team!r}: {e}",
                          file=sys.stderr, flush=True)

            # The submitter's keys are injected into their sandboxed run: NDIF_API_KEY
            # for nnsight remote traces (required, so always present here), HF_TOKEN
            # for loading gated HF models they have access to. (The token can't reach
            # the private eval/labels — that's the organizers' org, not the participant's.)
            extra_env = {"NDIF_API_KEY": ndif_api_key}
            if hf_token:
                extra_env["HF_TOKEN"] = hf_token

            with tempfile.TemporaryDirectory(prefix="aletheia-upload-") as tmp:
                zpath = Path(tmp) / "submission.zip"
                zpath.write_bytes(data)
                # run_zip is heavy (venv, pip, notebook execution, NDIF traces). Run
                # it off the event loop in a worker thread, bounded by the semaphore:
                # a burst of submissions queues here (the loop stays free to serve the
                # leaderboard/health endpoints) and at most MAX_CONCURRENT_SUBMISSIONS
                # run at once.
                async with submit_slots:
                    try:
                        records = await run_in_threadpool(
                            pipeline.run_zip, zpath, team, config, extra_env=extra_env)
                    except (FileNotFoundError, ValueError) as e:
                        raise HTTPException(400, f"invalid submission: {e}")

            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for r in records:
                r.submitted_at = stamp
            async with bucket_lock:
                await run_in_threadpool(store.append, records)
            lb_cache.invalidate()

            # Report one score per notebook = mean across datasets (dataset names
            # are never returned); a notebook with no successful dataset is a failure.
            by_nb: dict[str, list] = {}
            for r in records:
                by_nb.setdefault(r.notebook, []).append(r)
            scores, failures = {}, []
            for nb, recs in by_nb.items():
                oks = [x.score for x in recs if x.ok and x.score is not None]
                if oks:
                    scores[nb.split("/")[-1]] = sum(oks) / len(oks)
                else:
                    failures.append({"notebook": nb.split("/")[-1], "error": recs[0].error})
            return {
                "team": team,
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
