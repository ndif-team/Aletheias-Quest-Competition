"""Fargate task entrypoint: run ONE submission and report results to S3.

The trusted runner, relocated from the Space into a per-submission microVM. It
reads the submission zip from S3, runs the existing sandboxed pipeline against
the BAKED datasets (inputs arrow cache + labels are in the image; no HF org token
at runtime), and writes progress + results back to S3 for the Space to tail and
finalize.

S3 layout (bucket = $RUN_BUCKET, prefix = runs/$RUN_ID):
    input.zip       (in)  — the dispatcher uploaded the submission
    status.json     (out) — {phase: queued|running|done|error, ...}; the heartbeat
    progress.jsonl  (out) — one raw progress event per line (organizer-only keys)
    result.jsonl    (out) — the ResultRecords (same schema the bucket store persists)
    csv/<...>.csv   (out) — each produced submission.csv, archived

Env in: RUN_BUCKET, RUN_ID, TEAM, and the submitter's forwarded keys
(NDIF_API_KEY, optional HF_TOKEN, optional ALETHEIA_LIMIT). The runner config
(datasets, ndif_host, sandbox/egress knobs) is the baked $RUNNER_CONFIG.

Exit code: 0 whenever we produced a verdict (even a logical submission failure is
a valid, recorded outcome); non-zero only on an infrastructure fault (bad input,
config load failure) so the dispatcher can distinguish "scored" from "broke".
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import boto3

from aletheia_runner import pipeline
from aletheia_runner.config import RunnerConfig
from aletheia_runner.results import _to_jsonl

BUCKET = os.environ["RUN_BUCKET"]
RUN_ID = os.environ["RUN_ID"]
PREFIX = f"runs/{RUN_ID}"
TEAM = os.environ.get("TEAM", "")
s3 = boto3.client("s3")


def _put(key: str, body: bytes, content_type: str = "application/json") -> None:
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}", Body=body,
                  ContentType=content_type)


def _status(phase: str, **extra) -> None:
    _put("status.json", json.dumps({"phase": phase, "run_id": RUN_ID, **extra}).encode())


def main() -> int:
    _status("running")
    cfg_path = os.environ.get("RUNNER_CONFIG", "/baked/runner.yaml")
    config = RunnerConfig.from_yaml(cfg_path)
    if not config.datasets:
        _status("error", error="runner has no datasets configured")
        return 2

    extra_env: dict[str, str] = {}
    if os.environ.get("NDIF_API_KEY"):
        extra_env["NDIF_API_KEY"] = os.environ["NDIF_API_KEY"]
    if os.environ.get("HF_TOKEN"):
        extra_env["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("ALETHEIA_LIMIT"):
        extra_env["ALETHEIA_LIMIT"] = os.environ["ALETHEIA_LIMIT"]

    with tempfile.TemporaryDirectory(prefix="aletheia-task-") as tmp:
        tmp = Path(tmp)
        zpath = tmp / "input.zip"
        try:
            s3.download_file(BUCKET, f"{PREFIX}/input.zip", str(zpath))
        except Exception as e:  # noqa: BLE001
            _status("error", error=f"could not fetch submission: {e}")
            return 2

        root = tmp / "unpacked"
        try:
            pipeline.unpack(zpath, root)
            pipeline.validate_submission(root)
        except (FileNotFoundError, ValueError) as e:
            # A structurally-invalid submission is a real, recordable verdict, but
            # there are no per-dataset records to emit — report it as a failure.
            _put("result.jsonl", b"")
            _status("done", ok=False, records=0, error=f"invalid submission: {e}")
            return 0

        # Stream raw progress events to S3 (organizer-only bucket); the Space redacts
        # dataset keys to codenames before relaying them to the participant.
        events: list[dict] = []

        def on_progress(ev: dict) -> None:
            events.append(ev)
            _put("progress.jsonl", ("".join(json.dumps(e) + "\n" for e in events)).encode())

        def on_csv(notebook: str, dataset_key: str, csv_bytes: bytes) -> None:
            safe = f"{dataset_key}__{notebook}".replace("/", "__")
            _put(f"csv/{safe}.csv", csv_bytes, content_type="text/csv")

        try:
            records = pipeline.run_pipeline(
                root, TEAM, config, extra_env=extra_env,
                on_submission_csv=on_csv, on_progress=on_progress, cancel=None)
        except (FileNotFoundError, ValueError) as e:
            _put("result.jsonl", b"")
            _status("done", ok=False, records=0, error=f"invalid submission: {e}")
            return 0
        except Exception as e:  # noqa: BLE001 — infra fault, not a submission verdict
            _status("error", error=f"runner crashed: {e}\n{traceback.format_exc()[-2000:]}")
            return 2

        _put("result.jsonl", _to_jsonl(records).encode())
        ok = bool(records) and all(r.ok for r in records)
        _status("done", ok=ok, records=len(records))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — last-ditch: never leave the Space hanging
        try:
            _status("error", error=f"fatal: {e}\n{traceback.format_exc()[-2000:]}")
        except Exception:  # noqa: BLE001
            pass
        print(f"[runner_main] fatal: {e}", file=sys.stderr)
        sys.exit(3)
