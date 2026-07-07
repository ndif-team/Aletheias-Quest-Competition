"""Run a submission on AWS Fargate instead of in-process.

A drop-in for ``pipeline.run_pipeline`` from the Space's point of view: the Space
uploads the submission zip to S3, launches ONE Fargate task (the baked runner),
tails the ``progress.jsonl`` the task writes (forwarding each event to the same
``on_progress`` callback the in-process path uses), then reads ``result.jsonl``
and returns the ``ResultRecord`` list. State serialization on the Space is
unchanged — this only relocates execution.

``FargateBackend.run`` is blocking (S3/ECS polling); the Space calls it in a
threadpool worker exactly like it called ``run_pipeline``. ``FargateCanceller``
mirrors ``sandbox.Canceller`` so ``/admin/cancel`` works unchanged: before the
task launches a cancel latches; after, it ``StopTask``s.

Config comes from the environment (Space variables/secrets); ``from_env`` returns
``None`` when Fargate isn't configured, so the Space falls back to in-process.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import RunnerConfig
from .results import ResultRecord, _parse_jsonl

# How often to poll ECS + S3 while a run is in flight, and a backstop so a wedged
# task can never make the Space poll forever (the task's own submission_timeout
# should fire first; this is belt-and-suspenders).
_POLL_SECONDS = 5.0
_MAX_WALL_SECONDS = 12 * 3600


@dataclass
class FargateConfig:
    region: str
    cluster: str
    task_family: str
    subnets: list[str]
    security_group: str
    run_bucket: str

    @classmethod
    def from_env(cls, env=None) -> "FargateConfig | None":
        import os
        env = env if env is not None else os.environ
        cluster = env.get("FARGATE_CLUSTER")
        family = env.get("FARGATE_TASK_FAMILY")
        subnets = [s for s in env.get("FARGATE_SUBNETS", "").split(",") if s]
        sg = env.get("FARGATE_SECURITY_GROUP")
        bucket = env.get("FARGATE_RUN_BUCKET")
        if not (cluster and family and subnets and sg and bucket):
            return None
        return cls(region=env.get("AWS_REGION", "us-east-1"), cluster=cluster,
                   task_family=family, subnets=subnets, security_group=sg,
                   run_bucket=bucket)


class FargateCanceller:
    """Cancel handle for a Fargate run. Same interface as ``sandbox.Canceller``
    (``cancel()`` / ``cancelled``) so the admin endpoints treat both alike."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._ecs = None
        self._cluster = None
        self._task_arn: str | None = None

    def bind(self, ecs, cluster: str, task_arn: str) -> bool:
        """Attach the live task. Returns False if a cancel already arrived (the
        caller should stop the just-launched task immediately)."""
        with self._lock:
            self._ecs, self._cluster, self._task_arn = ecs, cluster, task_arn
            return not self._cancelled

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            ecs, cluster, arn = self._ecs, self._cluster, self._task_arn
        if ecs is not None and arn is not None:
            try:
                ecs.stop_task(cluster=cluster, task=arn, reason="cancelled by operator")
            except Exception:  # noqa: BLE001 — best-effort; the run reports failure anyway
                pass

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


class FargateBackend:
    def __init__(self, cfg: FargateConfig):
        import boto3
        self.cfg = cfg
        self.ecs = boto3.client("ecs", region_name=cfg.region)
        self.s3 = boto3.client("s3", region_name=cfg.region)

    def _key(self, run_key: str, name: str) -> str:
        return f"runs/{run_key}/{name}"

    def _get(self, run_key: str, name: str) -> str | None:
        try:
            obj = self.s3.get_object(Bucket=self.cfg.run_bucket,
                                     Key=self._key(run_key, name))
            return obj["Body"].read().decode()
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:  # noqa: BLE001 — transient S3 error; treat as "not ready yet"
            return None

    def run(self, config: RunnerConfig, run_key: str, zip_bytes: bytes, team: str,
            extra_env: dict[str, str] | None, on_progress=None,
            cancel: "FargateCanceller | None" = None) -> list[ResultRecord]:
        """Launch the task, tail progress, return records. Blocking (threadpool)."""
        env = [{"name": "RUN_BUCKET", "value": self.cfg.run_bucket},
               {"name": "RUN_ID", "value": run_key},
               {"name": "TEAM", "value": team}]
        for k, v in (extra_env or {}).items():
            env.append({"name": k, "value": v})

        # Upload the submission, then launch one task.
        self.s3.put_object(Bucket=self.cfg.run_bucket,
                           Key=self._key(run_key, "input.zip"), Body=zip_bytes)
        resp = self.ecs.run_task(
            cluster=self.cfg.cluster, launchType="FARGATE",
            taskDefinition=self.cfg.task_family,
            networkConfiguration={"awsvpcConfiguration": {
                "subnets": self.cfg.subnets,
                "securityGroups": [self.cfg.security_group],
                "assignPublicIp": "ENABLED"}},
            overrides={"containerOverrides": [{"name": "runner", "environment": env}]})
        if not resp.get("tasks"):
            raise RuntimeError(f"could not launch run task: {resp.get('failures')}")
        task_arn = resp["tasks"][0]["taskArn"]

        # Wire cancellation; if a cancel already arrived, stop the task now.
        if cancel is not None and not cancel.bind(self.ecs, self.cfg.cluster, task_arn):
            try:
                self.ecs.stop_task(cluster=self.cfg.cluster, task=task_arn,
                                   reason="cancelled by operator")
            except Exception:  # noqa: BLE001
                pass

        # Poll ECS for completion while forwarding new progress events.
        seen = 0
        deadline = time.monotonic() + _MAX_WALL_SECONDS
        while True:
            time.sleep(_POLL_SECONDS)
            if on_progress is not None:
                text = self._get(run_key, "progress.jsonl") or ""
                lines = [ln for ln in text.splitlines() if ln.strip()]
                for ln in lines[seen:]:
                    try:
                        on_progress(json.loads(ln))
                    except (json.JSONDecodeError, Exception):  # noqa: BLE001
                        pass
                seen = len(lines)

            desc = self.ecs.describe_tasks(cluster=self.cfg.cluster, tasks=[task_arn])
            tasks = desc.get("tasks") or []
            status = tasks[0]["lastStatus"] if tasks else "STOPPED"
            if status == "STOPPED":
                break
            if time.monotonic() > deadline:
                try:
                    self.ecs.stop_task(cluster=self.cfg.cluster, task=task_arn,
                                       reason="space-side wall-clock backstop")
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError("run task exceeded the Space-side wall-clock backstop")

        # Task stopped: prefer the recorded verdict, else it's an infra failure.
        status_json = self._get(run_key, "status.json")
        phase = None
        if status_json:
            try:
                phase = json.loads(status_json).get("phase")
            except json.JSONDecodeError:
                pass
        result_text = self._get(run_key, "result.jsonl")

        if phase == "done":
            records = _parse_jsonl(result_text or "")
            if records:
                return records
            # done with no records = a validated-away submission; surface its reason.
            err = "invalid submission"
            try:
                err = json.loads(status_json).get("error") or err
            except Exception:  # noqa: BLE001
                pass
            raise ValueError(err)

        # No clean verdict: the task crashed / OOMed / was killed.
        detail = ""
        if status_json:
            try:
                detail = json.loads(status_json).get("error", "") or ""
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(f"the run task did not finish cleanly (phase={phase}). {detail}"[:500])

    def new_canceller(self) -> FargateCanceller:
        return FargateCanceller()
