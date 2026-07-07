#!/usr/bin/env python3
"""Drive one fixture submission through the Fargate runner, end to end.

Mimics what the Space dispatcher will do: zip the smoke submission, upload it to
S3, RunTask the registered ``aletheia-runner`` task def, wait, then read the
status/result/progress the task wrote back. Proves the whole path (S3 handoff,
sandbox, offline dataset load, scoring, result handoff) on real Fargate.

    AWS_PROFILE=AdministratorAccess-797161732516 AWS_REGION=us-east-1 \
        leaderboard/fargate/.venv/bin/python leaderboard/fargate/smoke_test.py
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import boto3

HERE = Path(__file__).parent
OUT = json.loads((HERE / "infra-outputs.json").read_text())
SUBMISSION = HERE / "smoke_submission"


def _zip_submission() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(SUBMISSION.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(SUBMISSION).as_posix())
    return buf.getvalue()


def _read(s3, bucket, key) -> str | None:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    except s3.exceptions.NoSuchKey:
        return None


def main() -> int:
    s3 = boto3.client("s3", region_name=OUT["region"])
    ecs = boto3.client("ecs", region_name=OUT["region"])
    logs = boto3.client("logs", region_name=OUT["region"])
    bucket = OUT["run_bucket"]
    run_id = f"smoke-{int(time.time())}"
    prefix = f"runs/{run_id}"

    print(f"run_id={run_id}")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/input.zip", Body=_zip_submission())
    print(f"uploaded submission → s3://{bucket}/{prefix}/input.zip")

    resp = ecs.run_task(
        cluster=OUT["cluster"], launchType="FARGATE",
        taskDefinition=OUT["task_family"],
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": OUT["subnets"], "securityGroups": [OUT["security_group"]],
            "assignPublicIp": "ENABLED"}},
        overrides={"containerOverrides": [{"name": "runner", "environment": [
            {"name": "RUN_BUCKET", "value": bucket},
            {"name": "RUN_ID", "value": run_id},
            {"name": "TEAM", "value": "smoke-team"}]}]})
    if not resp.get("tasks"):
        print("run_task failed:", resp.get("failures"))
        return 2
    task_arn = resp["tasks"][0]["taskArn"]
    task_id = task_arn.split("/")[-1]
    print(f"task {task_id} launched; waiting…")

    ecs.get_waiter("tasks_stopped").wait(
        cluster=OUT["cluster"], tasks=[task_arn],
        WaiterConfig={"Delay": 6, "MaxAttempts": 200})
    d = ecs.describe_tasks(cluster=OUT["cluster"], tasks=[task_arn])["tasks"][0]
    cont = d["containers"][0]
    print(f"\nstopped: {d.get('stoppedReason','')}  exitCode={cont.get('exitCode')}  "
          f"reason={cont.get('reason','')}")

    print("\n=== status.json ===");   print(_read(s3, bucket, f"{prefix}/status.json"))
    print("=== progress.jsonl ===");  print(_read(s3, bucket, f"{prefix}/progress.jsonl"))
    print("=== result.jsonl ===");    print(_read(s3, bucket, f"{prefix}/result.jsonl"))

    print("\n--- container logs (tail) ---")
    stream = f"task/runner/{task_id}"
    try:
        ev = logs.get_log_events(logGroupName=OUT["log_group"], logStreamName=stream,
                                 startFromHead=False, limit=40)
        for e in ev["events"]:
            print(e["message"])
    except logs.exceptions.ResourceNotFoundException:
        print("(no log stream)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
