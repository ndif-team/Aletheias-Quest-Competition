#!/usr/bin/env python3
"""Register a task def, run one Fargate task, wait for it, and dump its logs.

A thin reusable harness over ECS RunTask for smoke-testing the runner image
(the probe, the sandbox self-test, a fixture submission). Reads resource ids
from ``infra-outputs.json``.

    AWS_PROFILE=AdministratorAccess-797161732516 AWS_REGION=us-east-1 \
        leaderboard/fargate/.venv/bin/python leaderboard/fargate/run_task.py \
            --tag probe --command python /app/probe.py

The task runs in the existing public subnets with a public IP (to pull the image
from ECR and reach the internet); egress is otherwise restricted in-process by
the runner's own sandbox.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3

OUT = json.loads((Path(__file__).parent / "infra-outputs.json").read_text())


def register(ecs, *, family: str, image: str, cpu: str, memory: str,
             command: list[str] | None) -> str:
    container = {
        "name": "runner",
        "image": image,
        "essential": True,
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": OUT["log_group"],
                "awslogs-region": OUT["region"],
                "awslogs-stream-prefix": "task",
            },
        },
    }
    if command:
        container["command"] = command
    r = ecs.register_task_definition(
        family=family,
        requiresCompatibilities=["FARGATE"],
        networkMode="awsvpc",
        cpu=cpu, memory=memory,
        executionRoleArn=OUT["execution_role_arn"],
        taskRoleArn=OUT["task_role_arn"],
        containerDefinitions=[container],
    )
    arn = r["taskDefinition"]["taskDefinitionArn"]
    print(f"registered {arn}")
    return arn


def run_and_wait(ecs, logs, task_def: str, environment: list[dict] | None) -> int:
    overrides = {}
    if environment:
        overrides = {"containerOverrides": [{"name": "runner", "environment": environment}]}
    resp = ecs.run_task(
        cluster=OUT["cluster"], launchType="FARGATE", taskDefinition=task_def,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": OUT["subnets"],
            "securityGroups": [OUT["security_group"]],
            "assignPublicIp": "ENABLED"}},
        overrides=overrides)
    if not resp.get("tasks"):
        print(f"run_task failed: {resp.get('failures')}", file=sys.stderr)
        return 2
    task_arn = resp["tasks"][0]["taskArn"]
    task_id = task_arn.split("/")[-1]
    print(f"running task {task_id} … (pull + boot can take a minute)")

    ecs.get_waiter("tasks_stopped").wait(
        cluster=OUT["cluster"], tasks=[task_arn],
        WaiterConfig={"Delay": 6, "MaxAttempts": 200})

    d = ecs.describe_tasks(cluster=OUT["cluster"], tasks=[task_arn])["tasks"][0]
    cont = d["containers"][0]
    exit_code = cont.get("exitCode")
    print(f"\nstopped: {d.get('stoppedReason','')}  "
          f"container={cont.get('lastStatus')}  exitCode={exit_code}  "
          f"reason={cont.get('reason','')}")

    # Dump the CloudWatch log stream for this task.
    stream = f"task/runner/{task_id}"
    print(f"\n--- logs ({OUT['log_group']} :: {stream}) ---")
    try:
        token = None
        for _ in range(50):
            kw = {"logGroupName": OUT["log_group"], "logStreamName": stream,
                  "startFromHead": True}
            if token:
                kw["nextToken"] = token
            ev = logs.get_log_events(**kw)
            for e in ev["events"]:
                print(e["message"])
            if ev.get("nextForwardToken") == token:
                break
            token = ev.get("nextForwardToken")
    except logs.exceptions.ResourceNotFoundException:
        print("(no log stream — task may have failed before the container started)")
    print("--- end logs ---")
    return exit_code if exit_code is not None else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="image tag in the ECR repo (e.g. probe)")
    ap.add_argument("--family", help="task-def family suffix (default: aletheia-runner-<tag>)")
    ap.add_argument("--cpu", default="256")
    ap.add_argument("--memory", default="512")
    ap.add_argument("--command", nargs=argparse.REMAINDER,
                    help="container command override (everything after --command)")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE env var (repeatable)")
    args = ap.parse_args()

    session = boto3.Session(region_name=OUT["region"])
    ecs = session.client("ecs")
    logs = session.client("logs")

    image = f"{OUT['ecr_repo_uri']}:{args.tag}"
    family = args.family or f"aletheia-runner-{args.tag}"
    environment = [{"name": k, "value": v}
                   for k, v in (kv.split("=", 1) for kv in args.env)]

    task_def = register(ecs, family=family, image=image, cpu=args.cpu,
                        memory=args.memory, command=args.command)
    return run_and_wait(ecs, logs, task_def, environment)


if __name__ == "__main__":
    sys.exit(main())
