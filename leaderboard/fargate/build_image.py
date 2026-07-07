#!/usr/bin/env python3
"""Bake the eval data into the runner image, push it, register the task def.

The "easy button": when datasets change, re-run this. It (1) builds each dataset's
inputs Arrow cache + LoRA configs via the real ``prepare_inputs``, (2) normalizes
each dataset's private labels to a local ``index,deceptive`` CSV, (3) writes a
baked ``runner.yaml`` pointing at them, (4) ``docker build`` + push to ECR, and
(5) registers a task-def revision.

    # fast fixture image (no torch/nnsight) for the smoke test:
    AWS_PROFILE=AdministratorAccess-797161732516 AWS_REGION=us-east-1 \
        leaderboard/fargate/.venv/bin/python leaderboard/fargate/build_image.py --fixture

    # full production image from the real runner.yaml:
    ... build_image.py --config leaderboard/runner.yaml --heavy --tag prod

Needs an HF org token (read from ~/.cache/huggingface/token) to fetch the private
inputs + labels at BUILD time; the token never enters the image or AWS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
LEADERBOARD = HERE.parent
BUILD_CTX = HERE / "build-context"          # COPYed into the image as /baked
OUT = json.loads((HERE / "infra-outputs.json").read_text())

FIXTURE = [{"name": "NDIF/aletheia-fake-eval",
            "labels_uri": "NDIF/aletheia-fake-eval-labels",
            "id_column": "id", "label_column": "deceptive"}]

# Fields carried from a source runner.yaml into the baked, execution-only config.
# Bucket/rate-limit/store settings are Space concerns the task never touches, so
# they're dropped; datasets, cache_dir, and sandbox are set by the bake itself.
_CARRY = ("notebook_timeout", "submission_timeout", "mem_mb", "ndif_host",
          "egress_allowlist", "egress_get_only_suffixes", "install_allowlist",
          "enforce_egress", "redact_errors", "score_partial", "confine")


def _hf_token() -> str:
    p = Path.home() / ".cache/huggingface/token"
    if not p.exists():
        sys.exit("no HF token at ~/.cache/huggingface/token (needed to fetch private data)")
    return p.read_text().strip()


def bake(datasets: list[dict], base_cfg: dict) -> None:
    """Populate build-context/: inputs cache, normalized labels CSVs, runner.yaml."""
    sys.path.insert(0, str(LEADERBOARD / "src"))
    from aletheia_runner.config import SPLIT, DatasetConfig, RunnerConfig
    from aletheia_runner.data import prepare_inputs
    from datasets import load_dataset
    import pandas as pd

    token = _hf_token()
    if BUILD_CTX.exists():
        shutil.rmtree(BUILD_CTX)
    cache = BUILD_CTX / "cache"
    labels_dir = BUILD_CTX / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inputs arrow cache + LoRA configs (the trusted parent, online, with token).
    src_dsets = [DatasetConfig(**d) for d in datasets]
    prep_cfg = RunnerConfig(datasets=src_dsets, cache_dir=str(cache), hf_token=token)
    print(f"baking inputs cache for {len(src_dsets)} dataset(s) → {cache}")
    prepare_inputs(prep_cfg)

    # 2. Normalize each dataset's labels to a local index,deceptive CSV.
    baked = []
    for d in src_dsets:
        raw = load_dataset(d.labels_uri, split=SPLIT, token=token).to_pandas()
        idx = raw[d.id_column].tolist() if d.id_column in raw.columns else list(range(len(raw)))
        out_csv = labels_dir / (d.name.replace("/", "__") + "-labels.csv")
        pd.DataFrame({"index": idx,
                      "deceptive": raw[d.label_column].astype(int)}).to_csv(out_csv, index=False)
        print(f"  labels {d.name} → {out_csv.name} ({len(idx)} rows)")
        baked.append({"name": d.name, "labels_uri": f"/baked/labels/{out_csv.name}",
                      "id_column": "index", "label_column": "deceptive"})

    # 3. Baked runner.yaml: datasets + cache + sandbox set here, everything else
    # (egress allowlist, timeouts, mem_mb, ndif_host, redaction) carried from source.
    cfg = {"datasets": baked, "sandbox": True, "cache_dir": "/baked/cache"}
    cfg.update({k: base_cfg[k] for k in _CARRY if k in base_cfg})
    (BUILD_CTX / "runner.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {BUILD_CTX/'runner.yaml'} (carried: "
          f"{[k for k in _CARRY if k in base_cfg]})")


def docker_build_push(tag: str, heavy: bool) -> str:
    image = f"{OUT['ecr_repo_uri']}:{tag}"
    print(f"\ndocker build → {image} (heavy={heavy})")
    subprocess.run(
        ["docker", "build", "-f", str(HERE / "Dockerfile.runner"),
         "--build-arg", f"INSTALL_HEAVY={'1' if heavy else '0'}",
         "-t", image, str(LEADERBOARD)], check=True)
    subprocess.run(["docker", "push", image], check=True)
    return image


def register(image: str, tag: str, cpu: str, memory: str,
             family: str | None = None) -> str:
    import boto3
    ecs = boto3.client("ecs", region_name=OUT["region"])
    r = ecs.register_task_definition(
        family=family or OUT["task_family"],
        requiresCompatibilities=["FARGATE"], networkMode="awsvpc",
        cpu=cpu, memory=memory,
        executionRoleArn=OUT["execution_role_arn"],
        taskRoleArn=OUT["task_role_arn"],
        containerDefinitions=[{
            "name": "runner", "image": image, "essential": True,
            "logConfiguration": {"logDriver": "awslogs", "options": {
                "awslogs-group": OUT["log_group"], "awslogs-region": OUT["region"],
                "awslogs-stream-prefix": "task"}}}])
    arn = r["taskDefinition"]["taskDefinitionArn"]
    print(f"registered {arn}")
    return arn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="store_true",
                    help="bake the NDIF/aletheia-fake-eval fixture (smoke test)")
    ap.add_argument("--config", help="bake datasets from this runner.yaml")
    ap.add_argument("--heavy", action="store_true", help="also install torch+nnsight")
    ap.add_argument("--tag", default=None, help="ECR image tag (default: fixture|prod)")
    ap.add_argument("--cpu", default="1024")
    ap.add_argument("--memory", default="4096")
    ap.add_argument("--task-family", default=None,
                    help="register under this task-def family (default: aletheia-runner)")
    ap.add_argument("--no-register", action="store_true", help="build+push only")
    args = ap.parse_args()

    if args.fixture:
        datasets, base_cfg = FIXTURE, {"notebook_timeout": 300}
        tag = args.tag or "fixture"
    elif args.config:
        data = yaml.safe_load(Path(args.config).read_text())
        datasets, base_cfg = data["datasets"], data
        tag = args.tag or "prod"
    else:
        sys.exit("pass --fixture or --config <runner.yaml>")

    bake(datasets, base_cfg)
    image = docker_build_push(tag, args.heavy)
    if not args.no_register:
        register(image, tag, args.cpu, args.memory, args.task_family)
    print(f"\ndone. image tag: {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
