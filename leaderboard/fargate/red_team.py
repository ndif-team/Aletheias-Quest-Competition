#!/usr/bin/env python3
"""Red-team the runner on REAL Fargate: run an adversarial notebook inside the
actual sandbox (Landlock + seccomp egress gate + allowlist proxy) and report what
it could and couldn't do. Verifies, on the deployed image:

  - AWS credential theft is blocked (ECS task-role endpoint 169.254.170.2 + IMDS)
  - non-allowlisted egress is denied; allowlisted egress works
  - dataset exfil via HF upload (POST) is denied (HF is GET-only via MITM)
  - the child can't read the parent's /proc environ (token theft)
  - the child can't read the baked private labels

The notebook writes its findings to submission.csv (so the task archives them to
S3 under csv/); we read them back. Runs against the aletheia-runner-dev image,
which carries the real egress allowlist.

    AWS_PROFILE=AdministratorAccess-797161732516 AWS_REGION=us-east-1 \
        leaderboard/fargate/.venv/bin/python leaderboard/fargate/red_team.py
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import boto3
import nbformat

OUT = json.loads((Path(__file__).parent / "infra-outputs.json").read_text())
TASK_FAMILY = "aletheia-runner-dev"     # dev image = real egress allowlist baked in

# The adversarial notebook body. Every probe is wrapped so the notebook still
# produces submission.csv (its exfil channel back to us).
REDTEAM_SRC = r'''
import os, socket, csv, urllib.request

findings = []
def rec(name, result):
    findings.append((name, result)); print(f"[REDTEAM] {name}: {result}")

def raw_connect(host, port, timeout=4):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return "REACHED  <-- LEAK"
    except Exception as e:
        return f"blocked ({type(e).__name__})"

def http(url, method="GET", data=None, timeout=8):
    import urllib.error
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method=method, data=data), timeout=timeout)
        return f"REACHED HTTP {r.status} ({method})"
    except urllib.error.HTTPError as e:
        # Reached a server and got a status. 405 from the MITM = method blocked at the
        # proxy (good); a forwarded 2xx/401 would mean the method reached the origin.
        return f"HTTP {e.code} ({method})"
    except Exception as e:
        return f"blocked ({type(e).__name__})"

# --- AWS credential theft (Fargate-specific) --------------------------------
rec("aws.ecs_task_creds_endpoint(169.254.170.2:80)", raw_connect("169.254.170.2", 80))
rec("aws.imds(169.254.169.254:80)", raw_connect("169.254.169.254", 80))
rel = os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
rec("aws.creds_env_var_exposed", ("PRESENT "+rel+"  <-- LEAK") if rel else "absent (good)")
rec("aws.access_key_in_env", "PRESENT  <-- LEAK" if os.environ.get("AWS_ACCESS_KEY_ID") else "absent (good)")
if rel:  # if the path leaked, try the actual theft
    rec("aws.fetch_task_creds", http("http://169.254.170.2" + rel))

# --- egress allow/deny ------------------------------------------------------
rec("egress.raw_connect_1.1.1.1:443", raw_connect("1.1.1.1", 443))
rec("egress.http_nonallowlisted(example.org)", http("https://example.org/"))
rec("egress.http_allowlisted(huggingface.co GET)", http("https://huggingface.co/api/whoami-v2"))

# --- dataset exfil via HF upload (POST) must be denied (GET-only MITM) -------
rec("exfil.hf_post(create repo)", http("https://huggingface.co/api/repos/create", method="POST", data=b"{}"))

# --- secret / label reads ---------------------------------------------------
try:
    open(f"/proc/{os.getppid()}/environ", "rb").read(64)
    rec("read.parent_proc_environ", "READ  <-- LEAK")
except Exception as e:
    rec("read.parent_proc_environ", f"denied ({type(e).__name__})")
try:
    rec("read.list_/baked", str(os.listdir("/baked"))[:80])
except Exception as e:
    rec("read.list_/baked", f"denied ({type(e).__name__})")
import glob
try:
    hits = glob.glob("/baked/labels/*.csv")
    if hits:
        open(hits[0]).read(64); rec("read.baked_labels", "READ  <-- LEAK")
    else:
        rec("read.baked_labels", "no path visible (good)")
except Exception as e:
    rec("read.baked_labels", f"denied ({type(e).__name__})")

# Forwarded submitter keys ARE expected in the child env (by design) — note them.
rec("note.ndif_key_forwarded", "present (expected)" if os.environ.get("NDIF_API_KEY") else "absent")
rec("note.hf_token_forwarded", "present (expected)" if os.environ.get("HF_TOKEN") else "absent")

with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["index", "vector", "result"])
    for i, (n, r) in enumerate(findings):
        w.writerow([i, n, r])
print("[REDTEAM] wrote", len(findings), "findings")
'''


def build_zip() -> bytes:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(REDTEAM_SRC)]
    nb.metadata["kernelspec"] = {"display_name": "python3", "language": "python", "name": "python3"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission/redteam.ipynb", nbformat.writes(nb))
    return buf.getvalue()


def main() -> int:
    s3 = boto3.client("s3", region_name=OUT["region"])
    ecs = boto3.client("ecs", region_name=OUT["region"])
    bucket = OUT["run_bucket"]
    run_key = f"redteam-{int(time.time())}"
    prefix = f"runs/{run_key}"
    s3.put_object(Bucket=bucket, Key=f"{prefix}/input.zip", Body=build_zip())
    print(f"run_key={run_key}; launching red-team task on {TASK_FAMILY} …")

    resp = ecs.run_task(
        cluster=OUT["cluster"], launchType="FARGATE", taskDefinition=TASK_FAMILY,
        networkConfiguration={"awsvpcConfiguration": {
            "subnets": OUT["subnets"], "securityGroups": [OUT["security_group"]],
            "assignPublicIp": "ENABLED"}},
        overrides={"containerOverrides": [{"name": "runner", "environment": [
            {"name": "RUN_BUCKET", "value": bucket},
            {"name": "RUN_ID", "value": run_key},
            {"name": "TEAM", "value": "red-team"},
            # forward a dummy NDIF/HF key so the "forwarded key" checks are realistic
            {"name": "NDIF_API_KEY", "value": "dummy-ndif"},
            {"name": "HF_TOKEN", "value": "dummy-hf"}]}]})
    if not resp.get("tasks"):
        print("run_task failed:", resp.get("failures")); return 2
    arn = resp["tasks"][0]["taskArn"]
    ecs.get_waiter("tasks_stopped").wait(cluster=OUT["cluster"], tasks=[arn],
                                         WaiterConfig={"Delay": 6, "MaxAttempts": 200})

    def get(name):
        try:
            return s3.get_object(Bucket=bucket, Key=f"{prefix}/{name}")["Body"].read().decode()
        except s3.exceptions.NoSuchKey:
            return None
    print("status:", get("status.json"))
    # The findings CSV is archived under csv/<dataset>__submission__redteam.ipynb.csv
    keys = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/csv/").get("Contents", [])
    if not keys:
        print("!! no findings CSV archived — check logs"); return 1
    csv_text = s3.get_object(Bucket=bucket, Key=keys[0]["Key"])["Body"].read().decode()
    print("\n=== RED-TEAM FINDINGS ===")
    import csv as _csv
    for row in _csv.reader(csv_text.splitlines()[1:]):
        if len(row) >= 3:
            flag = "  ❌LEAK" if "LEAK" in row[2] or "REACHED" in row[2] else ""
            print(f"  {row[1]:<48} {row[2]}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
