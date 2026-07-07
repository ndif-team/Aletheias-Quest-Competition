#!/usr/bin/env python3
"""Idempotent AWS setup for the Fargate submission runner.

Creates ONLY the new resources the migration needs; the VPC + subnets already
exist (CDK-managed ``NDIF-Hackathon-Network``) and are referenced read-only.

    python infra.py            # create/update everything (safe to re-run)
    python infra.py --destroy  # tear it all down

Run with the AdministratorAccess SSO profile:
    AWS_PROFILE=AdministratorAccess-797161732516 AWS_REGION=us-east-1 \
        leaderboard/fargate/.venv/bin/python leaderboard/fargate/infra.py

Writes ``fargate/infra-outputs.json`` (resource ids, safe to read) and, when a
dispatcher access key is minted, ``fargate/dispatcher-credentials.json`` (the
Space's AWS key — gitignored, set it as Space secrets and then delete it).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
VPC_ID = "vpc-0ab211670ffaa5d7b"
# Public subnets (route 0.0.0.0/0 -> IGW, mapPublicIp=true): tasks run here with
# assignPublicIp=ENABLED so they reach ECR/S3/NDIF/HF/PyPI with no NAT. Egress is
# restricted in-process by the runner's own sandbox, not at the network layer.
PUBLIC_SUBNETS = ["subnet-088d0bb1c08922ac1", "subnet-09c2ab55c3b8b2fff"]

NAME = "aletheia-runner"                 # ECR repo + ECS cluster + task-def family
LOG_GROUP = f"/ecs/{NAME}"
SG_NAME = f"{NAME}-task"
EXEC_ROLE = f"{NAME}-execution"          # pulls image, writes logs
TASK_ROLE = f"{NAME}-task"               # the running task's own perms (S3 run I/O)
DISPATCHER_USER = "aletheia-space-dispatcher"   # the Space's scoped AWS identity
RUN_ARTIFACT_TTL_DAYS = 30               # S3 lifecycle: expire run I/O after this

OUT_PATH = Path(__file__).parent / "infra-outputs.json"
CRED_PATH = Path(__file__).parent / "dispatcher-credentials.json"

ECS_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                   "Action": "sts:AssumeRole"}],
}


def _code(e: ClientError) -> str:
    return e.response.get("Error", {}).get("Code", "")


def account_id(session) -> str:
    return session.client("sts").get_caller_identity()["Account"]


# ---------------------------------------------------------------- create -----
def ensure_ecr(ecr) -> str:
    try:
        r = ecr.create_repository(
            repositoryName=NAME,
            imageScanningConfiguration={"scanOnPush": True},
            imageTagMutability="MUTABLE")
        uri = r["repository"]["repositoryUri"]
        print(f"  + ECR repo {NAME}")
    except ClientError as e:
        if _code(e) != "RepositoryAlreadyExistsException":
            raise
        uri = ecr.describe_repositories(repositoryNames=[NAME])[
            "repositories"][0]["repositoryUri"]
        print(f"  = ECR repo {NAME} (exists)")
    return uri


def ensure_cluster(ecs) -> str:
    ecs.create_cluster(clusterName=NAME)   # create_cluster is idempotent
    print(f"  = ECS cluster {NAME}")
    return f"arn:aws:ecs:{REGION}:{ACCOUNT}:cluster/{NAME}"


def ensure_log_group(logs) -> None:
    try:
        logs.create_log_group(logGroupName=LOG_GROUP)
        logs.put_retention_policy(logGroupName=LOG_GROUP, retentionInDays=30)
        print(f"  + log group {LOG_GROUP}")
    except ClientError as e:
        if _code(e) != "ResourceAlreadyExistsException":
            raise
        print(f"  = log group {LOG_GROUP} (exists)")


def ensure_bucket(s3) -> str:
    bucket = f"aletheia-leaderboard-runs-{ACCOUNT}"
    try:
        s3.create_bucket(Bucket=bucket)    # us-east-1: no LocationConstraint
        print(f"  + S3 bucket {bucket}")
    except ClientError as e:
        if _code(e) not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
        print(f"  = S3 bucket {bucket} (exists)")
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={"BlockPublicAcls": True,
                                        "IgnorePublicAcls": True,
                                        "BlockPublicPolicy": True,
                                        "RestrictPublicBuckets": True})
    # Run artifacts (input.zip, progress.jsonl, result.json) are ephemeral; expire
    # them so the bucket doesn't grow without bound.
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={"Rules": [{
            "ID": "expire-run-artifacts",
            "Filter": {"Prefix": "runs/"},
            "Status": "Enabled",
            "Expiration": {"Days": RUN_ARTIFACT_TTL_DAYS}}]})
    return bucket


def ensure_security_group(ec2) -> str:
    existing = ec2.describe_security_groups(Filters=[
        {"Name": "vpc-id", "Values": [VPC_ID]},
        {"Name": "group-name", "Values": [SG_NAME]}])["SecurityGroups"]
    if existing:
        print(f"  = security group {SG_NAME} (exists)")
        return existing[0]["GroupId"]
    sg = ec2.create_security_group(
        GroupName=SG_NAME, VpcId=VPC_ID,
        Description="Aletheia Fargate runner: no inbound, all egress (restricted in-process)")
    gid = sg["GroupId"]
    # Default egress (all outbound) is what we want; strip nothing. No ingress rules
    # are added, so inbound is denied.
    print(f"  + security group {SG_NAME} ({gid})")
    return gid


def _ensure_role(iam, name: str, trust: dict, description: str) -> str:
    try:
        r = iam.create_role(RoleName=name,
                            AssumeRolePolicyDocument=json.dumps(trust),
                            Description=description)
        print(f"  + role {name}")
        return r["Role"]["Arn"]
    except ClientError as e:
        if _code(e) != "EntityAlreadyExists":
            raise
        print(f"  = role {name} (exists)")
        return iam.get_role(RoleName=name)["Role"]["Arn"]


def ensure_execution_role(iam) -> str:
    arn = _ensure_role(iam, EXEC_ROLE, ECS_TRUST,
                       "ECS agent: pull image from ECR, write logs")
    iam.attach_role_policy(
        RoleName=EXEC_ROLE,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy")
    return arn


def ensure_task_role(iam, bucket: str) -> str:
    arn = _ensure_role(iam, TASK_ROLE, ECS_TRUST,
                       "Aletheia runner task: read input + write results in the run bucket")
    iam.put_role_policy(
        RoleName=TASK_ROLE, PolicyName="run-bucket-io",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow",
                 "Action": ["s3:GetObject", "s3:PutObject"],
                 "Resource": f"arn:aws:s3:::{bucket}/runs/*"},
                {"Effect": "Allow", "Action": "s3:ListBucket",
                 "Resource": f"arn:aws:s3:::{bucket}",
                 "Condition": {"StringLike": {"s3:prefix": "runs/*"}}}]}))
    return arn


def ensure_dispatcher_user(iam, cluster_arn: str, bucket: str,
                           exec_arn: str, task_arn: str) -> None:
    """The Space's scoped identity: launch/stop/describe THIS cluster's tasks,
    pass the two task roles, read/write run I/O, and tail task logs. Mints one
    access key if the user has none (AWS allows 2)."""
    try:
        iam.create_user(UserName=DISPATCHER_USER)
        print(f"  + user {DISPATCHER_USER}")
    except ClientError as e:
        if _code(e) != "EntityAlreadyExists":
            raise
        print(f"  = user {DISPATCHER_USER} (exists)")

    # Trailing wildcard (no ':') so it covers every task-def family that starts with
    # NAME AND every revision — e.g. aletheia-runner:2 and aletheia-runner-dev:2.
    task_def = f"arn:aws:ecs:{REGION}:{ACCOUNT}:task-definition/{NAME}*"
    task_res = f"arn:aws:ecs:{REGION}:{ACCOUNT}:task/{NAME}/*"
    log_arn = f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{LOG_GROUP}:*"
    iam.put_user_policy(
        UserName=DISPATCHER_USER, PolicyName="dispatch",
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {"Sid": "RunTask", "Effect": "Allow", "Action": "ecs:RunTask",
                 "Resource": task_def,
                 "Condition": {"ArnEquals": {"ecs:cluster": cluster_arn}}},
                {"Sid": "ControlTasks", "Effect": "Allow",
                 "Action": ["ecs:StopTask", "ecs:DescribeTasks"],
                 "Resource": task_res,
                 "Condition": {"ArnEquals": {"ecs:cluster": cluster_arn}}},
                {"Sid": "PassRoles", "Effect": "Allow", "Action": "iam:PassRole",
                 "Resource": [exec_arn, task_arn],
                 "Condition": {"StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}}},
                {"Sid": "RunIO", "Effect": "Allow",
                 "Action": ["s3:GetObject", "s3:PutObject"],
                 "Resource": f"arn:aws:s3:::{bucket}/runs/*"},
                {"Sid": "TailLogs", "Effect": "Allow",
                 "Action": ["logs:GetLogEvents", "logs:FilterLogEvents",
                            "logs:DescribeLogStreams"],
                 "Resource": log_arn}]}))

    keys = iam.list_access_keys(UserName=DISPATCHER_USER)["AccessKeyMetadata"]
    if keys:
        print(f"  = dispatcher already has {len(keys)} access key(s); not minting "
              f"a new one. (Delete an old one to rotate.)")
        return
    k = iam.create_access_key(UserName=DISPATCHER_USER)["AccessKey"]
    CRED_PATH.write_text(json.dumps({
        "AWS_ACCESS_KEY_ID": k["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": k["SecretAccessKey"],
        "note": "Set these as HF Space secrets, then DELETE this file. Rotate via infra.py."},
        indent=2))
    print(f"  + minted dispatcher access key -> {CRED_PATH.name} "
          f"(SET AS SPACE SECRETS, THEN DELETE)")


def create(session) -> None:
    ec2 = session.client("ec2")
    ecr = session.client("ecr")
    ecs = session.client("ecs")
    logs = session.client("logs")
    s3 = session.client("s3")
    iam = session.client("iam")

    print("Provisioning Aletheia Fargate runner infra:")
    repo_uri = ensure_ecr(ecr)
    cluster_arn = ensure_cluster(ecs)
    ensure_log_group(logs)
    bucket = ensure_bucket(s3)
    sg_id = ensure_security_group(ec2)
    exec_arn = ensure_execution_role(iam)
    task_arn = ensure_task_role(iam, bucket)
    ensure_dispatcher_user(iam, cluster_arn, bucket, exec_arn, task_arn)

    outputs = {
        "region": REGION, "account": ACCOUNT, "vpc_id": VPC_ID,
        "subnets": PUBLIC_SUBNETS, "security_group": sg_id,
        "ecr_repo_uri": repo_uri, "cluster": NAME, "cluster_arn": cluster_arn,
        "task_family": NAME, "log_group": LOG_GROUP, "run_bucket": bucket,
        "execution_role_arn": exec_arn, "task_role_arn": task_arn,
        "dispatcher_user": DISPATCHER_USER,
    }
    OUT_PATH.write_text(json.dumps(outputs, indent=2))
    print(f"\nWrote {OUT_PATH.name}. Next: build_image.py to bake data + register the task def.")


# --------------------------------------------------------------- destroy -----
def destroy(session) -> None:
    ec2 = session.client("ec2")
    ecr = session.client("ecr")
    ecs = session.client("ecs")
    logs = session.client("logs")
    s3 = session.client("s3")
    iam = session.client("iam")
    bucket = f"aletheia-leaderboard-runs-{ACCOUNT}"
    print("Tearing down Aletheia Fargate runner infra (best-effort):")

    # Dispatcher user: keys + inline policy, then the user.
    try:
        for k in iam.list_access_keys(UserName=DISPATCHER_USER)["AccessKeyMetadata"]:
            iam.delete_access_key(UserName=DISPATCHER_USER, AccessKeyId=k["AccessKeyId"])
        for p in iam.list_user_policies(UserName=DISPATCHER_USER)["PolicyNames"]:
            iam.delete_user_policy(UserName=DISPATCHER_USER, PolicyName=p)
        iam.delete_user(UserName=DISPATCHER_USER)
        print(f"  - user {DISPATCHER_USER}")
    except ClientError as e:
        if _code(e) != "NoSuchEntity":
            print(f"  ! user: {e}")

    # Deregister task-def revisions in the family, then roles.
    try:
        arns = ecs.list_task_definitions(familyPrefix=NAME)["taskDefinitionArns"]
        for arn in arns:
            ecs.deregister_task_definition(taskDefinition=arn)
        if arns:
            print(f"  - deregistered {len(arns)} task-def revision(s)")
    except ClientError as e:
        print(f"  ! task defs: {e}")

    for role, managed in ((EXEC_ROLE, ["arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"]),
                          (TASK_ROLE, [])):
        try:
            for arn in managed:
                iam.detach_role_policy(RoleName=role, PolicyArn=arn)
            for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
                iam.delete_role_policy(RoleName=role, PolicyName=p)
            iam.delete_role(RoleName=role)
            print(f"  - role {role}")
        except ClientError as e:
            if _code(e) != "NoSuchEntity":
                print(f"  ! role {role}: {e}")

    try:
        ecs.delete_cluster(cluster=NAME)
        print(f"  - cluster {NAME}")
    except ClientError as e:
        print(f"  ! cluster: {e}")

    try:
        sgs = ec2.describe_security_groups(Filters=[
            {"Name": "vpc-id", "Values": [VPC_ID]},
            {"Name": "group-name", "Values": [SG_NAME]}])["SecurityGroups"]
        for sg in sgs:
            ec2.delete_security_group(GroupId=sg["GroupId"])
            print(f"  - security group {sg['GroupId']}")
    except ClientError as e:
        print(f"  ! security group: {e}")

    try:
        logs.delete_log_group(logGroupName=LOG_GROUP)
        print(f"  - log group {LOG_GROUP}")
    except ClientError as e:
        if _code(e) != "ResourceNotFoundException":
            print(f"  ! log group: {e}")

    try:
        ecr.delete_repository(repositoryName=NAME, force=True)
        print(f"  - ECR repo {NAME}")
    except ClientError as e:
        if _code(e) != "RepositoryNotFoundException":
            print(f"  ! ECR repo: {e}")

    # S3 last: empty then delete. Left in place unless --delete-bucket to avoid
    # nuking archived run artifacts by accident.
    print(f"  . S3 bucket {bucket} left in place (delete by hand if wanted)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destroy", action="store_true", help="tear everything down")
    args = ap.parse_args()

    session = boto3.Session(region_name=REGION)
    global ACCOUNT
    ACCOUNT = account_id(session)
    who = session.client("sts").get_caller_identity()["Arn"]
    print(f"Account {ACCOUNT} / {REGION} as {who}\n")

    if args.destroy:
        destroy(session)
    else:
        create(session)
    return 0


if __name__ == "__main__":
    sys.exit(main())
