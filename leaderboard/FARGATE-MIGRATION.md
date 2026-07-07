# Fargate migration — leaderboard submission execution

Moving submission **execution** off the HF Space and onto **ECS Fargate** (one task
per submission), while keeping the HF Space as the front door (dispatcher + viewer).

Status: **runner path proven end-to-end on Fargate** (fixture submission scored). Target AWS
account `797161732516`, region `us-east-1`. Remaining: dispatcher rework (Space) + production
image (heavy deps + real 29 datasets) + red-team egress check.

Tooling (all in `leaderboard/fargate/`, venv `.venv`): `infra.py` (provision/--destroy),
`build_image.py` (bake data + build + push + register task def; `--fixture` or `--config`,
`--heavy`), `runner_main.py` (task entrypoint), `run_task.py` (generic register+run+logs),
`smoke_test.py` (fixture end-to-end), `Dockerfile.runner`, `Dockerfile.probe`.
Ids in gitignored `infra-outputs.json`; dispatcher key in gitignored `dispatcher-credentials.json`.

## Why

- The Space runs submissions **in-process, serially** (`MAX_CONCURRENT_SUBMISSIONS=1`);
  a full run is ~2 h over ~29 datasets, so submissions queue for the whole competition.
- Same-kernel sandbox on the Space (no userns/KVM) is defense-in-depth, not a hard boundary.
- A wedged run can stall the single serial slot for the whole event.

Fargate gives one microVM per submission (strong isolation, no serial bottleneck),
with NDIF — not Fargate — remaining the true compute bottleneck.

## Decisions (locked)

- **Dispatch:** ECS **RunTask per submission** (one ephemeral task each). Not SQS —
  low volume + long untrusted runs favor per-task isolation, clean `StopTask` cancel,
  and no at-least-once double-run hazard. SQS is a documented future upgrade if volume explodes.
- **Front door:** HF **Space stays**, becomes a thin **dispatcher**: validate → tier-gate →
  registry resolve → rate-limit → `RunTask` → tail progress → finalize. Sheds torch/nnsight/sandbox.
- **State store:** **keep the HF bucket**, **serialized by the Space** (single process, single
  writer). The Space still owns every write to `results.jsonl` / `teams.json` / `rate_limits.json`
  under its existing `bucket_lock`. **This write path does not change.** The Fargate task never
  touches shared state; it only emits a per-run `result.json` (unique key, no contention).
- **Space ↔ task transport:** **S3** (in-account, presigned). Space uploads `input.zip`;
  task reads it, writes `progress.jsonl` + `result.json`; Space tails/reads them. S3 is transport
  + raw-artifact staging only, NOT a system of record.
- **Baked image:** torch + nnsight (pinned SHA) + transformers/peft + the runner package +
  **baked eval inputs (Arrow cache), LoRA adapter configs, and private labels**, all into a
  **private ECR image**. Consequence: **the task needs no HF org token at runtime** — the only
  org secret is used at *build* time on a trusted machine. Rebuild = the "easy button" when
  datasets change (`build_runner_image.py`).
- **Egress / sandbox:** **reuse the existing in-process egress sandbox inside the task**,
  unchanged (AllowProxy + seccomp connect() user-notif gate + HF GET-only MITM + Landlock +
  rlimits + label isolation). **No squid, no Network Firewall, no NAT.** Task runs in a public
  subnet with `assignPublicIp=ENABLED`, inbound denied, egress restricted in-process — a faithful
  lift-and-shift of today's model.
  - **Pivot risk:** depends on the Fargate managed kernel supporting seccomp user-notif +
    Landlock. **Verify first (phase 1).** Fallback if not: squid-NAT for egress allowlisting +
    different-UID/file-perms for label isolation.
- **IaC:** **Terraform** for durable infra; Python/boto3 for image build/push + the RunTask
  dispatch (which lives in the Space).

## Verified on the Fargate kernel (2026-07-07)

Ran `probe.py` + `landlock.py` selftest as real Fargate tasks (`run_task.py`):
- Kernel **6.1.174**. `seccomp_bpf` ✓, **`seccomp_user_notif` ✓** (egress proxy works),
  **Landlock ✓ but ABI 2** (Space had ABI 6), user namespaces ✗ (expected, unused).
- `landlock.py` **degrades gracefully** — `_fs_mask(abi)` masks handled rights to the ABI;
  the load-bearing guarantees are ABI-1 rights. **No code change needed.**
- Selftest on Fargate: read-secret-outside **DENIED**, read-parent-environ **DENIED**,
  RW-scratch allowed, RO-dir write denied. **Label isolation + token-theft prevention hold.**
- Still to confirm (with the smoke test): live egress allow/deny through the in-proc proxy.

## Open items

- [x] ~~Reuse existing NDIF VPC~~ `vpc-0ab211670ffaa5d7b`, public subnets, `assignPublicIp`.
- [x] ~~Confirm Landlock/seccomp on the Fargate kernel~~ — done, see above.
- [x] ~~Deployer creds~~ — SSO profile `AdministratorAccess-797161732516`.
- [ ] Set the dispatcher access key (`fargate/dispatcher-credentials.json`) as Space secrets, then delete it.
- [ ] Fargate on-demand vCPU **quota bump** (default ~6) once task size is fixed.
- [ ] Task sizing (vCPU / memory / ephemeral storage) — start ~4 vCPU / 32 GB / 30 GB.
- [ ] Runner image runs as **root** in-container by default — add a non-root user (mirror the Space's uid 1000).

## Target architecture

```
HF Space (dispatcher)                             AWS (acct 797161732516, us-east-1)
POST /submit                                      S3  s3://…/runs/<id>/{input.zip,progress.jsonl,result.json}
  whoami tier-gate ┐                              ECR image (deps + baked inputs/loras/labels)
  registry.resolve │ asyncio lock, HF bucket      ECS Fargate task (the runner)
  ratelimit consume┘  (UNCHANGED)                   read input.zip (S3)
  upload input.zip (S3)                             per-job venv + pip  [in-proc egress: PyPI]
  RunTask (env: run id, submitter NDIF key +        run notebook CHILD (sandboxed, submitter keys)
          HF token, S3 URLs)                          [in-proc egress: NDIF + HF GET-only]  ──▶ NDIF / HF
  tail progress.jsonl (S3) → SSE                    score vs BAKED labels (task, no org token)
on task exit:                                       write progress.jsonl + result.json (S3)
  read result.json → store.append() [asyncio lock, HF bucket, UNCHANGED]
/admin/cancel → StopTask(taskArn)
```

## Components

| Component | Where | Notes |
|---|---|---|
| Dispatcher API + leaderboard | HF Space | slimmed image (FastAPI + boto3 + huggingface_hub + config/results/registry/ratelimit) |
| Runner image | ECR | deps + baked data; `build_runner_image.py`; tagged by dataset-epoch + content hash |
| Execution | Fargate (RunTask) | one microVM/submission; existing pipeline+sandbox lifted in |
| Run I/O transport | S3 | `input.zip` in; `progress.jsonl` + `result.json` out (presigned) |
| Durable state | HF bucket | Space-serialized; write path unchanged |
| Logs | CloudWatch | task stdout/stderr; org-only error detail |
| Dispatch creds | IAM user `aletheia-space-dispatcher` | scoped key as Space secret (RunTask/StopTask/DescribeTasks + S3 + logs) |

## Phased rollout

1. ✅ **Infra** (boto3 `infra.py`): ECR, ECS cluster, S3 run bucket, SG, IAM roles + dispatcher user, log group.
2. ✅ **Runner image + entrypoint:** `build_image.py` bakes inputs/loras/labels + `runner_main.py`;
   fixture image built/pushed; task def registered. Landlock/seccomp probed on Fargate.
3. ✅ **Fixture smoke test passed** (`smoke_test.py`): S3 handoff → sandbox → offline dataset load →
   score vs baked labels (no org token) → result.jsonl, exit 0. Every migration assumption validated.
4. ⏳ **Production image:** `build_image.py --config runner.yaml --heavy` (torch/nnsight + 29 datasets).
   Needs a Fargate vCPU quota bump + task sizing (~4 vCPU / 32 GB). One NDIF-tracing submission run.
5. ⏳ **Red-team egress** (task #6): explicit allow/deny on Fargate.
6. ⏳ **Dispatcher:** rework `app.py` `/submit`: `launch_task → tail S3 progress → read result.jsonl
   → _finalize`. `submit_slots` caps concurrent tasks (1 → ~5–10). `runs[id]` stores `taskArn`;
   `/admin/cancel` → `StopTask`. Keep in-process path behind a flag. Slim Space image + boto3 + AWS secrets.
7. ⏳ **Cutover:** point prod Space at the dispatcher path; in-process path is one-flag rollback.

## Code reorg (contained)

- Space `app.py` `/submit`: everything through rate-limit consume **unchanged**; swap
  `run_in_threadpool(run_pipeline, …)` for launch/tail/finalize.
- New `runner_main.py` baked into the task: read `input.zip` (S3) → existing `run_pipeline`
  (progress callback → `progress.jsonl`) → write `result.json` → exit code = success.
- `scoring.load_labels`: read baked local labels instead of HF (no runtime org token).
- Repo split: shared core + `web/` (Space) + `runner/` (task). Modest.
