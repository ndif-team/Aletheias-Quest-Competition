# Aletheia's Quest — Leaderboard Runner

Backend for the Aletheia's Quest deception-detection competition. It runs
untrusted participant notebooks against a private eval set, scores them against
held-out labels, and publishes a leaderboard — deployed as a **HuggingFace Docker
Space** that wraps this package.

This lives at `leaderboard/` inside the participant repo (the repo root holds
`submit.py` + `submission/`). Participants submit with `submit.py` (zip → POST);
this runner receives, sandboxes, executes, and scores those zips. The code being
visible to participants is fine — security rests on the sandbox, the private
datasets, and tokens, not on hiding this code.

## Architecture

```
zip (submission/*.ipynb + extras)
  → unpack
  → predownload each dataset's inputs (offline arrow cache)
  → for each dataset in runner config:
        set DATASET_NAME (+ the submitter's NDIF_API_KEY)
        run each notebook sandboxed (venv, no token, offline data) ── nnsight ──▶ NDIF
        read submission.csv (id, prediction)
        join to private labels (separate dataset) by id
        score (AUROC, swappable)
  → write result records to the results bucket
  → render the leaderboard page
```

CPU-only: heavy model compute runs remotely on NDIF when a notebook uses
`remote=True` (the participant's choice).

## Layout

| Path                          | What                                             |
|-------------------------------|--------------------------------------------------|
| Path                            | What                                           |
|---------------------------------|------------------------------------------------|
| `src/aletheia_runner/config`    | dataset list, labels/results locations, metric, sandbox knobs |
| `src/aletheia_runner/data`      | predownload inputs (parent); offline child cache env |
| `src/aletheia_runner/scoring`   | align predictions ↔ labels, metric registry    |
| `src/aletheia_runner/results`   | result records + leaderboard store (local/bucket) |
| `src/aletheia_runner/executor`  | in-process notebook runner (non-sandbox fallback) |
| `src/aletheia_runner/pipeline`  | unpack → run (sandboxed) → score → records      |
| `src/aletheia_runner/app`       | FastAPI: `/submit`, leaderboard, probe endpoints |
| `src/aletheia_runner/sandbox/`  | **confinement subpackage:**                     |
| &nbsp;&nbsp;`sandbox/runner`    | per-job venv + sandboxed notebook execution     |
| &nbsp;&nbsp;`sandbox/landlock`  | filesystem confinement (ctypes)                 |
| &nbsp;&nbsp;`sandbox/seccomp`   | syscall blocklist (ctypes BPF)                  |
| &nbsp;&nbsp;`sandbox/egress`    | connect() allowlist (seccomp user-notif)        |
| &nbsp;&nbsp;`sandbox/probe`     | capability prober (`/api/sandbox-probe`)        |
| &nbsp;&nbsp;`sandbox/_run_entry`| in-sandbox notebook entrypoint                  |
| `tests/`                        | offline tests (no NDIF/HF needed)               |

The execution sandbox (Landlock + seccomp + rlimits + egress allowlist + per-job
venv + predownloaded read-only data) is built and verified live — see
[`DESIGN.md`](DESIGN.md). Remaining: submit auth/rate-limiting, eval-vs-test
phase gating, real (non-fake) eval datasets.

## Develop

```bash
pip install -e ".[dev]"
pytest
```

The tests use a deterministic fixture notebook and a local labels CSV, so they
run fully offline.
