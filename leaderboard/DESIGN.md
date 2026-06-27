Title: Leaderboard framework for NDIF

Goal: NDIF has an upcoming hackathon / competition. We need some way for participants to sumbit their code, have it run agianst a private datatset and push the results to a huggingface Leaderboard.

Relevant Repos: /disk/u/jadenfk/wd/ndif /disk/u/jadenfk/wd/nnsight /disk/u/jadenfk/wd/Aletheias-Quest-Competition

Discussion: We need some way for participants of out competition to submit their method for evaluation without leaking the dataset itself. Im thinking it works like this. Have a Huggingface docker space on the NDIF HF org.  We have this repo /disk/u/jadenfk/wd/Aletheias-Quest-Competition. I want aprticipants to clone / fork this repo. In the repo there will be a folder called submissions. Expects a directoy full of ipynb files. At the repo root there will be submit.py. Sumbit.py will simply compress the repo and submit it via http to the HF space. It can contain other files like probe weights etc, whatever they need for the competition. The hf space is configurable to have a list of HF ids for datasets (private datasets that only this org has access to). The space will then have an outer loop of running over all submission and an inner loop of ruinning of each dataset. It will then set the dataset as an enviropnement variable so the participants submission know which to use. It will then run their notebooks end to end while spapping out this env var.  it will assume after running each notebook a results.csv is created which contains two columns, row id from the datstset and 1 or 0 for detection classicaition. it will then store then compare this to the labels (in a seperate private datset with a different key so their scripts can never get access?? not sure) to generate a score. Finally after all are done, it will push the scores to the leaderboard (which also a part of the docker container? html / js and all that). (results stored in a HF bucket?) Anotehr important note, id like to be a able to configure the  runner app to only allow requests to specific urls. Main one being the NDIF api at api.ndif.us

Agent Notes:

## Decisions (locked 2026-06-08)

- **Execution model:** HF **Docker Space** runner (long-running FastAPI + Docker on the NDIF HF org). NOT GitHub Actions — the starter.ipynb wording about "GitHub Action" is superseded by this.
- **Submit flow:** `submit.py` at repo root compresses the cloned/forked repo and **POSTs the zip over HTTP** to the Space endpoint. No GitHub coupling required.
- **Score format / metric:** notebook produces `submission.csv` with `id,prediction`, `prediction` a probability in `[0,1]`. Metric = **AUROC** (treated as a *placeholder* — keep the scoring function swappable).
- **Threat model:** started at "trusted participants" but **upgraded to a real per-notebook sandbox** (semi-trusted: named hackathon participants, defending against accidents + casual misuse, not nation-state). Kept fully **on the HF Space** — no VM/namespaces (the Space blocks userns/KVM). See the **Execution sandbox** section below. Both original hooks are now implemented: (1) **egress allowlist** enforced via seccomp user-notif on `connect()`, (2) **label isolation** via a predownloaded offline inputs cache (the child's forwarded HF token can't reach the organizers' private eval/labels).
- **Persistence:** results written to a **HF "bucket"** (results dataset/repo on the org). Private **labels live in a SEPARATE dataset** from the eval inputs, keyed by row id, so the inputs the notebook sees never carry labels.
- **Leaderboard:** **score-on-submit** — POST → run + score immediately → update a **static leaderboard page** served by the same Space (HTML/JS reading the results store).
- **Space compute:** **CPU-only**; heavy model work runs on NDIF when a notebook uses `remote=True` (participant's choice — the runner no longer forces a `REMOTE` env var). Space just runs notebook glue + scoring.
- **Dataset scope:** runner loops over a **configurable list of datasets** (the spec's inner loop). **One repo == one dataset** — no HF config subsets, single `test` split (constant `SPLIT="test"`). Sets only `DATASET_NAME` env per run. Seed with one eval set, generalize from there.

## Runner architecture (target)

```
participant repo (fork of this)        HF Docker Space (NDIF org)
  submissions/*.ipynb            zip     ┌─────────────────────────────┐
  submit.py  ───────────────────POST───▶ │ FastAPI /submit             │
  (probe weights, etc.)                  │  → unzip to scratch dir     │
                                         │  → predownload inputs (RO)  │  ◀── eval inputs (private)
                                         │  → for each dataset cfg:    │
                                         │      set DATASET_NAME        │
                                         │      for each notebook:     │
                                         │        run SANDBOXED ········│  ──▶ NDIF api.ndif.us
                                         │        (see Execution        │      (nnsight remote)
                                         │         sandbox below)       │
                                         │        read submission.csv  │
                                         │  → join preds w/ labels      │  ◀── labels dataset (private,
                                         │      (parent holds token)    │       parent-only)
                                         │  → AUROC (swappable metric)  │
                                         │  → write scores to results   │  ──▶ results HF bucket
                                         │  → render leaderboard page   │
                                         └─────────────────────────────┘
                                                static / leaderboard
```

## Execution sandbox (built + verified live on the Space)

Participant notebooks are untrusted code. They run **on the HF Space** (no VM —
the Space blocks unprivileged user namespaces and exposes no `/dev/kvm`, so
Firecracker/gVisor/Kata are all out). Instead each notebook runs as a confined
child process built from **unprivileged Linux primitives the Space *does* allow**
(probed live via `/api/sandbox-probe`): seccomp-bpf, seccomp **user-notification**,
**Landlock ABI 6**, and rlimits. Namespaces and KVM are unavailable.

This is same-kernel, same-container **defense-in-depth** — strong for
semi-trusted participants, *not* a hardware boundary. For a genuinely adversarial
setting the escalation path is an off-Space Firecracker host (needs KVM).

### Per-notebook flow (`sandbox.run_notebook_sandboxed`)

```
parent (trusted)                              child (confined, per job)
─────────────────                             ──────────────────────────
predownload inputs  ─ data.prepare_inputs ─▶  canonical arrow cache (built once,
  (in-process; labels NOT here)                 in-process via load_dataset cache_dir)
copy submission → scratch/work                → writable copy of it in scratch
create venv (--system-site-packages)
                                              ── PHASE 1: pip install -r requirements.txt
                                                 egress: PyPI only
                                              ── PHASE 2: run notebook (nbclient, venv kernel)
 preexec applies, in order:                      egress: api.ndif.us + HF + DNS + loopback
   • rlimits (CPU/FSIZE/NOFILE/NPROC)            load_dataset(...) reads the cache copy
   • Landlock (RO sys, RW scratch,                datasets OFFLINE; HF hub live
       deny everything else incl. /proc)        writes submission.csv → scratch/work
   • connect() user-notif (fd → parent)
   • seccomp blocklist (ptrace/mount/...)
 egress supervisor thread serves the fd ◀──── connect() notifications
read scratch/work/submission.csv
score vs labels (parent holds the token)
```

### The four confinement layers

1. **Landlock (`landlock.py`)** — filesystem confinement by path. Grants
   **read+exec** on `/usr,/lib,/lib64,/bin,/sbin,/etc` and the venv;
   **read-write** only on the job scratch (which holds the per-job dataset cache
   copy, + `/dev/null,zero,urandom,...`); **denies everything else, including
   `/proc`** — which is what stops a same-uid child from reading the parent's
   `/proc/<pid>/environ` to steal the HF token.
2. **seccomp blocklist (`seccomp.py`)** — a BPF filter returning EPERM for ~37
   dangerous syscalls (`ptrace`, `process_vm_readv/writev`, `mount`, `kexec`,
   `bpf`, `setns`, `unshare`, `keyctl`, …), shrinking kernel attack surface and
   closing cross-process memory-read vectors Landlock can't.
3. **rlimits** — `RLIMIT_CPU`, `FSIZE`, `NOFILE`, `NPROC` (`RLIMIT_AS` available
   but off by default — it breaks some ML allocators) + a parent-enforced
   wall-clock timeout that SIGKILLs the whole process group.
4. **Egress allowlist (`egress.py`)** — a seccomp **user-notification** filter
   turns every `connect()` into a notification handled by a supervisor thread in
   the *parent*. It reads the target `sockaddr` (`/proc/<pid>/mem`) and allows
   only: `AF_UNIX`, loopback, **DNS (port 53)**, or an IP currently resolving
   from an allowed hostname; else EPERM. Two-phase: **install → PyPI**, **run →
   `api.ndif.us` + `huggingface.co`** (+ loopback for the kernel's ZMQ sockets).
   Caveats: filters TCP `connect()` only (UDP exfil is a residual channel),
   address-allowlisting has a TOCTOU window, install phase trusts PyPI.

### Label isolation (how the child can load inputs but never the labels)

`data.prepare_inputs` (trusted parent) builds each dataset's **inputs** arrow
cache once, in-process. Each job gets a **writable copy** in its scratch dir; the
child loads it with `HF_DATASETS_OFFLINE=1`, so unchanged
`load_dataset(DATASET_NAME, split="test")` reads the prebuilt arrow from cache
(self-contained; no hub fetch). **Labels are never in that cache** — the parent
loads them separately (with the org token) and scores in-process.

The child *is* given the **submitter's own** `HF_TOKEN` (forwarded like the NDIF
key) so it can load gated HF models it has access to. **Invariant:** the
competition eval/labels datasets are private to the organizers' org, and
participants are external, so a participant's token **cannot** read them — that's
what preserves isolation now that the child has a token. (If an *org member*
submitted, their token could; fine for a real competition, not for org-internal
testing.) Datasets stay forced-offline so even a stray fetch of the private eval
set uses the cache, and `/proc` denial still blocks reading the parent's token.

### Dependencies

The base image pre-installs the competition's heavy required deps — **CPU
`torch` + `nnsight`** (plus `datasets`, `numpy`, `pandas`, `scikit-learn`,
`nbclient`). Each job's venv uses `--system-site-packages` to inherit them, so a
standard probe needs no `requirements.txt`; participants only list extras, which
are pip-installed into the per-job venv during phase 1.

### Config knobs (`runner.yaml` → `RunnerConfig`)

`sandbox: true`, `cache_dir`, `cpu_seconds`, `mem_mb`, `notebook_timeout`
(wall), `enforce_egress`, `egress_allowlist` (run phase), `install_allowlist`
(install phase). `sandbox: false` falls back to the in-process executor (used by
tests).

### What the live Space actually allows (probed) / blocks (tested)

- Allowed: seccomp-bpf ✓, seccomp user-notif ✓, Landlock ABI 6 ✓, rlimits ✓.
- Blocked on the Space: unprivileged user namespaces ✗ (no bwrap/nsjail), KVM ✗.
- Verified end-to-end live: example notebook scores under full confinement;
  `/proc/<parent>/environ` read → DENIED; `connect()` to `1.1.1.1` → BLOCKED,
  `api.ndif.us` → allowed.

## Key parameters / contracts

- Notebook input contract: reads `DATASET_NAME` from env; loads with `load_dataset(DATASET_NAME, split="test")`; writes `submission.csv` (`id,prediction`), one row per example **in dataset order**. No `DATASET_CONFIG`/`DATASET_SPLIT`, no forced `REMOTE` (the notebook chooses `remote=True`). The eval set is a predownloaded offline arrow cache (`HF_DATASETS_OFFLINE`); the submitter's `NDIF_API_KEY` and `HF_TOKEN` are injected for the run phase (the HF token lets them load gated models but can't reach the organizers' private eval/labels).
- Eval/test inputs are **label-free**; grader joins by `id` against the private labels dataset (separate repo, same `test` split).
- `DatasetConfig`: `name` + `labels_uri` (+ `id_column`, `label_column`). Runner config (yaml): list of those, results uri, metric, timeouts/limits, sandbox + egress knobs.

## Resolved

- **Notebook error handling:** recorded as an `ok=False` result with the captured
  error/phase (install|run|read); never crashes the pipeline. No score for that run.
- **`id` alignment:** `scoring.align` fails loudly if prediction count ≠ label
  count or any label id has no matching prediction; `load_predictions` rejects
  duplicate ids and out-of-range probabilities.
- **Resource limits:** `cpu_seconds`/`notebook_timeout`/`nofile`/`nproc`/`fsize`
  via rlimits + wall-clock group-kill (see Execution sandbox).
- **Egress + label isolation:** implemented (see Execution sandbox).
- **Zip size cap:** `submit.py` guards at 200 MB; the Space `/submit` rejects > `MAX_UPLOAD_MB` (250).

## Open questions / to confirm later

- HF Space **submit auth** (currently just a `team` form field — no auth/rate
  limiting; Space is private so endpoints need a bearer token to reach at all).
- **eval vs test phase gating** (test scored "once at the end") — not built.
- Real **private eval + labels** datasets (currently `NDIF/aletheia-fake-*` with
  random labels).
- Results **bucket** is wired (`bucket://NDIF/leaderboard-dev-storage`); revisit
  read-modify-write under concurrent submissions if volume grows.
- Egress hardening: filter UDP `sendto` (close the UDP exfil channel); tighten
  install-phase trust in PyPI.
