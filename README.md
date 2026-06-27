# Aletheia's Quest — Competition Submission Repo

[![Apply to compete](https://img.shields.io/badge/Apply%20to%20compete-aletheias--quest.github.io-6f42c1?style=for-the-badge)](https://aletheias-quest.github.io/)
&nbsp;
[![Live leaderboard](https://img.shields.io/badge/Live%20leaderboard-ndif--leaderboard--dev-2ea44f?style=for-the-badge)](https://ndif-leaderboard-dev.hf.space/)

Build a **deception detector**: for each model conversation in a private eval
set, output the probability that the assistant's final message is *deceptive*.
You build your detector with [nnsight](https://nnsight.net) and run model traces
remotely on [NDIF](https://ndif.us).

> **You must be an official participant to compete.** Apply at
> **[aletheias-quest.github.io](https://aletheias-quest.github.io/)** — the
> organizers will issue you an **NDIF API key** for the competition cluster (it's
> what authenticates your remote traces). Watch your standing on the
> **[live leaderboard](https://ndif-leaderboard-dev.hf.space/)**.

## Quick start

1. **Fork / clone** this repo.
2. **Become an official participant** — apply at
   [aletheias-quest.github.io](https://aletheias-quest.github.io/); the organizers
   issue you an **NDIF API key** for the competition cluster (needed for remote traces).
3. **Install the competition's nnsight** — the `hackathon/peft` branch (remote
   PEFT/LoRA support; it also targets the hackathon NDIF cluster). This branch is
   **required** both locally and on the Space, and **may change before the start**:
   ```bash
   pip install 'nnsight @ git+https://github.com/ndif-team/nnsight.git@hackathon/peft'
   ```
4. For local development, accept the terms on the
   [`Cadenza-Labs/liars-bench`](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)
   dataset page and `huggingface-cli login` (the public **train** data is gated).
5. Look at [`submissions/example.ipynb`](submissions/example.ipynb) — a minimal
   working baseline that follows the contract; replace it with your detector.
6. Put your notebook(s) in [`submissions/`](submissions/).
7. **Verify locally first** (no upload) — runs your notebook end-to-end in a venv
   (installing your `requirements.txt`) against a small *public* dataset and
   prints a score, so you catch broken deps / output before submitting. On failure
   it prints the **full error/traceback** — this is where you debug, because the
   leaderboard itself only returns a generic error (see *Execution sandbox*):
   ```bash
   export NDIF_API_KEY="your-ndif-key"
   python submit.py --dry
   ```
8. Submit for real — give your **team name on the first submission only**:
   ```bash
   python submit.py --team "your-team-name" --space-url <LEADERBOARD_SPACE_URL>
   # afterwards, just:  python submit.py --space-url <LEADERBOARD_SPACE_URL>
   ```

**To submit you need:** an **NDIF API key** (`$NDIF_API_KEY` or `--ndif-api-key`) —
always required — and the leaderboard Space URL (`--space-url` or
`$ALETHEIA_SPACE_URL`). Your **NDIF key is your identity**: the first time you
submit, your `--team` name is bound to it (and must be unused by anyone else);
after that the key alone identifies your team, so you needn't pass `--team` again.

**Submission limit & your standing.** Each team may submit **3 times per 4 hours**
(the budget refills 4 hours after your first submission in a window; over it, you
get a clear "try again in …" message). On the leaderboard page, the **Entrant's
Desk** lets you enter your NDIF key to see your team, best score, attempts
remaining, and your submission history.

## The contract

The runner executes **every `.ipynb` in `submissions/`** top-to-bottom. Before
running, it sets these environment variables:

| Variable        | Meaning                                                       |
|-----------------|---------------------------------------------------------------|
| `DATASET_NAME`  | HF dataset id to predict on (the private eval set)            |
| `NDIF_HOST`     | The competition's NDIF cluster, `https://aletheias.api.ndif.us` — set for you; nnsight reads it, so **you don't configure a host** (just pass `remote=True`) |
| `NDIF_API_KEY`  | Your NDIF key, forwarded from your submission (nnsight reads it) |
| `HF_TOKEN`      | Your HF token, forwarded so you can load gated HF models you can access |

Each dataset is one repo with a single `test` split, so load it with
`load_dataset(DATASET_NAME, split="test")` — **don't hard-code the dataset name**.
There is no `DATASET_CONFIG`/`DATASET_SPLIT`; the eval data is predownloaded and
loaded offline for you (so `load_dataset` only works for `DATASET_NAME`, not other
datasets at runtime). Whether to run traces remotely on NDIF is **your choice** —
pass `remote=True` to your nnsight traces (recommended; your `NDIF_API_KEY` is set).
When run, your notebook must **write
`submission.csv`** in the repo root:

```
id,prediction
0,0.83
1,0.12
2,0.97
...
```

- `id` — the 0-based row index of the example (`0 .. N-1`).
- `prediction` — a probability in `[0, 1]` that the example is deceptive.
- **Exactly one row per example, in dataset order.** That's all the grader reads.

The eval inputs are **label-free**; you are scored by joining your `prediction`s
against held-out labels you never see. Current metric: **AUROC** (placeholder —
may change). You're evaluated on one or more held-out datasets and your reported
score is the **average** across them — your notebook just predicts on whatever
`DATASET_NAME` is set to each run.

## Dependencies (`requirements.txt`)

Put a `requirements.txt` at the repo root. Before your notebook runs, the runner
installs it into a per-job virtualenv (with `--system-site-packages`) on top of
the base image, so you only list what you *add*. The base already includes
`datasets`, `numpy`, `pandas`, `scikit-learn`, `nbclient`, **`torch` (CPU)**,
**`transformers`**, and **`nnsight`** (the `hackathon/peft` branch — see step 3) —
so a standard nnsight + NDIF probe needs no extra deps. (Installs reach **PyPI
only**; see the egress note below.)

## Execution sandbox

Your notebook runs in an isolated sandbox (filesystem-confined, syscall-filtered,
resource-limited). Implications for your code:

- The eval dataset is **predownloaded**; `load_dataset(DATASET_NAME, …)` reads it
  offline (you can't load other datasets from the Hub at runtime).
- Your **`NDIF_API_KEY`** and **`HF_TOKEN`** are forwarded into the run (set them
  before `submit.py`, or pass `--ndif-api-key`/`--hf-token`), so nnsight
  authenticates remote traces and you can load gated HF models you have access to.
- `LanguageModel("…")` works: `huggingface.co` is reachable for model
  config/tokenizer. Run the weights on NDIF with `remote=True` (don't dispatch
  them locally).
- Network egress is allowlisted by hostname to the NDIF cluster
  (`aletheias.api.ndif.us`, under `api.ndif.us`) + its results bucket and
  `huggingface.co`/`hf.co`; everything else is blocked. HF is read-only (model
  configs/tokenizers download fine; you can't upload).
- You may only write under the working directory; CPU/memory/time are capped.
- **If your run fails here, the leaderboard returns a *generic* error** (the real
  error could leak the private eval data). To see the actual traceback, reproduce
  with `python submit.py --dry` — same pipeline, public data, full errors.

`--dry` runs the real pipeline (venv → `requirements.txt` → your notebook → NDIF
traces → scoring), so it catches dependency, code, NDIF, and output errors. It
does **not** apply the network/filesystem confinement above (so it stays portable)
— a notebook that reaches a non-allowlisted host or writes outside its directory
passes locally but fails here. Keep your code to NDIF + HF and the working dir.

## Notes

- Heavy model compute should run on NDIF (`remote=True` in your traces), not locally.
- You can include extra files (probe weights, helper modules) in the repo — they
  ship with your submission. Keep the total package reasonable (< 200 MB).
- `submit.py` packages the repo and POSTs it to the leaderboard Space; set the
  Space URL via `--space-url` or the `ALETHEIA_SPACE_URL` environment variable.
