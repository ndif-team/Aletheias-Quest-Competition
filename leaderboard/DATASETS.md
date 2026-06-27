# Eval datasets — layout & NDIF processing

How the deception-detection eval datasets are structured, how the leaderboard
consumes them, and the NDIF-side setup required to make them loadable from a
remote `model.session(...)` submission.

## Naming & layout

Each eval is **two** private HuggingFace datasets under the `aletheias-quest/`
org, sharing a stem and split on `index`:

```
aletheias-quest/<collection>-<task>-<model>[-<lora>]            # inputs
aletheias-quest/<collection>-<task>-<model>[-<lora>]-labels     # held-out labels
```

- `<collection>` — `dev` (rehearsal), `dev-test` / `validation` (the real run).
- `<task>` — e.g. `instructed-deception`, `soft-trigger`, `convincing-game`,
  `insider-trading`.
- `<model>` / `<lora>` — the model (and optional LoRA) that generated the
  conversations; encoded in the name so the matrix is legible at a glance.

Every dataset has a single **`test`** split.

## Schema

**Inputs** (`...`):

| column | meaning |
|---|---|
| `index` | stable row id — the **join key** to the labels and to a submission's `index`. |
| `model` | base model that produced the conversation (e.g. `Qwen/Qwen3.5-27B`, `google/gemma-3-27b-it`, `mlabonne/gemma-3-27b-it-abliterated`). |
| `lora` | LoRA adapter to attach, or `None` for the base model. |
| `messages` | the conversation — a list of `{"role", "content"}` turns. |
| `deceptive` | **always null in the inputs repo** — the real label is held out in `-labels`. |
| `temperature`, `meta`, `canary` | generation metadata; not needed for scoring. |

**Labels** (`...-labels`): just `index` + `deceptive` (bool, `true` = deceptive).
Labels are pulled **only by the trusted parent** for scoring and never enter the
sandbox or NDIF (see `data.py`).

## How the leaderboard consumes them

- **Inputs** are loaded by the participant's notebook. In the session-based
  example, `load_dataset(DATASET_NAME, split="test")` runs **on NDIF** inside
  `model.session(remote=True)`, so the heavy load + tokenize + forward passes all
  happen as one remote job; only per-row `index` + `score` come back.
- The notebook reads `model` / `lora` from row 0 and builds
  `VisionLanguageModel(model, peft=lora)`.
- **Labels** are loaded separately by the parent and joined to the submission on
  `index` to compute balanced accuracy / AUROC / recall / FPR.

`runner.yaml` lists the datasets for the real Space run; `dry.yaml` lists the
datasets `python submit.py --dry` rehearses on. Each entry is
`{name, labels_uri, id_column: index, label_column: deceptive}`.

## NDIF-side processing (required for `model.session`)

Because the inputs/LoRAs are **private** and the session loads them **on the NDIF
worker**, the worker must be able to read them. A participant's forwarded
`HF_TOKEN` cannot (it has no access to `aletheias-quest/` private repos), so
access is provisioned on the deployment side, not in the notebook:

1. **Base models deployed on NDIF** — `Qwen/Qwen3.5-27B`, `google/gemma-3-27b-it`,
   and `mlabonne/gemma-3-27b-it-abliterated` must each be served on the cluster.
2. **Datasets pre-downloaded on the workers** — each inputs repo cached locally so
   `load_dataset` resolves from disk. Without it the session fails with
   `DatasetNotFoundError: ... doesn't exist on the Hub or cannot be accessed`.
3. **LoRA adapters pre-downloaded on the workers** — the **full** snapshot
   (`adapter_config.json` **and** the weight file). The two families differ:
   `botc` / `echoblast` ship `adapter_model.safetensors`; the `collusion` /
   `hidden-goal` gemma adapters ship `adapter_model.bin`.
4. **HF token on the NDIF deployment** — peft resolves a LoRA by **repo id**, not
   a local path, and makes a live `file_exists` probe (safetensors vs `.bin`)
   before consulting the cache. For a private repo that probe fails
   unauthenticated, so peft falls back to requesting `adapter_model.bin` and 404s
   (`Repository Not Found ... If you are trying to access a private or gated
   repo, make sure you are authenticated`). A read token on the deployment lets
   that probe succeed and the correct weight file load. This keeps submissions
   **token-free** — the deployment authenticates, not the participant.

### Datasets are kept private end to end

The data side never exposes labels: the `deceptive` column is nulled in the
inputs repo, the `-labels` repo is read only by the trusted parent, and the
inputs the notebook sees carry no answer. Worker-side caching + a deployment
token are what make these private repos loadable without handing a participant
the ability to read them.

## Verifying

`python submit.py --dry` runs the **real pipeline** (per-job venv →
`requirements.txt` → notebook → offline label load → scoring) against the
datasets in `dry.yaml`. A green dry run confirms the datasets load on NDIF, the
model + LoRA attach, and the grader scores all four metrics. The session example
scored all six `dev-instructed-deception-*` datasets (22,668 rows) in one session
each.
