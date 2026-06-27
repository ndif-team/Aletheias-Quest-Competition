# NOTES — first-time participant / agent walkthrough

Field notes from going through Aletheia's Quest end-to-end as a *first-time blue-team
participant driving the repo with a coding agent*, then submitting an
efficiency-optimized version of an existing probe. The goal of this file is to make the
on-ramp smoother for the next person: what was unclear, what blocked me, what broke, and
concrete suggestions to improve the docs, the toolkit, and the leaderboard.

Legend: 🟥 blocker / bug · 🟧 friction / unclear · 🟩 worked well · 💡 suggestion

---

## 0. TL;DR of the most actionable items

0. 🟥 **The `ndif-leaderboard-dev` Space can't read its own private eval data, so no submission
   can score.** A real submit is rejected `400 invalid submission: Dataset 'aletheias-quest/
   dev-test-…' doesn't exist or cannot be accessed`. The Space loads eval data with its own
   `HF_TOKEN` secret (`data.py:78` + `config.py:164`), which is unset/unauthorized for the org's
   private datasets. **Fix:** set the Space's `HF_TOKEN` secret to an `aletheias-quest` token with
   read access to the private eval + labels repos. **(Resolved in-session once the organizer set
   the Space `HF_TOKEN`.)** See §6 → "Real submission attempt." (Also: that 400 path leaks a real
   private dataset name, and would spend an attempt if the limiter were enabled — both detailed
   there.) **Then a second blocker appeared — see 0b.**
0b. 🟥 **`run_full_session` loads the dataset *on NDIF*, so every eval dataset must be cached on
   NDIF — and the example template depends on this.** `util.run_full_session` calls
   `load_examples(dataset_name)` *inside* `with model.session(remote=True)`, so `load_dataset`
   executes on NDIF, not in the local sandbox. If the dataset isn't already cached on NDIF, its
   `download_and_prepare` trips NDIF's sandbox guard and you get a baffling
   `RemoteException: … RuntimeError: Operation 'shutil.rmtree' is blocked inside the sandbox`.
   My public-dev `--dry` passed only because those sets were already on NDIF; the private
   `dev-test-*`/`validation-*` sets are not (yet), so the real submission failed there. Two
   consequences: (a) **organizers must pre-cache every scored dataset on NDIF** or the official
   `example.ipynb` (which uses `run_full_session`) fails for everyone; (b) a submission that loads
   **locally** instead (like the baselines, or my `run_local_session` variant) only needs the
   *model* on NDIF and works regardless. The contract says eval data is "predownloaded and read
   offline" — that's true for the local sandbox, but `run_full_session` bypasses it by loading
   remotely, which is surprising and the error is opaque (redacted to a generic message on the
   board). See §6 → "Second blocker."
1. 🟥 **`baselines/mean_probe.ipynb` is broken against the current contract.** It writes
   `id,prediction` on row position, but the grader requires `index,deceptive,score`
   joined on the dataset's `index` column. As shipped it would be **rejected** ("missing
   columns") — and it's pointed to by name in the README/llms.txt as *the* probe baseline.
   (Every *other* baseline correctly uses `util.write_submission`.) See §3.
2. 🟥 **`mean_probe.npz` carries standardization metadata (`mu`, `sigma`, `normalize=True`,
   `bias`, `score_std`) that *no* notebook or tutorial actually applies.** The shipped
   `probe_proba` projects *raw* activations, which is not how the probe was trained.
   Applying the saved standardization changed nothing about whether it runs but is the
   correct scoring (and scored very well — see §6). See §3.
3. 🟧 **None of the shipped baselines use `run_full_session`** — the one-session helper the
   README and `llms.txt` push hard as *the* efficient path. They all do one
   `model.trace(..., remote=True)` per batch. So the "efficient" pattern has no worked
   example in `baselines/`; you have to reverse-engineer it from `util.py` + `example.ipynb`.
   See §4.
4. 🟧 **All docs say `python submit.py`, but a clean machine often only has `python3`.**
   And the runner needs **Python ≥3.10** (system was 3.8). The happy path really depends on
   a correctly-provisioned env that the docs under-specify. See §1.
5. 💡 The single highest-value addition would be a **dry-run row cap** flag (I added an env
   knob to my own notebook) so a first `--dry` doesn't run 4,000+ remote forward passes. See §5.

---

## 0d. Resolution log — changes made this session

These items were addressed in the repo (verified with `--dry`):

- **#1 + #2 (mean_probe baseline):** rewrote `baselines/mean_probe.ipynb` to write the correct
  `index,deceptive,score` contract via `write_submission` and to apply the probe's saved
  standardization (`mu`/`sigma`/`normalize`/`bias`/`score_std`), not a raw projection. Loads
  data **locally** + one remote trace per batch (the simple "regular" shape).
- **#3 (no `run_full_session` example):** added `baselines/mean_probe_fast.ipynb` — the same
  probe as **one** `run_full_session` session with a `tracer.stop()` early-exit after layer 26
  (the scored submission). It's the worked example of the efficient path; `llms.txt §5/§6` now
  point at it and document `tracer.stop()` as an efficiency lever.
- **#4 (env / paths):** added `requirements-dev.txt` + `setup_dev.sh` (venv, Python-≥3.10 check)
  for a one-command local dev + `--dry` environment; **moved the submission's `requirements.txt`
  into `submission/`** (runner + README/llms.txt updated); **renamed `submissions/` → `submission/`**
  across the repo and runner.
- **#5 (dry-run row cap):** added a real **`submit.py --limit N`** that threads through to the
  notebook as `$ALETHEIA_LIMIT` (via `extra_env`, the same channel as the keys — so unlike a
  bare shell var it actually reaches the sandbox). `example.ipynb` (and both probes) read it and
  pass `limit=` to `run_full_session`. Replaces the dead `ALETHEIA_MAX_ROWS` knob.
- **Bug found while verifying #5:** the `--limit` feature exposed a latent alignment bug — the
  notebooks joined `len(scores)` results against the *full* index column (`ValueError: indices
  (4157) and scores (8) differ`). Fixed by slicing `examples["index"][:len(scores)]`. (Harmless
  before, because a full run always had `len(scores) == len(examples)`.)
- Minor: `tutorials/3.3-did-you-lie-probe.ipynb` ships as a **0-byte file** (empty placeholder) —
  worth filling in or removing.

**⚠ Deploy note:** the rename and the requirements move change the **runner**, so the live
`ndif-leaderboard-dev` Space must be **redeployed from this updated repo**. Until then the
deployed Space still expects `submissions/` + a root `requirements.txt` and will reject a package
built by the updated `submit.py` (which ships `submission/`). The earlier `aletheia-blue` score
stands; new submits need the redeployed runner.

**Verification:** `submit.py --dry --limit 8` runs `example.ipynb` end-to-end on the renamed
layout and **partial-scores** the subset (Bal.Acc 0.625 on 8 rows, random baseline); the runner
unit suite is **67 passed / 2 skipped**; the rebuilt regular `mean_probe.ipynb` passes an offline
contract/tokenization/scoring check (its remote trace pattern is identical to the leaderboard-scored
fast notebook — a live re-score was blocked only by a transient NDIF "model deployment" outage).

(The path references below still say `submissions/` where they describe the *original* state.)

---

## 1. Setup / environment (§ "Quick start", "Build with an agent")

- 🟧 **`python` vs `python3`.** Every command in the README and `llms.txt` is
  `python submit.py …`. On a stock Ubuntu box `python` doesn't exist (only `python3`), so
  the literal copy-paste fails with `python: command not found`. Either tell people to use
  `python3`, or note that a venv/conda env puts `python` on PATH.
- 🟥 **Python version floor is implicit.** `leaderboard/pyproject.toml` says
  `requires-python = ">=3.10"`, and `--dry` builds the per-job venv from `sys.executable`,
  so the interpreter you launch `submit.py` with must be ≥3.10. The system Python here was
  **3.8** — `--dry` would have failed in a confusing way. The README's step 3 only mentions
  installing nnsight; it should also state the **minimum Python** (3.10+; the runner image
  appears to be 3.12) up front. 💡 Add a one-line `python3 --version` precheck to `submit.py`
  that fails fast with a clear message.
- 🟧 **`requirements.txt` and the base image are documented, but "what must already be on
  my machine to even run `--dry`" is not.** `--dry` needs nnsight **plus** the runner deps
  (`datasets, nbclient, ipykernel, scikit-learn, pyyaml`) in the *launching* interpreter,
  on top of a 3.10+ Python. A copy-paste "create the dev env" block (conda/venv + the exact
  pip line + the runner deps) would remove a lot of guesswork.
- 🟩 **Once a correct env exists, it's smooth.** Here a preconfigured `ndif-test` conda env
  (Python 3.12) already had nnsight (`0.7.1.dev…`, the hackathon build), all runner deps,
  and the NDIF key saved in nnsight's `CONFIG`. With that, everything below "just worked."
- 🟧 **NDIF key discovery.** The key lived only in nnsight's persisted `CONFIG`
  (`CONFIG.API.APIKEY`), not in `$NDIF_API_KEY`. `submit.py` requires `--ndif-api-key`/
  `$NDIF_API_KEY`. So even though nnsight itself was authenticated, `submit.py` would error
  with "`--ndif-api-key … is required`" until I exported it. 💡 Have `submit.py` fall back to
  `nnsight.CONFIG.API.APIKEY` when `$NDIF_API_KEY` is unset (it already special-cases the HF
  token via `huggingface_hub.get_token()` — do the symmetric thing for NDIF).

---

## 2. Understanding the task & data (§3 of llms.txt)

- 🟩 `llms.txt` is genuinely good as an agent briefing — the repo map, the `detect_fn`
  rules, and the error table are exactly what an agent needs and saved real time.
- 🟧 **`lora` is a real `None` for base rows, but it's easy to fear it's the string
  `"None"`** (the dataset card / a casual `print` shows `None`, and dataset repo names
  literally end in `-None`). Worth one sentence: "`row['lora']` is Python `None` for base
  models." If it *were* the string `"None"`, `build_model(model_id, lora_id)` would do
  `peft="None"` (truthy!) and try to attach a nonexistent adapter — a nasty trap. Confirmed
  it's real `None`, so the example template is safe; but the trap is one bad data export away.
- 🟧 **Dataset size isn't mentioned anywhere.** The dev `gemma-3-27b-it-None` set is **4,157
  rows**. A newcomer's first instinct is `python submit.py --dry`, which then quietly kicks
  off thousands of remote 27B forward passes across *six* datasets. People should be warned,
  and given a cap (see §5).
- 🟧 **`index` is a string.** `row['index']` is e.g. `'35387'` (str), and the labels join on
  it. `util.write_submission` passes it through fine, but anyone hand-rolling the CSV or
  doing integer math on `index` could get bitten. A note in the contract section would help.

---

## 3. The probe baselines (§ "baselines/", llms.txt §6)

This is where the biggest issues are. I picked the **mean-difference probe** because it's the
cheapest method (a single forward pass + a single-layer read; no generation, no judging),
which best fits the "fast and efficient" goal.

- 🟥 **Contract mismatch in `baselines/mean_probe.ipynb`.** Final cell:
  ```python
  pd.DataFrame({"id": np.arange(len(predictions)), "prediction": predictions}).to_csv(...)
  ```
  The grader (`leaderboard/.../scoring.py::load_predictions`) requires columns
  `{"index","deceptive","score"}` and joins on `index`. So this baseline produces
  `id,prediction` keyed on **row position** and would be rejected with
  `submission.csv missing columns: ['deceptive', 'index', 'score']`. The README/llms.txt
  even name this notebook as the canonical activation-probe example, so it's the most likely
  thing a newcomer copies. **Fix:** use `util.write_submission(examples["index"], predictions)`
  like every other baseline does.
- 🟥 **`mean_probe.npz` ≠ what `mean_probe.ipynb` uses.** The shipped `.npz` has
  `direction, mu, sigma, normalize=True, bias, score_std, layer=26, model_id, max_seq_len=300,
  use_chat_template`. But `probe_proba` does a raw `X @ direction` and **ignores
  mu/sigma/normalize/bias/score_std**. Standardization is *not* rank-preserving (it rescales
  each feature before the dot product), so this changes both the AUROC ranking and the
  `score>=0.5` binary call. The tutorial `tutorials/3.1-mean-probe.ipynb` that supposedly
  produced it saves only `direction, layer, model_id` and also scores raw — i.e. the **shipped
  npz was made by a newer/different training script than the tutorial that documents it**, and
  the consumer notebook was never updated to match. Net: the baseline and its weights are out
  of sync with each other *and* with the tutorial. (Applying the full standardization scored
  Bal.Acc ≈ 0.91 on dev — see §6 — so the metadata is clearly meant to be used.)
- 🟧 **`mean_probe.ipynb` reads `os.environ.get("LORA_ADAPTER")`**, but `LORA_ADAPTER` is not
  one of the documented contract env vars (`DATASET_NAME, NDIF_HOST, NDIF_API_KEY, HF_TOKEN`).
  So that branch is dead on the leaderboard and just adds confusion about whether the probe is
  supposed to track each row's LoRA. (It isn't — the probe is a fixed gemma feature extractor.)
- 💡 **The baselines need a CI smoke test** that runs each notebook's final cell against the
  scoring validator on a tiny fixture, so a contract drift like this can't ship. There's
  already `leaderboard/tests/` with fixtures — extend it to lint the baselines' output schema.

---

## 4. Making it fast/efficient — what I built

Goal: keep the *exact* probe, but minimize leaderboard runtime. My notebook is
`submissions/mean_probe_fast.ipynb`. Three levers, in impact order:

1. 🟩 **One NDIF session for the whole dataset** via `util.run_full_session` (one queue wait)
   instead of N per-batch `model.trace(remote=True)` calls like the stock baselines. This is
   the pattern the docs recommend but, oddly, **no baseline demonstrates** — so this notebook
   doubles as the missing worked example of `run_full_session` + a custom `preprocess` + a
   custom `detect`.
2. 🟩 **Early-stop the forward pass at the probe's layer** with `tracer.stop()`. The probe
   reads layer 26; `gemma-3-27b-it` has ~62 decoder layers, so stopping right after layer 26
   skips ~35 layers **and the LM head** — well over half the forward compute, for free, with
   no effect on the score. (`tracer.stop()` is only shown once, buried in
   `tutorials/2-predicting.ipynb`; it deserves to be called out in `llms.txt §6 "Efficiency"`
   as a first-class lever for single-layer probes — it's the biggest single win and easy to miss.)
3. 🟩 **Faithful, correct output**: `index,deceptive,score` via `write_submission`, plus the
   probe's full saved standardization (fixing the two bugs in §3).

Other notes:
- The probe is a **fixed feature extractor** trained on `google/gemma-3-27b-it`, so I force
  that model (no LoRA) for *every* dataset — including Qwen/Nemotron-generated ones — matching
  how `mean_probe.ipynb` applies it (it hardcodes the npz's `model_id`). Whether running gemma
  over another model's text is the *best* choice is a method question, but it's the faithful
  one, and dev numbers held up.
- 🟧 **Shipping a custom `preprocess`/`detect` to NDIF depends on `inspect.getsource`.**
  nnsight cloudpickles them *by value via source code*. A function defined on a bare
  `python - <<EOF` heredoc (no source file) fails with
  `PicklingError: Cannot serialize function … source code unavailable`. Inside a real notebook
  it's fine (IPython registers cell source), and from an imported `.py` it's fine. Worth a line
  in `llms.txt`: "your `detect`/`preprocess` must have retrievable source — define them in the
  notebook or an imported module, never via `exec`/`-c`/heredoc."
- 🟧 **`run_full_session` runs `preprocess` *inside* the session**, i.e. tokenization happens
  remotely. So `preprocess` is also cloudpickled and subject to the same source/whitelist rules
  as `detect`. The docstring says this, but it's easy to miss that a "preprocessing" function
  is not client-side.

---

## 5. `--dry` rehearsal (§ "Test locally, then submit")

- 🟩 `--dry` does exactly what it promises: real venv → `requirements.txt` → notebook →
  remote NDIF traces → scoring, with **real** errors and **real** dev dataset names. Great.
- 🟥/💡 **No row cap.** `--dry` scores *every* row of *every* dataset in `dry.yaml`
  (default: 6 datasets × thousands of rows). A first rehearsal is therefore enormous and
  burns real NDIF compute. There's `limit=` in `run_full_session`, but nothing surfaces it to
  `--dry`. I added an `ALETHEIA_MAX_ROWS` env knob to my notebook (`limit=` when set, full
  otherwise) so I could rehearse on 64 rows first. 💡 Bake a `--limit N` / `max_rows:` into
  `submit.py --dry` + `dry.yaml`, and recommend a small cap for the first run in the README.
- 🟧 **`dry.yaml` defaults to all six dev datasets.** Combined with no cap, the out-of-the-box
  `--dry` is the most expensive possible rehearsal. Defaulting to one small dataset (with a
  comment on how to add more) would be a friendlier default.
- 🟧 **Attempts accounting is well documented but scary in combination with the above.**
  "3 per 4h, an errored real run still costs one" is clear — but because the *honest*
  pre-submit rehearsal is so heavy, people may be tempted to skip `--dry` and gamble an
  attempt. A cheap-by-default `--dry` would fix the incentive.
- 🟩 Nice touch: `--dry` prints real dataset names while the leaderboard shows opaque
  `Dataset N` — and that distinction is documented. Good.

---

## 6. Results observed

Capped `--dry` (`ALETHEIA_MAX_ROWS=64`) across **all six** dev datasets, one session each,
normalized probe, early-stop @ layer 26, total runtime **4m 39s** (incl. venv + model load):

```
mean_probe_fast.ipynb   MEAN  Bal.Acc 0.8279  AUROC 0.9790  Recall 0.6870  FPR 0.0312
  gemma-3-27b-it-None                         Bal.Acc 0.9138  AUROC 0.9817  Recall 0.8908  FPR 0.0632
  gemma-3-27b-it-abliterated-collusion        Bal.Acc 0.9108  AUROC 0.9769  Recall 0.8969  FPR 0.0753
  gemma-3-27b-it-hidden-goal-model-organism   Bal.Acc 0.9021  AUROC 0.9819  Recall 0.8420  FPR 0.0378
  Qwen3.5-27B-None                            Bal.Acc 0.7924  AUROC 0.9886  Recall 0.5892  FPR 0.0043
  Qwen3.5-27B-echoblast-model-organism        Bal.Acc 0.7867  AUROC 0.9854  Recall 0.5790  FPR 0.0057
  Qwen3.5-27B-botc-latest-checkpoint          Bal.Acc 0.6618  AUROC 0.9597  Recall 0.3242  FPR 0.0007
```

🟩🟧 **Big finding — the threshold, not the signal, is the cross-model bottleneck.** On the
**gemma** sets (the probe's home base) balanced accuracy is ~0.90–0.91. On the **Qwen** sets it
falls to 0.66–0.79 — but **AUROC stays ~0.96–0.99**. So the probe's *ranking* transfers across
base models almost perfectly; what collapses is the **0.5 binary threshold**: FPR ≈ 0 and recall
is low (0.32–0.59), i.e. the gemma-calibrated sigmoid sits below 0.5 for Qwen activations and the
detector under-fires. Since the **leaderboard ranks by balanced accuracy** (a thresholded metric)
while the signal lives in AUROC, *calibrating the binary call* is the cheapest lever to raise
standing — no extra compute, no probe change. I kept the faithful 0.5 here (to stay true to "an
existing probe" and not overfit dev's instructed-deception-only distribution), but a lower global
threshold (or a percentile rule for `deceptive`) is the first thing I'd tune to climb the board.

(64-row-per-dataset sample; full-set numbers will differ. Each dataset includes a fresh model
load, so steady-state throughput is better than the wall-clock suggests.)

### Real submission attempt — blocked by a Space-side data-access bug 🟥

Submitted for real (`--team aletheia-blue --space-url https://ndif-leaderboard-dev.hf.space`,
NDIF key `123` per the dev-Space instruction). Result:

```
✓ packaged 33 files  ·  1.3 MB
▸ entering the lists as aletheia-blue  →  https://ndif-leaderboard-dev.hf.space/submit
✗ submission rejected  [400]
   invalid submission: Dataset 'aletheias-quest/dev-test-instructed-deception-Qwen3.5-27B-None'
   doesn't exist on the Hub or cannot be accessed.
```

My package is fine (it passed structure validation and runs clean on all six public dev sets).
The failure is **server-side**: the Space can't load its *own* private eval data. Traced it
through the runner:

- `data.py:78` predownloads each eval set with `load_dataset(cfg.name, token=config.hf_token)`.
- `config.py:164` sets `config.hf_token = self.hf_token or os.environ["HF_TOKEN"]` — i.e. the
  **Space's own `HF_TOKEN` secret**, *not* the participant's (by design — `app.py:279-280`:
  "the token can't reach the private eval/labels — that's the organizers' org").
- So on `ndif-leaderboard-dev`, the `HF_TOKEN` secret is **missing or lacks read access** to the
  private `aletheias-quest/dev-test-*` datasets. I confirmed those datasets *do* exist and are
  private, and that *my* cached HF token can read them — so it's purely the Space's token/secret.

**Organizer fix:** set the `HF_TOKEN` secret on the `ndif-leaderboard-dev` Space to an
`aletheias-quest` org token with read access to the private eval **and** labels repos. (That same
`config.hf_token` also backs the results store, team registry, and rate-limit buckets at
`app.py:353-357`, so it's load-bearing well beyond data.) Until then **no submission can score** —
this is the single hard blocker to "getting on the leaderboard," and it's invisible from the
participant side except as a confusing 400.

Two further issues this surfaced:

- 🟥 **The 400 may still cost an attempt.** The rate-limit slot is consumed at `app.py:247`
  *before* `run_pipeline` (`app.py:292`), and this dataset-load failure happens *inside*
  `run_pipeline` → 400 at `:296`. So a failure the participant cannot influence (the Space
  can't read its own data) appears to **spend an attempt**, contradicting the docs' promise that
  "a submission rejected up front … costs nothing." Up-front *structure* rejection (`:239`) is
  indeed free, but a data/pipeline rejection is not. (Couldn't verify the counter — `POST
  /api/me` returned HF's generic 404 page on this Space, so the Entrant's-Desk API may be
  mis-routed here too. 💡 data-availability should be checked *before* consuming the slot, or the
  slot refunded on a `400`.) **Update:** `runner.yaml` has `rate_limit_max: 0` (limiter disabled),
  so on this dev Space attempts aren't actually charged right now — the concern is real for when
  the limit is turned on, not a live problem today. (And the failed run showed **no submission** on
  the Entrant's Desk, consistent with failures not being stored.)
- 🟧 **The up-front 400 leaks a real private dataset name** (`dev-test-instructed-deception-
  Qwen3.5-27B-None`). `llms.txt §3` promises failures say "failed on Dataset N", never the real
  name, precisely to keep the private split private. The structural/predownload rejection path
  bypasses that redaction. (The *run-time* failure path **does** redact — see the second blocker
  below, which correctly reported "failed on Dataset Doris".) Minor, but the up-front path
  contradicts a stated guarantee.

### Second blocker — datasets not cached on NDIF (resolved) 🟥→🟩

After the organizer set the Space `HF_TOKEN`, the submission got past data access and **ran**, then
failed on one dataset with the generic redacted message ("failed on **Dataset Doris**" — note the
redaction works here). Reproducing locally by pointing `dry.yaml` at the 12 private leaderboard
datasets gave the real cause:

```
RemoteException: … util.py:241  ds = load_examples(dataset_name)   (inside model.session(remote=True))
  → load_dataset → download_and_prepare → shutil.rmtree
  RuntimeError: Operation 'shutil.rmtree' is blocked inside the sandbox   [on NDIF]
```

i.e. **`run_full_session` loads the dataset on NDIF, and the private sets weren't cached there yet**
(see TL;DR 0b for why this is a design sharp-edge). The organizer pre-downloaded the 12 datasets
onto NDIF, after which **all 12 run and score**. This is the moment the whole pipeline finally
worked end-to-end. Takeaways for the docs: (a) every scored dataset must be on NDIF for the
`run_full_session`/`example.ipynb` path; (b) the failure is a redacted generic on the board, and
only reproducible via `--dry` *if the participant can load that dataset* — for the private sets that
means org access, which a normal blue-teamer doesn't have, so they'd be stuck. A submission that
**loads locally** sidesteps the dependency entirely.

### 🟧 The `--dry` row-cap I added never actually caps

I added an `ALETHEIA_MAX_ROWS` env knob to the notebook to rehearse on a subset. It silently does
nothing under the real pipeline: the sandbox's `_child_env` (`sandbox/runner.py:155`) builds a
**fresh, sanitized environment** (PATH/HOME/cache + `DATASET_NAME` + the forwarded keys) and does
**not** inherit arbitrary parent env vars. So my "capped" `--dry` runs were in fact **full** runs —
which is exactly why their metrics match the leaderboard's to 4 decimals. Implication for everyone:
**you cannot speed up `--dry` with an env var** (or any local env state); the only knobs are
trimming `dry.yaml` to fewer datasets or hard-coding a `limit` in the notebook. 💡 If quick
rehearsal matters, `submit.py --dry` should grow a real `--limit N` that the runner injects into
`DATASET_*`-style env the sandbox *does* pass through.

### Full private-set rehearsal = the live leaderboard result (calibration story, sharpened)

`--dry` over the **actual 12 leaderboard datasets** (full, 476–5784 rows each), runtime 7m53s — and
the subsequent real submission returned the **identical** numbers in 8m32s:

```
MEAN  Bal.Acc 0.6796  AUROC 0.8989
  dev-test instructed (gemma ×3)      Bal.Acc 0.909–0.921   AUROC ~0.98     ← probe's home base
  dev-test instructed (Qwen ×3)       Bal.Acc 0.679–0.802   AUROC 0.95–0.99
  validation-convincing-game (gemma)  Bal.Acc 0.628         AUROC 0.914
  validation-insider-trading (gemma)  Bal.Acc 0.500         AUROC 0.377  ← probe ANTI-correlated
  validation-soft-trigger ×4          Bal.Acc 0.500 (each)  AUROC 0.90–0.91  ← signal there, 0 fire
```

This is the §6 finding made undeniable: **4 of 12 datasets sit at exactly 0.5 balanced accuracy
with AUROC ~0.91** — the gemma probe ranks deception well on the soft-trigger sets but its raw
scores never cross 0.5, so the binary call fires on nothing. A calibrated/percentile threshold
would lift those four from 0.50 toward ~0.8+ — by far the biggest available gain, and it costs
nothing (no probe change, no extra compute). **`insider-trading` is the exception**: AUROC 0.377
means the instructed-deception direction is genuinely *inverted* for that scenario — a calibration
fix can't save it; it needs a different/ensemble probe or a scenario-aware sign. Headline lesson for
blue-teamers: **a probe trained on one scenario/base transfers in *ranking* (AUROC) far better than
in *thresholded* decisions (balanced accuracy), and the leaderboard ranks on the latter — so
calibration of the binary call is as important as the probe itself.**

### 🟩 On the leaderboard — final result

Team **`aletheia-blue`** (NDIF key `123`), `mean_probe_fast.ipynb`:

```
MEAN  Bal.Acc 0.6796   AUROC 0.8989   Recall 0.3729   FPR 0.0136     runtime 8m 32s  (12 full datasets)
```

End-to-end success: the efficient form of an existing probe — **one NDIF session + early-stop at
layer 26** (runs 44% of a 62-layer model, skips the LM head) + correct `index,deceptive,score`
contract + faithful standardized scoring — scored all 12 datasets (≈20k conversations total) in
**under 9 minutes**. The headline `0.68` is held back almost entirely by the binary-threshold
calibration gap (5 of 12 datasets degenerate to 0.5 despite AUROC ~0.9), not by the probe's signal
or the runtime — see the calibration discussion above. That's the single clear next lever to climb.

---

## 7. Misc smaller things

- 🟧 `submit.py` requires `--space-url`/`$ALETHEIA_SPACE_URL` for a real submit, but the URL
  isn't in the repo anywhere (the README links the Space *page* but the submit endpoint base
  URL isn't spelled out as the value to pass). 💡 Either default `--space-url` to the known
  Space, or print the exact value to use in the README's submit step.
- 🟩 The `submit.py` UX (banner, spinner, per-dataset metric table, "no attempt spent on a
  rejected package") is polished and reassuring.
- 🟧 `submissions/` ships `example.ipynb`; the moment you add your own notebook you have two
  `.ipynb` and the package is rejected. This is documented, but a `--dry`/submit that detects
  "you left example.ipynb in place" and says so by name would be friendlier than the generic
  "exactly one notebook" error. (I removed `example.ipynb`.)
