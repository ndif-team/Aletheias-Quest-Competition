# Datasets — layout & how to (re)build them

How the eval datasets on the [`aletheias-quest`](https://huggingface.co/aletheias-quest)
HF org are organized, and the exact procedure to regenerate them when the
**source** datasets change. The `runner.yaml` / `dry.yaml` dataset lists point at
the *derived* datasets this process produces, so re-run this whenever the sources
are updated and then refresh those configs (last section).

## The three sources

The org holds three **source** datasets — leave these untouched, they are the
inputs to the conversion:

| Source                       | Holds                |
|------------------------------|----------------------|
| `aletheias-quest/dev`        | dev data             |
| `aletheias-quest/dev-test`   | dev-test eval data   |
| `aletheias-quest/validation` | validation eval data |

Each source is a single repo with **one or more config subsets** — each subset is
a **task** (e.g. `instructed-deception`, `varied-deception`, `lie-auditors-sd`,
`soft-trigger`). Do **not** hardcode the subset list; discover it from the repo's
parquet files, because it changes between updates. Every row carries at least
these columns: `model`, `lora`, `index`, `messages`, `deceptive` (+ task-specific
extras like `temperature`, `meta`, `canary`).

## The derived layout (what we build)

The competition needs **one dataset per `(task, model, lora)` combination**, with
its labels split off into a companion repo:

For every distinct `(source, task, model, lora)` in the sources, create:

1. **Main** `aletheias-quest/<name>` — all source columns **except `deceptive`**,
   single `test` split.
2. **Labels** `aletheias-quest/<name>-labels` — exactly two columns, `index` and
   `deceptive`, single `test` split.

Predictions join to labels on `index` (this is the `id_column` in the configs).

- **Both repos are private.**
- Both are added to a **collection** named `<source>-<task>` (private). A
  collection therefore contains all the mains *and* labels for one source/task.

### Naming

```
<name> = <source>-<task>-<model_base>-<lora_base>
```

- `model_base` = the part of the model id after the last `/`
  (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` → `NVIDIA-Nemotron-3-Super-120B-A12B-BF16`).
- `lora_base` = `None` when the row's `lora` is null, else the part after the last `/`.

**HF repo names are capped at 96 chars.** The cap must hold for the *labels* repo
too, i.e. `len(<name> + "-labels") <= 96`. When a name is too long, shorten the
**lora** segment (only as needed), in this order:

1. If `lora_base` starts with `model_base` (the lora repeats the model name), strip
   that redundant prefix.
2. Then drop the substring `-model-organism`.
3. Assert the result fits; if it still doesn't, stop and pick a new rule rather
   than silently truncating.

The same shortened `<name>` is used for both the main and the `-labels` repo, so
the `<name>` → `<name>-labels` convention always holds (the config generator and
any tooling rely on appending `-labels`).

## Rebuild procedure

Prereqs:

- An HF token with **write** access to the `aletheias-quest` org
  (`hf auth whoami` should list it). The datasets are private, so the token is
  also needed to *read* them back.
- A Python env with `huggingface_hub`, `datasets`, and `pyarrow`. (Some subsets
  have a declared-vs-actual schema mismatch — e.g. a `meta` column declared `null`
  but holding strings — which makes `datasets.load_dataset` choke. We sidestep
  that by reading the parquet files directly with pyarrow.)

The script below is **idempotent and self-discovering**: it lists each source's
subsets and `(model, lora)` combos from the parquet files, (over)writes every
derived dataset, fixes collection membership, then **deletes orphans** (derived
datasets/collection items that are no longer produced — e.g. a task or lora that
the source dropped) and removes now-empty collections. `push_to_hub` fully
replaces a repo's data, so dropped rows are handled by the overwrite; only whole
combos that disappear need deleting.

```python
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from datasets import Dataset

api = HfApi(); ORG = "aletheias-quest"
SOURCES = ["dev", "dev-test", "validation"]

def base(x):
    return "None" if x is None else x.split("/")[-1]

def make_name(src, task, model, lora):
    mb, lb = base(model), base(lora)
    n = f"{src}-{task}-{mb}-{lb}"
    if len(n + "-labels") > 96:                      # keep room for the -labels repo
        if lb.startswith(mb):                        # 1) lora repeats the model name
            lb = lb[len(mb):].lstrip("-")
        lb = lb.replace("-model-organism", "")       # 2) drop a known redundant token
        n = f"{src}-{task}-{mb}-{lb}"
    assert len(n + "-labels") <= 96, f"still too long: {n}"
    return n

# Prefixes used to recognize *derived* datasets (incl. tasks that may have been
# dropped) so we can find orphans. Add any historical task names here.
KNOWN_TASKS = ["instructed-deception", "varied-deception", "lie-auditors-sd",
               "soft-trigger", "convincing-game", "insider-trading"]
DERIVED_PREFIXES = [f"{ORG}/{s}-{t}-" for s in SOURCES for t in KNOWN_TASKS]

desired = set()
live_collections = []

# --- 1. create / overwrite every derived dataset, populate collections ---
for src in SOURCES:
    files = [f for f in api.list_repo_files(f"{ORG}/{src}", repo_type="dataset")
             if f.endswith(".parquet")]
    tasks = sorted({f.split("/")[0] for f in files if "/" in f})   # discover subsets
    for task in tasks:
        title = f"{src}-{task}"
        col = api.create_collection(title, namespace=ORG, private=True, exists_ok=True)
        live_collections.append(title)
        tf = [f for f in files if f.startswith(task + "/")]
        table = pa.concat_tables(
            [pq.read_table(hf_hub_download(f"{ORG}/{src}", f, repo_type="dataset")) for f in tf])
        models = table.column("model").to_pylist()
        loras = table.column("lora").to_pylist()
        combos = {}
        for i, (m, l) in enumerate(zip(models, loras)):
            combos.setdefault((m, l), []).append(i)
        for (m, l), idx in combos.items():
            rid = f"{ORG}/{make_name(src, task, m, l)}"
            lrid = rid + "-labels"
            sub = table.take(pa.array(idx))
            Dataset(sub.drop_columns(["deceptive"])).push_to_hub(rid,  split="test", private=True)
            Dataset(sub.select(["index", "deceptive"])).push_to_hub(lrid, split="test", private=True)
            api.add_collection_item(col.slug, rid,  item_type="dataset", exists_ok=True)
            api.add_collection_item(col.slug, lrid, item_type="dataset", exists_ok=True)
            desired.add(rid); desired.add(lrid)
        print(f"{title}: {len(combos)} mains (+labels)")

# --- 2. drop stale items from the collections we touched ---
for title in live_collections:
    slug = next(c.slug for c in api.list_collections(owner=ORG) if c.title == title)
    for it in api.get_collection(slug).items:
        if it.item_type == "dataset" and it.item_id not in desired:
            api.delete_collection_item(slug, it.item_object_id)

# --- 3. delete orphan derived datasets ---
existing = [d.id for d in api.list_datasets(author=ORG)
            if any(d.id.startswith(p) for p in DERIVED_PREFIXES)]
for o in sorted(set(existing) - desired):
    api.delete_repo(o, repo_type="dataset", missing_ok=True)
    print("deleted orphan", o)

# --- 4. delete collections whose task no longer exists ---
for c in api.list_collections(owner=ORG):
    if any(c.title == f"{s}-{t}" for s in SOURCES for t in KNOWN_TASKS) \
       and c.title not in live_collections:
        api.delete_collection(c.slug, missing_ok=True)
        print("deleted collection", c.title)
```

> ⚠️ **Destructive.** Step 3/4 delete datasets and collections. When the source
> structure changes a lot, this can be dozens of repos. Before running, diff
> `desired` against `existing` and confirm the orphan list — especially any
> repos you didn't create, which may predate this process.

After it runs, sanity-check: every main has a matching `-labels`, mains lack a
`deceptive` column, labels are exactly `index, deceptive`, all repos are private,
and each collection holds an equal count of mains and labels.

## Updating the runner configs

The dataset lists in [`runner.yaml`](runner.yaml) (the leaderboard run) and
[`../dry.yaml`](../dry.yaml) (local `submit.py --dry` rehearsal) are derived from
the collections. Each entry is one **main** dataset paired with its labels repo:

```yaml
  - name: aletheias-quest/<name>
    labels_uri: aletheias-quest/<name>-labels
    id_column: index
    label_column: deceptive
```

- `dry.yaml` = the **dev** collections (`dev-instructed-deception` +
  `dev-varied-deception`).
- `runner.yaml` = the **dev-test** and **validation** collections.

Regenerate the entries from the collections (don't hand-edit), preserving each
file's comment header above `datasets:`:

```python
from huggingface_hub import HfApi
api = HfApi(); ORG = "aletheias-quest"
slugs = {c.title: c.slug for c in api.list_collections(owner=ORG)}

def entries(titles, include_nemotron=True):
    out = []
    for t in titles:
        items = [i.item_id for i in api.get_collection(slugs[t]).items if i.item_type == "dataset"]
        for d in sorted(m for m in items if not m.endswith("-labels")):
            if not include_nemotron and "nemotron" in d.lower():
                continue
            out += [f"  - name: {d}",
                    f"    labels_uri: {d}-labels",
                    f"    id_column: index",
                    f"    label_column: deceptive"]
    return "\n".join(out) + "\n"

# dry.yaml    -> entries(["dev-instructed-deception", "dev-varied-deception"])
# runner.yaml -> entries(["dev-test-instructed-deception", "dev-test-varied-deception",
#                         "validation-lie-auditors-sd", "validation-soft-trigger"])
```

Notes:

- The list-item keys (`labels_uri`, `id_column`, `label_column`) are indented
  **4 spaces** so they align under `name` (which sits at column 4 after `- `).
  6-space indentation is invalid YAML here.
- `include_nemotron` is a knob because we have toggled it before. The Nemotron
  combos are real datasets; include them unless a task explicitly says otherwise.
- Always `yaml.safe_load` the result and assert every `name`/`labels_uri` exists
  on the hub (`api.list_datasets(author=ORG)`) before committing.
