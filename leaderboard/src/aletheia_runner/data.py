"""Predownload dataset INPUTS and LoRA adapters so sandboxed notebooks load them
without a large in-sandbox write (which trips RLIMIT_FSIZE -> EFBIG).

Datasets: the trusted parent builds each dataset's Arrow cache under
``cache_dir/datasets`` via ``load_dataset``; per job the child gets a writable
copy plus ``HF_DATASETS_OFFLINE`` and loads it with no token (keeps the private
eval set readable without exposing it — labels are scored separately).

LoRA adapters: the child would hit ``OSError: [Errno 27] File too large``
downloading multi-GB adapter safetensors itself. So the parent predownloads each
adapter as **flat real files** under ``cache_dir/adapters/<repo>`` (real files,
not the HF cache's symlink layout — the persistent store may be a bucket mount
that doesn't preserve symlinks) plus a small manifest of the commit sha + per-file
etags. Per job we then reconstruct a valid HF hub cache in the child's (symlink-
capable) scratch: ``blobs/<etag>`` symlinks point at the flat files and
``snapshots/<sha>/<file>`` symlinks point at the blobs. When peft/nnsight loads
the adapter by repo id it's a cache hit — served from the reconstructed cache, no
download, no large write. The hub stays online for live base-config fetches.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SPLIT, RunnerConfig

_MANIFEST = "_aletheia_manifest.json"


@dataclass
class DataLayout:
    """Predownloaded inputs: dataset Arrow cache (copied writable per job) plus the
    flat per-repo adapter downloads (reconstructed into an HF cache per job)."""

    datasets_cache: Path
    adapters_dir: Path | None = None

    def child_env(self, job_scratch: Path, offline: bool = True) -> dict[str, str]:
        """Env for a sandboxed child to load the inputs. ``offline`` forces the
        datasets library offline (private data from the copied cache, no token);
        the HF *hub* stays online so notebooks fetch model configs live and the
        reconstructed adapter cache validates as a cache hit."""
        scratch = Path(job_scratch)
        ds_copy = scratch / "hf_datasets_cache"
        if ds_copy.exists():
            shutil.rmtree(ds_copy)
        shutil.copytree(self.datasets_cache, ds_copy)  # datasets needs a writable lock

        # Reconstruct the HF hub cache for each predownloaded adapter, in the
        # child's writable (symlink-capable) scratch, pointing at the flat files on
        # the (possibly symlink-less) persistent store. Cheap: a handful of symlinks.
        hub_copy = scratch / "hf_hub_cache"
        if hub_copy.exists():
            shutil.rmtree(hub_copy)
        hub_copy.mkdir(parents=True, exist_ok=True)
        if self.adapters_dir and Path(self.adapters_dir).is_dir():
            for adir in sorted(Path(self.adapters_dir).iterdir()):
                man = adir / _MANIFEST
                if not man.is_file():
                    continue
                try:
                    _reconstruct_cache_entry(hub_copy, adir, json.loads(man.read_text()))
                except Exception as exc:  # noqa: BLE001 - one bad adapter mustn't break the job
                    print(f"[child_env] adapter cache reconstruct failed for "
                          f"{adir.name}: {exc}", flush=True)

        env = {
            "HF_DATASETS_CACHE": str(ds_copy),
            "HF_HUB_CACHE": str(hub_copy),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Classic HTTP download path (hf_xet doesn't route through the proxy).
            "HF_HUB_DISABLE_XET": "1",
            # Downloads go through the MITM egress proxy — give generous budgets.
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_HUB_ETAG_TIMEOUT": "60",
        }
        if offline:
            env["HF_DATASETS_OFFLINE"] = "1"
        return env


def _reconstruct_cache_entry(hub_copy: Path, adir: Path, manifest: dict) -> None:
    """Build ``models--org--repo/{refs,blobs,snapshots}`` under ``hub_copy`` from a
    flat adapter download + its manifest, using symlinks to the flat files."""
    repo = manifest["repo"]
    sha = manifest["sha"]
    etags: dict[str, str] = manifest["etags"]
    org, name = repo.split("/", 1)
    root = hub_copy / f"models--{org}--{name}"
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(sha)
    blobs = root / "blobs"
    blobs.mkdir(exist_ok=True)
    snap = root / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    for rel, etag in etags.items():
        src = adir / rel                       # real file on the persistent store
        if not src.exists():
            continue
        blob = blobs / etag
        if not blob.exists():
            os.symlink(src, blob)              # blob -> flat file
        ptr = snap / rel
        ptr.parent.mkdir(parents=True, exist_ok=True)
        if not ptr.exists():
            os.symlink(os.path.relpath(blob, ptr.parent), ptr)  # snapshot -> blob


def _predownload_adapter(repo: str, adapters_dir: Path, token: str | None) -> None:
    """Download an adapter repo as flat real files + write a manifest (commit sha +
    per-file etags) so it can be reconstructed into an HF cache per job."""
    from huggingface_hub import HfApi, snapshot_download

    local = adapters_dir / repo.replace("/", "__")
    snapshot_download(repo, local_dir=str(local), token=token)
    rels = [
        str(p.relative_to(local).as_posix())
        for p in local.rglob("*")
        if p.is_file() and ".cache/" not in p.relative_to(local).as_posix()
        and p.name != _MANIFEST
    ]
    api = HfApi()
    sha = api.model_info(repo, token=token).sha
    etags = {}
    for pi in api.get_paths_info(repo, rels, repo_type="model", token=token):
        lfs = getattr(pi, "lfs", None)
        etags[pi.path] = lfs.sha256 if lfs else pi.blob_id
    (local / _MANIFEST).write_text(json.dumps({"repo": repo, "sha": sha, "etags": etags}))


def prepare_inputs(config: RunnerConfig) -> DataLayout:
    """Build the dataset Arrow cache and predownload LoRA adapters (in-process, in
    the trusted parent). Idempotent via a marker."""
    cache = Path(config.cache_dir)
    datasets_cache = cache / "datasets"
    adapters_dir = cache / "adapters"
    marker = cache / ".prepared"
    layout = DataLayout(datasets_cache=datasets_cache, adapters_dir=adapters_dir)

    key = json.dumps(sorted(d.name for d in config.datasets), sort_keys=True)
    if marker.exists() and marker.read_text() == key:
        return layout

    datasets_cache.mkdir(parents=True, exist_ok=True)
    adapters_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    # Load each eval dataset once (builds the Arrow cache) and, in the same pass,
    # collect the distinct LoRA adapter repos it references. The labels dataset is
    # never loaded here.
    loras: set[str] = set()
    for cfg in config.datasets:
        ds = load_dataset(cfg.name, split=SPLIT, cache_dir=str(datasets_cache),
                          token=config.hf_token)
        if "lora" in ds.column_names:
            for value in ds.unique("lora"):
                if isinstance(value, str) and "/" in value:
                    loras.add(value)

    # Predownload each adapter. Fail-safe: a failed adapter degrades only THAT
    # dataset (child downloads it live / hits FSIZE), never the whole submission;
    # skip the marker so the next run retries.
    all_ok = True
    for repo in sorted(loras):
        try:
            _predownload_adapter(repo, adapters_dir, config.hf_token)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, never abort
            all_ok = False
            print(f"[prepare_inputs] adapter predownload failed for {repo}: {exc}",
                  flush=True)

    if all_ok:
        marker.write_text(key)
    return layout
