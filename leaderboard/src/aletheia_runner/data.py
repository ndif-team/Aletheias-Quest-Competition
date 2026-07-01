"""Predownload dataset INPUTS (and LoRA adapters) so sandboxed notebooks load
them offline / from cache, no large writes.

The trusted parent, on first run, builds each dataset's **Arrow cache** under
``cache_dir/datasets`` via ``load_dataset(cache_dir=...)``. A prebuilt Arrow cache
is self-contained, so per job the child gets a **writable copy** plus
``HF_DATASETS_OFFLINE`` and loads it with no token — that's what keeps the private
eval set readable without exposing it (labels are scored separately by the parent
and never enter this cache).

The parent ALSO predownloads every LoRA adapter referenced by the datasets (their
``lora`` column) into an HF hub cache under ``cache_dir/hf_hub``. The parent runs
without rlimits, so it can write multi-GB adapter files; the sandboxed child is
capped by ``RLIMIT_FSIZE`` and would hit ``OSError: [Errno 27] File too large``
downloading them itself. Per job we seed the child's ``HF_HUB_CACHE`` with symlinks
to those predownloaded repos, so when nnsight/peft loads an adapter it's a cache
hit — served from the read-only predownload, no re-download, no large write.

Base model configs/tokenizers are NOT pre-cached: the HF hub stays reachable
through the egress proxy, so notebooks fetch them live (using the submitter's
forwarded ``HF_TOKEN`` for gated repos); weights run on NDIF.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SPLIT, RunnerConfig


@dataclass
class DataLayout:
    """Predownloaded inputs: dataset Arrow cache (copied writable per job) plus a
    shared, read-only HF hub cache holding the LoRA adapters (symlinked per job)."""

    datasets_cache: Path
    hub_cache: Path | None = None

    def child_env(self, job_scratch: Path, offline: bool = True) -> dict[str, str]:
        """Env for a sandboxed child to load the inputs.

        ``offline=True`` (confined server) forces the datasets library offline so
        the private eval set loads from the predownloaded Arrow copy with no token.
        The HF *hub* is left online (reachable via the egress proxy) so notebooks
        can fetch model configs live and validate the (cache-hit) adapters.
        ``offline=False`` (``--dry``) forces nothing.
        """
        scratch = Path(job_scratch)
        ds_copy = scratch / "hf_datasets_cache"
        if ds_copy.exists():
            shutil.rmtree(ds_copy)
        shutil.copytree(self.datasets_cache, ds_copy)  # datasets needs a writable lock

        # Seed the child's HF hub cache with the predownloaded adapter repos. We
        # symlink each ``models--*`` dir (rather than copy: adapters are multi-GB,
        # up to ~40GB) into the per-job writable cache. The child reads the adapter
        # through the symlink (RO target, Landlock-allowed) — a cache hit, so no
        # download and no large write — while live base-config fetches still write
        # into the writable cache dir itself.
        hub_copy = scratch / "hf_hub_cache"
        if hub_copy.exists():
            shutil.rmtree(hub_copy)
        hub_copy.mkdir(parents=True, exist_ok=True)
        if self.hub_cache and Path(self.hub_cache).exists():
            for entry in Path(self.hub_cache).iterdir():
                if entry.name.startswith("models--"):
                    link = hub_copy / entry.name
                    if not link.exists():
                        os.symlink(entry, link)

        env = {
            "HF_DATASETS_CACHE": str(ds_copy),
            "HF_HUB_CACHE": str(hub_copy),  # writable; live model dl + seeded adapters
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Force the classic HTTP download path. The hf_xet (Rust) client doesn't
            # route all its sockets through HTTPS_PROXY, so under the loopback-only
            # seccomp egress gate its direct connects stall and tokenizer/model
            # downloads hang. Classic HTTPS honours the proxy and works.
            "HF_HUB_DISABLE_XET": "1",
            # HF model/tokenizer/LoRA-adapter downloads go through the egress proxy
            # in MITM mode, which adds latency — the 10s default read budget trips a
            # ReadTimeout at model-load (intermittently). Give it generous budgets.
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "HF_HUB_ETAG_TIMEOUT": "60",
        }
        if offline:
            env["HF_DATASETS_OFFLINE"] = "1"      # private data from cache, no token
        return env


def prepare_inputs(config: RunnerConfig) -> DataLayout:
    """Build the dataset Arrow cache and predownload LoRA adapters (in-process,
    in the trusted parent). Idempotent via a marker."""
    datasets_cache = Path(config.cache_dir) / "datasets"
    hub_cache = Path(config.cache_dir) / "hf_hub"
    marker = Path(config.cache_dir) / ".prepared"
    layout = DataLayout(datasets_cache=datasets_cache, hub_cache=hub_cache)

    # Content-addressed marker: re-prepare only when the dataset set changes.
    key = json.dumps(sorted(d.name for d in config.datasets), sort_keys=True)
    if marker.exists() and marker.read_text() == key:
        return layout

    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    # Load each eval dataset once (builds the Arrow cache) and, in the same pass,
    # collect the distinct LoRA adapter repos it references (its ``lora`` column).
    # The labels dataset is never loaded here — labels must not enter the cache.
    loras: set[str] = set()
    for cfg in config.datasets:
        ds = load_dataset(cfg.name, split=SPLIT, cache_dir=str(datasets_cache),
                          token=config.hf_token)
        if "lora" in ds.column_names:
            for value in ds.unique("lora"):
                if isinstance(value, str) and "/" in value:
                    loras.add(value)

    # Predownload every referenced adapter into the shared hub cache. The parent
    # has no RLIMIT_FSIZE, so multi-GB adapters download fine here; the sandboxed
    # child then loads them from cache (no large write). snapshot_download is
    # idempotent — an already-cached, unchanged adapter is a no-op.
    from huggingface_hub import snapshot_download
    for repo in sorted(loras):
        snapshot_download(repo, cache_dir=str(hub_cache), token=config.hf_token)

    marker.write_text(key)
    return layout
