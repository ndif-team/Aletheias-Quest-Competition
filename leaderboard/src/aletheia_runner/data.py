"""Predownload dataset INPUTS so sandboxed notebooks load them offline, no token.

The trusted parent, on first run, builds each dataset's **Arrow cache** under
``cache_dir/datasets`` via ``load_dataset(cache_dir=...)``. A prebuilt Arrow cache
is self-contained, so per job the child gets a **writable copy** plus
``HF_DATASETS_OFFLINE`` and loads it with no token — that's what keeps the private
eval set readable without exposing it (labels are scored separately by the parent
and never enter this cache).

Model configs/tokenizers are NOT pre-cached: the HF hub is reachable through the
egress proxy, so notebooks fetch them live (using the submitter's forwarded
``HF_TOKEN`` for gated repos / higher rate limits); weights run on NDIF.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import SPLIT, RunnerConfig


@dataclass
class DataLayout:
    """Canonical predownloaded dataset Arrow cache (copied, writable, per job)."""

    datasets_cache: Path

    def child_env(self, job_scratch: Path, offline: bool = True) -> dict[str, str]:
        """Env for a sandboxed child to load the inputs.

        ``offline=True`` (confined server) forces the datasets library offline so
        the private eval set loads from the predownloaded Arrow copy with no token.
        The HF *hub* is left online (reachable via the egress proxy) so notebooks
        can fetch model configs live. ``offline=False`` (``--dry``) forces nothing.
        """
        scratch = Path(job_scratch)
        ds_copy = scratch / "hf_datasets_cache"
        if ds_copy.exists():
            shutil.rmtree(ds_copy)
        shutil.copytree(self.datasets_cache, ds_copy)  # datasets needs a writable lock
        env = {
            "HF_DATASETS_CACHE": str(ds_copy),
            "HF_HUB_CACHE": str(scratch / "hf_hub_cache"),  # writable; live model dl
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Force the classic HTTP download path. The hf_xet (Rust) client doesn't
            # route all its sockets through HTTPS_PROXY, so under the loopback-only
            # seccomp egress gate its direct connects stall and tokenizer/model
            # downloads hang. Classic HTTPS honours the proxy and works.
            "HF_HUB_DISABLE_XET": "1",
        }
        if offline:
            env["HF_DATASETS_OFFLINE"] = "1"      # private data from cache, no token
        return env


def prepare_inputs(config: RunnerConfig) -> DataLayout:
    """Build the dataset Arrow cache (in-process). Idempotent via a marker."""
    datasets_cache = Path(config.cache_dir) / "datasets"
    marker = Path(config.cache_dir) / ".prepared"
    layout = DataLayout(datasets_cache=datasets_cache)

    # Content-addressed marker: re-prepare only when the dataset set changes.
    key = json.dumps(sorted(d.name for d in config.datasets), sort_keys=True)
    if marker.exists() and marker.read_text() == key:
        return layout

    datasets_cache.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    for cfg in config.datasets:
        load_dataset(cfg.name, split=SPLIT, cache_dir=str(datasets_cache),
                     token=config.hf_token)
    marker.write_text(key)
    return layout
