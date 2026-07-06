# HuggingFace Docker Space image for the Aletheia's Quest leaderboard.
# CPU-only: heavy model compute runs remotely on NDIF.
# Python 3.12+ is required by the NDIF client (the server rejects older clients).
FROM python:3.12-slim

# Non-root user (HF Spaces convention) with a writable home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

WORKDIR /app
COPY . /app

# Install the runner + web extras, then register a "python3" Jupyter kernel
# so nbclient can launch notebooks.
RUN pip install --no-cache-dir ".[space]" \
    && python -m ipykernel install --name python3 --display-name python3

# Pre-install the competition's required heavy deps (CPU torch + nnsight) so
# participant notebooks inherit them via the per-job venv (--system-site-packages)
# instead of reinstalling ~1GB per job. Heavy compute still runs remotely on NDIF.
# transformers is pinned to a release: nnsight resolves the model architecture
# locally on meta before tracing, so the version must recognise the traced models.
# 5.12.1 recognises Nemotron-3-Super-120B / Qwen3.5-27B / gemma-3-27b-it (verified
# via `--dry`). NOTE: the soft-trigger LoRA-on-sharded-120B failure was NOT a
# transformers issue and NOT fixed by transformers main (main's WeightConverter also
# lacks the kwarg) — it's peft: keep the NDIF *workers* on peft<0.19 (peft 0.19.x
# calls transformers' WeightConverter with a distributed_operation kwarg no
# transformers release or main accepts). The WeightConverter fix itself lives on the
# workers (the client only resolves on meta; the workers do the remote weight loading),
# but we pin the client to peft<0.19 too so client and workers stay on the same peft line.
# nnsight is pinned to the competition's hackathon branch (PEFT/LoRA remote support
# + the hackathon NDIF cluster). Pinned to an explicit commit SHA (the HEAD of
# hackathon/peft as of this deploy) rather than the bare branch name: it guarantees
# the newest commit is installed AND changes this line so the Docker layer cache is
# busted on rebuild (a bare @hackathon/peft could otherwise be served from a stale
# cached layer). Re-pin to the branch's new HEAD to pull later updates.
# git + a C toolchain: the hackathon/peft branch builds a C extension
# (nnsight._c.py_mount) from source, so it needs gcc + headers (build-essential).
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir "transformers==5.12.1" "peft<0.19" \
       "nnsight @ git+https://github.com/ndif-team/nnsight.git@e4f37fa3e7c6bb936b4a37e2b5402dca2330610c"

# RESULTS_URI is left unset so runner.yaml's bucket:// results store is used.
# (Set RESULTS_URI=/data/results.jsonl + attach a volume to use local storage instead.)
ENV RUNNER_CONFIG=/app/runner.yaml \
    PORT=7860
# /data holds the predownloaded input cache (runner.yaml cache_dir). On this Space
# /data is a PERSISTENT HF bucket mount, so the cache survives rebuilds/restarts —
# to refresh it after re-pushing a dataset, clear the bucket's cache/ prefix
# (see leaderboard-dev-storage), not a rebuild.
RUN mkdir -p /data && chown -R user /app /data
USER user

EXPOSE 7860
CMD ["uvicorn", "aletheia_runner.app:app", "--host", "0.0.0.0", "--port", "7860"]
