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
# transformers is pinned: nnsight resolves the model architecture locally (meta
# weights) before tracing, so the version must recognise the traced models. 5.10.1
# maps gemma3 for image-text-to-text; leaving it unpinned drifts to a build that
# raises "Unrecognized model" for google/gemma-3-*.
# nnsight is pinned to the competition's hackathon branch (PEFT/LoRA remote support
# + the hackathon NDIF cluster). NOTE: this branch may change before the start.
# git + a C toolchain: the hackathon/peft branch builds a C extension
# (nnsight._c.py_mount) from source, so it needs gcc + headers (build-essential).
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir "transformers==5.10.1" peft \
       "nnsight @ git+https://github.com/ndif-team/nnsight.git@hackathon/peft"

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
