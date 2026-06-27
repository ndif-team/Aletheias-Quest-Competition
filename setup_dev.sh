#!/usr/bin/env bash
# Create a local development environment for Aletheia's Quest.
#
# Builds a Python venv at ./.venv and installs requirements-dev.txt into it —
# everything needed to develop your method against the public dev data and to run
# `python submit.py --dry` (the real leaderboard pipeline, locally).
#
#   ./setup_dev.sh            # create ./.venv and install
#   source .venv/bin/activate # then work in it
#   python submit.py --dry
#
# Prefer conda? The equivalent is:
#   conda create -n aletheia python=3.12 -y && conda activate aletheia
#   pip install -r requirements-dev.txt
set -euo pipefail

cd "$(dirname "$0")"

# Pick a Python >= 3.10 (the runner targets 3.12). Try common names.
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "error: need Python >= 3.10 on PATH (the runner targets 3.12). Found:" >&2
  command -v python3 python 2>/dev/null | while read -r p; do echo "  $p -> $("$p" --version 2>&1)"; done >&2
  exit 1
fi
echo "Using $("$PY" --version) at $(command -v "$PY")"

VENV="${VENV:-.venv}"
if [ ! -d "$VENV" ]; then
  echo "Creating venv at $VENV ..."
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip
echo "Installing requirements-dev.txt (this pulls the hackathon nnsight from git) ..."
pip install -r requirements-dev.txt

cat <<'EOF'

✓ Dev environment ready.

  source .venv/bin/activate
  export NDIF_API_KEY="your-ndif-key"     # from your competition signup
  huggingface-cli login                   # so HF model configs/tokenizers load
  python submit.py --dry                  # rehearse on the datasets in dry.yaml

EOF
