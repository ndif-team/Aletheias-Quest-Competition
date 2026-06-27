"""In-sandbox notebook entrypoint, executed by the per-job venv's Python.

Runs under Landlock + seccomp + rlimits (applied by the parent's preexec). Pins
nbclient's kernel to *this* interpreter (the venv) via a generated kernelspec, so
the participant's ``requirements.txt`` packages are importable.

    <venv>/bin/python <path-to-this-file> <notebook.ipynb>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    nb_path = sys.argv[1]
    cell_timeout = int(os.environ.get("ALETHEIA_CELL_TIMEOUT", "600"))

    # A kernelspec whose argv is THIS python (the venv) → participant deps load.
    jupyter_dir = Path(os.environ["ALETHEIA_JUPYTER"])
    ks_dir = jupyter_dir / "kernels" / "job"
    ks_dir.mkdir(parents=True, exist_ok=True)
    (ks_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "job", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(jupyter_dir)

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(
        nb, timeout=cell_timeout, kernel_name="job",
        resources={"metadata": {"path": os.getcwd()}},
    ).execute()
    return 0


if __name__ == "__main__":
    sys.exit(main())
