"""Notebook execution: run each submission notebook end-to-end and collect its
``submission.csv``.

Notebooks run with CWD = the submission repo root (so local imports and bundled
files resolve) and a per-run environment. They are run **sequentially**; the
shared ``submission.csv`` is snapshotted right after each notebook, so multiple
notebooks in one submission don't clobber each other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

SUBMISSION_FILENAME = "submission.csv"


@dataclass
class NotebookResult:
    notebook: str                 # path relative to the submission root
    ok: bool
    submission_csv: Path | None   # snapshot location of the produced csv
    error: str | None = None


def list_notebooks(root: Path) -> list[Path]:
    """The submission's notebook(s) under ``submission/``. A submission is **one
    notebook at a time**: more than one ``.ipynb`` is rejected (``ValueError``)
    rather than silently scored together. (Returns an empty list when there are
    none — the caller decides how to report that.)"""
    notebooks = sorted((root / "submission").glob("*.ipynb"))
    if len(notebooks) > 1:
        names = ", ".join(nb.name for nb in notebooks)
        raise ValueError(
            f"submission/ must contain exactly one notebook — found "
            f"{len(notebooks)} ({names}). Submit one notebook at a time.")
    return notebooks


def run_notebook(nb_path: Path, root: Path, env: dict[str, str],
                 timeout: int, snapshot_dir: Path) -> NotebookResult:
    """Execute one notebook; snapshot its submission.csv. Never raises."""
    rel = nb_path.relative_to(root).as_posix()
    stale = root / SUBMISSION_FILENAME
    if stale.exists():
        stale.unlink()

    run_env = {**os.environ, **env}
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    try:
        # The kernel inherits this process's env at launch.
        prev = dict(os.environ)
        os.environ.update(run_env)
        try:
            client.execute()
        finally:
            os.environ.clear()
            os.environ.update(prev)
    except CellExecutionError as e:
        return NotebookResult(rel, ok=False, submission_csv=None,
                              error=f"{e.ename}: {e.evalue}"[:2000])
    except Exception as e:  # timeouts, kernel death, etc.
        return NotebookResult(rel, ok=False, submission_csv=None,
                              error=f"{type(e).__name__}: {e}"[:2000])

    produced = root / SUBMISSION_FILENAME
    if not produced.exists():
        return NotebookResult(rel, ok=False, submission_csv=None,
                              error="notebook did not write submission.csv")

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    safe = rel.replace("/", "__")
    snap = snapshot_dir / f"{safe}.submission.csv"
    snap.write_bytes(produced.read_bytes())
    return NotebookResult(rel, ok=True, submission_csv=snap)


def run_submission(root: Path, env: dict[str, str], timeout: int,
                   snapshot_dir: Path) -> list[NotebookResult]:
    """Run every notebook under ``root/submission`` with the given env."""
    notebooks = list_notebooks(root)
    if not notebooks:
        raise FileNotFoundError("submission/ contains no .ipynb files")
    return [run_notebook(nb, root, env, timeout, snapshot_dir) for nb in notebooks]
