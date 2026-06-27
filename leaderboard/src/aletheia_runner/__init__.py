"""Aletheia's Quest competition runner.

Core, web-free building blocks for the leaderboard Space:

- ``config``   — what to run on and where labels/results live
- ``executor`` — run submission notebooks end-to-end, collect submission.csv
- ``scoring``  — align predictions with held-out labels, compute the metric
- ``results``  — result records + the leaderboard store
- ``pipeline`` — unpack → run → score → records
"""

from .config import DatasetConfig, RunnerConfig
from .pipeline import run_pipeline, run_zip
from .results import ResultRecord, ResultStore, make_store

__all__ = [
    "DatasetConfig", "RunnerConfig",
    "run_pipeline", "run_zip",
    "ResultRecord", "ResultStore", "make_store",
]
