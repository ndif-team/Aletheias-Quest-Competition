"""Scoring: load predictions + private labels, align by id, compute the metric.

The metric is intentionally swappable (``METRICS`` registry) — AUROC is the
current placeholder. Label loading is isolated here so notebook code never
touches it (Phase 1: in-process; Phase 4: separate process/credential).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import SPLIT, DatasetConfig


class ScoringError(ValueError):
    """Raised when a submission can't be scored (bad format, misaligned ids)."""


# --- metric registry ------------------------------------------------------

def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score

    return float(accuracy_score(y_true, (y_score >= 0.5).astype(int)))


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": _auroc,
    "accuracy": _accuracy,
}


# --- predictions ----------------------------------------------------------

def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a submission.csv and validate its shape."""
    p = Path(path)
    if not p.exists():
        raise ScoringError(f"no submission.csv was produced at {p}")
    df = pd.read_csv(p)
    missing = {"id", "prediction"} - set(df.columns)
    if missing:
        raise ScoringError(f"submission.csv missing columns: {sorted(missing)}")
    df = df[["id", "prediction"]].copy()
    if df["prediction"].isna().any():
        raise ScoringError("submission.csv has NaN predictions")
    if df["id"].duplicated().any():
        raise ScoringError("submission.csv has duplicate ids")
    lo, hi = df["prediction"].min(), df["prediction"].max()
    if lo < 0 or hi > 1:
        raise ScoringError(f"predictions must be in [0, 1] (got [{lo}, {hi}])")
    return df


# --- labels ---------------------------------------------------------------

def load_labels(cfg: DatasetConfig, hf_token: str | None = None) -> pd.DataFrame:
    """Load held-out labels for a dataset config as a DataFrame[id, label].

    ``labels_uri`` is a local ``.csv`` path or an HF dataset id.
    """
    uri = cfg.labels_uri
    if uri.endswith(".csv") and Path(uri).exists():
        raw = pd.read_csv(uri)
    else:
        from datasets import load_dataset

        ds = load_dataset(uri, split=SPLIT, token=hf_token)
        raw = ds.to_pandas()

    if cfg.label_column not in raw.columns:
        raise ScoringError(f"labels missing column {cfg.label_column!r}")
    if cfg.id_column in raw.columns:
        ids = raw[cfg.id_column].to_numpy()
    else:  # fall back to row order (label-free eval inputs are 0..N-1)
        ids = np.arange(len(raw))
    return pd.DataFrame({"id": ids, "label": raw[cfg.label_column].astype(int).to_numpy()})


# --- alignment + scoring --------------------------------------------------

def align(preds: pd.DataFrame, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Join predictions to labels by id; fail loudly on mismatch."""
    if len(preds) != len(labels):
        raise ScoringError(
            f"prediction count {len(preds)} != label count {len(labels)}")
    merged = labels.merge(preds, on="id", how="left")
    if merged["prediction"].isna().any():
        n = int(merged["prediction"].isna().sum())
        raise ScoringError(f"{n} label ids had no matching prediction")
    return merged["label"].to_numpy(), merged["prediction"].to_numpy()


def score(preds: pd.DataFrame, labels: pd.DataFrame, metric: str = "auroc") -> float:
    if metric not in METRICS:
        raise ScoringError(f"unknown metric {metric!r}; have {sorted(METRICS)}")
    y_true, y_score = align(preds, labels)
    return METRICS[metric](y_true, y_score)
