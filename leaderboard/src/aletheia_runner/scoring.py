"""Scoring: load predictions + private labels, align by index, compute metrics.

A submission reports, per row: ``index`` (the join key), ``deceptive`` (the
participant's binary call) and ``score`` (a continuous value in ``[0, 1]``). We
report four metrics per dataset:

- **balanced_accuracy**, **recall**, **fpr** — from the binary ``deceptive`` column
  (the participant's own threshold), with "deceptive" as the positive class.
- **auroc** — from the continuous ``score`` column.

Label loading is isolated here so notebook code never touches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (METRIC_KEYS, PRIMARY_METRIC, SECONDARY_METRIC,  # noqa: F401
                     SPLIT, DatasetConfig)


class ScoringError(ValueError):
    """Raised when a submission can't be scored (bad format, misaligned ids)."""


# --- metrics --------------------------------------------------------------

def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """(tp, tn, fp, fn) with "deceptive" (== 1) as the positive class."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    tp, _, _, fn = _confusion(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) else None          # TP / (TP + FN)


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    _, tn, fp, _ = _confusion(y_true, y_pred)
    return fp / (fp + tn) if (fp + tn) else None          # FP / (FP + TN)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    tpr = _recall(y_true, y_pred)                          # = sensitivity
    fpr = _false_positive_rate(y_true, y_pred)
    if tpr is None or fpr is None:                         # only one class present
        return None
    return (tpr + (1.0 - fpr)) / 2.0                       # (sensitivity + specificity) / 2


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:                         # AUROC undefined on one class
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def _round(x: float | None) -> float | None:
    return None if x is None else float(x)


# --- predictions ----------------------------------------------------------

def _coerce_binary(s: pd.Series) -> pd.Series:
    """Coerce a ``deceptive`` column to integer 0/1, accepting bool, 0/1, or
    true/false strings. Raises ScoringError on anything else."""
    if s.isna().any():
        raise ScoringError("submission.csv has NaN deceptive values")
    if s.dtype == bool:
        return s.astype(int)
    if np.issubdtype(s.dtype, np.number):
        vals = set(pd.unique(s))
        if not vals <= {0, 1, 0.0, 1.0}:
            raise ScoringError("deceptive must be true/false (or 0/1)")
        return s.astype(int)
    t = s.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0, "t": 1, "f": 0}
    if not t.isin(mapping).all():
        raise ScoringError("deceptive must be true/false (or 0/1)")
    return t.map(mapping).astype(int)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read a submission.csv and validate its shape: ``index, deceptive, score``."""
    p = Path(path)
    if not p.exists():
        raise ScoringError(f"no submission.csv was produced at {p}")
    df = pd.read_csv(p)
    missing = {"index", "deceptive", "score"} - set(df.columns)
    if missing:
        raise ScoringError(f"submission.csv missing columns: {sorted(missing)}")
    df = df[["index", "deceptive", "score"]].copy()
    if df["index"].duplicated().any():
        raise ScoringError("submission.csv has duplicate index values")

    df["deceptive"] = _coerce_binary(df["deceptive"])

    score = pd.to_numeric(df["score"], errors="coerce")
    if score.isna().any():
        raise ScoringError("submission.csv has missing/non-numeric score values")
    lo, hi = float(score.min()), float(score.max())
    if lo < 0 or hi > 1:
        raise ScoringError(f"score must be in [0, 1] (got [{lo}, {hi}])")
    df["score"] = score
    return df


# --- labels ---------------------------------------------------------------

def load_labels(cfg: DatasetConfig, hf_token: str | None = None) -> pd.DataFrame:
    """Load held-out labels for a dataset config as a DataFrame[index, label].

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
    return pd.DataFrame({"index": ids,
                         "label": raw[cfg.label_column].astype(int).to_numpy()})


# --- alignment + scoring --------------------------------------------------

def align(preds: pd.DataFrame, labels: pd.DataFrame, *, partial: bool = False) -> pd.DataFrame:
    """Join predictions to labels by ``index``; fail loudly on mismatch.

    Returns the merged frame with columns ``label, deceptive, score``.

    ``partial=True`` (local ``--dry`` only) scores just the rows the submission
    predicted — an inner join on ``index`` — so a capped rehearsal (e.g. ``--limit
    N``) still yields a score on its N rows. The real leaderboard uses ``partial=
    False``: a submission MUST predict every row (no skipping easy ones).
    """
    if partial:
        merged = labels.merge(preds, on="index", how="inner")
        if merged.empty:
            raise ScoringError("no predicted index values matched the labels")
        return merged
    if len(preds) != len(labels):
        raise ScoringError(
            f"prediction count {len(preds)} != label count {len(labels)}")
    merged = labels.merge(preds, on="index", how="left")
    if merged["score"].isna().any() or merged["deceptive"].isna().any():
        n = int(merged["score"].isna().sum())
        raise ScoringError(f"{n} label index values had no matching prediction")
    return merged


def compute_metrics(preds: pd.DataFrame, labels: pd.DataFrame, *,
                    partial: bool = False) -> dict[str, float | None]:
    """All four metrics for a produced submission against held-out labels.

    Binary metrics use ``deceptive``; AUROC uses the continuous ``score``. A metric
    is ``None`` when it's undefined (e.g. AUROC with only one label class present).
    ``partial`` is forwarded to :func:`align` (score only predicted rows, --dry only).
    """
    merged = align(preds, labels, partial=partial)
    y_true = merged["label"].to_numpy()
    y_pred = merged["deceptive"].to_numpy()
    y_score = merged["score"].to_numpy()
    return {
        "balanced_accuracy": _round(_balanced_accuracy(y_true, y_pred)),
        "auroc": _round(_auroc(y_true, y_score)),
        "recall": _round(_recall(y_true, y_pred)),
        "fpr": _round(_false_positive_rate(y_true, y_pred)),
    }
