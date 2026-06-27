import pandas as pd
import pytest

from aletheia_runner import scoring
from aletheia_runner.config import DatasetConfig

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _preds(ids, p):
    return pd.DataFrame({"id": ids, "prediction": p})


def test_auroc_perfect_separation():
    labels = scoring.load_labels(DatasetConfig(
        name="x", labels_uri=str(FIXTURES / "labels.csv")))
    preds = _preds([0, 1, 2, 3, 4], [0.9, 0.8, 0.2, 0.1, 0.6])
    assert scoring.score(preds, labels, "auroc") == 1.0


def test_accuracy_metric():
    labels = scoring.load_labels(DatasetConfig(
        name="x", labels_uri=str(FIXTURES / "labels.csv")))
    preds = _preds([0, 1, 2, 3, 4], [0.9, 0.8, 0.2, 0.1, 0.6])
    assert scoring.score(preds, labels, "accuracy") == 1.0


def test_out_of_range_predictions_rejected(tmp_path):
    p = tmp_path / "submission.csv"
    _preds([0, 1], [1.4, 0.2]).to_csv(p, index=False)
    with pytest.raises(scoring.ScoringError):
        scoring.load_predictions(p)


def test_missing_id_fails_alignment():
    labels = pd.DataFrame({"id": [0, 1, 2], "label": [1, 0, 1]})
    preds = _preds([0, 1, 9], [0.9, 0.1, 0.5])  # id 9 not in labels, 2 missing
    with pytest.raises(scoring.ScoringError):
        scoring.align(preds, labels)


def test_unknown_metric():
    labels = pd.DataFrame({"id": [0, 1], "label": [1, 0]})
    with pytest.raises(scoring.ScoringError):
        scoring.score(_preds([0, 1], [0.9, 0.1]), labels, "f1-but-not-registered")
