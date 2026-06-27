import pandas as pd
import pytest

from aletheia_runner import scoring
from aletheia_runner.config import DatasetConfig

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def _preds(index, deceptive, score):
    return pd.DataFrame({"index": index, "deceptive": deceptive, "score": score})


def _labels():
    return scoring.load_labels(DatasetConfig(
        name="x", labels_uri=str(FIXTURES / "labels.csv")))   # labels = [1,1,0,0,1]


def test_metrics_all_perfect():
    preds = _preds([0, 1, 2, 3, 4], [True, True, False, False, True],
                   [0.9, 0.8, 0.2, 0.1, 0.6])
    m = scoring.compute_metrics(preds, _labels())
    assert m == {"balanced_accuracy": 1.0, "auroc": 1.0, "recall": 1.0, "fpr": 0.0}


def test_auroc_uses_score_binary_metrics_use_deceptive():
    # `deceptive` is perfect (binary metrics = 1.0) but `score` is fully inverted,
    # so AUROC (which uses score) must be 0.0 — proving the two columns drive
    # different metrics.
    preds = _preds([0, 1, 2, 3, 4], [True, True, False, False, True],
                   [0.1, 0.2, 0.8, 0.9, 0.4])
    m = scoring.compute_metrics(preds, _labels())
    assert m["balanced_accuracy"] == 1.0
    assert m["recall"] == 1.0
    assert m["fpr"] == 0.0
    assert m["auroc"] == 0.0


def test_auroc_none_when_single_class():
    labels = pd.DataFrame({"index": [0, 1, 2], "label": [1, 1, 1]})  # one class
    preds = _preds([0, 1, 2], [True, False, True], [0.9, 0.1, 0.5])
    assert scoring.compute_metrics(preds, labels)["auroc"] is None


def test_out_of_range_score_rejected(tmp_path):
    p = tmp_path / "submission.csv"
    _preds([0, 1], [True, False], [1.4, 0.2]).to_csv(p, index=False)
    with pytest.raises(scoring.ScoringError):
        scoring.load_predictions(p)


def test_missing_column_rejected(tmp_path):
    p = tmp_path / "submission.csv"
    pd.DataFrame({"index": [0, 1], "deceptive": [True, False]}).to_csv(p, index=False)
    with pytest.raises(scoring.ScoringError, match="missing"):
        scoring.load_predictions(p)


def test_non_binary_deceptive_rejected(tmp_path):
    p = tmp_path / "submission.csv"
    _preds([0, 1], [0.3, 0.7], [0.3, 0.7]).to_csv(p, index=False)
    with pytest.raises(scoring.ScoringError, match="true/false"):
        scoring.load_predictions(p)


def test_string_booleans_accepted(tmp_path):
    p = tmp_path / "submission.csv"
    p.write_text("index,deceptive,score\n0,true,0.9\n1,false,0.1\n")
    df = scoring.load_predictions(p)
    assert list(df["deceptive"]) == [1, 0]


def test_missing_index_fails_alignment():
    labels = pd.DataFrame({"index": [0, 1, 2], "label": [1, 0, 1]})
    preds = _preds([0, 1, 9], [True, False, True], [0.9, 0.1, 0.5])  # 9 absent, 2 missing
    with pytest.raises(scoring.ScoringError):
        scoring.align(preds, labels)
