import pytest

from aletheia_runner.results import (BucketResultStore, LocalResultStore,
                                     ResultRecord, make_store, summarize_submission)


def _rec(team, bal_acc, ok=True, notebook="n.ipynb", at=None, dataset="d",
         auroc=None, runtime=None):
    """A scored (or failed) per-dataset record. ``bal_acc`` is the balanced accuracy;
    other metrics default off it. ``bal_acc=None`` -> a failed run (no metrics)."""
    metrics = {} if bal_acc is None else {
        "balanced_accuracy": bal_acc,
        "auroc": auroc if auroc is not None else bal_acc,
        "recall": bal_acc, "fpr": 0.0}
    return ResultRecord(team=team, notebook=notebook, dataset_key=dataset,
                        metrics=metrics, ok=ok, submitted_at=at, runtime_seconds=runtime)


def test_make_store_local(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    assert isinstance(s, LocalResultStore)


def test_make_store_bucket():
    s = make_store("bucket://NDIF/leaderboard-dev-storage/results.jsonl")
    assert isinstance(s, BucketResultStore)
    assert s.bucket_id == "NDIF/leaderboard-dev-storage"
    assert s.remote_path == "results.jsonl"


def test_make_store_bucket_default_path():
    s = make_store("bucket://NDIF/some-bucket")
    assert s.remote_path == "results.jsonl"


def test_make_store_bucket_malformed():
    with pytest.raises(ValueError):
        make_store("bucket://onlyorg")


def test_unparseable_lines_are_skipped(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"team": "a", "notebook": "n", "dataset_key": "d", '
                 '"metrics": {"balanced_accuracy": 0.7}, "ok": true}\n'
                 'not json at all\n'
                 '{"legacy": "schema", "score": 0.5}\n')   # old schema -> skipped
    recs = make_store(str(p)).all()
    assert len(recs) == 1 and recs[0].team == "a"


def test_leaderboard_orders_by_balanced_accuracy_and_skips_failures(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([_rec("a", 0.7, at="t1"), _rec("b", 0.9, at="t1"),
              _rec("a", 0.8, at="t2"), _rec("c", None, ok=False, at="t1")])
    board = s.leaderboard()
    assert [(r["team"], r["balanced_accuracy"]) for r in board] == [("b", 0.9), ("a", 0.8)]
    assert "datasets" in board[0]                          # per-dataset breakdown surfaced


_DEVTEST = "aletheias-quest/dev-test-instructed-deception-Qwen3.5-27B-None"
_VALID = "aletheias-quest/validation-soft-trigger-gemma-3-27b-it-gemma-3-27b-it-lora-greeting"


def test_headline_mean_and_rank_use_validation_only(tmp_path):
    # dev-test scores high, validation low: the headline mean must be the validation
    # value only (0.5), NOT the average of both (0.7). Both are still in the breakdown.
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([_rec("a", 0.9, at="t1", dataset=_DEVTEST),
              _rec("a", 0.5, at="t1", dataset=_VALID)])
    row = s.leaderboard()[0]
    assert row["balanced_accuracy"] == pytest.approx(0.5)
    counted = {d["dataset"]: d["counted"] for d in row["datasets"]}
    assert counted == {_DEVTEST: False, _VALID: True}     # both shown, only validation counts


def test_headline_mean_falls_back_to_all_when_no_validation(tmp_path):
    # A dev-only run (e.g. local --dry) has no validation datasets -> average all,
    # so the rehearsal still reports a score (and every row counts).
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([_rec("a", 0.6, at="t1", dataset=_DEVTEST),
              _rec("a", 0.8, at="t1", dataset="aletheias-quest/dev-test-instructed-deception-gemma-3-27b-it-None")])
    row = s.leaderboard()[0]
    assert row["balanced_accuracy"] == pytest.approx(0.7)
    assert all(d["counted"] for d in row["datasets"])


def test_leaderboard_means_metrics_across_datasets_with_breakdown(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([_rec("a", 0.6, at="t1", dataset="d1", auroc=0.5, runtime=120.0),
              _rec("a", 0.8, at="t1", dataset="d2", auroc=0.9, runtime=120.0)])
    board = s.leaderboard()
    assert len(board) == 1
    row = board[0]
    assert row["balanced_accuracy"] == pytest.approx(0.7)
    assert row["auroc"] == pytest.approx(0.7)
    assert row["runtime_seconds"] == 120.0
    assert {d["dataset"] for d in row["datasets"]} == {"d1", "d2"}


def test_failed_dataset_fails_whole_submission(tmp_path):
    # all-or-nothing: one dataset scored, another failed -> not ranked at all.
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([_rec("a", 0.9, at="t1", dataset="d1"),
              _rec("a", None, ok=False, at="t1", dataset="d2")])
    assert s.leaderboard() == []


def test_summarize_all_or_nothing_surfaces_failed_dataset():
    recs = [_rec("a", 0.9, dataset="d1"),
            ResultRecord("a", "n", "d2", ok=False, error="boom", runtime_seconds=42.0)]
    summ = summarize_submission(recs)
    assert summ["ok"] is False
    assert summ["failed_dataset"] == "d2" and summ["error"] == "boom"
    assert summ["datasets"] == []
    assert all(v is None for v in summ["metrics"].values())
    assert summ["runtime_seconds"] == 42.0


def test_leaderboard_keeps_best_submission_mean_not_best_per_dataset(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    # sub1 mean=(0.9+0.3)/2=0.6 ; sub2 mean=(0.5+0.7)/2=0.6 -> tie at 0.6, not 0.8.
    s.append([_rec("a", 0.9, at="t1", dataset="d1"), _rec("a", 0.3, at="t1", dataset="d2"),
              _rec("a", 0.5, at="t2", dataset="d1"), _rec("a", 0.7, at="t2", dataset="d2")])
    board = s.leaderboard()
    assert board[0]["balanced_accuracy"] == pytest.approx(0.6)


def test_leaderboard_keys_by_team_and_notebook(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([
        _rec("a", 0.7, notebook="x.ipynb", at="2026-06-08T10:00:00+00:00"),
        _rec("a", 0.9, notebook="y.ipynb", at="2026-06-08T11:00:00+00:00"),
        _rec("a", 0.6, notebook="x.ipynb", at="2026-06-08T12:00:00+00:00"),  # worse resubmit
    ])
    board = s.leaderboard()
    assert [(r["team"], r["notebook"], r["balanced_accuracy"]) for r in board] == [
        ("a", "y.ipynb", 0.9), ("a", "x.ipynb", 0.7)]
    x = next(r for r in board if r["notebook"] == "x.ipynb")
    assert x["submitted_at"] == "2026-06-08T10:00:00+00:00"
