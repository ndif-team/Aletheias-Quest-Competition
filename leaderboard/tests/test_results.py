import pytest

from aletheia_runner.results import (BucketResultStore, LocalResultStore,
                                     ResultRecord, make_store)


def _rec(team, score, ok=True, notebook="n.ipynb", at=None, dataset="d"):
    return ResultRecord(team=team, notebook=notebook, dataset_key=dataset,
                        metric="auroc", score=score, ok=ok, submitted_at=at)


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


def test_local_store_leaderboard_keeps_best_and_skips_failures(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    # team a submits twice (distinct stamps = distinct submissions); keeps its best.
    s.append([_rec("a", 0.7, at="t1"), _rec("b", 0.9, at="t1"),
              _rec("a", 0.8, at="t2"), _rec("c", None, ok=False, at="t1")])
    board = s.leaderboard()
    assert [(r["team"], r["score"]) for r in board] == [("b", 0.9), ("a", 0.8)]
    assert "dataset" not in board[0]


def test_leaderboard_reports_mean_across_datasets(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    # One submission, notebook scored on two datasets -> reported score is the mean.
    s.append([_rec("a", 0.6, at="t1", dataset="d1"),
              _rec("a", 0.8, at="t1", dataset="d2")])
    board = s.leaderboard()
    assert len(board) == 1
    assert board[0]["score"] == pytest.approx(0.7)
    assert "dataset" not in board[0]


def test_leaderboard_keeps_best_submission_mean_not_best_per_dataset(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    # sub1 mean=(0.9+0.3)/2=0.6 ; sub2 mean=(0.5+0.7)/2=0.6 -> tie at 0.6, not 0.8.
    s.append([_rec("a", 0.9, at="t1", dataset="d1"), _rec("a", 0.3, at="t1", dataset="d2"),
              _rec("a", 0.5, at="t2", dataset="d1"), _rec("a", 0.7, at="t2", dataset="d2")])
    board = s.leaderboard()
    assert board[0]["score"] == pytest.approx(0.6)   # not 0.8 (no cross-submission cherry-pick)


def test_leaderboard_keys_by_team_and_notebook(tmp_path):
    s = make_store(str(tmp_path / "r.jsonl"))
    s.append([
        _rec("a", 0.7, notebook="x.ipynb", at="2026-06-08T10:00:00+00:00"),
        _rec("a", 0.9, notebook="y.ipynb", at="2026-06-08T11:00:00+00:00"),
        _rec("a", 0.6, notebook="x.ipynb", at="2026-06-08T12:00:00+00:00"),  # worse resubmit
    ])
    board = s.leaderboard()
    # Two rows (one per notebook); x keeps its best (0.7) with that run's time.
    assert [(r["team"], r["notebook"], r["score"]) for r in board] == [
        ("a", "y.ipynb", 0.9), ("a", "x.ipynb", 0.7)]
    x = next(r for r in board if r["notebook"] == "x.ipynb")
    assert x["submitted_at"] == "2026-06-08T10:00:00+00:00"
