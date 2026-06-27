import shutil
from pathlib import Path

from aletheia_runner import pipeline
from aletheia_runner.config import DatasetConfig, RunnerConfig
from aletheia_runner.results import ResultStore

FIXTURES = Path(__file__).parent / "fixtures"


def _make_submission(tmp_path: Path) -> Path:
    """Build a submission root with submissions/fixture.ipynb."""
    root = tmp_path / "submission"
    (root / "submissions").mkdir(parents=True)
    shutil.copy(FIXTURES / "fixture.ipynb", root / "submissions" / "fixture.ipynb")
    return root


def _config() -> RunnerConfig:
    ds = DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))
    return RunnerConfig(datasets=[ds], metric="auroc")


def test_pipeline_runs_and_scores(tmp_path):
    root = _make_submission(tmp_path)
    records = pipeline.run_pipeline(root, team="team-a", config=_config())

    assert len(records) == 1
    r = records[0]
    assert r.ok and r.error is None
    assert r.team == "team-a"
    assert r.notebook == "submissions/fixture.ipynb"
    assert r.dataset_key == "dummy"
    assert r.score == 1.0


def test_pipeline_records_notebook_failure(tmp_path):
    root = tmp_path / "submission"
    (root / "submissions").mkdir(parents=True)
    (root / "submissions" / "broken.ipynb").write_text(_BROKEN_NB)

    records = pipeline.run_pipeline(root, team="team-b", config=_config())
    assert len(records) == 1
    assert records[0].ok is False
    assert records[0].score is None
    assert records[0].error


def test_sandboxed_pipeline_sets_up_once_per_request(tmp_path, monkeypatch):
    """The sandbox path builds the venv/dataset copy once per request and reuses
    it across every notebook (confine=False so no Landlock/seccomp needed here)."""
    from aletheia_runner import data
    from aletheia_runner import sandbox

    # Stub dataset prep: a copyable empty cache dir (the fixture nb reads no data),
    # so we don't touch the Hub.
    cache = tmp_path / "dscache"
    cache.mkdir()
    (cache / "marker").write_text("x")
    monkeypatch.setattr(data, "prepare_inputs",
                        lambda config: data.DataLayout(datasets_cache=cache))

    setups = {"n": 0}
    real_setup = sandbox.setup_job
    monkeypatch.setattr(sandbox, "setup_job",
                        lambda *a, **k: (setups.__setitem__("n", setups["n"] + 1)
                                         or real_setup(*a, **k)))

    root = tmp_path / "submission"
    (root / "submissions").mkdir(parents=True)
    shutil.copy(FIXTURES / "fixture.ipynb", root / "submissions" / "a.ipynb")
    shutil.copy(FIXTURES / "fixture.ipynb", root / "submissions" / "b.ipynb")

    # Use an HF-style "org/name" key (with a slash) so snapshot-path flattening
    # is exercised — a bare name wouldn't catch the slash bug.
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="org/dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        metric="auroc", sandbox=True, confine=False, enforce_egress=False,
        cache_dir=str(tmp_path / "cache"))
    records = pipeline.run_pipeline(root, team="team-a", config=cfg)

    assert setups["n"] == 1                      # one setup for both notebooks
    assert len(records) == 2 and all(r.ok and r.score == 1.0 for r in records)
    assert {r.notebook for r in records} == {"submissions/a.ipynb", "submissions/b.ipynb"}


def test_leaderboard_keeps_best(tmp_path):
    store = ResultStore(str(tmp_path / "results.jsonl"))
    root = _make_submission(tmp_path)
    store.append(pipeline.run_pipeline(root, team="team-a", config=_config()))
    store.append(pipeline.run_pipeline(root, team="team-a", config=_config()))

    board = store.leaderboard()
    assert len(board) == 1  # de-duped to one row per (team, dataset)
    assert board[0]["score"] == 1.0


def test_execution_error_is_generic_and_does_not_leak_notebook_output(tmp_path, capsys):
    """Exfil-via-error: a notebook can read the private inputs and raise them in its
    traceback, but the participant must only get a generic message — the real error
    (which it controls) is logged server-side, never returned."""
    root = tmp_path / "submission"
    (root / "submissions").mkdir(parents=True)
    (root / "submissions" / "leak.ipynb").write_text(_LEAK_NB)

    records = pipeline.run_pipeline(root, team="t", config=_config())
    assert len(records) == 1 and not records[0].ok
    assert _SECRET not in (records[0].error or "")     # not returned to the participant
    assert "--dry" in records[0].error                 # generic, actionable guidance
    assert _SECRET in capsys.readouterr().err          # but logged for maintainers
    # The full real error is kept in error_detail — persisted to the bucket so the
    # failure can be diagnosed from S3, but never surfaced in the participant response.
    assert _SECRET in (records[0].error_detail or "")


def test_dry_run_config_shows_real_error(tmp_path):
    """With redact_errors=False (how --dry runs), the real notebook error is
    returned so the participant can debug locally."""
    from dataclasses import replace

    root = tmp_path / "submission"
    (root / "submissions").mkdir(parents=True)
    (root / "submissions" / "leak.ipynb").write_text(_LEAK_NB)

    records = pipeline.run_pipeline(root, team="t", config=replace(_config(), redact_errors=False))
    assert len(records) == 1 and not records[0].ok
    assert _SECRET in (records[0].error or "")         # real error surfaced for --dry


# A notebook whose single cell raises at run time.
_BROKEN_NB = (
    '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},'
    '"outputs":[],"source":["raise RuntimeError(\\"boom\\")"]}],'
    '"metadata":{"kernelspec":{"display_name":"Python 3","language":"python",'
    '"name":"python3"},"language_info":{"name":"python"}},'
    '"nbformat":4,"nbformat_minor":5}'
)

# A notebook that raises with a secret marker (stand-in for an exfiltrated row).
_SECRET = "S3CR3T_EVAL_ROW_MARKER"
_LEAK_NB = (
    '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},'
    '"outputs":[],"source":["raise RuntimeError(\\"' + _SECRET + '\\")"]}],'
    '"metadata":{"kernelspec":{"display_name":"Python 3","language":"python",'
    '"name":"python3"},"language_info":{"name":"python"}},'
    '"nbformat":4,"nbformat_minor":5}'
)
