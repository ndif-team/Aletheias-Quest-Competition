"""Runner robustness: bounded post-kill reap, whole-tree kill, cooperative cancel,
and the overall per-submission wall-clock budget.

Regression cover for the wedge that hung the serial eval queue: a run whose kernel
(launched by nbclient in its own session) escaped the killed process group and held
the stdout pipe open, wedging the reaping ``communicate()`` forever.
"""

import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aletheia_runner import pipeline
from aletheia_runner.config import DatasetConfig, RunnerConfig
from aletheia_runner.sandbox import Canceller, runner

FIXTURES = Path(__file__).parent / "fixtures"
_HAS_PROC = Path("/proc").exists()


def _config() -> RunnerConfig:
    ds = DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))
    return RunnerConfig(datasets=[ds], redact_errors=False)


# --- Canceller ------------------------------------------------------------------

class _FakeProc:
    def __init__(self, pid): self.pid = pid


def test_canceller_kills_registered_proc(monkeypatch):
    killed = []
    monkeypatch.setattr(runner, "_kill_tree", killed.append)
    c = Canceller()
    c.register(_FakeProc(4242))
    assert not killed and not c.cancelled
    c.cancel()
    assert killed == [4242] and c.cancelled


def test_canceller_latches_and_kills_next_registration(monkeypatch):
    """A cancel that arrives between runs (nothing registered) is latched: the next
    ``register`` kills immediately, so a queued/just-starting run can't slip through."""
    killed = []
    monkeypatch.setattr(runner, "_kill_tree", killed.append)
    c = Canceller()
    c.cancel()                       # cancelled while idle
    assert killed == []              # nothing live to kill yet
    c.register(_FakeProc(99))        # next run starts -> killed on arrival
    assert killed == [99]


def test_canceller_unregister_clears_target(monkeypatch):
    killed = []
    monkeypatch.setattr(runner, "_kill_tree", killed.append)
    c = Canceller()
    p = _FakeProc(7)
    c.register(p)
    c.unregister(p)
    c.cancel()                       # no live proc -> nothing killed, but latched
    assert killed == [] and c.cancelled


# --- process-tree discovery / kill ----------------------------------------------

@pytest.mark.skipif(not _HAS_PROC, reason="requires /proc")
def test_descendants_finds_escaped_child():
    """A child launched in its OWN session (escaping our process group) is still a
    descendant by ppid, so the /proc walk finds it — which is what lets _kill_tree
    reap the orphaned kernel the group-kill would miss."""
    script = ("import subprocess, time; "
              "subprocess.Popen(['sleep', '30'], start_new_session=True); "
              "time.sleep(30)")
    p = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    try:
        time.sleep(1.0)
        desc = runner._descendants(p.pid)
        assert desc, "expected the escaped grandchild among descendants"
    finally:
        runner._kill_tree(p.pid)
        p.wait(timeout=10)


@pytest.mark.skipif(not _HAS_PROC, reason="requires /proc")
def test_kill_tree_does_not_raise_on_dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait(timeout=10)
    runner._kill_tree(p.pid)          # already gone -> no exception


# --- overall submission budget (pure logic) -------------------------------------

def test_budget_abort_on_spent_deadline():
    cfg = replace(_config(), submission_timeout=100)
    rec = pipeline._budget_abort("t", "nb", "d", cfg,
                                 deadline=time.monotonic() - 1, cancel=None)
    assert rec is not None and not rec.ok and "budget" in (rec.error or "")


def test_budget_abort_on_cancel():
    c = Canceller(); c.cancel()
    rec = pipeline._budget_abort("t", "nb", "d", _config(), deadline=None, cancel=c)
    assert rec is not None and "cancelled" in (rec.error or "")


def test_budget_abort_none_when_ok():
    assert pipeline._budget_abort(
        "t", "nb", "d", _config(),
        deadline=time.monotonic() + 100, cancel=Canceller()) is None


def test_run_timeout_clamps_to_remaining_budget():
    cfg = replace(_config(), notebook_timeout=1800)
    assert pipeline._run_timeout(cfg, None) == 1800          # no cap -> per-run budget
    clamped = pipeline._run_timeout(cfg, time.monotonic() + 100)
    assert 90 <= clamped <= 100                              # clamped to remaining
    assert pipeline._run_timeout(cfg, time.monotonic() + 0.001) >= 1   # never < 1


def test_pipeline_cancel_aborts_before_next_run(tmp_path):
    """A pre-cancelled Canceller threaded through run_pipeline stops at the first run
    boundary with a cancelled record — no subprocess needed."""
    root = tmp_path / "submission"
    (root / "submission").mkdir(parents=True)
    shutil.copy(FIXTURES / "fixture.ipynb", root / "submission" / "a.ipynb")
    c = Canceller(); c.cancel()
    records = pipeline.run_pipeline(root, team="t", config=_config(), cancel=c)
    assert len(records) == 1 and not records[0].ok
    assert "cancelled" in (records[0].error or "")


# --- end-to-end: the wedge regression -------------------------------------------

# A notebook that hangs forever AND spawns a child that escapes the process group,
# reproducing the orphaned-kernel wedge. The reaping communicate() must not hang.
_HANG_NB = (
    '{"cells":[{"cell_type":"code","execution_count":null,"metadata":{},'
    '"outputs":[],"source":['
    '"import subprocess, time\\n",'
    '"subprocess.Popen([\\"sleep\\", \\"300\\"], start_new_session=True)\\n",'
    '"time.sleep(300)"]}],'
    '"metadata":{"kernelspec":{"display_name":"Python 3","language":"python",'
    '"name":"python3"},"language_info":{"name":"python"}},'
    '"nbformat":4,"nbformat_minor":5}'
)


@pytest.mark.skipif(not _HAS_PROC, reason="requires /proc (Linux sandbox path)")
def test_hanging_notebook_is_killed_and_releases_within_timeout(tmp_path, monkeypatch):
    """Pre-fix this wedged forever (the escaped kernel held the pipe open); now the
    per-run timeout kills the whole tree and the run returns promptly."""
    from aletheia_runner import data

    cache = tmp_path / "dscache"
    cache.mkdir()
    (cache / "marker").write_text("x")
    monkeypatch.setattr(data, "prepare_inputs",
                        lambda config: data.DataLayout(datasets_cache=cache))

    root = tmp_path / "submission"
    (root / "submission").mkdir(parents=True)
    (root / "submission" / "hang.ipynb").write_text(_HANG_NB)

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="d", labels_uri=str(FIXTURES / "labels.csv"))],
        sandbox=True, confine=False, enforce_egress=False,
        notebook_timeout=3, cache_dir=str(tmp_path / "cache"), redact_errors=False)

    start = time.monotonic()
    records = pipeline.run_pipeline(root, team="t", config=cfg)
    elapsed = time.monotonic() - start

    assert elapsed < 60, f"run did not release promptly ({elapsed:.0f}s) — wedged?"
    assert len(records) == 1 and not records[0].ok
    assert "timed out" in (records[0].error or "")
