import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aletheia_runner.app import create_app
from aletheia_runner.config import DatasetConfig, RunnerConfig
from aletheia_runner.registry import TeamRegistry
from aletheia_runner.results import ResultStore

FIXTURES = Path(__file__).parent / "fixtures"


def _zip_with_fixture() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("submissions/fixture.ipynb",
                    (FIXTURES / "fixture.ipynb").read_text())
    return buf.getvalue()


def _post(client, *, team=None, key="key-1"):
    data = {}
    if team is not None:
        data["team"] = team
    headers = {"X-NDIF-API-Key": key} if key else {}
    return client.post("/submit", data=data,
                       files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
                       headers=headers)


@pytest.fixture
def client(tmp_path):
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        metric="auroc")
    store = ResultStore(str(tmp_path / "results.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    return TestClient(create_app(cfg, store, registry))


def test_submit_scores_and_appears_on_leaderboard(client):
    res = _post(client, team="team-a", key="key-a")
    assert res.status_code == 200, res.text
    body = res.json()
    # Scores are keyed by notebook (mean across datasets), never by dataset name.
    assert body["scores"]["fixture.ipynb"] == 1.0 and not body["failures"]
    board = client.get("/api/leaderboard").json()
    assert board["results"][0]["team"] == "team-a"
    assert "dataset" not in board["results"][0]


def test_ndif_key_required(client):
    assert _post(client, team="team-a", key=None).status_code == 400


def test_new_key_requires_team(client):
    assert _post(client, team="  ", key="brand-new-key").status_code == 400


def test_key_remembers_team_and_team_is_unique(client):
    # First submit binds key-a -> team-a.
    assert _post(client, team="team-a", key="key-a").status_code == 200
    # Same key, no team -> remembered.
    r = _post(client, team=None, key="key-a")
    assert r.status_code == 200 and r.json()["team"] == "team-a"
    # A different key can't claim the same team name.
    assert _post(client, team="team-a", key="key-b").status_code == 400


def test_index_and_health(client):
    assert "Aletheia" in client.get("/").text
    assert client.get("/api/health").json()["datasets"] == ["dummy"]


def test_me_endpoint_reports_team_history_and_limit(client):
    assert _post(client, team="team-a", key="key-a").status_code == 200
    r = client.post("/api/me", headers={"X-NDIF-API-Key": "key-a"})
    assert r.status_code == 200
    d = r.json()
    assert d["registered"] and d["team"] == "team-a" and d["pending"] == 0
    assert d["rate_limit"]["enabled"] is False           # no limiter on the fixture
    assert len(d["submissions"]) == 1
    assert d["submissions"][0]["ok"] and d["submissions"][0]["score"] == 1.0
    # An unknown key is reported as unregistered (and isn't registered as a side effect).
    u = client.post("/api/me", headers={"X-NDIF-API-Key": "never-seen"})
    assert u.status_code == 200 and u.json()["registered"] is False


def test_me_requires_a_key(client):
    assert client.post("/api/me").status_code == 400


def test_submission_zip_is_archived(tmp_path):
    """Every uploaded zip is stored (under the team) for later retrieval."""
    import glob

    from aletheia_runner.archive import SubmissionArchive

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        metric="auroc")
    store = ResultStore(str(tmp_path / "r.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    archive = SubmissionArchive(str(tmp_path / "archive"))
    client = TestClient(create_app(cfg, store, registry, archive=archive))

    assert _post(client, team="team-a", key="key-a").status_code == 200
    zips = glob.glob(str(tmp_path / "archive" / "**" / "*.zip"), recursive=True)
    assert len(zips) == 1 and "team-a" in zips[0]


def test_submit_503_when_no_datasets(tmp_path):
    app = create_app(RunnerConfig(datasets=[]),
                     ResultStore(str(tmp_path / "r.jsonl")),
                     TeamRegistry(str(tmp_path / "teams.json")))
    assert _post(TestClient(app), team="x", key="k").status_code == 503


def test_rate_limit_returns_429_after_max(tmp_path):
    """A team gets `max` submissions per window; the next is 429'd with Retry-After.
    A different team has its own budget."""
    from aletheia_runner.ratelimit import RateLimiter

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        metric="auroc")
    store = ResultStore(str(tmp_path / "r.jsonl"))
    registry = TeamRegistry(str(tmp_path / "teams.json"))
    limiter = RateLimiter(str(tmp_path / "rl.json"), max_submissions=2,
                          window_seconds=3600)
    client = TestClient(create_app(cfg, store, registry, limiter))

    assert _post(client, team="team-a", key="key-a").status_code == 200
    assert _post(client, team="team-a", key="key-a").status_code == 200
    blocked = _post(client, team="team-a", key="key-a")
    assert blocked.status_code == 429
    assert blocked.headers.get("Retry-After")
    assert _post(client, team="team-b", key="key-b").status_code == 200  # own budget


def test_concurrent_submissions_bounded_and_all_recorded(tmp_path, monkeypatch):
    """Submissions run off the event loop, capped by the semaphore, and the
    bucket lock keeps concurrent result writes from clobbering each other."""
    import asyncio
    import threading
    import time

    import httpx

    from aletheia_runner import app as app_module
    from aletheia_runner.results import ResultRecord

    monkeypatch.setattr(app_module, "MAX_CONCURRENT_SUBMISSIONS", 2)

    state = {"active": 0, "peak": 0}
    slock = threading.Lock()

    def fake_run_zip(zip_path, team, config, extra_env=None):
        with slock:                       # runs in a threadpool thread
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.25)                  # hold the slot so runs actually overlap
        with slock:
            state["active"] -= 1
        return [ResultRecord(team=team, notebook="n.ipynb", dataset_key="dummy",
                             metric="auroc", score=1.0, ok=True)]

    monkeypatch.setattr(app_module.pipeline, "run_zip", fake_run_zip)

    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="dummy", labels_uri=str(FIXTURES / "labels.csv"))],
        metric="auroc")
    store = ResultStore(str(tmp_path / "results.jsonl"))
    app = create_app(cfg, store, TeamRegistry(str(tmp_path / "teams.json")))

    async def fire(n):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            async def one(i):
                return await ac.post(
                    "/submit", data={"team": f"team-{i}"},
                    files={"file": ("s.zip", _zip_with_fixture(), "application/zip")},
                    headers={"X-NDIF-API-Key": f"key-{i}"})
            return await asyncio.gather(*[one(i) for i in range(n)])

    resps = asyncio.run(fire(4))
    assert all(r.status_code == 200 for r in resps), [r.text for r in resps]
    assert state["peak"] == 2          # semaphore caps concurrency (and runs overlapped)
    assert len(store.all()) == 4       # bucket lock: all four writes landed, none lost
