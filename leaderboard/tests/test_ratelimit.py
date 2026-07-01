from aletheia_runner.ratelimit import RateLimiter


def test_fixed_window_allows_max_then_blocks_then_resets(tmp_path):
    rl = RateLimiter(str(tmp_path / "rl.json"), max_submissions=3, window_seconds=100)
    t0 = 1000.0
    for i in range(3):                       # 3 allowed within the window
        ok, retry = rl.check_and_consume("team-a", now=t0 + i)
        assert ok and retry == 0
    ok, retry = rl.check_and_consume("team-a", now=t0 + 3)   # 4th blocked
    assert not ok and 0 < retry <= 100
    # the window opened at t0, so it resets at t0+100
    ok, _ = rl.check_and_consume("team-a", now=t0 + 99)
    assert not ok
    ok, retry = rl.check_and_consume("team-a", now=t0 + 101)
    assert ok and retry == 0                 # fresh window


def test_limit_is_per_team(tmp_path):
    rl = RateLimiter(str(tmp_path / "rl.json"), max_submissions=1, window_seconds=100)
    assert rl.check_and_consume("a", now=10)[0]
    assert not rl.check_and_consume("a", now=11)[0]    # a is out
    assert rl.check_and_consume("b", now=11)[0]        # b has its own budget


def test_status_reports_usage_without_consuming(tmp_path):
    rl = RateLimiter(str(tmp_path / "rl.json"), max_submissions=3, window_seconds=100)
    s = rl.status("a", now=1000)
    assert s["enabled"] and s["used"] == 0 and s["remaining"] == 3 and s["resets_at"] is None
    rl.check_and_consume("a", now=1000)
    rl.check_and_consume("a", now=1001)
    s = rl.status("a", now=1002)
    assert s["used"] == 2 and s["remaining"] == 1
    assert abs(s["resets_at"] - 1100) < 1            # window_start(1000) + window(100)
    assert rl.status("a", now=1002)["used"] == 2     # status() didn't consume


def test_status_disabled(tmp_path):
    s = RateLimiter(str(tmp_path / "rl.json"), 0, 0).status("a")
    assert s["enabled"] is False and s["remaining"] is None


def _exempt(tmp_path, names):
    import json
    p = tmp_path / "exempt.json"
    p.write_text(json.dumps(names))
    return str(p)


def test_exempt_team_is_never_rate_limited(tmp_path):
    rl = RateLimiter(str(tmp_path / "rl.json"), max_submissions=1, window_seconds=100,
                     exempt_uri=_exempt(tmp_path, ["dev-team", "Jaden"]))
    # a normal team is capped at 1...
    assert rl.check_and_consume("someone", now=10)[0]
    assert not rl.check_and_consume("someone", now=11)[0]
    # ...but an exempt dev team can submit repeatedly
    for i in range(5):
        ok, retry = rl.check_and_consume("dev-team", now=10 + i)
        assert ok and retry == 0
    assert rl.status("dev-team")["exempt"] is True
    assert rl.status("dev-team")["remaining"] is None


def test_exempt_list_is_read_only_and_survives_a_bad_file(tmp_path):
    # missing file -> nobody exempt, no error
    rl = RateLimiter(str(tmp_path / "rl.json"), 1, 100,
                     exempt_uri=str(tmp_path / "nope.json"))
    assert rl.exempt_teams() == set()
    assert not rl.is_exempt("dev-team")
    # malformed file -> treated as empty, never raises (must not block submissions)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rl2 = RateLimiter(str(tmp_path / "rl.json"), 1, 100, exempt_uri=str(bad))
    assert rl2.exempt_teams() == set()
    assert rl2.check_and_consume("anyone", now=1)[0]


def test_disabled_when_max_or_window_zero(tmp_path):
    for kwargs in (dict(max_submissions=0, window_seconds=100),
                   dict(max_submissions=3, window_seconds=0)):
        rl = RateLimiter(str(tmp_path / "rl.json"), **kwargs)
        assert rl.enabled is False
        for i in range(10):
            assert rl.check_and_consume("t", now=1000 + i) == (True, 0)
