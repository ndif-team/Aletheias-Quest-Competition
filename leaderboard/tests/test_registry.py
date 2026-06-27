from aletheia_runner.registry import TeamRegistry


def test_registry_binds_key_to_team_and_enforces_uniqueness(tmp_path):
    reg = TeamRegistry(str(tmp_path / "teams.json"))

    # New key with no team -> error.
    team, err = reg.resolve("ndif-key-1", "")
    assert team is None and err

    # New key + team -> bound.
    team, err = reg.resolve("ndif-key-1", "alpha")
    assert team == "alpha" and err is None

    # Same key, no team -> remembered (submitted name ignored).
    team, err = reg.resolve("ndif-key-1", "")
    assert team == "alpha" and err is None
    team, err = reg.resolve("ndif-key-1", "different")
    assert team == "alpha" and err is None

    # Different key can't reuse a taken team name.
    team, err = reg.resolve("ndif-key-2", "alpha")
    assert team is None and "taken" in err

    # Different key + free name -> bound.
    team, err = reg.resolve("ndif-key-2", "beta")
    assert team == "beta" and err is None


def test_lookup_is_read_only(tmp_path):
    reg = TeamRegistry(str(tmp_path / "teams.json"))
    assert reg.lookup("k") is None               # unknown key
    reg.resolve("k", "alpha")
    assert reg.lookup("k") == "alpha"            # known key -> team
    assert reg.lookup("unseen") is None          # still read-only...
    assert reg.lookup("k") == "alpha"            # ...nothing got registered


def test_registry_persists_across_instances(tmp_path):
    uri = str(tmp_path / "teams.json")
    TeamRegistry(uri).resolve("k", "gamma")
    # A fresh instance reads the same file; the raw key is never stored.
    reg2 = TeamRegistry(uri)
    assert reg2.resolve("k", "") == ("gamma", None)
    import json
    stored = json.loads((tmp_path / "teams.json").read_text())
    assert "gamma" in stored.values() and "k" not in stored  # keyed by hash
    assert TeamRegistry.key_hash("k") in stored
