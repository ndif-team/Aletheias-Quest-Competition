"""Landlock confinement self-test (skips if Landlock unavailable on this host)."""

import json
import subprocess
import sys

import pytest


def _landlock_available() -> bool:
    code = ("from aletheia_runner.sandbox.landlock import abi_version, LandlockUnavailable\n"
            "import sys\n"
            "try:\n"
            "    abi_version(); print('YES')\n"
            "except LandlockUnavailable:\n"
            "    print('NO')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return "YES" in out.stdout


pytestmark = pytest.mark.skipif(not _landlock_available(),
                                reason="Landlock not available on this host")


def test_landlock_enforces_confinement():
    proc = subprocess.run([sys.executable, "-m", "aletheia_runner.sandbox.landlock"],
                          capture_output=True, text=True, timeout=30)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"], result
    r = result["results"]
    assert r["read_ro_file"].startswith("ALLOWED")
    assert r["write_ro_dir"].startswith("DENIED")
    assert r["write_rw_dir"].startswith("ALLOWED")
    # The token-theft vector (reading another proc's environ) must be blocked.
    assert r["read_parent_environ"].startswith("DENIED")
    # A secret (e.g. eval labels) on the parent disk outside scratch is unreadable.
    assert r["read_secret_outside"].startswith("DENIED")
