"""Egress allowlist self-test (skips if seccomp user-notif is unavailable)."""

import json
import subprocess
import sys

import pytest


def _user_notif_available() -> bool:
    code = ("from aletheia_runner.sandbox import egress\n"
            "import os\n"
            "pid=os.fork()\n"
            "if pid==0:\n"
            "    try: os.close(egress.install_connect_notifier()); os.write(1, b'YES')\n"
            "    except Exception: os.write(1, b'NO')\n"
            "    os._exit(0)\n"
            "os.waitpid(pid,0)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return "YES" in out.stdout


def test_allowproxy_host_ok_suffix_matching():
    from aletheia_runner.sandbox.egress import AllowProxy
    p = AllowProxy(["api.ndif.us", "huggingface.co", "hf.co"])
    try:
        assert p.host_ok("api.ndif.us")
        assert p.host_ok("huggingface.co")
        assert p.host_ok("cdn-lfs.huggingface.co")
        assert p.host_ok("cas-bridge.xethub.hf.co")
        assert p.host_ok("HF.CO")                      # case-insensitive
        assert not p.host_ok("evil.com")
        assert not p.host_ok("nothuggingface.co")      # not a subdomain
        assert not p.host_ok("api.ndif.us.evil.com")   # suffix-injection rejected
    finally:
        p.stop()


@pytest.mark.skipif(not _user_notif_available(),
                    reason="seccomp user-notif unavailable on this host")
def test_egress_allows_loopback_blocks_public():
    proc = subprocess.run([sys.executable, "-m", "aletheia_runner.sandbox.egress"],
                          capture_output=True, text=True, timeout=30)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "error" not in res, res
    assert res["loopback"].startswith("allowed")   # passed to kernel (ECONNREFUSED)
    assert res["public"] == "denied"               # EPERM from the supervisor
