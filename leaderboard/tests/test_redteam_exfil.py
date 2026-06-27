"""Red-team: a malicious submission trying to steal the private eval data/labels.

The runner's whole security premise is that an untrusted participant notebook
can read the (label-free) eval INPUTS to make predictions, but can never reach
the held-out LABELS — not from its environment, not from the filesystem, not over
the network, and not from another process's memory. These tests pin the controls
that enforce that, framed as the attacks they defeat.

The kernel-dependent controls (Landlock fs confinement, seccomp connect()/ptrace
gates) have their own self-test suites in test_landlock.py / test_egress.py and a
seccomp self-test below; the rest here are deterministic and need no kernel.
"""

import json
import socket
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aletheia_runner import data, scoring
from aletheia_runner.config import DatasetConfig, RunnerConfig
from aletheia_runner.data import DataLayout
from aletheia_runner.sandbox import egress, runner

# A token that, on the real Space, belongs to the ORGANIZERS' org and CAN read
# the private labels. It must never appear in anything a participant notebook sees.
ORG_TOKEN = "hf_ORG_SECRET_can_read_private_labels"
SUBMITTER_TOKEN = "hf_submitter_token_only_their_own_repos"
LABELS_URI = "NDIF/aletheia-fake-eval-labels"   # the crown jewels' location


def _child_view(tmp_path) -> dict:
    """The exact environment a sandboxed participant notebook is handed at run
    time: token-free base env + DATASET_NAME + the submitter's forwarded keys.

    Mirrors runner.run_notebook's ``run_env`` without spinning up a venv."""
    ds_cache = tmp_path / "dscache"
    ds_cache.mkdir()
    (ds_cache / "inputs.arrow").write_text("eval inputs, no labels")
    layout = DataLayout(datasets_cache=ds_cache)

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    data_env = layout.child_env(scratch, offline=True)
    base_env = runner._child_env(scratch, data_env, venv=scratch / "venv")

    ds = DatasetConfig(name="NDIF/aletheia-fake-eval", labels_uri=LABELS_URI)
    extra_env = {"NDIF_API_KEY": "ndif-key", "HF_TOKEN": SUBMITTER_TOKEN}
    return {**base_env, **ds.env(), **extra_env}


# --- attack 1: read the labels straight out of the environment ---------------

def test_org_token_never_reaches_the_child_env(tmp_path):
    """A notebook that reads os.environ must not find the organizers' token —
    otherwise it could ``load_dataset(<any NDIF private repo>, token=...)``."""
    env = _child_view(tmp_path)
    blob = "\n".join(f"{k}={v}" for k, v in env.items())
    assert ORG_TOKEN not in blob
    # The only HF token present is the submitter's own (harmless: their own access).
    assert env.get("HF_TOKEN") == SUBMITTER_TOKEN


def test_child_env_does_not_reveal_the_labels_location(tmp_path):
    """The child is pointed at the INPUTS dataset only; the labels URI/column are
    never leaked into its environment, so it can't even name what to fetch."""
    env = _child_view(tmp_path)
    blob = "\n".join(f"{k}={v}" for k, v in env.items())
    assert LABELS_URI not in blob
    assert "labels" not in blob.lower()
    assert env["DATASET_NAME"] == "NDIF/aletheia-fake-eval"   # inputs, not labels


def test_child_env_forces_offline_dataset_reads(tmp_path):
    """HF_DATASETS_OFFLINE pins the private eval set to the predownloaded cache, so
    a notebook can't fetch a *different* (e.g. labelled) split from the hub."""
    env = _child_view(tmp_path)
    assert env.get("HF_DATASETS_OFFLINE") == "1"


def test_low_level_child_env_helper_is_token_free(tmp_path):
    """runner._child_env builds the base env from scratch — it has no path by
    which any HF/NDIF credential could enter (no config, no os.environ pass-through)."""
    scratch = tmp_path / "s"
    scratch.mkdir()
    env = runner._child_env(scratch, data_env={"HF_DATASETS_OFFLINE": "1"},
                            venv=scratch / "venv")
    assert "HF_TOKEN" not in env
    assert not any("token" in k.lower() or "key" in k.lower() for k in env)


def test_dataset_config_env_exposes_only_the_inputs_name():
    """DatasetConfig.env() is the only dataset info the notebook gets — it must
    carry the inputs name and nothing about where labels live."""
    ds = DatasetConfig(name="NDIF/aletheia-fake-eval", labels_uri=LABELS_URI,
                       label_column="deceptive")
    assert ds.env() == {"DATASET_NAME": "NDIF/aletheia-fake-eval"}


# --- attack 2: exfiltrate over the network to an attacker host ----------------

def test_proxy_blocks_exfil_to_attacker_hosts():
    """A notebook tries to POST stolen data to its own server. The hostname
    allowlist (the only egress path) must reject everything off-list, including
    classic allowlist-bypass tricks."""
    p = egress.AllowProxy(["api.ndif.us", "huggingface.co", "hf.co",
                           "prod-ndif-results.s3.amazonaws.com"])
    try:
        # Legit destinations the run actually needs.
        assert p.host_ok("api.ndif.us")
        assert p.host_ok("cas-bridge.xethub.hf.co")
        # Exfil destinations / bypass attempts — all denied.
        for bad in ["evil.com",
                    "huggingface.co.evil.com",     # suffix injection
                    "evilhuggingface.co",          # missing dot boundary
                    "hf.co.attacker.net",
                    "attacker.net",
                    "169.254.169.254",             # cloud metadata IP literal
                    "10.0.0.5"]:                   # internal IP literal
            assert not p.host_ok(bad), bad
    finally:
        p.stop()


def test_loopback_gate_is_fail_closed_even_for_allowed_hosts():
    """The seccomp connect() gate permits ONLY loopback (the proxy) + AF_UNIX, so a
    notebook that ignores the proxy env and dials a host directly is blocked — even
    the *allowed* hosts' real IPs. Egress is possible only via the filtering proxy."""
    AF_UNIX, AF_INET, AF_INET6 = egress.AF_UNIX, egress.AF_INET, egress.AF_INET6
    assert egress._loopback_only(AF_UNIX, "")
    assert egress._loopback_only(AF_INET, "127.0.0.1")
    assert egress._loopback_only(AF_INET6, "::1")
    # Direct connects to anything routable — denied (must go through the proxy).
    for ip in ["8.8.8.8", "1.1.1.1", "13.226.0.1", "169.254.169.254", "10.0.0.5"]:
        assert not egress._loopback_only(AF_INET, ip), ip


# --- attack 3: read another process's memory to steal the org token -----------

def _seccomp_available() -> bool:
    code = ("from aletheia_runner.sandbox import seccomp\n"
            "import os\n"
            "pid=os.fork()\n"
            "if pid==0:\n"
            "    try: seccomp.install_blocklist(); os.write(1,b'YES')\n"
            "    except Exception: os.write(1,b'NO')\n"
            "    os._exit(0)\n"
            "os.waitpid(pid,0)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return "YES" in out.stdout


@pytest.mark.skipif(not _seccomp_available(),
                    reason="seccomp filter install unavailable on this host")
def test_seccomp_blocks_cross_process_memory_peek_and_udp_egress():
    """Two exfil routes seccomp must close:
    - cross-process memory peek (ptrace/process_vm_readv) — steal the org token /
      read the labels out of the parent's memory;
    - UDP/DNS egress — tunnel the private eval inputs out (the TCP gate forces
      everything through the loopback proxy, but UDP sendto would bypass it).
    TCP (the proxy) and AF_UNIX datagrams must stay usable."""
    proc = subprocess.run([sys.executable, "-m", "aletheia_runner.sandbox.seccomp"],
                          capture_output=True, text=True, timeout=30)
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["ok"], res
    r = res["results"]
    assert r["ptrace_blocked"].startswith("EPERM")
    assert r["udp_inet_blocked"].startswith("DENIED")
    assert r["udp_inet6_blocked"].startswith("DENIED")
    assert r["tcp_inet_allowed"] == "ALLOWED"
    assert r["unix_dgram_allowed"] == "ALLOWED"


# --- attack 4: smuggle labels in via a poisoned INPUTS dataset ----------------

def test_scoring_inputs_must_not_carry_the_label_column():
    """If the eval INPUTS the notebook reads ever included the label column, the
    notebook could read labels straight from its (legitimate) cache. This guards
    the data-hygiene invariant: a dataset handed to the child as inputs is rejected
    by the loader if it doubles as the label table."""
    # Stand in for what the child can see: an inputs frame. It must not be usable
    # as a label source — i.e. label loading from inputs is a misconfiguration we
    # can detect (here: inputs carry only an id, no label column).
    inputs = pd.DataFrame({"id": [0, 1, 2], "text": ["a", "b", "c"]})
    csv = Path(__import__("tempfile").mkstemp(suffix=".csv")[1])
    inputs.to_csv(csv, index=False)
    cfg = DatasetConfig(name="x", labels_uri=str(csv), label_column="deceptive")
    with pytest.raises(scoring.ScoringError, match="missing column"):
        scoring.load_labels(cfg)


# --- attack 5: reconstruct labels through the scoring oracle -------------------

def test_auroc_is_a_per_example_label_oracle():
    """KNOWN, sandbox-independent leak: the returned AUROC is a function of the
    hidden labels, so repeated crafted submissions reconstruct them. This test
    DEMONSTRATES the leak (it is not defended by the sandbox) to motivate a
    leaderboard-side mitigation (score quantization / rate limiting / secret
    holdout). If a mitigation lands, replace this with the bound it enforces."""
    labels = pd.DataFrame({"index": list(range(6)),
                           "label": [1, 0, 1, 0, 1, 0]})
    base = np.full(6, 0.5)

    def auroc(scores):
        preds = pd.DataFrame({"index": labels["index"],
                              "deceptive": scores >= 0.5, "score": scores})
        return scoring.compute_metrics(preds, labels)["auroc"]

    # Probing one example at a time, the score moves in opposite directions
    # depending on that example's hidden label — i.e. it leaks the bit.
    up = base.copy(); up[0] = 0.9      # raise example 0
    down = base.copy(); down[0] = 0.1  # lower example 0
    # Example 0 is positive → raising it should not *hurt* AUROC vs lowering it.
    assert auroc(up) >= auroc(down)
    # And the two differ, so an attacker observing the score learns the label.
    assert auroc(up) != auroc(down)


# --- attack 6: smuggle the labels in through the predownloaded INPUTS cache ----

def test_prepare_inputs_never_loads_the_labels_dataset(tmp_path, monkeypatch):
    """The predownload builds the cache that is copytree'd into each child. It must
    fetch only the INPUTS repo, never the labels repo — otherwise the answers would
    sit one copy away from the notebook."""
    loaded = []

    def fake_load_dataset(name, *a, **k):
        loaded.append(name)
        return types.SimpleNamespace(column_names=["id", "text"])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    cfg = RunnerConfig(
        datasets=[DatasetConfig(name="NDIF/aletheia-fake-eval", labels_uri=LABELS_URI)],
        cache_dir=str(tmp_path / "cache"))
    data.prepare_inputs(cfg)
    assert loaded == ["NDIF/aletheia-fake-eval"]     # labels repo never touched
    assert LABELS_URI not in loaded


def test_proxy_serves_403_on_a_real_connect_to_an_attacker_host():
    """End-to-end (not just host_ok): drive the running CONNECT proxy over a real
    socket and confirm an off-allowlist CONNECT is refused before any tunnel opens,
    and a non-CONNECT verb is rejected too."""
    proxy = egress.AllowProxy(["huggingface.co", "api.ndif.us"])
    port = proxy.start()
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        c.sendall(b"CONNECT evil.example.com:443 HTTP/1.1\r\n\r\n")
        resp = c.recv(256); c.close()
        assert b"403" in resp, resp

        c = socket.create_connection(("127.0.0.1", port), timeout=5)
        c.sendall(b"GET http://evil.example.com/ HTTP/1.1\r\n\r\n")
        resp = c.recv(256); c.close()
        assert b"405" in resp, resp
    finally:
        proxy.stop()


# --- attack 7: push the (legit-to-read) eval inputs out to an allowed host -----

def test_mitm_ca_mints_per_host_leaf_certs():
    pytest.importorskip("cryptography")
    ca = egress._CertAuthority()
    try:
        assert b"BEGIN CERTIFICATE" in ca.cert_pem
        ctx = ca.server_context("huggingface.co")
        import ssl as _ssl
        assert isinstance(ctx, _ssl.SSLContext)
        assert ca.server_context("huggingface.co") is ctx     # cached per host
    finally:
        ca.close()


def test_mitm_proxy_allows_reads_but_blocks_uploads_to_hf():
    """The eval INPUTS are readable by the notebook but must not be exfiltratable.
    huggingface.co stays reachable for model configs (GET/HEAD), but every upload
    verb (POST/PUT/PATCH/DELETE) is refused — so a notebook can't push the inputs
    to a repo it controls. Upstream is stubbed, so no network is needed."""
    pytest.importorskip("cryptography")
    requests = pytest.importorskip("requests")
    import os as _os
    import tempfile

    proxy = egress.AllowProxy(["models.test"], mitm_suffixes=["models.test"])

    def fake_upstream(tls, host, port, method, path, headers, body):
        payload = b"" if method == "HEAD" else b"config-bytes"
        tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n"
                    b"Connection: close\r\n\r\n%s" % (len(payload), payload))
    proxy._relay_upstream = fake_upstream
    port = proxy.start()

    fd, ca_path = tempfile.mkstemp(suffix=".pem")
    _os.write(fd, proxy.ca_cert_pem); _os.close(fd)
    proxies = {"https": f"http://127.0.0.1:{port}"}
    try:
        r = requests.get("https://models.test/api/models/x", proxies=proxies,
                         verify=ca_path, timeout=10)
        assert r.status_code == 200 and r.text == "config-bytes"
        assert requests.head("https://models.test/x", proxies=proxies,
                             verify=ca_path, timeout=10).status_code == 200
        # Every write verb — what an upload uses — is refused before reaching upstream.
        for verb in ("post", "put", "patch", "delete"):
            resp = getattr(requests, verb)("https://models.test/repo/commit",
                                           proxies=proxies, verify=ca_path, timeout=10)
            assert resp.status_code == 403, (verb, resp.status_code)
    finally:
        proxy.stop()
        _os.unlink(ca_path)
