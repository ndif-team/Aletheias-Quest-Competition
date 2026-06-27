#!/usr/bin/env python3
"""Submit your competition entry to the Aletheia's Quest leaderboard Space.

This compresses the repository (everything except git/cache cruft) into a zip
and POSTs it to the leaderboard Space, which runs every notebook in
``submissions/`` against the private eval data and returns your score.

Usage:
    python submit.py --team "my-team-name"

Configuration (flags override environment):
    --space-url   Leaderboard Space base URL   (env: ALETHEIA_SPACE_URL)
    --team        Team / display name          (env: ALETHEIA_TEAM)
    --root        Repo root to package          (default: this file's directory)
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import itertools
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("submit.py needs `requests` — install it with: pip install requests")

# Files/dirs we never ship to the runner ("leaderboard" is the runner's own code).
EXCLUDE_DIRS = {".git", ".claude", "leaderboard", "__pycache__",
                ".ipynb_checkpoints", ".venv", "venv", "node_modules",
                ".mypy_cache", ".pytest_cache"}
EXCLUDE_GLOBS = ["*.pyc", "*.pyo", ".DS_Store", "submission.csv"]

MAX_ZIP_MB = 200  # guardrail; the Space may enforce its own limit.


# ── Tiny ANSI styling + spinner (no deps; off when piped or NO_COLOR is set) ──
_STYLE = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _STYLE else s


def _bold(s): return _c("1", s)
def _dim(s): return _c("2", s)
def _ital(s): return _c("3", s)
def _ox(s): return _c("38;5;124", s)        # oxblood
def _grey(s): return _c("38;5;245", s)
def _green(s): return _c("38;5;71", s)
def _gold(s): return _c("38;5;179", s)


_RULE = _ox("─" * 44)


def _banner() -> None:
    print()
    print("  " + _gold("✦") + "  " + _bold("ALETHEIA’S QUEST")
          + _grey("   ·   a ledger of deception detection"))
    print("  " + _dim(_ital("ἀλήθεια — "
                            "disclosure; the state of not being hidden")))
    print("  " + _RULE)


def _info(msg): print("  " + _grey("▸ ") + msg)
def _ok(msg):   print("  " + _green("✓ ") + msg)
def _bad(msg):  print("  " + _ox("✗ ") + msg)


def _shorten(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


class Spinner:
    """Braille spinner + elapsed timer shown around a blocking call. When output
    isn't a TTY it just prints one static line, so logs stay clean."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, text: str):
        self.text = text
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.time()

    def __enter__(self):
        if _STYLE:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            print("  • " + self.text + " …")
        return self

    def _spin(self):
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            el = int(time.time() - self._t0)
            clock = f"{el // 60}m {el % 60:02d}s" if el >= 60 else f"{el}s"
            line = f"  {_ox(frame)}  {self.text}" + (_dim(f"   {clock}") if el >= 1 else "")
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
            time.sleep(0.09)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        if _STYLE:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def _excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    return any(fnmatch.fnmatch(rel.name, pat) for pat in EXCLUDE_GLOBS)


def build_zip(root: Path) -> bytes:
    """Package ``root`` into an in-memory zip, skipping cruft."""
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _excluded(rel):
                continue
            zf.write(path, rel.as_posix())
            n += 1
    data = buf.getvalue()
    size_mb = len(data) / 1e6
    if size_mb > MAX_ZIP_MB:
        sys.exit(f"package is {size_mb:.0f} MB (> {MAX_ZIP_MB} MB limit). "
                 "Remove large files (e.g. heavy weights) before submitting.")
    _ok(f"packaged {_bold(str(n))} files  {_dim('·')}  {_bold(f'{size_mb:.1f} MB')}")
    return data


def submit(space_url: str, team: str, payload: bytes,
           ndif_api_key: str | None, hf_token: str | None) -> None:
    url = space_url.rstrip("/") + "/submit"
    data = {"team": team or ""}   # empty -> the runner uses this key's registered team
    files = {"file": ("submission.zip", payload, "application/zip")}
    # Your NDIF key is passed through to your sandboxed run so nnsight can
    # authenticate remote traces (it reads NDIF_API_KEY from the environment).
    headers = {"X-NDIF-API-Key": ndif_api_key} if ndif_api_key else {}
    if hf_token:
        # Authorization lets an org member reach a private Space (harmless if
        # public); X-HF-Token is forwarded into your run as HF_TOKEN so your
        # notebook can load gated HF models you have access to.
        headers["Authorization"] = f"Bearer {hf_token}"
        headers["X-HF-Token"] = hf_token
    _info("entering the lists as " + _bold(team or "(remembered team)")
          + _dim("   →   " + url))

    err = None
    with Spinner("running your submission  "
                 + _dim("venv · deps · notebooks · NDIF traces")):
        try:
            resp = requests.post(url, data=data, files=files, headers=headers,
                                 timeout=60 * 30)
        except requests.RequestException as e:
            resp, err = None, e
    if resp is None:
        _bad("could not reach the Space  " + _dim("— " + _shorten(str(err), 160)))
        sys.exit(1)

    if resp.status_code != 200:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except ValueError:
            pass
        _bad(_bold(f"submission rejected  [{resp.status_code}]"))
        print("     " + _dim(_shorten(detail, 200)))
        sys.exit(1)

    try:
        body = resp.json()
    except ValueError:
        print(resp.text)
        return
    _render_results("the ledger answers", body.get("scores") or {},
                    body.get("failures") or [], body.get("message"))


def _render_results(title: str, scores: dict, failures: list, message: str | None) -> None:
    print()
    print("  " + _ox(title))
    print("  " + _RULE)
    for nb, score in scores.items():
        try:
            shown = f"{float(score):.4f}"
        except (TypeError, ValueError):
            shown = str(score)
        _ok(f"{nb:<26}  {_gold(_bold(shown))}")
    seen = []
    for f in failures:
        _bad(_bold(f.get("notebook") or "?"))
        msg = (f.get("error") or "").strip()
        if msg and msg not in seen:
            seen.append(msg)
            print("       " + _dim(_shorten(msg, 180)))
    if not scores and not failures:
        print("  " + _grey("no notebooks ran"))
    if message:
        print("  " + _RULE)
        print("  " + _grey(message))


def _resolve_hf_token(arg_token: str | None) -> str | None:
    """--hf-token / $HF_TOKEN, else the cached huggingface-cli login."""
    if arg_token:
        return arg_token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def run_dry(root: Path, ndif_api_key: str | None, hf_token: str | None) -> None:
    """Rehearse the submission locally via the leaderboard runner, no upload."""
    runner_src = root / "leaderboard" / "src"
    if not runner_src.is_dir():
        sys.exit("--dry needs the bundled runner at leaderboard/ (run from the repo root).")
    sys.path.insert(0, str(runner_src))
    try:
        from aletheia_runner.dryrun import dry_run
    except ImportError as e:
        sys.exit(f"--dry needs the runner's deps (datasets, nbclient, ipykernel, "
                 f"scikit-learn, pyyaml): {e}")

    _info("rehearsing locally  "
          + _dim("venv · requirements.txt · public dry-run data"))
    print("  " + _dim("  server-side Landlock/seccomp/egress is skipped here; "
                      "it runs on the Space"))
    with Spinner("running your notebooks locally"):
        records = dry_run(root, ndif_api_key, hf_token)

    print()
    print("  " + _ox("dry-run results"))
    print("  " + _RULE)
    ok = 0
    for r in records:
        name = r.notebook.split("/")[-1]
        if r.ok:
            ok += 1
            _ok(f"{name:<26}  {_gold(_bold(f'{r.score:.4f}'))}")
        else:
            _bad(_bold(name))
            # --dry shows the real error (public data, local) — print it in full.
            for ln in (r.error or "").strip().splitlines() or ["(no error captured)"]:
                print("       " + _dim(ln))
    print("  " + _RULE)
    if not ok:
        _bad("no notebook produced a valid submission.csv — fix the errors above")
        sys.exit(1)
    _ok(f"{_bold(f'{ok}/{len(records)}')} run(s) OK  "
        + _dim("· submit for real by dropping --dry"))


def main(argv: list[str] | None = None) -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--space-url", default=os.environ.get("ALETHEIA_SPACE_URL"))
    p.add_argument("--team", default=os.environ.get("ALETHEIA_TEAM"),
                   help="team name — required only on your NDIF key's FIRST "
                        "submission, then remembered (default: $ALETHEIA_TEAM)")
    p.add_argument("--root", default=str(here), type=Path)
    p.add_argument("--ndif-api-key", default=os.environ.get("NDIF_API_KEY"),
                   help="your NDIF API key — required (default: $NDIF_API_KEY)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF token to reach a private Space (default: $HF_TOKEN "
                        "or your cached huggingface-cli login)")
    p.add_argument("--dry", action="store_true",
                   help="run the full pipeline locally against a public dataset to "
                        "verify your submission works — no upload")
    args = p.parse_args(argv)

    if not args.ndif_api_key:
        p.error("--ndif-api-key (or $NDIF_API_KEY) is required")

    _banner()
    hf_token = _resolve_hf_token(args.hf_token)

    if args.dry:
        run_dry(Path(args.root).resolve(), args.ndif_api_key, hf_token)
        return

    if not args.space_url:
        p.error("--space-url (or ALETHEIA_SPACE_URL) is required")
    # --team is required only on this key's first submission; the runner enforces it.

    payload = build_zip(Path(args.root).resolve())
    submit(args.space_url, args.team, payload, args.ndif_api_key, hf_token)


if __name__ == "__main__":
    main()
