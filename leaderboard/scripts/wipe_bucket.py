#!/usr/bin/env python3
"""Wipe the leaderboard's HuggingFace storage bucket.

The bucket (``bucket://<org>/<name>`` in runner.yaml) holds three things:

  * ``results.jsonl`` — the scored leaderboard entries
  * ``teams.json``    — the NDIF-key -> team-name registry
  * ``cache/``        — the predownloaded dataset INPUT cache (Arrow), rebuilt
                        automatically on the next run if removed

By default this RESETS the leaderboard state (empties ``results.jsonl`` and
``teams.json``) but keeps the dataset cache, so the next submission doesn't have
to re-download inputs. Use ``--cache`` to also drop the cache (forces a refresh —
the documented way to pick up a re-pushed dataset), or ``--all`` to delete every
object in the bucket.

    # reset leaderboard + teams, keep the dataset cache (default)
    python scripts/wipe_bucket.py

    # also drop the predownloaded dataset cache (force re-download next run)
    python scripts/wipe_bucket.py --cache

    # nuke everything in the bucket
    python scripts/wipe_bucket.py --all

    python scripts/wipe_bucket.py --bucket NDIF/leaderboard-dev-storage --token hf_xxx --yes

The token must have WRITE access to the bucket's org.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER_YAML = ROOT / "runner.yaml"

RESULTS_FILE = "results.jsonl"
TEAMS_FILE = "teams.json"
RATE_LIMITS_FILE = "rate_limits.json"
CACHE_PREFIX = "cache/"


def default_bucket_id() -> str | None:
    """Parse ``org/name`` from runner.yaml's ``results_uri: bucket://org/name/...``."""
    if not RUNNER_YAML.is_file():
        return None
    m = re.search(r"^\s*results_uri:\s*bucket://([^/\s]+/[^/\s]+)",
                  RUNNER_YAML.read_text(), re.MULTILINE)
    return m.group(1) if m else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bucket", default=default_bucket_id(),
                   help="bucket id 'org/name' (default: parsed from runner.yaml)")
    p.add_argument("--token", default=None, help="HF token with write access to the org")
    p.add_argument("--cache", action="store_true",
                   help="also delete the predownloaded dataset cache (cache/ prefix)")
    p.add_argument("--all", action="store_true",
                   help="delete EVERY object in the bucket (implies --cache)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args(argv)

    if not args.bucket:
        p.error("--bucket is required (couldn't parse results_uri from runner.yaml)")

    from huggingface_hub import batch_bucket_files, list_bucket_tree

    tree = list(list_bucket_tree(args.bucket, recursive=True, token=args.token))
    paths = [it.path for it in tree if getattr(it, "path", None)]

    # Decide which objects to delete vs reset.
    if args.all:
        to_delete = paths
        to_reset: list[tuple[bytes, str]] = []
    else:
        to_delete = [pp for pp in paths if pp.startswith(CACHE_PREFIX)] if args.cache else []
        # Reset the state files in place (empty registry, results, rate limits).
        to_reset = [(b"{}", TEAMS_FILE), (b"", RESULTS_FILE), (b"{}", RATE_LIMITS_FILE)]

    print(f"bucket: {args.bucket}  ({len(paths)} objects)")
    if to_delete:
        print(f"  DELETE {len(to_delete)} object(s):")
        for pp in to_delete[:20]:
            print(f"    - {pp}")
        if len(to_delete) > 20:
            print(f"    ... and {len(to_delete) - 20} more")
    if to_reset:
        print(f"  RESET  {', '.join(dst for _, dst in to_reset)} (emptied)")
    if not to_delete and not to_reset:
        print("nothing to do.")
        return

    if not args.yes:
        verb = "DELETE EVERYTHING in" if args.all else "modify"
        resp = input(f"\nThis will {verb} {args.bucket}. Type 'wipe' to proceed: ")
        if resp.strip() != "wipe":
            sys.exit("aborted.")

    if to_delete:
        batch_bucket_files(args.bucket, delete=to_delete, token=args.token)
        print(f"deleted {len(to_delete)} object(s).")
    if to_reset:
        batch_bucket_files(args.bucket, add=to_reset, token=args.token)
        print("reset leaderboard state (results.jsonl + teams.json).")
    print("done.")


if __name__ == "__main__":
    main()
