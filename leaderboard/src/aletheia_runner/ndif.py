"""Minimal NDIF account helpers.

The leaderboard authenticates a submitter by their NDIF API key, but only keys
with a usable tier can actually drive NDIF traces. ``whoami`` lets the runner ask
NDIF which account + tiers a key maps to, so it can (a) reject a key NDIF doesn't
recognise and (b) decide whether to thread the submitter's own key into their run
or substitute the shared leaderboard key.

NDIF's ``/whoami`` does NOT use HTTP error codes for bad keys:
  - valid key          -> 200 ``{"email": "you@x", "tiers": ["tier_1", ...]}``
  - unknown (well-formed) key -> 200 ``{"email": null, "tiers": []}``
  - malformed key / outage    -> connection hangs (timeout), no HTTP response
So "not a valid key" is signalled by a null ``email`` in a 200 response, while a
timeout/connection failure is an *unknown* result (treated as a transient error,
not a rejection).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

# A key needs this tier to run remote traces on NDIF. Recognised keys without it
# get the shared leaderboard key threaded into their run instead.
USABLE_TIER = "tier_1"

# Fallback whoami host if the runner has no ``ndif_host`` configured.
DEFAULT_NDIF_HOST = "https://api.ndif.us"


def whoami(api_key: str, ndif_host: str | None = None,
           timeout: float = 10.0) -> dict | None:
    """Return NDIF's whoami payload for ``api_key``, or ``None`` if unreachable.

    Calls ``GET {ndif_host}/whoami`` with the ``ndif-api-key`` header.

    Returns the parsed JSON (``{"email", "tiers"}``) for any HTTP 200 — note a
    200 with ``email: null`` means NDIF does not recognise the key (the caller
    should treat that as invalid). Returns ``None`` on a network/timeout error,
    a non-200 response, or unparseable body — an *unknown* result the caller can
    treat as transient rather than a definitive rejection.
    """
    host = (ndif_host or DEFAULT_NDIF_HOST).rstrip("/")
    req = urllib.request.Request(f"{host}/whoami",
                                 headers={"ndif-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:  # noqa: BLE001
        print(f"[ndif] whoami failed (treated as unknown): "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return None


def is_recognized(info: dict | None) -> bool:
    """True iff ``info`` is a whoami payload for a key NDIF recognises.

    ``None`` (transient/unknown) is NOT recognised-or-rejected — it's unknown;
    callers distinguish it from an explicit rejection (a dict with null email)."""
    return bool(info) and bool(info.get("email"))


def has_usable_tier(info: dict | None) -> bool:
    """True iff the whoami payload carries the tier required to run NDIF traces."""
    return bool(info) and USABLE_TIER in (info.get("tiers") or [])
