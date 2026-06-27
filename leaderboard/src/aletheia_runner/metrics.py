"""InfluxDB metrics for the /submit endpoint.

Push-based: each event is written as a point to a standalone InfluxDB (its own
container, separate from the NDIF platform's). Set INFLUXDB_URL (+ TOKEN / ORG /
BUCKET) to enable; leave it unset and every record_* call is a no-op, so the app
runs without InfluxDB in dev/test — same pattern as configure_tracing/logging.

Measurements (all tagged ``service`` and ``env``; never ``team`` — that would
explode cardinality, so per-team detail stays in traces/logs):

    submission   tag outcome={scored,failed,rejected}
                 fields count(=1), duration_seconds?, upload_bytes?
    reject       tag reason={missing_api_key,too_large,registry_error,
                             rate_limited,invalid_submission}   field count(=1)
    queue_wait   field seconds          # time spent waiting for a run slot
    in_flight    field value            # gauge: submissions queued-or-running
"""
from __future__ import annotations

import os

_SERVICE = os.environ.get("SERVICE_NAME", "aletheia-runner")
_ENV = os.environ.get("ENV", "dev")

_write = None          # influxdb WriteApi, or None when not configured
_bucket: str | None = None
_org: str | None = None


def configure_metrics() -> None:
    """Open the InfluxDB write client (batched, non-blocking). Idempotent; a no-op
    when INFLUXDB_URL is unset so tests/dev need no InfluxDB."""
    global _write, _bucket, _org
    if _write is not None:
        return
    url = os.environ.get("INFLUXDB_URL")
    if not url:
        return
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import WriteOptions

    _org = os.environ.get("INFLUXDB_ORG", "aletheia")
    _bucket = os.environ.get("INFLUXDB_BUCKET", "submissions")
    client = InfluxDBClient(url=url, token=os.environ.get("INFLUXDB_TOKEN", ""), org=_org)
    # Batched background writes: points queue and flush every 2s, so a slow or down
    # InfluxDB never blocks a request; failures are retried then dropped internally.
    _write = client.write_api(write_options=WriteOptions(batch_size=50, flush_interval=2000))


def _point(measurement: str):
    from influxdb_client import Point
    return Point(measurement).tag("service", _SERVICE).tag("env", _ENV)


def _emit(point) -> None:
    if _write is None:
        return
    try:
        _write.write(bucket=_bucket, org=_org, record=point)
    except Exception:        # never let a metrics hiccup sink the request
        pass


def record_submission(outcome: str, duration_s: float | None = None,
                      upload_bytes: int | None = None) -> None:
    """One terminal submission. ``outcome`` is scored/failed/rejected; duration and
    size are recorded for runs that actually executed."""
    p = _point("submission").tag("outcome", outcome).field("count", 1)
    if duration_s is not None:
        p = p.field("duration_seconds", float(duration_s))
    if upload_bytes is not None:
        p = p.field("upload_bytes", int(upload_bytes))
    _emit(p)


def record_reject(reason: str) -> None:
    """A submission bounced before running. Also counts as a ``rejected`` outcome so
    submissions_total stays complete."""
    record_submission("rejected")
    _emit(_point("reject").tag("reason", reason).field("count", 1))


def record_queue_wait(seconds: float) -> None:
    _emit(_point("queue_wait").field("seconds", float(seconds)))


def set_in_flight(value: int) -> None:
    _emit(_point("in_flight").field("value", int(value)))
