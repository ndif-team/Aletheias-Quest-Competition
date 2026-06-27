"""Structured JSON logging for Loki ingestion.

stdout: newline-delimited JSON — Promtail/Alloy can scrape this in production.
Loki push: set LOKI_URL (e.g. http://localhost:3100) to also push directly.

``service`` and ``env`` become Loki stream labels (set via SERVICE_NAME / ENV
env vars). All extra fields (team, notebook, dataset, phase, …) are indexed as
structured metadata within the stream.

Usage:
    from .log import get_logger
    log = get_logger(__name__)
    log.info("submission.scored", extra={"team": "foo", "balanced_accuracy": 0.73})
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import sys
import threading
import urllib.error
import urllib.request

from opentelemetry import trace

_SERVICE = os.environ.get("SERVICE_NAME", "aletheia-runner")
_ENV = os.environ.get("ENV", "dev")

_SKIP = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


def _now_ns() -> int:
    return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1e9)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        doc: dict = {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond:06d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "service": _SERVICE,
            "env": _ENV,
        }
        # Correlate with Tempo: emit the active span's trace/span id (hex, no 0x)
        # so Grafana's logs<->traces links resolve. Threadpool workers see the
        # handed-off context (run_zip attaches it), so their logs link too.
        sc = trace.get_current_span().get_span_context()
        if sc.is_valid:
            doc["trace_id"] = trace.format_trace_id(sc.trace_id)
            doc["span_id"] = trace.format_span_id(sc.span_id)
        if record.exc_info:
            doc["exc_info"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k not in _SKIP and not k.startswith("_"):
                doc[k] = v
        return json.dumps(doc, default=str)


class _LokiHandler(logging.Handler):
    """Non-blocking Loki push handler.

    Log records are queued in-process; a background daemon thread drains the
    queue and POSTs to Loki's push API in small batches. Push failures are
    silently dropped so a Loki outage never affects the app.

    Stream labels are low-cardinality (service, env, level); all extra fields
    ride along as the JSON log line which Loki indexes as structured metadata.
    """

    _PUSH_URL_SUFFIX = "/loki/api/v1/push"
    _BATCH = 64           # max entries per push
    _INTERVAL = 2.0       # seconds between flushes when idle
    _TIMEOUT = 5          # HTTP timeout

    def __init__(self, loki_url: str) -> None:
        super().__init__()
        self._url = loki_url.rstrip("/") + self._PUSH_URL_SUFFIX
        self._q: queue.Queue[tuple[int, str, str]] = queue.Queue(maxsize=4096)
        self._formatter = _JsonFormatter()
        t = threading.Thread(target=self._drain, daemon=True, name="loki-drain")
        t.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self._formatter.format(record)
            self._q.put_nowait((_now_ns(), record.levelname.lower(), line))
        except queue.Full:
            pass                       # Loki backpressure: drop rather than block the app
        except Exception:
            self.handleError(record)   # a real bug (e.g. formatter); surface, don't swallow

    def _drain(self) -> None:
        while True:
            entries: list[tuple[int, str, str]] = []
            try:
                entries.append(self._q.get(timeout=self._INTERVAL))
            except queue.Empty:
                continue
            while len(entries) < self._BATCH:
                try:
                    entries.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._push(entries)

    def _push(self, entries: list[tuple[int, str, str]]) -> None:
        # Group by level so each Loki stream has {service, env, level} labels.
        by_level: dict[str, list] = {}
        for ns, level, line in entries:
            by_level.setdefault(level, []).append([str(ns), line])

        streams = [
            {"stream": {"service": _SERVICE, "env": _ENV, "level": level},
             "values": vals}
            for level, vals in by_level.items()
        ]
        body = json.dumps({"streams": streams}).encode()
        req = urllib.request.Request(
            self._url, data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT):
                pass
        except (urllib.error.URLError, OSError):
            pass


def configure_logging(level: int = logging.INFO,
                      loki_url: str | None = None) -> None:
    """Install JSON stdout + optional Loki push handler on the root logger.

    Idempotent: subsequent calls are no-ops so tests that build multiple app
    instances don't stack handlers. Pass ``loki_url`` or set ``LOKI_URL`` in
    the environment to enable direct push.
    """
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) and
           isinstance(getattr(h, "formatter", None), _JsonFormatter)
           for h in root.handlers):
        return

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(_JsonFormatter())
    root.setLevel(level)
    root.handlers = [stdout_handler]

    url = loki_url or os.environ.get("LOKI_URL")
    if url:
        root.addHandler(_LokiHandler(url))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
