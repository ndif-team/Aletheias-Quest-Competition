"""OTEL tracing setup for Tempo ingestion.

Call configure_tracing() once at startup (app.py). Set OTEL_EXPORTER_OTLP_ENDPOINT
(e.g. http://localhost:4318) to enable export; omit it and traces are no-ops so
the app works without Tempo in dev/test.

Usage:
    from .trace import get_tracer, current_context
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("my.operation") as span:
        span.set_attribute("team", team)
        ...

Thread handoff (async → threadpool):
    ctx = current_context()                        # capture in async handler
    await run_in_threadpool(fn, ctx, ...)          # pass to thread
    # inside fn:
    token = attach_context(ctx)
    try:
        ...                                        # spans here nest under root
    finally:
        detach_context(token)
"""

from __future__ import annotations

import os

from opentelemetry import context, trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_SERVICE = os.environ.get("SERVICE_NAME", "aletheia-runner")
_ENV = os.environ.get("ENV", "dev")

_provider: TracerProvider | None = None


def configure_tracing() -> None:
    """Install the OTEL tracer provider. Idempotent."""
    global _provider
    if _provider is not None:
        return

    resource = Resource.create({"service.name": _SERVICE, "deployment.environment": _ENV})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _provider = provider


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def current_context() -> Context:
    """Capture the active OTEL context — call this in the async handler before
    handing off to a thread pool so the thread can attach it."""
    return context.get_current()


def attach_context(ctx: Context):
    """Attach a captured context in the current thread. Returns a token for detach."""
    return context.attach(ctx)


def detach_context(token) -> None:
    context.detach(token)
