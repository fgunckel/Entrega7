"""Observability layer: active tracing of every execution to Arize Phoenix.

Initializes the OpenInference LangChain instrumentor so every graph node,
LLM call and tool call is exported as a trace to the Phoenix collector.
Also exposes a decorator for ad-hoc LLM callables that are not covered by
the automatic instrumentation.
"""

import functools
import inspect
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

_TRACING_INITIALIZED = False
TRACING_ENABLED = False

F = TypeVar("F", bound=Callable[..., Any])


def init_observability() -> bool:
    """Initialize OpenInference instrumentation and export traces to Phoenix.

    Reads configuration from environment variables (see .env.example):
    - PHOENIX_COLLECTOR_ENDPOINT: OTLP endpoint of the Phoenix collector.
    - PHOENIX_PROJECT_NAME: project name shown in the dashboard.

    Returns True when tracing is active, False when it is disabled or the
    endpoint is unreachable (the API must not crash without monitoring).
    """
    global _TRACING_INITIALIZED, TRACING_ENABLED

    if _TRACING_INITIALIZED:
        return TRACING_ENABLED

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    project = os.getenv("PHOENIX_PROJECT_NAME", "ada-api")

    if not endpoint:
        logger.warning("Observability disabled: PHOENIX_COLLECTOR_ENDPOINT not set")
        return False

    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource({"project.name": project}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)

        LangChainInstrumentor().instrument()
        _TRACING_INITIALIZED = True
        logger.info("Observability active: %s (project=%s)", endpoint, project)
        return True
    except Exception as error:  # pragma: no cover - depends on network
        logger.error("Failed to init observability: %s", error)
        return False


def set_job_context(job_id: str) -> None:
    """Attach the job_id to the current trace context (session attribute).

    Phoenix groups spans by session when openinference.session.id is set,
    so every execution can be followed end-to-end in the dashboard.
    """
    try:
        from opentelemetry import baggage, context, trace

        ctx = baggage.set_baggage("openinference.session.id", job_id)
        context.attach(ctx)
        tracer = trace.get_tracer("ada")
        span = tracer.start_span("ada.job")
        span.set_attribute("ada.job_id", job_id)
        span.end()
    except Exception:  # pragma: no cover - telemetry must never break a job
        logger.debug("set_job_context failed (telemetry off?)", exc_info=True)


def traced(name: str) -> Callable[[F], F]:
    """Decorator that adds an OpenTelemetry span around a sync or async call.

    Used for functions that OpenInference does not instrument automatically
    (e.g. custom scoring helpers), so every node of the graph leaves a trace.
    """

    def decorator(func: F) -> F:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = trace.get_tracer("ada")
                with tracer.start_as_current_span(name) as span:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as error:
                        span.record_exception(error)
                        span.set_status(Status(StatusCode.ERROR, str(error)))
                        raise

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = trace.get_tracer("ada")
            with tracer.start_as_current_span(name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    raise

        return cast(F, sync_wrapper)

    return decorator
