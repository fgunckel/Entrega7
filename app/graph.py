"""ADA orchestrator (Módulo 6) with Redis persistence and HITL.

Wraps the multi-agent StateGraph from the previous delivery and adds:
- a Redis checkpointer (RedisSaver) so every execution can be resumed,
- a Human-in-the-loop interrupt node (see hitl.py) wired between START
  and the supervisor: critical tasks pause there (real interrupt)
  before any specialist runs; non-critical tasks pass through.

The graph itself is unchanged: supervisor-routed hierarchical workflow
(Supervisor -> Researcher -> Analyst -> Validator -> Synthesizer).
"""

import logging

from app.hitl import make_hitl_node

logger = logging.getLogger(__name__)

# Checkpointer compartido por proceso (API y worker tienen uno cada uno).
# Lazy: se crea en el primer uso y se reutiliza en todas las ejecuciones.
_shared_checkpointer = None


def get_checkpointer():
    """Return the process-wide checkpointer (created once, reused).

    Uses RedisSaver when REDIS_URL is set; falls back to an in-memory
    MemorySaver only when Redis is NOT configured (tests, local runs
    without Docker, or ADA_USE_FAKEREDIS=1). If Redis is configured but
    the checkpointer cannot initialize, raise: the worker must not run
    the graph silently without persistence. Sharing one instance per
    process is required for HITL: the resume must read the checkpoint
    written by the initial run.
    """
    global _shared_checkpointer
    if _shared_checkpointer is None:
        import os

        fake_redis = os.getenv("ADA_USE_FAKEREDIS", "").strip() in ("1", "true")
        redis_url = os.getenv("REDIS_URL", "").strip()
        # Override explícito: ADA_CHECKPOINTER=memory fuerza MemorySaver
        # (solo pruebas locales sin Redis Stack; los checkpoints viven
        # mientras vive el proceso worker y no hay persistencia real).
        force_memory = (
            os.getenv("ADA_CHECKPOINTER", "").strip().lower() == "memory"
        )
        if redis_url and not fake_redis and not force_memory:
            checkpointer = build_checkpointer()
            if checkpointer is None:
                # Redis configurado pero no inicializable: no ejecutar el
                # grafo sin persistencia (los checkpoints HITL se perderían).
                raise RuntimeError(
                    "REDIS_URL está configurado pero el checkpointer RedisSaver "
                    "no pudo inicializarse; el worker no ejecuta el grafo sin "
                    "persistencia. Revisá la conectividad a Redis."
                )
            _shared_checkpointer = checkpointer
        else:
            from langgraph.checkpoint.memory import MemorySaver

            if force_memory:
                logger.warning(
                    "ADA_CHECKPOINTER=memory: usando MemorySaver en memoria "
                    "(override explícito; los checkpoints no persisten)"
                )
            else:
                logger.warning("REDIS_URL no configurado: usando MemorySaver en memoria")
            _shared_checkpointer = MemorySaver()
    return _shared_checkpointer


def build_checkpointer():
    """Create the Redis checkpointer for LangGraph (RedisSaver).

    Returns None when REDIS_URL is not configured, so the graph still
    runs in-memory (unit tests, CLI without Redis). The API and the
    worker each hold one long-lived instance; call .close() on shutdown.
    """
    import os

    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None

    try:
        from langgraph.checkpoint.redis import RedisSaver

        # from_conn_string returns a context manager; entering it builds
        # the underlying sync Redis client. setup() creates the indices.
        cm = RedisSaver.from_conn_string(url)
        saver = cm.__enter__()
        saver.setup()
        return saver
    except Exception as error:  # pragma: no cover - depends on network
        logger.error("RedisSaver init failed: %s", error)
        return None


def build_graph(chat_model, encoder=None, documents=None, checkpointer=None):
    """Build the ADA StateGraph with Redis checkpointer and HITL node.

    Reuses the graph builder from the Módulo 6 package (ada.graph) and
    wires the human-approval node between START and supervisor:
    - critical task -> real interrupt() (checkpoint saved, job paused),
    - rejected task -> limited_finish (safe stop, no specialist runs),
    - normal task -> supervisor (behavior identical to Módulo 6).
    """

    from ada.graph import build_graph as build_m6_graph

    graph = build_m6_graph(chat_model, encoder=encoder, documents=documents)

    if chat_model is not None:
        graph.add_node("hitl_approval", make_hitl_node())
        # START -> hitl_approval (en vez de START -> supervisor directo).
        graph.edges.remove(("__start__", "supervisor"))
        graph.edges.add(("__start__", "hitl_approval"))

        def route_after_hitl(state: dict) -> str:
            """Rejected tasks stop safely; the rest continue to the supervisor."""
            if state.get("hitl_rejected"):
                return "limited_finish"
            return "supervisor"

        graph.add_conditional_edges(
            "hitl_approval",
            route_after_hitl,
            {"supervisor": "supervisor", "limited_finish": "limited_finish"},
        )

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
