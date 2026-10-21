"""Worker independiente: consume la cola Redis y ejecuta el grafo ADA.

Separado de los endpoints HTTP: la API solo encola; este proceso (o tarea
de fondo de larga vida) consume la cola, ejecuta el grafo y actualiza el
estado del trabajo en Redis.

Ciclo de vida: PENDING -> RUNNING -> (WAITING_APPROVAL) -> DONE | FAILED.

Cola: Redis Streams (XADD/XREADGROUP) con consumer group y claim de
mensajes pendientes (XAUTOCLAIM) para recuperar trabajos de procesos
muertos. La reanudación HITL usa una cola separada (resume) para no
mezclar ejecuciones nuevas con reanudaciones.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Estados de un job en Redis.
PENDING = "PENDING"
RUNNING = "RUNNING"
WAITING_APPROVAL = "WAITING_APPROVAL"
DONE = "DONE"
FAILED = "FAILED"

# Claves Redis.
STREAM_KEY = "ada:jobs"          # Stream de trabajos nuevos.
RESUME_STREAM_KEY = "ada:jobs:resume"  # Stream de reanudaciones HITL.
GROUP = "ada-workers"
CONSUMER = "worker-1"
JOB_TTL_SECONDS = 24 * 3600

# Estados terminales (el polling del cliente se detiene aquí).
TERMINAL_STATES = (DONE, FAILED)


class JobStore:
    """Estado operacional del trabajo en Redis (hash ada:job:{id})."""

    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    async def create_job(self, user_request: str, critical: bool = False) -> str:
        """Create a PENDING job, enqueue it and return its id."""
        job_id = uuid.uuid4().hex
        key = f"ada:job:{job_id}"
        await self.redis.hset(
            key,
            mapping={
                "status": PENDING,
                "user_request": user_request,
                "critical": "true" if critical else "false",
                "created_at": str(time.time()),
                "updated_at": str(time.time()),
            },
        )
        await self.redis.expire(key, JOB_TTL_SECONDS)
        # Encolar en el stream (XADD). El worker lo consume con XREADGROUP.
        await self.redis.xadd(STREAM_KEY, {"job_id": job_id})
        return job_id

    async def enqueue_resume(self, job_id: str, approved: bool) -> None:
        """Enqueue a HITL resume request on the resume stream."""
        await self.redis.xadd(
            RESUME_STREAM_KEY, {"job_id": job_id, "approved": "true" if approved else "false"}
        )

    async def get_job(self, job_id: str) -> dict | None:
        """Return the job fields, or None when the job does not exist."""
        data = await self.redis.hgetall(f"ada:job:{job_id}")
        if not data:
            return None
        return {
            key.decode() if isinstance(key, bytes) else key: value.decode()
            if isinstance(value, bytes)
            else value
            for key, value in data.items()
        }

    async def update_status(self, job_id: str, status: str, **fields: Any) -> None:
        """Update the job status (and optional extra fields) in Redis."""
        mapping: dict[str, Any] = {"status": status, "updated_at": str(time.time()), **fields}
        await self.redis.hset(f"ada:job:{job_id}", mapping=mapping)

    async def claim_resume(self, job_id: str, expected_status: str) -> bool:
        """Atomically claim the approval of a job in WAITING_APPROVAL.

        Uses optimistic locking (WATCH/MULTI/EXEC): the status is read
        and, only if it still matches `expected_status` at commit time,
        the transition to RESUME_QUEUED is applied — as a single atomic
        transaction (no race between read and write). Returns True when
        this call won the claim; False otherwise (double approval,
        premature approval, or job in another state).
        """
        key = f"ada:job:{job_id}"
        async with self.redis.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            current = await pipe.hget(key, "status")
            if current != expected_status:
                # El estado cambió (o el job no existe): abortar sin escribir.
                await pipe.unwatch()
                return False
            pipe.multi()
            pipe.hset(key, "status", "RESUME_QUEUED")
            await pipe.execute()
        return True

    async def claim_pending(
        self, consumer: str, min_idle_ms: int = 60_000
    ) -> list[tuple[str, tuple]]:
        """Recover pending stream messages idle for too long (XAUTOCLAIM).

        Covers the failure mode where a worker crashed after XREADGROUP
        delivered a message but before finishing the job. Returns
        (stream_key, (msg_id, payload)) tuples so the caller can XACK
        the message on its original stream after processing.
        """
        claimed: list[tuple[str, tuple]] = []
        for key in (STREAM_KEY, RESUME_STREAM_KEY):
            cursor = "0-0"
            while True:
                result = await self.redis.xautoclaim(
                    key, GROUP, consumer, min_idle_ms, cursor, count=10
                )
                # redis-py 5: (next_cursor, messages, deleted_ids)
                next_cursor, messages = result[0], result[1]
                claimed.extend((key, message) for message in messages)
                if next_cursor in ("0-0", 0, b"0-0"):
                    break
                cursor = next_cursor
        return claimed


def build_job_store(redis_client) -> JobStore:
    """Build a JobStore over an async Redis client."""
    return JobStore(redis_client)


def _build_chat_model():
    """Create the Groq chat model from environment settings."""
    from langchain_groq import ChatGroq

    from ada.config import get_settings

    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def _initial_state(user_request: str) -> dict:
    """Build the initial AgentState for a new execution (same as the CLI)."""
    from ada.decision import build_criteria, extract_decision

    decision_spec = extract_decision(user_request).model_dump()
    requested = decision_spec["requested_criteria"]
    criteria = (
        [c.model_dump() for c in build_criteria(requested)] if requested else []
    )
    return {
        "user_request": user_request,
        "system_profile": {},
        "criteria": criteria,
        "decision_spec": decision_spec,
        "next_agent": "researcher",
        "routing_reason": "inicio",
        "steps": 0,
        "retries": 0,
        "task_completed": False,
    }


def _run_graph_sync(graph, initial_state: dict, config: dict, recorder) -> dict:
    """Run the graph synchronously (called from a worker thread).

    Accumulates the token usage reported by the provider into `recorder`
    (UsageRecorder) from the messages emitted by each node update.
    """
    final_state: dict = {}
    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node, update in (chunk or {}).items():
            if node == "__interrupt__":
                # El grafo quedó pausado en el HITL: propagar la señal.
                final_state["__interrupt__"] = update
            elif isinstance(update, dict):
                final_state = {**final_state, **update}
            for message in _iter_messages(update):
                if getattr(message, "usage_metadata", None):
                    recorder.add(message.usage_metadata)
    return final_state


def _resume_graph_sync(graph, resume_value: dict, config: dict, recorder) -> dict:
    """Resume a checkpointed graph with Command(resume=...) (worker thread)."""
    from langgraph.types import Command

    final_state: dict = {}
    for chunk in graph.stream(
        Command(resume=resume_value), config=config, stream_mode="updates"
    ):
        for node, update in (chunk or {}).items():
            if node == "__interrupt__":
                final_state["__interrupt__"] = update
            elif isinstance(update, dict):
                final_state = {**final_state, **update}
            for message in _iter_messages(update):
                if getattr(message, "usage_metadata", None):
                    recorder.add(message.usage_metadata)
    return final_state


def _iter_messages(update: Any):
    """Yield message objects from a node update (lists with messages)."""
    if not isinstance(update, dict):
        return
    for value in update.values():
        if isinstance(value, list):
            for item in value:
                if hasattr(item, "usage_metadata"):
                    yield item


async def process_new_job(job_id: str, store: JobStore) -> None:
    """Execute a new job: run the graph and persist DONE/FAILED/WAITING_APPROVAL."""
    from app.graph import build_graph, get_checkpointer

    data = await store.get_job(job_id)
    if not data:
        logger.warning("Job %s desapareció de Redis; se omite", job_id)
        return
    if data.get("status") not in (PENDING, RUNNING):
        # Ya procesado o reanudado por otro camino: evitar doble ejecución.
        return

    await store.update_status(job_id, RUNNING)
    logger.info("Job %s: ejecutando grafo", job_id)
    try:
        from app.cost import UsageRecorder
        from app.observability import set_job_context

        set_job_context(job_id)
        chat_model = _build_chat_model()
        checkpointer = get_checkpointer()
        graph = build_graph(chat_model, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": job_id}}
        initial_state = _initial_state(data["user_request"])
        if data.get("critical") == "true":
            # Tarea crítica: el nodo HITL pausará el grafo (interrupt real).
            initial_state["hitl_required"] = True
        recorder = UsageRecorder()
        final_state = await asyncio.to_thread(
            _run_graph_sync, graph, initial_state, config, recorder
        )
        await _persist_outcome(job_id, final_state, store, recorder)
    except Exception as error:
        logger.exception("Job %s failed", job_id)
        await store.update_status(job_id, FAILED, error=str(error))


async def process_resume(job_id: str, approved: bool, store: JobStore) -> None:
    """Resume a WAITING_APPROVAL job with the human decision (same thread_id)."""
    from app.graph import build_graph, get_checkpointer

    data = await store.get_job(job_id)
    if not data:
        logger.warning("Job %s desapareció de Redis; se omite", job_id)
        return
    if data.get("status") != "RESUME_QUEUED":
        # Doble aprobación o trabajo ya reanudado: no reanudar dos veces.
        logger.info(
            "Job %s no está en RESUME_QUEUED (%s); se omite", job_id, data.get("status")
        )
        return

    await store.update_status(job_id, RUNNING, approved="true" if approved else "false")
    try:
        from app.cost import UsageRecorder
        from app.observability import set_job_context

        set_job_context(job_id)
        chat_model = _build_chat_model()
        checkpointer = get_checkpointer()
        graph = build_graph(chat_model, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": job_id}}
        recorder = UsageRecorder()
        final_state = await asyncio.to_thread(
            _resume_graph_sync, graph, {"approved": approved}, config, recorder
        )
        await _persist_outcome(job_id, final_state, store, recorder)
    except Exception as error:
        logger.exception("Resume of job %s failed", job_id)
        await store.update_status(job_id, FAILED, error=str(error))


async def _persist_outcome(
    job_id: str, final_state: dict, store: JobStore, recorder
) -> None:
    """Persist the graph outcome: DONE, WAITING_APPROVAL or FAILED.

    Also persists the token/cost summary (UsageRecorder) when the
    provider reported usage; otherwise marks the data as unavailable.
    """
    usage_summary = recorder.summary(model=_current_model_name())
    usage_fields = {}
    for key, value in usage_summary.items():
        if isinstance(value, bool):
            usage_fields[f"usage_{key}"] = "true" if value else "false"
        elif isinstance(value, (int, float, str)):
            usage_fields[f"usage_{key}"] = value
        else:
            usage_fields[f"usage_{key}"] = json.dumps(value, ensure_ascii=False)
    # El grafo quedó pausado en el interrupt HITL: registrar WAITING_APPROVAL.
    if final_state.get("__interrupt__"):
        await store.update_status(
            job_id,
            WAITING_APPROVAL,
            hitl_reason=json.dumps(
                final_state.get("hitl_reason") or "Requiere aprobación humana",
                ensure_ascii=False,
            ),
            **usage_fields,
        )
        return
    report = final_state.get("final_report") or {}
    await store.update_status(
        job_id,
        DONE,
        result_json=json.dumps(report, ensure_ascii=False, default=str),
        **usage_fields,
    )


def _current_model_name() -> str | None:
    """Return the configured LLM model name (for cost rates), or None."""
    try:
        from ada.config import get_settings

        return get_settings().llm_model
    except Exception:
        return None


async def run_worker(poll_interval: float = 1.0) -> None:
    """Long-lived worker loop: consume both streams and process jobs.

    Runs forever; designed to be started as a separate process
    (python -m app.worker) or inside the API container as a background task.
    """
    from app.observability import init_observability

    init_observability()
    store = await make_store()
    # Consumer group: crear si no existe (ignorar BUSYGROUP).
    for key in (STREAM_KEY, RESUME_STREAM_KEY):
        try:
            await store.redis.xgroup_create(key, GROUP, id="0", mkstream=True)
        except Exception:
            pass  # BUSYGROUP: el grupo ya existe.

    logger.info("Worker iniciado (streams=%s)", [STREAM_KEY, RESUME_STREAM_KEY])
    while True:
        try:
            # Recuperar primero trabajos pendientes de procesos muertos.
            for stream, message in await store.claim_pending(CONSUMER):
                payload = message[1]
                await _dispatch(payload, store)
                # Confirmar el mensaje: el job quedó en un estado controlado
                # (DONE/FAILED/WAITING_APPROVAL) o fue omitido con seguridad.
                await store.redis.xack(stream, GROUP, message[0])
            # Leer nuevos mensajes de ambos streams.
            result = await store.redis.xreadgroup(
                GROUP, CONSUMER, {STREAM_KEY: ">", RESUME_STREAM_KEY: ">"}, count=10,
                block=int(poll_interval * 1000),
            )
            for stream, messages in result or []:
                for msg_id, payload in messages:
                    await _dispatch(payload, store)
                    # XACK tras procesar: si el worker muere antes, el
                    # mensaje queda pendiente y XAUTOCLAIM lo recupera.
                    await store.redis.xack(stream, GROUP, msg_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error en el loop del worker; reintentando")
            await asyncio.sleep(poll_interval)


async def _dispatch(payload: dict, store: JobStore) -> None:
    """Route a stream message to new-job or resume processing."""
    job_id = payload.get("job_id")
    if not job_id:
        return
    if payload.get("approved") is not None:
        approved = payload["approved"] in ("true", True, b"true")
        await process_resume(job_id, approved, store)
    else:
        await process_new_job(job_id, store)


async def make_store() -> JobStore:
    """Create the JobStore over a real or fake Redis client (shared config)."""
    import os

    if os.getenv("ADA_USE_FAKEREDIS", "").strip() in ("1", "true"):
        import fakeredis

        client = fakeredis.FakeAsyncRedis(decode_responses=True)
    else:
        import redis.asyncio as redis

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(url, decode_responses=True)
    return JobStore(client)
