"""FastAPI application: async endpoints for the ADA multi-agent system.

POST /tasks              -> enqueue the task and return a job_id (202, non-blocking)
GET  /tasks/{id}         -> job status (polling; never blocks the event loop)
POST /tasks/{id}/approve -> external human approval for HITL tasks

The API never runs the graph inside the HTTP request: it persists the job
in Redis (PENDING) and enqueues it on a Redis Stream. A separate worker
(app/worker.py) consumes the stream and executes the graph.
"""

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.worker import (
    PENDING,
    RUNNING,
    WAITING_APPROVAL,
    JobStore,
    make_store,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ADA API",
    description="API asíncrona del asistente multi-agente de decisiones de arquitectura",
    version="0.7.0",
)

_store: JobStore | None = None


async def get_store() -> JobStore:
    """Return the Redis-backed job store (created lazily, shared)."""
    global _store
    if _store is None:
        _store = await make_store()
    return _store


class TaskCreate(BaseModel):
    """Request body for POST /tasks."""

    user_request: str = Field(
        ..., min_length=10, description="Escenario técnico a analizar"
    )
    critical: bool = Field(
        default=False,
        description=(
            "Marca la tarea como crítica: el grafo se pausa (interrupt HITL) "
            "y espera aprobación humana antes de ejecutar especialistas."
        ),
    )


class ApprovalRequest(BaseModel):
    """Request body for POST /tasks/{id}/approve."""

    approved: bool = Field(..., description="true = aprobar y reanudar; false = rechazar")
    comment: str | None = Field(default=None, max_length=2000, description="Comentario opcional")


class TaskCreated(BaseModel):
    job_id: str
    status: str


class TaskStatus(BaseModel):
    job_id: str
    status: str
    user_request: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    hitl_reason: str | None = None
    error: str | None = None
    result: dict | None = None
    usage: dict | None = None


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Central error handler: log the traceback, never leak it to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health")
async def health() -> dict:
    """Liveness probe (also reports Redis connectivity)."""
    store = await get_store()
    try:
        await store.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok", "redis": redis_ok}


@app.post("/tasks", status_code=202, response_model=TaskCreated)
async def create_task(payload: TaskCreate) -> TaskCreated:
    """Receive the task, enqueue it and return its id (202, non-blocking)."""
    store = await get_store()
    job_id = await store.create_job(user_request=payload.user_request, critical=payload.critical)
    return TaskCreated(job_id=job_id, status=PENDING)


@app.get("/tasks/{job_id}", response_model=TaskStatus)
async def get_task(job_id: str) -> TaskStatus:
    """Return the job state (polling endpoint; never blocks)."""
    store = await get_store()
    data = await store.get_job(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="job not found")
    hitl_reason = data.get("hitl_reason")
    if hitl_reason:
        try:
            hitl_reason = json.loads(hitl_reason)
        except (TypeError, ValueError):
            pass
    # Resumen de tokens/costo (UsageRecorder) persistido por el worker.
    usage: dict | None = None
    if data.get("usage_tokens_available") is not None:
        raw_available = data.get("usage_tokens_available")
        available = raw_available in ("true", True, "True", "1")
        usage = {
            "tokens_available": available,
            "input_tokens": _as_int(data.get("usage_input_tokens")),
            "output_tokens": _as_int(data.get("usage_output_tokens")),
            "total_tokens": _as_int(data.get("usage_total_tokens")),
            "llm_calls": _as_int(data.get("usage_llm_calls")),
            "model": data.get("usage_model"),
            "estimated_cost_usd": _as_float(data.get("usage_estimated_cost_usd")),
            "cost_note": data.get("usage_cost_note"),
        }
    return TaskStatus(
        job_id=job_id,
        status=data.get("status") or PENDING,
        user_request=data.get("user_request"),
        created_at=_as_float(data.get("created_at")),
        updated_at=_as_float(data.get("updated_at")),
        hitl_reason=hitl_reason if data.get("status") == WAITING_APPROVAL else None,
        error=data.get("error"),
        result=json.loads(data["result_json"]) if data.get("result_json") else None,
        usage=usage,
    )


@app.post("/tasks/{job_id}/approve", response_model=TaskCreated)
async def approve_task(job_id: str, payload: ApprovalRequest) -> TaskCreated:
    """Register the human decision and enqueue the graph resume.

    - 404 when the job does not exist.
    - 409 when the job is not WAITING_APPROVAL (covers double approval).
    - The resume itself runs in the worker (same thread_id, Command(resume)).
    """
    store = await get_store()
    data = await store.get_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    # Transición atómica WAITING_APPROVAL -> RESUME_QUEUED con Lua:
    # verifica el estado y reserva la aprobación en un solo paso, sin
    # carrera entre el GET y la escritura. Solo la primera llamada con el
    # job en WAITING_APPROVAL gana; el resto recibe 409 (doble aprobación
    # o aprobación prematura) y el grafo no se reanuda dos veces.
    claimed = await store.claim_resume(job_id, WAITING_APPROVAL)
    if not claimed:
        raise HTTPException(status_code=409, detail="job is not awaiting approval")
    # Conservar la decisión y encolar la reanudación (mismo thread_id).
    await store.update_status(
        job_id, "RESUME_QUEUED", approved="true" if payload.approved else "false"
    )
    await store.enqueue_resume(job_id, approved=payload.approved)
    return TaskCreated(job_id=job_id, status=RUNNING)


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None
