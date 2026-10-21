"""Integration tests: API, worker, HITL and concurrency.

All tests use fakeredis (in-memory Redis) and fake chat models: no real
Redis server, no Groq key, no network. They verify the full job lifecycle
through the FastAPI app and the worker functions.
"""

import os
import unittest.mock

os.environ.setdefault("ADA_USE_FAKEREDIS", "1")
os.environ.pop("REDIS_URL", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.main import app  # noqa: E402
from app.worker import (  # noqa: E402
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    WAITING_APPROVAL,
    JobStore,
    _initial_state,
    process_new_job,
    process_resume,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeChatModel:
    """Routes deterministically from the supervisor context text."""

    def invoke(self, messages):
        text = str(messages[-1][1] if isinstance(messages[-1], tuple) else messages[-1])
        if "Research artifact present: False" in text:
            nxt = "researcher"
        elif "Analysis artifact present: False" in text:
            nxt = "analyst"
        elif "Validation result: none" in text:
            nxt = "validator"
        elif "is_valid=True" in text:
            nxt = "synthesizer"
        else:
            nxt = "limited_finish"

        class Response:
            content = (
                f'{{"next_agent": "{nxt}", "reason": "fake routing", '
                f'"task_completed": false}}'
            )

        return Response()


class BrokenChatModel:
    def invoke(self, messages):
        raise ConnectionError("network down")


USER_REQUEST = (
    "Decidir si una API .NET debe ser monolito modular o microservicios. "
    "El equipo es pequeño y el dominio está en evolución."
)


@pytest.fixture
def store():
    """A JobStore over a fresh fakeredis client."""
    import fakeredis

    return JobStore(fakeredis.FakeAsyncRedis(decode_responses=True))


@pytest.fixture
def client():
    """FastAPI TestClient with a fresh fakeredis-backed store."""
    import fakeredis

    import app.main as main_module

    fake_store = JobStore(fakeredis.FakeAsyncRedis(decode_responses=True))
    main_module._store = fake_store
    with TestClient(app) as test_client:
        yield test_client
    main_module._store = None


# ---------------------------------------------------------------------------
# API: POST /tasks, GET /tasks/{id}
# ---------------------------------------------------------------------------


class TestApi:
    def test_create_task_returns_202_and_job_id(self, client):
        response = client.post("/tasks", json={"user_request": USER_REQUEST})
        assert response.status_code == 202
        body = response.json()
        assert body["job_id"]
        assert body["status"] == PENDING

    def test_create_task_validates_payload(self, client):
        # user_request demasiado corto -> 422 de Pydantic.
        response = client.post("/tasks", json={"user_request": "corto"})
        assert response.status_code == 422

    def test_get_task_unknown_returns_404(self, client):
        response = client.get("/tasks/no-existe")
        assert response.status_code == 404

    def test_get_task_reflects_status(self, client):
        # Crear el job vía API (misma instancia de fakeredis que la app).
        created = client.post("/tasks", json={"user_request": USER_REQUEST})
        job_id = created.json()["job_id"]
        response = client.get(f"/tasks/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == PENDING
        assert data["user_request"] == USER_REQUEST
        assert data["created_at"] is not None

    def test_health_reports_redis(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["redis"] is True


# ---------------------------------------------------------------------------
# Worker: ciclo de vida del job
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    @pytest.mark.asyncio
    async def test_new_job_completes_with_done(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await process_new_job(job_id, store)
        data = await store.get_job(job_id)
        assert data["status"] == DONE
        # result_json contiene el informe final (final_report) serializado.
        assert '"status": "complete"' in data["result_json"]

    @pytest.mark.asyncio
    async def test_agent_exception_does_not_hang(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=BrokenChatModel()):
            await process_new_job(job_id, store)
        data = await store.get_job(job_id)
        # ADA captura el error del LLM y termina de forma segura
        # (limited_finish): el job nunca queda colgado en RUNNING.
        assert data["status"] in (DONE, FAILED)
        assert data["status"] != RUNNING

    @pytest.mark.asyncio
    async def test_worker_crash_marks_failed(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST)
        # Fallo fuera del grafo (p.ej. construyendo el modelo): FAILED.
        with unittest.mock.patch(
            "app.worker._build_chat_model", side_effect=RuntimeError("boom")
        ):
            await process_new_job(job_id, store)
        data = await store.get_job(job_id)
        assert data["status"] == FAILED
        assert "boom" in data["error"]

    @pytest.mark.asyncio
    async def test_double_execution_is_prevented(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await process_new_job(job_id, store)
            first = await store.get_job(job_id)
            # Segunda ejecución del mismo job: no debe reprocesar (status DONE).
            await process_new_job(job_id, store)
            second = await store.get_job(job_id)
        assert first["status"] == DONE
        assert second["status"] == DONE

    @pytest.mark.asyncio
    async def test_missing_job_is_skipped(self, store):
        # No debe lanzar excepción aunque el job no exista.
        await process_new_job("job-inexistente", store)


# ---------------------------------------------------------------------------
# HITL: pausa, aprobación, rechazo, duplicado
# ---------------------------------------------------------------------------


class TestHitl:
    @pytest.mark.asyncio
    async def test_critical_task_pauses_and_resumes(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST, critical=True)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await process_new_job(job_id, store)
            data = await store.get_job(job_id)
            assert data["status"] == WAITING_APPROVAL

            # Aprobación: encolar y reanudar (mismo thread_id).
            await store.enqueue_resume(job_id, approved=True)
            await store.redis.hset(f"ada:job:{job_id}", "status", "RESUME_QUEUED")
            await process_resume(job_id, True, store)
            data = await store.get_job(job_id)
            assert data["status"] == DONE

    @pytest.mark.asyncio
    async def test_rejection_stops_safely(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST, critical=True)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await process_new_job(job_id, store)
            assert (await store.get_job(job_id))["status"] == WAITING_APPROVAL

            await store.enqueue_resume(job_id, approved=False)
            await store.redis.hset(f"ada:job:{job_id}", "status", "RESUME_QUEUED")
            await process_resume(job_id, False, store)
            data = await store.get_job(job_id)
            assert data["status"] == DONE
            # El rechazo termina en limited_finish (status "limited").
            assert '"status": "limited"' in data["result_json"] or "limited" in data["result_json"]

    @pytest.mark.asyncio
    async def test_double_resume_is_prevented(self, store):
        import unittest.mock

        job_id = await store.create_job(user_request=USER_REQUEST, critical=True)
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await process_new_job(job_id, store)
            await store.redis.hset(f"ada:job:{job_id}", "status", "RESUME_QUEUED")
            await process_resume(job_id, True, store)
            first = await store.get_job(job_id)
            # Segunda reanudación: el job ya no está en RESUME_QUEUED.
            await process_resume(job_id, True, store)
            second = await store.get_job(job_id)
        assert first["status"] == DONE
        assert second == first  # sin cambios

    def test_approve_endpoint_rejects_non_waiting_job(self, client):
        # Job en PENDING (no WAITING_APPROVAL): el endpoint debe dar 409.
        created = client.post("/tasks", json={"user_request": USER_REQUEST})
        job_id = created.json()["job_id"]
        response = client.post(f"/tasks/{job_id}/approve", json={"approved": True})
        assert response.status_code == 409

    def test_approve_endpoint_unknown_job_returns_404(self, client):
        response = client.post("/tasks/no-existe/approve", json={"approved": True})
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_approve_accepts_rejects_and_double_approval(self, store):
        """Aprobación, rechazo y doble aprobación vía endpoint (transición atómica)."""
        import unittest.mock

        import httpx

        import app.main as main_module

        main_module._store = store  # la API usa el mismo fakeredis del test
        try:
            transport = httpx.ASGITransport(app=main_module.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as http:
                # --- Aprobación ---
                job_id = (
                    await http.post(
                        "/tasks", json={"user_request": USER_REQUEST, "critical": True}
                    )
                ).json()["job_id"]
                with unittest.mock.patch(
                    "app.worker._build_chat_model", return_value=FakeChatModel()
                ):
                    await process_new_job(job_id, store)
                status = (await http.get(f"/tasks/{job_id}")).json()["status"]
                assert status == WAITING_APPROVAL
                # Aprobar: 200 y pasa a RESUME_QUEUED (reanudación encolada).
                ok = await http.post(f"/tasks/{job_id}/approve", json={"approved": True})
                assert ok.status_code == 200
                # Doble aprobación: 409 (la transición atómica ya consumió el estado).
                dup = await http.post(f"/tasks/{job_id}/approve", json={"approved": True})
                assert dup.status_code == 409

                # --- Rechazo ---
                job_id2 = (
                    await http.post(
                        "/tasks", json={"user_request": USER_REQUEST, "critical": True}
                    )
                ).json()["job_id"]
                with unittest.mock.patch(
                    "app.worker._build_chat_model", return_value=FakeChatModel()
                ):
                    await process_new_job(job_id2, store)
                assert (
                    (await http.get(f"/tasks/{job_id2}")).json()["status"]
                    == WAITING_APPROVAL
                )
                rejected = await http.post(
                    f"/tasks/{job_id2}/approve", json={"approved": False}
                )
                assert rejected.status_code == 200
                # Doble envío sobre el job rechazado: 409.
                dup2 = await http.post(
                    f"/tasks/{job_id2}/approve", json={"approved": False}
                )
                assert dup2.status_code == 409
        finally:
            main_module._store = None

    @pytest.mark.asyncio
    async def test_premature_then_valid_approval(self, store):
        """Aprobación prematura (job no pausado) -> 409; luego, con el job
        en WAITING_APPROVAL, la aprobación válida -> 200 y RESUME_QUEUED."""
        import unittest.mock

        import httpx

        import app.main as main_module

        main_module._store = store
        try:
            transport = httpx.ASGITransport(app=main_module.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as http:
                job_id = (
                    await http.post(
                        "/tasks", json={"user_request": USER_REQUEST, "critical": True}
                    )
                ).json()["job_id"]
                # --- Aprobación prematura: el job aún está en PENDING ---
                # (nadie lo procesó): la transición atómica debe rechazarla.
                early = await http.post(
                    f"/tasks/{job_id}/approve", json={"approved": True}
                )
                assert early.status_code == 409
                data = await store.get_job(job_id)
                assert data["status"] == PENDING  # sin transición espuria

                # --- Aprobación válida: pausar el job y aprobar ---
                with unittest.mock.patch(
                    "app.worker._build_chat_model", return_value=FakeChatModel()
                ):
                    await process_new_job(job_id, store)
                status = (await http.get(f"/tasks/{job_id}")).json()["status"]
                assert status == WAITING_APPROVAL
                ok = await http.post(f"/tasks/{job_id}/approve", json={"approved": True})
                assert ok.status_code == 200
                data = await store.get_job(job_id)
                assert data["status"] == "RESUME_QUEUED"
                assert data.get("approved") == "true"
        finally:
            main_module._store = None


# ---------------------------------------------------------------------------
# Checkpointer y estado inicial
# ---------------------------------------------------------------------------


class TestGraphAndState:
    def test_initial_state_shape(self):
        state = _initial_state(USER_REQUEST)
        assert state["user_request"] == USER_REQUEST
        assert state["next_agent"] == "researcher"
        assert state["steps"] == 0
        assert "hitl_required" not in state  # lo agrega el worker si es crítico

    def test_graph_compiles_with_checkpointer(self):
        from app.graph import build_graph

        graph = build_graph(FakeChatModel(), checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_normal_flow_reaches_report(self):
        from app.graph import build_graph

        graph = build_graph(FakeChatModel(), checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "it-normal"}}
        final = graph.invoke(_initial_state(USER_REQUEST), config=config)
        assert final.get("final_report")

    def test_graph_interrupt_and_resume(self):
        from app.graph import build_graph

        graph = build_graph(FakeChatModel(), checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "it-hitl"}}
        initial = {**_initial_state(USER_REQUEST), "hitl_required": True}
        paused = graph.invoke(initial, config=config)
        assert paused.get("__interrupt__"), "debe pausarse en el interrupt"
        resumed = graph.invoke(Command(resume={"approved": True}), config=config)
        assert resumed.get("final_report")
        assert resumed.get("hitl_approved") is True

    def test_checkpointer_fails_loudly_when_redis_configured_but_unreachable(self):
        """Redis configurado + inalcanzable: error, nunca correr sin persistencia."""
        import app.graph as graph_module

        graph_module._shared_checkpointer = None  # reset del singleton
        try:
            with unittest.mock.patch.dict(
                os.environ,
                {"REDIS_URL": "redis://localhost:9999/0", "ADA_USE_FAKEREDIS": ""},
            ):
                with unittest.mock.patch(
                    "app.graph.build_checkpointer", return_value=None
                ):
                    with pytest.raises(RuntimeError, match="no pudo inicializarse"):
                        graph_module.get_checkpointer()
        finally:
            graph_module._shared_checkpointer = None

    def test_checkpointer_memory_when_redis_not_configured(self):
        """Sin REDIS_URL: MemorySaver (tests / desarrollo sin Docker)."""
        import app.graph as graph_module

        graph_module._shared_checkpointer = None
        try:
            with unittest.mock.patch.dict(
                os.environ, {"REDIS_URL": "", "ADA_USE_FAKEREDIS": ""}
            ):
                checkpointer = graph_module.get_checkpointer()
                from langgraph.checkpoint.memory import MemorySaver

                assert isinstance(checkpointer, MemorySaver)
        finally:
            graph_module._shared_checkpointer = None


# ---------------------------------------------------------------------------
# Concurrencia: 5 jobs en paralelo
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_five_jobs_processed_concurrently(self, store):
        import asyncio
        import unittest.mock

        job_ids = [
            await store.create_job(user_request=USER_REQUEST) for _ in range(5)
        ]
        with unittest.mock.patch("app.worker._build_chat_model", return_value=FakeChatModel()):
            await asyncio.gather(*(process_new_job(job_id, store) for job_id in job_ids))
        statuses = [await store.get_job(job_id) for job_id in job_ids]
        assert all(data["status"] == DONE for data in statuses)
        # Cada job terminó exactamente una vez (5 hashes distintos).
        assert len(statuses) == 5


# ---------------------------------------------------------------------------
# Redis Streams: confirmación de mensajes (XACK)
# ---------------------------------------------------------------------------


class TestStreamAck:
    @pytest.mark.asyncio
    async def test_message_is_acked_after_processing(self, store):
        """XACK: el mensaje sale del PEL tras procesarse el job."""
        import unittest.mock

        from app.worker import GROUP, STREAM_KEY

        # Crear el grupo como lo hace el worker.
        try:
            await store.redis.xgroup_create(STREAM_KEY, GROUP, id="0", mkstream=True)
        except Exception:
            pass

        job_id = await store.create_job(user_request=USER_REQUEST)
        # Leer el mensaje (queda en el PEL, sin ACK).
        result = await store.redis.xreadgroup(
            GROUP, "consumer-test", {STREAM_KEY: ">"}, count=10
        )
        messages = result[0][1]
        assert len(messages) == 1
        msg_id = messages[0][0]
        # El mensaje está pendiente (no confirmado).
        pending = await store.redis.xpending(STREAM_KEY, GROUP)
        assert pending["pending"] == 1
        # Procesar y confirmar.
        with unittest.mock.patch(
            "app.worker._build_chat_model", return_value=FakeChatModel()
        ):
            await process_new_job(job_id, store)
        acked = await store.redis.xack(STREAM_KEY, GROUP, msg_id)
        assert acked == 1
        pending = await store.redis.xpending(STREAM_KEY, GROUP)
        assert pending["pending"] == 0
        # El job terminó correctamente.
        data = await store.get_job(job_id)
        assert data["status"] == DONE
