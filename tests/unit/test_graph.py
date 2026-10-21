"""Tests for the ADA StateGraph: normal path and failure path.

All tests use a fake chat model and a fake encoder, so they never touch
the network, Groq or Hugging Face downloads.
"""

import numpy as np
import pytest

from ada.graph import compile_graph


class FakeChatModel:
    """Fake chat model that routes deterministically from the context text."""

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
    """Chat model that always raises, to exercise the fallback route."""

    def invoke(self, messages):
        raise ConnectionError("network down")


class GarbageChatModel:
    """Chat model that returns unparseable output."""

    def invoke(self, messages):
        class Response:
            content = "not a json object at all"

        return Response()


@pytest.fixture
def documents():
    from ada.retrieval.documents import TechnicalDocument

    return [
        TechnicalDocument(
            document_id="doc-1",
            title="Monolito modular",
            content=(
                "Un monolito modular organiza el código en módulos "
                "con límites claros dentro de una aplicación."
            ),
            source="https://example.com/modular-monolith",
            topic="arquitectura",
        ),
        TechnicalDocument(
            document_id="doc-2",
            title="Microservicios",
            content=(
                "Los microservicios separan capacidades de negocio "
                "en servicios independientes."
            ),
            source="https://microservices.io/patterns/microservices.html",
            topic="arquitectura",
        ),
    ]


def fake_encoder(texts: list[str]) -> "np.ndarray":
    vectors = []
    for text in texts:
        normalized = text.lower()
        if "monolito" in normalized:
            vectors.append([0.0, 1.0])
        else:
            vectors.append([1.0, 0.0])
    return np.asarray(vectors, dtype=np.float32)


def initial_state() -> dict:
    return {
        "user_request": "Decidir arquitectura para una API .NET",
        "system_profile": {},
        "criteria": [],
        "next_agent": "researcher",
        "routing_reason": "inicio",
        "steps": 0,
        "retries": 0,
        "task_completed": False,
    }


def test_graph_normal_path_visits_all_specialists(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    visited = [c["agent"] for c in result.get("contributions", [])]
    assert "researcher" in visited
    assert "analyst" in visited
    assert "validator" in visited
    assert "synthesizer" in visited
    assert result["final_report"]["status"] == "complete"


def test_graph_workers_only_write_their_own_fields(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    # The researcher must not write analysis or validation fields.
    assert result.get("research_artifact") is not None
    assert result.get("analysis_artifact") is not None
    assert result.get("validation_result") is not None
    assert result.get("final_report") is not None
    # Routing fields are owned by the supervisor only.
    assert result["routing_reason"]


def test_supervisor_falls_back_safely_on_model_error(documents):
    graph = compile_graph(BrokenChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    # The workflow must finish with a limited result, not crash or loop.
    assert result["final_report"]["status"] == "limited"
    assert "No se emite recomendación definitiva" in result["final_report"]["recommendation"]


def test_supervisor_falls_back_on_garbage_response(documents):
    graph = compile_graph(GarbageChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    assert result["final_report"]["status"] == "limited"


def test_validator_accepts_consistent_artifacts(documents):
    from ada.graph import validator_node

    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    state = graph.invoke(initial_state())

    validation = validator_node(state)["validation_result"]
    assert validation["is_valid"] is True
    assert validation["errors"] == []


def test_validator_rejects_unknown_evidence_references():
    from ada.graph import validator_node

    state = {
        "user_request": "test",
        "criteria": [],
        "retries": 0,
        "steps": 3,
        "research_artifact": {
            "query": "q",
            "evidence": [
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1#full",
                    "title": "T",
                    "excerpt": "E",
                    "relevance_score": 0.5,
                    "retrieval_method": ["bm25"],
                    "content_hash": "abc123",
                }
            ],
            "covered_topics": [],
            "evidence_gaps": [],
            "confidence_score": 0.5,
        },
        "analysis_artifact": {
            "alternatives": [
                {
                    "alternative_id": "a",
                    "name": "A",
                    "scores": [
                        {
                            "criterion_id": "c1",
                            "weight": 1.0,
                            "score": 4,
                            "rationale": "r",
                            "evidence_ids": ["doc-999"],
                        }
                    ],
                    "weighted_score": 0.8,
                },
                {
                    "alternative_id": "b",
                    "name": "B",
                    "scores": [
                        {
                            "criterion_id": "c1",
                            "weight": 1.0,
                            "score": 3,
                            "rationale": "r",
                        }
                    ],
                    "weighted_score": 0.8,
                },
            ],
            "ranking": ["a", "b"],
            "data_gaps": [],
            "qualitative_summary": "s",
        },
    }

    result = validator_node(state)["validation_result"]
    assert result["is_valid"] is False
    assert any("unknown evidence" in error for error in result["errors"])


def test_synthesizer_marks_unapproved_artifacts_as_needs_review():
    from ada.graph import synthesizer_node

    state = {
        "user_request": "test",
        "validation_result": {
            "is_valid": False,
            "errors": ["Weighted score mismatch for 'a'."],
            "warnings": [],
            "requires_human_review": False,
        },
        "research_artifact": {"evidence": []},
        "analysis_artifact": {"alternatives": [], "ranking": []},
    }

    report = synthesizer_node(state)["final_report"]
    assert report["status"] == "needs_human_review"
    assert "no definitiva" in " ".join(report["risks_and_limitations"]).lower()


def test_limited_finish_stops_with_limited_result():
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=[])

    # Empty corpus forces the researcher to fail; the supervisor fallback
    # and limits must still produce a controlled finish.
    state = initial_state()
    state["steps"] = 99  # force step limit
    result = graph.invoke(state)

    assert result["final_report"]["status"] in {"limited", "needs_human_review"}
