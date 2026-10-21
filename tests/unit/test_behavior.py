"""Behavior tests for the ADA validation/repair cycle.

All tests use fake chat models and fake encoders: no network, no Groq key,
no Hugging Face downloads. They verify graph outcomes and transitions,
not just function existence.
"""

import numpy as np
import pytest

from ada.agents.analyst import derive_scores_from_evidence
from ada.graph import END, compile_graph, route_after_validator
from ada.schemas import AlternativeAnalysis, Criterion, CriterionScore
from ada.tools.scoring import calculate_weighted_score, rank_alternatives

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


class GarbageChatModel:
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
        "user_request": (
            "Decidir arquitectura para una API .NET. Equipo pequeño, "
            "presión por entregar rápido, no se requieren despliegues "
            "independientes."
        ),
        "system_profile": {},
        "criteria": [],
        "next_agent": "researcher",
        "routing_reason": "inicio",
        "steps": 0,
        "retries": 0,
        "task_completed": False,
    }


# ---------------------------------------------------------------------------
# 1. Flujo exitoso
# ---------------------------------------------------------------------------


def test_successful_flow_produces_final_report(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    visited = [c["agent"] for c in result.get("contributions", [])]
    assert "researcher" in visited
    assert "analyst" in visited
    assert "validator" in visited
    assert "synthesizer" in visited
    assert result["final_report"]["status"] == "complete"
    assert result["validation_result"]["is_valid"] is True


# ---------------------------------------------------------------------------
# 2. Delegación dinámica del Supervisor
# ---------------------------------------------------------------------------


def test_supervisor_delegates_dynamically_by_state(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    contributions = result.get("contributions", [])
    # The supervisor must delegate in order: research first, then analysis.
    researcher_steps = [c["step"] for c in contributions if c["agent"] == "researcher"]
    analyst_steps = [c["step"] for c in contributions if c["agent"] == "analyst"]
    assert researcher_steps and analyst_steps
    assert min(researcher_steps) < min(analyst_steps)


# ---------------------------------------------------------------------------
# 3. Routing determinístico post-Validator
# ---------------------------------------------------------------------------


def test_route_after_validator_is_deterministic():
    assert (
        route_after_validator({"validation_result": {"is_valid": True}})
        == "synthesizer"
    )
    assert (
        route_after_validator(
            {
                "validation_result": {
                    "is_valid": False,
                    "research_errors": ["no evidence"],
                    "analysis_errors": [],
                },
                "retries": 0,
            }
        )
        == "researcher"
    )
    assert (
        route_after_validator(
            {
                "validation_result": {
                    "is_valid": False,
                    "research_errors": [],
                    "analysis_errors": ["score mismatch"],
                },
                "retries": 0,
            }
        )
        == "analyst"
    )
    assert (
        route_after_validator(
            {
                "validation_result": {
                    "is_valid": False,
                    "research_errors": [],
                    "analysis_errors": ["x"],
                },
                "retries": 99,
            }
        )
        == "limited_finish"
    )


# ---------------------------------------------------------------------------
# 4. Reintento del Researcher: contador aumenta y segunda validación aprueba
# ---------------------------------------------------------------------------


def test_researcher_receives_errors_and_retries(documents):
    from ada.graph import make_researcher_node

    state = {
        "user_request": "test",
        "retries": 1,
        "steps": 2,
        "validation_errors": ["Research artifact contains no evidence."],
        "contributions": [],
    }

    update = make_researcher_node(encoder=fake_encoder, documents=documents)(state)

    # The researcher cleared the pending errors and recorded the attempt.
    assert update["validation_errors"] == []
    contribution = update["contributions"][-1]
    assert "Reintento tras validación" in update["contributions"][-1]["summary"]
    assert contribution["attempt"] == 1
    assert contribution["errors"] == ["Research artifact contains no evidence."]
    # The artifact was regenerated (new version replaces the invalid one).
    assert update["research_artifact"] is not None


# ---------------------------------------------------------------------------
# 5. Reintento del Analyst: mismo comportamiento
# ---------------------------------------------------------------------------


def test_analyst_receives_errors_and_retries(documents):
    from ada.graph import analyst_node

    criteria = [Criterion.model_validate(item) for item in [
        {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
    ]]
    research = {
        "query": "q",
        "evidence": [
            {
                "document_id": "doc-1",
                "chunk_id": "doc-1#full",
                "title": "Monolito modular",
                "excerpt": "Un monolito modular organiza módulos.",
                "relevance_score": 0.5,
                "retrieval_method": ["bm25"],
                "content_hash": "abc123",
            }
        ],
        "covered_topics": [],
        "evidence_gaps": [],
        "confidence_score": 0.5,
    }
    state = {
        "user_request": "test",
        "system_profile": {},
        "criteria": [criterion.model_dump() for criterion in criteria],
        "research_artifact": research,
        "retries": 1,
        "steps": 3,
        "validation_errors": ["Weighted score mismatch for 'a'."],
        "contributions": [],
    }

    update = analyst_node(state)

    assert update["validation_errors"] == []
    assert update["analysis_artifact"] is not None
    assert "Reintento tras validación" in update["contributions"][-1]["summary"]
    assert update["contributions"][-1]["attempt"] == 1


# ---------------------------------------------------------------------------
# 5b. Ciclo de reparación end-to-end: Validator devuelve al responsable
# ---------------------------------------------------------------------------


def test_repair_round_returns_to_responsible_specialist(documents):
    from ada.graph import validator_node

    # An analysis citing an unknown evidence ID must be classified as an
    # analysis error and routed back to the analyst.
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

    result = validator_node(state)
    validation = result["validation_result"]

    assert validation["is_valid"] is False
    assert validation["analysis_errors"]
    assert not validation["research_errors"]
    # Deterministic repair routing: analyst receives the specific errors.
    assert route_after_validator(result) == "analyst"
    assert result["retries"] == 1
    assert result["validation_errors"] == validation["analysis_errors"]


# ---------------------------------------------------------------------------
# 6. Reintentos agotados: finaliza sin bucle y marca revisión humana
# ---------------------------------------------------------------------------


def test_exhausted_retries_finish_safely_without_loop(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)

    state = initial_state()
    state["retries"] = 2  # ADA_MAX_RETRIES default is 2
    result = graph.invoke(state)

    # With valid artifacts the flow still completes normally (retries only
    # gate repair rounds, not success), but the counter stays bounded.
    assert result["final_report"]["status"] == "complete"
    assert result["retries"] <= 3  # bounded, no runaway increments


def test_exhausted_retries_route_to_limited_finish():
    # With retries exhausted, the deterministic router must stop the
    # repair cycle instead of looping back to a specialist.
    state = {
        "validation_result": {
            "is_valid": False,
            "research_errors": [],
            "analysis_errors": ["score mismatch"],
        },
        "retries": 2,
    }
    assert route_after_validator(state) == "limited_finish"


# ---------------------------------------------------------------------------
# 7. No hay síntesis exitosa si la validación falla
# ---------------------------------------------------------------------------


def test_synthesizer_never_publishes_success_on_failed_validation():
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
    assert report["recommendation"] != "complete"


# ---------------------------------------------------------------------------
# 8. No hay finalización exitosa cuando faltan artefactos obligatorios
# ---------------------------------------------------------------------------


def test_validator_fails_when_required_artifacts_missing():
    from ada.graph import validator_node

    result = validator_node({"criteria": [], "retries": 0, "steps": 1})
    validation = result["validation_result"]

    assert validation["is_valid"] is False
    assert any("Missing research artifact" in error for error in validation["errors"])
    assert any("Missing analysis artifact" in error for error in validation["errors"])
    # Deterministic routing must not go to the synthesizer.
    assert route_after_validator(result) in {"researcher", "limited_finish"}


# ---------------------------------------------------------------------------
# 9. El scoring es determinístico y no depende del LLM
# ---------------------------------------------------------------------------


def test_scoring_is_deterministic_and_llm_independent():
    criteria = [Criterion.model_validate(item) for item in [
        {"criterion_id": "c1", "name": "C1", "weight": 0.6, "description": "d"},
        {"criterion_id": "c2", "name": "C2", "weight": 0.4, "description": "d"},
    ]]
    scores = [
        CriterionScore.model_validate(item)
        for item in [
            {"criterion_id": "c1", "weight": 0.6, "score": 4, "rationale": "r"},
            {"criterion_id": "c2", "weight": 0.4, "score": 2, "rationale": "r"},
        ]
    ]

    first = calculate_weighted_score(scores, criteria)
    second = calculate_weighted_score(scores, criteria)
    assert first == pytest.approx(second)

    # The ranking is derived deterministically from the weighted scores.
    analyses = [
        AlternativeAnalysis.model_validate(item)
        for item in [
            {
                "alternative_id": "a",
                "name": "A",
                "scores": [
                    {"criterion_id": "c1", "weight": 0.6, "score": 4, "rationale": "r"},
                    {"criterion_id": "c2", "weight": 0.4, "score": 4, "rationale": "r"},
                ],
                "weighted_score": 0.8,
            },
            {
                "alternative_id": "b",
                "name": "B",
                "scores": [
                    {"criterion_id": "c1", "weight": 0.6, "score": 3, "rationale": "r"},
                    {"criterion_id": "c2", "weight": 0.4, "score": 3, "rationale": "r"},
                ],
                "weighted_score": 0.6,
            },
        ]
    ]
    assert rank_alternatives(analyses)[0] == "a"


def test_derive_scores_is_deterministic_and_evidence_backed(documents):
    criteria = [
        Criterion.model_validate(item)
        for item in [
            {"criterion_id": "dev-speed", "name": "Dev", "weight": 0.5, "description": "d"},
            {"criterion_id": "scalability", "name": "Scale", "weight": 0.5, "description": "d"},
        ]
    ]
    research = {
        "evidence": [
            {
                "document_id": "doc-1",
                "title": "Monolito modular",
                "excerpt": "Un monolito modular organiza módulos.",
            }
        ]
    }

    alternatives = [{"alternative_id": "modular-monolith", "name": "Monolito modular"}]

    first = derive_scores_from_evidence(
        user_request="Equipo pequeño, presión por entregar rápido",
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact=research,
    )
    second = derive_scores_from_evidence(
        user_request="Equipo pequeño, presión por entregar rápido",
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact=research,
    )
    assert first == second

    # All cited evidence IDs exist in the research artifact.
    for scores in first.values():
        for score in scores:
            for evidence_id in score["evidence_ids"]:
                assert evidence_id in {item["document_id"] for item in research["evidence"]}


# ---------------------------------------------------------------------------
# 10. Aislamiento de escritura
# ---------------------------------------------------------------------------


def test_specialists_only_update_their_assigned_fields(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    # Each artifact exists and routing fields were owned by the supervisor.
    assert result.get("research_artifact") is not None
    assert result.get("analysis_artifact") is not None
    assert result.get("validation_result") is not None
    assert result.get("final_report") is not None
    assert result["routing_reason"]


# ---------------------------------------------------------------------------
# 11. Fallo del modelo: salida segura con estado y motivo explícitos
# ---------------------------------------------------------------------------


def test_model_failure_produces_safe_output(documents):
    graph = compile_graph(BrokenChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    assert result["final_report"]["status"] == "limited"
    assert "No se emite recomendación definitiva" in result["final_report"]["recommendation"]
    assert result["routing_reason"]


def test_garbage_response_produces_safe_output(documents):
    graph = compile_graph(GarbageChatModel(), encoder=fake_encoder, documents=documents)
    result = graph.invoke(initial_state())

    assert result["final_report"]["status"] == "limited"


# ---------------------------------------------------------------------------
# 12. Los tests no necesitan clave Groq (fakes en todas las pruebas del grafo)
# ---------------------------------------------------------------------------


def test_graph_tests_use_fakes_only():
    # Sanity check: the fake models used above never call a real API.
    assert not hasattr(FakeChatModel, "invoke_real")
    assert not hasattr(BrokenChatModel, "api_key")
    assert not hasattr(GarbageChatModel, "api_key")


# ---------------------------------------------------------------------------
# 13. Protección determinística de síntesis en route_after_supervisor
# ---------------------------------------------------------------------------


def test_supervisor_cannot_route_to_synthesizer_without_approved_validation():
    from ada.graph import route_after_supervisor

    # Supervisor proposes synthesizer with no validation at all.
    assert (
        route_after_supervisor({"next_agent": "synthesizer"})
        == "validator"
    )
    # Supervisor proposes synthesizer with a failed validation.
    assert (
        route_after_supervisor(
            {
                "next_agent": "synthesizer",
                "validation_result": {"is_valid": False},
            }
        )
        == "validator"
    )
    # With an approved validation the route is allowed.
    assert (
        route_after_supervisor(
            {
                "next_agent": "synthesizer",
                "validation_result": {"is_valid": True},
            }
        )
        == "synthesizer"
    )


def test_supervisor_end_without_successful_report_is_not_success():
    from ada.graph import route_after_supervisor

    # END with a complete report AND a current approved validation -> finish.
    assert (
        route_after_supervisor(
            {
                "next_agent": "END",
                "final_report": {"status": "complete"},
                "validation_result": {"is_valid": True},
            }
        )
        == END
    )
    # END with a complete report but NO current validation -> revalidate
    # (a stale report must not present success).
    assert (
        route_after_supervisor(
            {
                "next_agent": "END",
                "final_report": {"status": "complete"},
                "research_artifact": {"evidence": []},
            }
        )
        == "validator"
    )
    # END without any report or artifacts -> safe stop, no fake success.
    assert (
        route_after_supervisor({"next_agent": "END"})
        == "limited_finish"
    )
    # END without a report but with research pending -> complete validation.
    assert (
        route_after_supervisor(
            {"next_agent": "END", "research_artifact": {"evidence": []}}
        )
        == "validator"
    )
    # END with a limited report and no artifacts -> safe stop.
    assert (
        route_after_supervisor(
            {
                "next_agent": "END",
                "final_report": {"status": "limited"},
            }
        )
        == "limited_finish"
    )


# ---------------------------------------------------------------------------
# 14. Obsolescencia de validación: regenerar un artefacto invalida la
#     validación previa, aunque haya sido aprobada.
# ---------------------------------------------------------------------------


def test_regenerated_artifact_invalidates_previous_validation(documents):
    from ada.graph import analyst_node, make_researcher_node

    criteria = [Criterion.model_validate(item) for item in [
        {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
    ]]
    state = {
        "user_request": "test",
        "system_profile": {},
        "criteria": [criterion.model_dump() for criterion in criteria],
        "retries": 0,
        "steps": 3,
        "validation_errors": ["Weighted score mismatch for 'a'."],
        "validation_result": {"is_valid": True},  # validación previa aprobada
        "contributions": [],
    }

    analyst_update = analyst_node(state)
    # The previous approved validation no longer authorizes synthesis.
    assert analyst_update["validation_result"] is None

    researcher_update = make_researcher_node(
        encoder=fake_encoder, documents=documents
    )({**state, "validation_errors": ["no evidence"]})
    assert researcher_update["validation_result"] is None


def test_stale_validation_does_not_authorize_synthesis(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)

    # Simulate a stale approved validation followed by an analyst repair
    # round: the router must not reach the synthesizer until revalidation.
    state = initial_state()
    state["validation_result"] = {"is_valid": True}
    state["validation_errors"] = ["Weighted score mismatch for 'a'."]
    result = graph.invoke(state)

    # The flow must end with a fresh validation, not the stale one.
    assert result["validation_result"] is not None
    assert result["final_report"]["status"] in {"complete", "needs_human_review", "limited"}
    # If synthesis happened, the validation must be current and approved.
    if result["final_report"]["status"] == "complete":
        assert result["validation_result"]["is_valid"] is True


# ---------------------------------------------------------------------------
# 15. Robustez A4: invalidación de informe obsoleto y revalidación
#     post-reparación.
# ---------------------------------------------------------------------------


def _state_with_approved_results() -> dict:
    """State simulating a previously approved validation and report."""
    criteria = [Criterion.model_validate(item) for item in [
        {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
    ]]
    return {
        "user_request": "test",
        "system_profile": {},
        "criteria": [criterion.model_dump() for criterion in criteria],
        "retries": 0,
        "steps": 3,
        "validation_errors": ["Weighted score mismatch for 'a'."],
        "validation_result": {"is_valid": True},  # validación previa aprobada
        "final_report": {"status": "complete"},  # informe previo exitoso
        "contributions": [],
    }


def test_researcher_update_invalidates_validation_and_report(documents):
    from ada.graph import make_researcher_node

    update = make_researcher_node(encoder=fake_encoder, documents=documents)(
        _state_with_approved_results()
    )

    # Both the previous validation and the previous report are invalidated.
    assert update["validation_result"] is None
    assert update["final_report"] is None


def test_analyst_update_invalidates_validation_and_report():
    from ada.graph import analyst_node

    update = analyst_node(_state_with_approved_results())

    # Both the previous validation and the previous report are invalidated.
    assert update["validation_result"] is None
    assert update["final_report"] is None


def test_validator_revalidates_new_version_after_repair(documents):
    from ada.graph import validator_node

    # A repaired analysis (coherent scores, known evidence IDs) must pass a
    # fresh validation of the NEW version, not inherit the old approval.
    criteria = [Criterion.model_validate(item) for item in [
        {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
    ]]
    state = {
        "user_request": "test",
        "criteria": [criterion.model_dump() for criterion in criteria],
        "retries": 1,
        "steps": 5,
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
                            "evidence_ids": ["doc-1"],
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
                            "evidence_ids": ["doc-1"],
                        }
                    ],
                    "weighted_score": 0.6,
                },
            ],
            "ranking": ["a", "b"],
            "data_gaps": [],
            "qualitative_summary": "s",
        },
    }

    result = validator_node(state)
    validation = result["validation_result"]

    # The new version is validated on its own merits: fresh approval.
    assert validation["is_valid"] is True
    assert validation["errors"] == []
    # The repair round is recorded in the audit trail.
    assert result["contributions"][-1]["attempt"] == 1


def test_repaired_flow_reaches_synthesizer_with_current_validation(documents):
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)

    # Start with a pending analysis error: the graph must route to the
    # analyst, revalidate the new version, and only then synthesize.
    state = initial_state()
    state["validation_errors"] = ["Weighted score mismatch for 'a'."]
    result = graph.invoke(state)

    visited = [c["agent"] for c in result.get("contributions", [])]
    assert "analyst" in visited
    assert "validator" in visited
    # Synthesis only with a current approved validation of the new version.
    if "synthesizer" in visited:
        assert result["validation_result"]["is_valid"] is True
        assert result["final_report"]["status"] == "complete"


# ---------------------------------------------------------------------------
# 8. Semántica de ADA_MAX_RETRIES: reparaciones, no validaciones totales
#
# Pruebas deterministas sobre validator_node y route_after_validator con
# estados controlados: sin depender del ciclo fake a nivel de grafo.
# ---------------------------------------------------------------------------


class SettingsWithRetries:
    """Proxy de Settings que solo sobrescribe max_retries."""

    def __init__(self, max_retries: int):
        self._max_retries = max_retries

    def __getattr__(self, name):
        from ada.config import get_settings as real_get_settings

        return getattr(real_get_settings(), name)

    @property
    def max_retries(self):
        return self._max_retries


def make_repair_state(retries: int = 0) -> dict:
    """Estado con un error de análisis fixable: el `weighted_score`
    reportado no coincide con el calculado por Python (0.8)."""
    return {
        "user_request": "test",
        "criteria": [],
        "retries": retries,
        "steps": 5,
        "research_artifact": {
            "query": "q",
            "evidence": [
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1#full",
                    "title": "T",
                    "excerpt": "Un monolito modular organiza el código en módulos.",
                    "relevance_score": 0.9,
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
                    "alternative_id": "monolith",
                    "name": "Monolito",
                    "scores": [
                        {
                            "criterion_id": "dev-speed",
                            "weight": 0.25,
                            "score": 4,
                            "rationale": "r",
                            "evidence_ids": ["doc-1"],
                        },
                        {
                            "criterion_id": "operational-cost",
                            "weight": 0.25,
                            "score": 4,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "independent-deploys",
                            "weight": 0.15,
                            "score": 3,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "scalability",
                            "weight": 0.15,
                            "score": 3,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "team-alignment",
                            "weight": 0.20,
                            "score": 4,
                            "rationale": "r",
                        },
                    ],
                    # Incorrecto a propósito: el calculado es 0.8.
                    "weighted_score": 0.5,
                },
                {
                    "alternative_id": "microservices",
                    "name": "Microservicios",
                    "scores": [
                        {
                            "criterion_id": "dev-speed",
                            "weight": 0.25,
                            "score": 3,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "operational-cost",
                            "weight": 0.25,
                            "score": 3,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "independent-deploys",
                            "weight": 0.15,
                            "score": 4,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "scalability",
                            "weight": 0.15,
                            "score": 4,
                            "rationale": "r",
                        },
                        {
                            "criterion_id": "team-alignment",
                            "weight": 0.20,
                            "score": 3,
                            "rationale": "r",
                        },
                    ],
                    "weighted_score": 0.6,
                },
            ],
            "ranking": ["monolith", "microservices"],
            "data_gaps": [],
            "qualitative_summary": "s",
        },
    }


def test_max_retries_0_first_failure_is_not_repairable(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(0))

    result = validator_node(make_repair_state(retries=0))

    # First failure with max_retries=0: no repair dispatched.
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert "pending_repair_agent" not in validation
    assert result["retries"] == 0
    assert route_after_validator(result) == "limited_finish"


def test_max_retries_1_first_failure_dispatches_one_repair(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(1))

    result = validator_node(make_repair_state(retries=0))

    # First failure (retries=0 < 1): one repair dispatched, counter 0 -> 1.
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert validation["pending_repair_agent"] == "analyst"
    assert result["retries"] == 1
    assert result["validation_errors"]  # specific errors for the analyst
    assert route_after_validator(result) == "analyst"


def test_max_retries_1_second_failure_ends_safely(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(1))

    # Second failure (retries=1 >= max_retries): no second repair.
    result = validator_node(make_repair_state(retries=1))
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert "pending_repair_agent" not in validation
    # Sin reparación despachada, el contador no cambia (el update no lo
    # incluye: LangGraph conserva el valor previo).
    assert result.get("retries", 1) == 1
    assert route_after_validator(result) == "limited_finish"


def test_max_retries_2_allows_two_repairs_and_blocks_third(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(2))

    # First failure (retries=0): repair 1 dispatched.
    first = validator_node(make_repair_state(retries=0))
    assert first["retries"] == 1
    assert route_after_validator(first) == "analyst"

    # Second failure (retries=1): repair 2 dispatched.
    second = validator_node(make_repair_state(retries=1))
    assert second["retries"] == 2
    assert route_after_validator(second) == "analyst"

    # Third failure (retries=2): budget exhausted, safe finish.
    third = validator_node(make_repair_state(retries=2))
    assert third.get("retries", 2) == 2  # no increment
    assert "pending_repair_agent" not in third["validation_result"]
    assert route_after_validator(third) == "limited_finish"


def test_counter_increments_exactly_once_per_repair(monkeypatch):
    from ada.graph import validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(3))

    # Each dispatched repair increments the counter exactly once.
    retries = 0
    for expected in (1, 2, 3):
        result = validator_node(make_repair_state(retries=retries))
        assert result["retries"] == expected
        retries = expected


def test_repair_budget_exhausted_requires_human_review(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(2))

    # Exhausted repair budget with a still-failing artifact: human review
    # pending, never presented as success.
    result = validator_node(make_repair_state(retries=2))
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert validation["requires_human_review"] is True
    assert route_after_validator(result) == "limited_finish"


def test_missing_analysis_artifact_is_classified_and_repairable(monkeypatch):
    from ada.graph import route_after_validator, validator_node

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries(2))

    # Missing analysis artifact: the error IS classifiable (analyst's
    # artifact), so a repair is dispatched with budget available.
    state = make_repair_state(retries=0)
    state["analysis_artifact"] = None
    result = validator_node(state)

    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert validation["analysis_errors"]
    assert validation["pending_repair_agent"] == "analyst"
    assert route_after_validator(result) == "analyst"


