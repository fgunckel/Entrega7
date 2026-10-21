"""Tests de regresión: alineación consulta-alternativas, pertinencia
semántica, scoring/ranking consistentes e informe coherente (SPEC)."""

import pytest

from ada.decision import build_criteria, extract_decision
from ada.schemas import CriterionScore, ValidationResult
from ada.tools.scoring import rank_alternatives

# ---------------------------------------------------------------------------
# 1. Extracción de decisión (Requisito 1)
# ---------------------------------------------------------------------------


def test_service_bus_vs_http_produces_exactly_those_alternatives():
    spec = extract_decision(
        "Necesito decidir cómo implementar la comunicación entre microservicios "
        "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
        "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
        "latencia, complejidad operativa y escalabilidad."
    )
    assert spec.alternatives_are_explicit is True
    ids = [a.alternative_id for a in spec.requested_alternatives]
    assert spec.decision_question
    assert "azure-service-bus" in ids
    assert "sync-http" in ids
    assert "monolith" not in ids
    assert "modular-monolith" not in ids


def test_other_decision_keeps_its_own_relevant_alternatives():
    spec = extract_decision(
        "Decidir si una API .NET debe ser monolito modular o microservicios"
    )
    assert spec.alternatives_are_explicit is True
    ids = [a.alternative_id for a in spec.requested_alternatives]
    assert "modular-monolith" in ids
    assert "microservices" in ids
    assert "azure-service-bus" not in ids
    assert "sync-http" not in ids


def test_query_without_explicit_alternatives_is_not_invented():
    spec = extract_decision("Necesito mejorar la observabilidad del sistema")
    assert spec.alternatives_are_explicit is False
    assert spec.requested_alternatives == []


# ---------------------------------------------------------------------------
# 2. Scoring y ranking consistentes (Requisito 4)
# ---------------------------------------------------------------------------


def _alternative(alternative_id: str, weighted_score: float):
    from ada.schemas import AlternativeAnalysis, CriterionScore

    return AlternativeAnalysis(
        alternative_id=alternative_id,
        name=alternative_id.title(),
        scores=[
            CriterionScore(
                criterion_id="c1",
                weight=1.0,
                score=3,
                rationale="r",
            )
        ],
        weighted_score=weighted_score,
    )


def test_equal_scores_produce_honest_tie_in_insertion_order():
    analyses = [
        _alternative("zulu", 0.8),
        _alternative("alpha", 0.8),
        _alternative("mike", 0.8),
    ]
    ranking = rank_alternatives(analyses)
    # Empate coherente: el orden es el de presentación, no alfabético.
    assert ranking == ["zulu", "alpha", "mike"]


def test_different_scores_are_ordered_correctly():
    analyses = [
        _alternative("low", 0.3),
        _alternative("high", 0.9),
        _alternative("mid", 0.6),
    ]
    ranking = rank_alternatives(analyses)
    assert ranking == ["high", "mid", "low"]


def test_tie_in_top_declares_tie_in_summary():
    from ada.agents.analyst import build_analysis
    from ada.schemas import Criterion

    criteria = [Criterion(criterion_id="c1", name="C1", weight=1.0, description="d")]
    alternatives = [
        {"alternative_id": "a", "name": "A"},
        {"alternative_id": "b", "name": "B"},
    ]
    scores = {
        "a": [{"criterion_id": "c1", "weight": 1.0, "score": 4, "rationale": "r"}],
        "b": [{"criterion_id": "c1", "weight": 1.0, "score": 4, "rationale": "r"}],
    }
    artifact = build_analysis(criteria, alternatives, scores)
    assert "Empate" in artifact.qualitative_summary
    assert artifact.ranking == ["a", "b"]


def test_missing_or_invalid_weights_are_handled_explicitly():
    from ada.schemas import Criterion
    from ada.tools.scoring import calculate_weighted_score

    criteria = [
        Criterion(criterion_id="c1", name="C1", weight=0.5, description="d"),
        Criterion(criterion_id="c2", name="C2", weight=0.4, description="d"),
    ]
    scores = [
        {"criterion_id": "c1", "weight": 0.5, "score": 4, "rationale": "r"},
        {"criterion_id": "c2", "weight": 0.4, "score": 3, "rationale": "r"},
    ]
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_weighted_score(criteria=criteria, scores=scores)


# ---------------------------------------------------------------------------
# 3. Validación de pertinencia semántica (Requisitos 2 y 3)
# ---------------------------------------------------------------------------


def _base_state() -> dict:
    return {
        "user_request": "Compará Azure Service Bus y la comunicación HTTP síncrona",
        "system_profile": {},
        "criteria": [
            {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
        ],
        "decision_spec": extract_decision(
            "Compará Azure Service Bus y la comunicación HTTP síncrona"
        ).model_dump(),
        "next_agent": "validator",
        "routing_reason": "r",
        "steps": 3,
        "retries": 0,
        "task_completed": False,
    }


def _consistent_artifacts() -> dict:
    return {
        "research_artifact": {
            "query": "Compará Azure Service Bus y la comunicación HTTP síncrona",
            "evidence": [
                {
                    "document_id": "doc-1",
                    "chunk_id": "doc-1#full",
                    "title": "Mensajería asíncrona",
                    "excerpt": "Azure Service Bus desacopla servicios con colas.",
                    "relevance_score": 0.9,
                    "retrieval_method": ["bm25"],
                    "content_hash": "abc123",
                }
            ],
            "covered_topics": ["mensajeria"],
            "evidence_gaps": [],
            "confidence_score": 0.5,
        },
        "analysis_artifact": {
            "alternatives": [
                {
                    "alternative_id": "azure-service-bus",
                    "name": "Azure Service Bus",
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
                    "alternative_id": "sync-http",
                    "name": "HTTP síncrono",
                    "scores": [
                        {
                            "criterion_id": "c1",
                            "weight": 1.0,
                            "score": 3,
                            "rationale": "r",
                            "evidence_ids": [],
                        }
                    ],
                    "weighted_score": 0.6,
                },
            ],
            "ranking": ["azure-service-bus", "sync-http"],
            "data_gaps": [],
            "qualitative_summary": "s",
        },
    }


def test_validator_accepts_artifacts_aligned_with_query():
    from ada.graph import validator_node

    state = {**_base_state(), **_consistent_artifacts()}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is True
    assert validation["errors"] == []


def test_validator_rejects_alternatives_not_requested():
    from ada.graph import validator_node

    artifacts = _consistent_artifacts()
    # Sustituir una alternativa solicitada por una no solicitada.
    artifacts["analysis_artifact"]["alternatives"][1] = {
        "alternative_id": "monolith",
        "name": "Monolito",
        "scores": [
            {"criterion_id": "c1", "weight": 1.0, "score": 3, "rationale": "r"}
        ],
        "weighted_score": 0.6,
    }
    artifacts["analysis_artifact"]["ranking"] = ["azure-service-bus", "monolith"]
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert any("not requested" in error for error in validation["errors"])
    assert any("not requested" in error for error in validation["analysis_errors"])


def test_validator_rejects_omitted_requested_alternative():
    from ada.graph import validator_node

    artifacts = _consistent_artifacts()
    artifacts["analysis_artifact"]["alternatives"] = [
        artifacts["analysis_artifact"]["alternatives"][0]
    ]
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    # Con una sola alternativa el contrato falla (min_length=2); el error
    # es accionable para el analyst en cualquier caso.
    assert any(
        "omits alternatives" in error or "contract violation" in error
        for error in validation["errors"]
    )


def test_validator_rejects_evidence_irrelevant_to_requested_alternatives():
    from ada.graph import validator_node

    artifacts = _consistent_artifacts()
    artifacts["research_artifact"]["evidence"][0] = {
        "document_id": "doc-9",
        "chunk_id": "doc-9#full",
        "title": "Topologías de equipos",
        "excerpt": "Conway's law y topologías de equipos de plataforma.",
        "relevance_score": 0.4,
        "retrieval_method": ["vector"],
        "content_hash": "def456",
    }
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert any(
        "No retrieved evidence mentions any requested alternative" in error
        for error in validation["errors"]
    )
    # El error es accionable para el especialista responsable.
    assert any(
        "No retrieved evidence" in error for error in validation["research_errors"]
    )


def test_validator_reports_unsupported_criteria():
    from ada.graph import validator_node

    artifacts = _consistent_artifacts()
    artifacts["research_artifact"]["evidence"] = []
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert validation["unsupported_criteria"] == ["c1"]


# ---------------------------------------------------------------------------
# 4. Flujo de reparación acotado y síntesis protegida (Requisitos 2 y 5)
# ---------------------------------------------------------------------------


def test_semantic_errors_follow_bounded_repair_flow(monkeypatch):
    from ada.graph import validator_node

    class SettingsWithRetries:
        max_retries = 1
        max_steps = 8

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries())

    artifacts = _consistent_artifacts()
    artifacts["analysis_artifact"]["alternatives"][1] = {
        "alternative_id": "monolith",
        "name": "Monolito",
        "scores": [
            {"criterion_id": "c1", "weight": 1.0, "score": 3, "rationale": "r"}
        ],
        "weighted_score": 0.6,
    }
    artifacts["analysis_artifact"]["ranking"] = ["azure-service-bus", "monolith"]
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    # La reparación se despacha al especialista responsable (analyst).
    assert result["validation_result"]["pending_repair_agent"] == "analyst"
    assert result["retries"] == 1
    assert result["validation_errors"]


def test_repair_budget_exhausted_ends_safely(monkeypatch):
    from ada.graph import validator_node

    class SettingsWithRetries:
        max_retries = 2
        max_steps = 8

    monkeypatch.setattr("ada.graph.get_settings", lambda: SettingsWithRetries())
    artifacts = _consistent_artifacts()
    artifacts["analysis_artifact"]["alternatives"][1] = {
        "alternative_id": "monolith",
        "name": "Monolito",
        "scores": [
            {"criterion_id": "c1", "weight": 1.0, "score": 3, "rationale": "r"}
        ],
        "weighted_score": 0.6,
    }
    artifacts["analysis_artifact"]["ranking"] = ["azure-service-bus", "monolith"]
    state = {**_base_state(), **artifacts, "retries": 2}
    result = validator_node(state)
    validation = result["validation_result"]
    assert validation["is_valid"] is False
    assert "pending_repair_agent" not in validation
    assert validation["requires_human_review"] is True


def test_synthesizer_never_runs_without_current_approved_validation():
    from ada.graph import route_after_supervisor

    state = {
        "next_agent": "synthesizer",
        "validation_result": None,
        "final_report": None,
        "research_artifact": {"evidence": [{"document_id": "d"}]},
    }
    assert route_after_supervisor(state) == "validator"


def test_synthesizer_rejects_failed_validation():
    from ada.graph import synthesizer_node

    state = {
        "user_request": "Compará Azure Service Bus y HTTP síncrono",
        "validation_result": ValidationResult(
            is_valid=False, errors=["Weighted score mismatch."]
        ).model_dump(),
        "research_artifact": {"evidence": []},
        "analysis_artifact": {},
    }
    result = synthesizer_node(state)
    report = result["final_report"]
    assert report["status"] == "needs_human_review"
    assert "no definitiva" in " ".join(report["risks_and_limitations"]).lower()


def test_report_cannot_declare_complete_with_missing_evidence():
    from ada.graph import route_after_validator, validator_node

    artifacts = _consistent_artifacts()
    artifacts["research_artifact"]["evidence"] = []
    state = {**_base_state(), **artifacts}
    result = validator_node(state)
    validation = result["validation_result"]
    # Sin evidencia obligatoria no hay validación aprobada, y el enrutador
    # no puede llevar el flujo a síntesis con validación fallida.
    assert validation["is_valid"] is False
    assert route_after_validator(result) != "synthesizer"


def test_artifact_repair_invalidates_validation_and_report():
    from ada.graph import analyst_node

    state = {
        "user_request": "Compará Azure Service Bus y la comunicación HTTP síncrona",
        "system_profile": {},
        "criteria": [
            {"criterion_id": "c1", "name": "C1", "weight": 1.0, "description": "d"}
        ],
        "decision_spec": extract_decision(
            "Compará Azure Service Bus y la comunicación HTTP síncrona"
        ).model_dump(),
        "validation_result": ValidationResult(is_valid=True).model_dump(),
        "final_report": {"status": "complete"},
        "validation_errors": ["Weighted score mismatch for 'sync-http'."],
        "contributions": [],
        "next_agent": "analyst",
        "routing_reason": "r",
        "steps": 3,
        "retries": 1,
        "task_completed": False,
    }
    update = analyst_node(state)
    # La corrección invalida la validación y el informe anteriores.
    assert update["validation_result"] is None
    assert update["final_report"] is None
    # Y evalúa las alternativas solicitadas, no la lista genérica.
    analysis_ids = {
        a["alternative_id"] for a in update["analysis_artifact"]["alternatives"]
    }
    assert analysis_ids == {"azure-service-bus", "sync-http"}


def test_analyst_uses_criteria_extracted_from_original_query():
    """El Analyst evalúa exactamente los cinco criterios de la consulta.

    La consulta original pide evaluar acoplamiento, resiliencia, latencia,
    complejidad operativa y escalabilidad; el artefacto de análisis debe
    contener exactamente esos criterios para todas las alternativas, no
    los predeterminados.
    """
    from ada.graph import analyst_node

    original_query = (
        "Necesito decidir cómo implementar la comunicación entre microservicios "
        "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
        "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
        "latencia, complejidad operativa y escalabilidad."
    )
    decision_spec = extract_decision(original_query).model_dump()
    state = {
        "user_request": original_query,
        "system_profile": {},
        "criteria": [
            criterion.model_dump()
            for criterion in build_criteria(decision_spec["requested_criteria"])
        ],
        "decision_spec": decision_spec,
        "research_artifact": {
            "query": original_query,
            "evidence": [],
            "covered_topics": [],
            "evidence_gaps": [],
            "confidence_score": 0.5,
        },
        "contributions": [],
        "next_agent": "analyst",
        "routing_reason": "r",
        "steps": 2,
        "retries": 0,
        "task_completed": False,
    }

    update = analyst_node(state)
    analysis = update["analysis_artifact"]

    assert analysis["alternatives"], "El análisis debe evaluar alternativas."
    expected_ids = {
        "coupling",
        "resilience",
        "latency",
        "operational-cost",
        "scalability",
    }
    for alternative in analysis["alternatives"]:
        actual_ids = {score["criterion_id"] for score in alternative["scores"]}
        assert actual_ids == expected_ids, (
            f"La alternativa '{alternative['alternative_id']}' debe evaluar "
            f"exactamente los cinco criterios de la consulta."
        )


def test_scoring_uses_extracted_criteria_and_weights():
    """El scoring usa exactamente los criterios y pesos extraídos.

    El weighted_score de cada alternativa debe ser la suma de
    score * weight sobre los cinco criterios extraídos (pesos
    normalizados a 1), no sobre los predeterminados.
    """
    from ada.tools.scoring import calculate_weighted_score

    original_query = (
        "Necesito decidir cómo implementar la comunicación entre microservicios "
        "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
        "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
        "latencia, complejidad operativa y escalabilidad."
    )
    decision_spec = extract_decision(original_query).model_dump()
    criteria = build_criteria(decision_spec["requested_criteria"])
    weights = {c.criterion_id: c.weight for c in criteria}

    # Pesos normalizados: cada peso base dividido por la suma de bases.
    base = {"coupling": 0.20, "resilience": 0.20, "latency": 0.15,
            "operational-cost": 0.15, "scalability": 0.15}
    total_base = sum(base.values())
    for criterion_id, expected_weight in base.items():
        assert abs(weights[criterion_id] - base[criterion_id] / total_base) < 1e-6

    # Scores idénticos 3 en los cinco criterios ->
    # suma(peso * 3/5) = 3/5 * suma(pesos) = 0.6 (escala normalizada 0-1).
    scores = [
        CriterionScore(
            criterion_id=criterion_id,
            score=3,
            weight=weights[criterion_id],
            rationale="j",
        )
        for criterion_id in weights
    ]
    weighted = calculate_weighted_score(scores, criteria)
    # Tolerancia 1e-5: build_criteria redondea cada peso a 6 decimales y la
    # suma puede desviarse hasta ~1e-6 (hallazgo de diagnóstico documentado).
    assert abs(weighted - 0.6) < 1e-5, (
        f"el scoring debe usar los pesos extraídos (resultado ~0.6), "
        f"obtuvo {weighted}"
    )


# ---------------------------------------------------------------------------
# 8. Scoring basado en datos recibidos (hallazgo de diagnóstico)
# ---------------------------------------------------------------------------


def test_derived_scores_reflect_received_data():
    """Las puntuaciones derivadas reflejan los datos recibidos.

    Con datos idénticos para ambas alternativas (mismo texto de consulta,
    sin evidencia), ambas reciben exactamente los mismos puntajes neutros:
    no se inventan diferencias entre alternativas.
    """
    from ada.agents.analyst import derive_scores_from_evidence

    original_query = (
        "Necesito decidir cómo implementar la comunicación entre microservicios "
        "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
        "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
        "latencia, complejidad operativa y escalabilidad."
    )
    decision_spec = extract_decision(original_query).model_dump()
    criteria = build_criteria(decision_spec["requested_criteria"])
    alternatives = [
        {"alternative_id": "azure-service-bus", "name": "Azure Service Bus"},
        {"alternative_id": "sync-http", "name": "Comunicación HTTP síncrona"},
    ]

    scores = derive_scores_from_evidence(
        user_request=original_query,
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact={"evidence": []},
    )

    # Cada alternativa evaluada exactamente con los 5 criterios recibidos.
    for alternative_scores in scores.values():
        ids = {s["criterion_id"] for s in alternative_scores}
        assert ids == {c.criterion_id for c in criteria}
        for score in alternative_scores:
            assert 1 <= score["score"] <= 5

    # Datos idénticos para ambas alternativas -> puntajes idénticos:
    # el scoring no inventa diferencias entre alternativas.
    assert scores["azure-service-bus"] == scores["sync-http"]

    # Pesos copiados en cada score coinciden con los criterios recibidos.
    for alternative_scores in scores.values():
        for score in alternative_scores:
            expected = next(
                c.weight for c in criteria if c.criterion_id == score["criterion_id"]
            )
            assert abs(score["weight"] - expected) < 1e-6


def test_derived_scores_change_with_signals_in_request():
    """Las señales presentes en la consulta empujan los puntajes de forma
    determinística y reflejan el caso recibido."""
    from ada.agents.analyst import derive_scores_from_evidence

    criteria = build_criteria(["coupling", "resilience", "latency"])
    alternatives = [{"alternative_id": "svc", "name": "Svc"}]

    with_messaging = derive_scores_from_evidence(
        user_request=(
            "Comunicación entre servicios con mensajería asincrónica y colas, "
            "evaluando acoplamiento y resiliencia."
        ),
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact={"evidence": []},
    )
    direct = derive_scores_from_evidence(
        user_request=(
            "Servicios que se comunican con llamadas directas HTTP síncronas, "
            "evaluando acoplamiento y latencia."
        ),
        system_profile={},
        criteria=criteria,
        alternatives=[{"alternative_id": "svc", "name": "Svc"}],
        research_artifact={"evidence": []},
    )

    def score_of(scores, criterion_id):
        return next(
            s["score"] for s in scores["svc"] if s["criterion_id"] == criterion_id
        )

    # 'mensajería' es señal positiva de coupling: puntaje > neutro 3.
    assert score_of(with_messaging, "coupling") > 3
    # 'llamadas directas' y 'acoplamiento' son dos señales negativas de
    # coupling en el texto: neutro 3 - 2 = 1.
    assert score_of(direct, "coupling") == 1
    # 'latencia' es señal negativa de latency: puntaje < neutro 3.
    assert score_of(direct, "latency") < 3
    # Determinismo: misma entrada -> mismo resultado.
    again = derive_scores_from_evidence(
        user_request=(
            "Comunicación entre servicios con mensajería asincrónica y colas, "
            "evaluando acoplamiento y resiliencia."
        ),
        system_profile={},
        criteria=criteria,
        alternatives=[{"alternative_id": "svc", "name": "Svc"}],
        research_artifact={"evidence": []},
    )
    assert again == with_messaging


def test_build_criteria_weights_sum_exactly_one():
    """Los pesos de build_criteria suman exactamente 1.0 (tolerancia 1e-6)."""

    criteria = build_criteria(
        ["coupling", "resilience", "latency", "operational-cost", "scalability"]
    )
    total = sum(c.weight for c in criteria)
    assert abs(total - 1.0) <= 1e-6, f"los pesos deben sumar 1.0, sumaron {total}"


def test_corpus_without_relevant_evidence_keeps_limited_result():
    """Corpus sin evidencia sobre las alternativas solicitadas.

    El validator rechaza, el status queda 'limited' y NO se emite una
    recomendación definitiva.
    """
    from ada.graph import compile_graph
    from ada.retrieval.documents import TechnicalDocument
    from tests.unit.test_graph import FakeChatModel, fake_encoder

    documents = [
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

    original_query = (
        "Necesito decidir cómo implementar la comunicación entre microservicios "
        "en una arquitectura .NET desplegada en Azure. Compará Azure Service Bus "
        "y la comunicación HTTP síncrona, evaluando acoplamiento, resiliencia, "
        "latencia, complejidad operativa y escalabilidad."
    )
    decision_spec = extract_decision(original_query).model_dump()
    criteria = build_criteria(decision_spec["requested_criteria"])

    # Corpus real de prueba: no menciona Azure Service Bus ni HTTP síncrono.
    graph = compile_graph(FakeChatModel(), encoder=fake_encoder, documents=documents)
    state = {
        "user_request": original_query,
        "system_profile": {},
        "criteria": [c.model_dump() for c in criteria],
        "decision_spec": decision_spec,
    }
    final_state = {}
    for chunk in graph.stream(state, stream_mode="updates"):
        for node, update in (chunk or {}).items():
            if isinstance(update, dict):
                final_state.update(update)

    validation = final_state.get("validation_result") or {}
    assert validation.get("is_valid") is False
    assert "irrelevant" in " ".join(validation.get("errors", [])).lower()

    report = final_state.get("final_report") or {}
    assert report.get("status") == "limited"


# ---------------------------------------------------------------------------
# 9. Scoring determinístico con datos controlados (hallazgo de diagnóstico)
# ---------------------------------------------------------------------------


def test_scoring_distinguishes_alternatives_only_with_specific_evidence():
    """Diferencias de puntuación solo con evidencia específica por alternativa.

    Caso (a): la evidencia menciona señales de una alternativa (mensajería
    asincrónica) y no de la otra (llamada HTTP síncrona). El mecanismo
    léxico debe reflejar esa diferencia: la alternativa con soporte léxico
    recibe el empujón acotado (+1) y evidence_ids trazables; la otra no.
    No se afirma ningún hecho técnico: se prueba el mecanismo de señales.
    """
    from ada.agents.analyst import derive_scores_from_evidence

    criteria = build_criteria(["coupling", "resilience"])
    research_artifact = {
        "evidence": [
            {
                "document_id": "doc-test-1",
                "title": "Mensajería asincrónica",
                "excerpt": (
                    "La mensajería asincrónica desacopla los servicios y "
                    "mejora la resiliencia ante fallos temporales."
                ),
            }
        ]
    }
    alternatives = [
        {"alternative_id": "azure-service-bus", "description": "mensajería"},
        {"alternative_id": "sync-http", "description": "llamada http directa"},
    ]
    scores = derive_scores_from_evidence(
        user_request="comparar alternativas de comunicación",
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact=research_artifact,
    )

    bus_scores = scores["azure-service-bus"]
    http_scores = scores["sync-http"]
    by_criterion_http = {s["criterion_id"]: s for s in http_scores}

    for score in bus_scores:
        criterion_id = score["criterion_id"]
        http_score = by_criterion_http[criterion_id]
        # La evidencia menciona señales de Service Bus: soporte trazable.
        assert score["evidence_ids"], (
            f"la evidencia menciona señales de azure-service-bus; "
            f"{criterion_id} debe citarla"
        )
        assert http_score["evidence_ids"] == [], (
            f"la evidencia no menciona señales de sync-http; "
            f"{criterion_id} no debe atribuirse evidencia inexistente"
        )
        # El empujón de corpus (+1 acotado) solo para la alternativa soportada:
        # la diferencia entre ambas debe existir y estar justificada por la
        # evidencia (no se asume un valor exacto: los textos de las
        # alternativas también activan señales de criterio).
        assert score["score"] > http_score["score"], (
            f"{criterion_id}: azure-service-bus ({score['score']}) debe "
            f"superar a sync-http ({http_score['score']}) porque la evidencia "
            "solo menciona señales de la primera"
        )


def test_derived_scores_equal_without_sufficient_information():
    """Sin información suficiente, las puntuaciones son iguales.

    Con una consulta neutra (sin señales de criterio) y evidencia que no
    menciona ninguna de las dos alternativas, ninguna recibe empujones:
    las puntuaciones de ambas alternativas deben ser iguales (neutro
    honesto), sin diferencias inventadas.
    """
    from ada.agents.analyst import derive_scores_from_evidence

    criteria = build_criteria(["coupling", "resilience"])
    research_artifact = {
        "evidence": [
            {
                "document_id": "doc-neutral",
                "title": "Contexto organizacional",
                "excerpt": (
                    "Los equipos de plataforma acompañan la adopción "
                    "de prácticas de entrega continua."
                ),
            }
        ]
    }
    alternatives = [
        {"alternative_id": "azure-service-bus", "description": "alternativa a"},
        {"alternative_id": "sync-http", "description": "alternativa b"},
    ]
    scores = derive_scores_from_evidence(
        user_request="decidir un mecanismo de integración",
        system_profile={},
        criteria=criteria,
        alternatives=alternatives,
        research_artifact=research_artifact,
    )

    bus_scores = scores["azure-service-bus"]
    http_scores = scores["sync-http"]
    assert len(bus_scores) == len(http_scores) == 2
    for bus_entry, http_entry in zip(bus_scores, http_scores, strict=True):
        assert bus_entry["criterion_id"] == http_entry["criterion_id"]
        assert bus_entry["score"] == http_entry["score"], (
            f"sin señales específicas por alternativa, las puntuaciones de "
            f"{bus_entry['criterion_id']} deben ser iguales; obtuvo "
            f"{bus_entry['score']} vs {http_entry['score']}"
        )
