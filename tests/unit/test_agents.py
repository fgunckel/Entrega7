"""Tests for the specialist agents (mocked encoder, no network)."""

from pathlib import Path

import numpy as np
import pytest

from ada.agents.analyst import build_analysis
from ada.agents.researcher import retrieve_evidence
from ada.retrieval.documents import TechnicalDocument
from ada.schemas import Criterion, CriterionScore


@pytest.fixture
def documents():
    return [
        TechnicalDocument(
            document_id="doc-1",
            title="Arquitectura de microservicios",
            content=(
                "Los microservicios separan capacidades de negocio "
                "en servicios independientes con despliegue autónomo."
            ),
            source="https://microservices.io/patterns/microservices.html",
            topic="arquitectura",
        ),
        TechnicalDocument(
            document_id="doc-2",
            title="Monolito modular",
            content=(
                "Un monolito modular organiza el código en módulos "
                "con límites claros dentro de una aplicación."
            ),
            source="https://www.kamilgrzybek.com/design/modular-monolith-primer/",
            topic="arquitectura",
            language="es",
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


def test_retrieve_evidence_builds_valid_artifact(documents):
    artifact = retrieve_evidence("monolito modular", encoder=fake_encoder, documents=documents)

    assert artifact.query == "monolito modular"
    assert artifact.evidence
    assert artifact.confidence_score > 0
    assert all(item.document_id for item in artifact.evidence)
    assert all(item.excerpt for item in artifact.evidence)


def test_researcher_reports_gaps_instead_of_fabricating(documents):
    artifact = retrieve_evidence(
        "tema completamente ajeno al corpus",
        encoder=fake_encoder,
        documents=documents,
    )

    # BM25/vector still return something; the artifact must not invent sources.
    assert all(item.document_id in {d.document_id for d in documents} for item in artifact.evidence)


def test_researcher_uses_default_corpus_when_none_given(monkeypatch):
    from ada.retrieval import corpus as corpus_module

    monkeypatch.setattr(
        corpus_module,
        "DEFAULT_CORPUS_PATH",
        Path(__file__).parent / "data" / "mini_corpus.json",
    )

    artifact = retrieve_evidence("monolito modular", encoder=fake_encoder)

    assert artifact.query == "monolito modular"
    assert artifact.confidence_score >= 0.0


def test_build_analysis_computes_totals_deterministically():
    criteria = [
        Criterion(criterion_id="c1", name="C1", weight=0.6, description="d1"),
        Criterion(criterion_id="c2", name="C2", weight=0.4, description="d2"),
    ]
    alternatives = [
        {"alternative_id": "a", "name": "A"},
        {"alternative_id": "b", "name": "B"},
    ]
    scores = {
        "a": [
            CriterionScore(criterion_id="c1", weight=0.6, score=5, rationale="r"),
            CriterionScore(criterion_id="c2", weight=0.4, score=3, rationale="r"),
        ],
        "b": [
            {"criterion_id": "c1", "weight": 0.6, "score": 4, "rationale": "r"},
            {"criterion_id": "c2", "weight": 0.4, "score": 4, "rationale": "r"},
        ],
    }

    artifact = build_analysis(criteria, alternatives, scores)

    assert artifact.ranking[0] == "a"
    a = next(a for a in artifact.alternatives if a.alternative_id == "a")
    assert abs(a.weighted_score - (0.6 * 1.0 + 0.4 * 0.6)) < 1e-6


def test_build_analysis_rejects_missing_alternative_scores():
    criteria = [Criterion(criterion_id="c1", name="C1", weight=1.0, description="d")]
    alternatives = [{"alternative_id": "a", "name": "A"}]

    with pytest.raises(ValueError, match="Missing proposed scores"):
        build_analysis(criteria, alternatives, {})
