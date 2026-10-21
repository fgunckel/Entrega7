"""Tests for the hybrid retrieval coordinator (mocked encoder, no network)."""

import numpy as np
import pytest

from ada.retrieval.documents import TechnicalDocument
from ada.retrieval.hybrid import HybridRetriever


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
            source="docs/microservices.md",
            topic="arquitectura",
        ),
        TechnicalDocument(
            document_id="doc-2",
            title="Monolito modular",
            content=(
                "Un monolito modular organiza el código en módulos "
                "con límites claros dentro de una aplicación."
            ),
            source="docs/modular-monolith.md",
            topic="arquitectura",
        ),
        TechnicalDocument(
            document_id="doc-3",
            title="Pruebas automatizadas",
            content=(
                "Las pruebas unitarias verifican componentes "
                "individuales del software."
            ),
            source="docs/testing.md",
            topic="testing",
        ),
    ]


def fake_encoder(texts: list[str]) -> "np.ndarray":
    """Return deterministic vectors keyed by topic keywords."""
    vectors = []
    for text in texts:
        normalized = text.lower()
        if "monolito" in normalized:
            vectors.append([0.0, 1.0])
        elif "pruebas" in normalized:
            vectors.append([-1.0, 0.0])
        else:
            vectors.append([1.0, 0.0])
    return np.asarray(vectors, dtype=np.float32)


def test_hybrid_fuses_rankings_and_preserves_provenance(documents):
    retriever = HybridRetriever(documents, encoder=fake_encoder)

    results = retriever.search("monolito modular con límites claros")

    assert results
    assert all(result.retriever for result in results)
    first = results[0]
    assert first.document.document_id == "doc-2"
    assert first.document.source == "docs/modular-monolith.md"
    assert first.document.content
    assert first.rank == 1


def test_hybrid_limits_results_to_top_k(documents):
    retriever = HybridRetriever(documents, encoder=fake_encoder, top_k=2)

    results = retriever.search("arquitectura de servicios")

    assert len(results) <= 2


def test_hybrid_annotates_contributing_strategies(documents):
    retriever = HybridRetriever(documents, encoder=fake_encoder)

    results = retriever.search("monolito modular")

    first = results[0]
    assert "bm25" in first.retriever or "vector" in first.retriever


def test_hybrid_rejects_empty_corpus():
    with pytest.raises(ValueError, match="corpus is empty"):
        HybridRetriever([], encoder=fake_encoder)


def test_hybrid_rejects_empty_query(documents):
    retriever = HybridRetriever(documents, encoder=fake_encoder)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("   ")


def test_hybrid_rejects_duplicate_document_ids(documents):
    duplicated = documents + [documents[0].model_copy()]

    try:
        HybridRetriever(duplicated, encoder=fake_encoder)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("Expected ValueError for duplicate document IDs.")


def test_hybrid_rejects_invalid_top_k(documents):
    retriever = HybridRetriever(documents, encoder=fake_encoder)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("monolito", top_k=0)
