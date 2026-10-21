
import numpy as np
import pytest

from ada.retrieval.bm25 import BM25Retriever
from ada.retrieval.documents import RetrievedDocument, TechnicalDocument
from ada.retrieval.fusion import reciprocal_rank_fusion
from ada.retrieval.vector import VectorRetriever


@pytest.fixture
def documents():
    return [
        TechnicalDocument(
            document_id="doc-1",
            title="Arquitectura de microservicios",
            content=(
                "Los microservicios separan capacidades de negocio "
                "en servicios independientes."
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


def fake_encoder(texts: list[str]) -> np.ndarray:
    """Return deterministic vectors for unit tests."""
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


def test_bm25_returns_relevant_document_first(documents):
    retriever = BM25Retriever(documents)

    results = retriever.search("microservicios servicios independientes")

    assert results
    assert results[0].document.document_id == "doc-1"
    assert results[0].retriever == "bm25"
    assert results[0].rank == 1


def test_bm25_preserves_source_traceability(documents):
    retriever = BM25Retriever(documents)

    results = retriever.search("monolito modular")

    assert results[0].document.source == "docs/modular-monolith.md"


def test_bm25_rejects_empty_query(documents):
    retriever = BM25Retriever(documents)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("  ")


def test_bm25_rejects_invalid_top_k(documents):
    retriever = BM25Retriever(documents)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search("arquitectura", top_k=0)


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError, match="At least one document"):
        BM25Retriever([])


def test_vector_retriever_returns_most_similar_document_first(documents):
    retriever = VectorRetriever(documents, encoder=fake_encoder)

    results = retriever.search("microservicios")

    assert results[0].document.document_id == "doc-1"
    assert results[0].retriever == "vector"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)


def test_vector_retriever_preserves_source(documents):
    retriever = VectorRetriever(documents, encoder=fake_encoder)

    results = retriever.search("monolito modular")

    assert results[0].document.document_id == "doc-2"
    assert results[0].document.source == "docs/modular-monolith.md"


def test_vector_retriever_rejects_empty_query(documents):
    retriever = VectorRetriever(documents, encoder=fake_encoder)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.search("  ")


def test_vector_retriever_rejects_wrong_embedding_dimension(documents):
    def invalid_encoder(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)

    retriever = VectorRetriever(documents, encoder=fake_encoder)
    retriever._encoder = invalid_encoder

    with pytest.raises(ValueError, match="invalid query embedding"):
        retriever.search("microservicios")


def test_rrf_ranks_document_present_in_both_rankings(documents):
    ranking_bm25 = [
        RetrievedDocument(
            document=documents[0],
            score=10.0,
            rank=1,
            retriever="bm25",
        ),
        RetrievedDocument(
            document=documents[1],
            score=8.0,
            rank=2,
            retriever="bm25",
        ),
    ]

    ranking_vector = [
        RetrievedDocument(
            document=documents[0],
            score=0.95,
            rank=1,
            retriever="vector",
        ),
        RetrievedDocument(
            document=documents[2],
            score=0.90,
            rank=2,
            retriever="vector",
        ),
    ]

    results = reciprocal_rank_fusion([ranking_bm25, ranking_vector])

    assert results[0].document.document_id == "doc-1"
    assert results[0].retriever == "rrf"
    assert results[0].rank == 1


def test_rrf_preserves_source_traceability(documents):
    ranking = [
        [
            RetrievedDocument(
                document=documents[1],
                score=1.0,
                rank=1,
                retriever="bm25",
            )
        ]
    ]

    results = reciprocal_rank_fusion(ranking)

    assert results[0].document.source == "docs/modular-monolith.md"
    assert results[0].document.document_id == "doc-2"


def test_rrf_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank_fusion([], k=0)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reciprocal_rank_fusion([], top_k=0)
