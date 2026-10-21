"""Hybrid retrieval coordinator: BM25 + vector search fused with RRF."""

from ada.config import get_settings
from ada.retrieval.bm25 import BM25Retriever
from ada.retrieval.documents import RetrievedDocument, TechnicalDocument
from ada.retrieval.fusion import reciprocal_rank_fusion
from ada.retrieval.vector import VectorRetriever


class HybridRetriever:
    """Run BM25 and vector retrieval over the same corpus and fuse with RRF.

    Both strategies run over the same `TechnicalDocument` list, so every
    fused result keeps full provenance (document_id, source, content).
    """

    def __init__(
        self,
        documents: list[TechnicalDocument],
        encoder=None,
        top_k: int | None = None,
    ) -> None:
        if not documents:
            raise ValueError(
                "The corpus is empty. Add documents to data/corpus/ before searching."
            )

        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Document IDs must be unique in the corpus.")

        if top_k is None:
            top_k = get_settings().top_k
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self._documents = documents
        self._top_k = top_k
        self._bm25 = BM25Retriever(documents)
        if encoder is None:
            from ada.retrieval.encoder import SentenceTransformerEncoder

            encoder = SentenceTransformerEncoder()
        self._vector = VectorRetriever(documents, encoder=encoder)

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        """Run hybrid retrieval and return RRF-fused results.

        Results are ordered by fused score and keep document_id, source,
        text and the retrieval strategies that matched each document.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k is None:
            top_k = self._top_k
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        bm25_ranking = self._bm25.search(query, top_k=top_k)
        vector_ranking = self._vector.search(query, top_k=top_k)

        fused = reciprocal_rank_fusion(
            [bm25_ranking, vector_ranking],
            top_k=top_k,
        )

        # Annotate which strategies contributed to each fused result.
        bm25_ids = {candidate.document.document_id for candidate in bm25_ranking}
        vector_ids = {candidate.document.document_id for candidate in vector_ranking}

        for result in fused:
            strategies = []
            if result.document.document_id in bm25_ids:
                strategies.append("bm25")
            if result.document.document_id in vector_ids:
                strategies.append("vector")
            if strategies:
                result.retriever = "+".join(strategies)

        return fused
