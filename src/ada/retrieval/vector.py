
"""Local vector retrieval using embeddings and cosine similarity."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ada.retrieval.documents import RetrievedDocument, TechnicalDocument

EmbeddingEncoder = Callable[[list[str]], NDArray[np.float32]]


class VectorRetriever:
    """Retrieve documents by semantic similarity."""

    def __init__(
        self,
        documents: list[TechnicalDocument],
        encoder: EmbeddingEncoder,
    ) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Document IDs must be unique.")

        self._documents = documents
        self._encoder = encoder

        texts = [
            f"{document.title}. {document.topic}. {document.content}"
            for document in documents
        ]
        embeddings = np.asarray(encoder(texts), dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise ValueError("Encoder must return one vector per document.")
        if embeddings.shape[1] == 0:
            raise ValueError("Embeddings must not be empty.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must contain finite values.")

        self._embeddings = self._normalize(embeddings)

    @staticmethod
    def _normalize(
        vectors: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1.0, norms)
        return vectors / safe_norms

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Return documents ordered by cosine similarity."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = np.asarray(
            self._encoder([query]),
            dtype=np.float32,
        )

        if (
            query_vector.ndim != 2
            or query_vector.shape[0] != 1
            or query_vector.shape[1] != self._embeddings.shape[1]
        ):
            raise ValueError("Encoder returned an invalid query embedding.")
        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding must contain finite values.")

        normalized_query = self._normalize(query_vector)[0]
        scores = self._embeddings @ normalized_query

        ranked_indices = sorted(
            range(len(self._documents)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            RetrievedDocument(
                document=self._documents[index],
                score=float(scores[index]),
                rank=rank,
                retriever="vector",
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
