
"""BM25 lexical retrieval over a local document collection."""

import re

from rank_bm25 import BM25Okapi

from ada.retrieval.documents import RetrievedDocument, TechnicalDocument


def tokenize(text: str) -> list[str]:
    """Normalize text into simple lowercase word tokens."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    """Retrieve documents using lexical relevance."""

    def __init__(self, documents: list[TechnicalDocument]) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        ids = [document.document_id for document in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Document IDs must be unique.")

        self._documents = documents
        self._tokenized_documents = [
            tokenize(f"{document.title} {document.topic} {document.content}")
            for document in documents
        ]
        self._index = BM25Okapi(self._tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Return the highest-ranked documents for a query."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Stable tie-breaking makes results deterministic.
        ranked_indices = sorted(
            range(len(self._documents)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            RetrievedDocument(
                document=self._documents[index],
                score=float(scores[index]),
                rank=rank,
                retriever="bm25",
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
