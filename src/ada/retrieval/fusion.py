
"""Reciprocal Rank Fusion for combining retrieval rankings."""

from collections import defaultdict

from ada.retrieval.documents import RetrievedDocument


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedDocument]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedDocument]:
    """Fuse ranked lists using reciprocal rank contributions."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, RetrievedDocument] = {}

    for ranking in rankings:
        seen_in_ranking: set[str] = set()

        for result in ranking:
            document_id = result.document.document_id

            if document_id in seen_in_ranking:
                continue

            seen_in_ranking.add(document_id)
            scores[document_id] += 1.0 / (k + result.rank)
            documents.setdefault(document_id, result)

    ordered_ids = sorted(
        scores,
        key=lambda document_id: (-scores[document_id], document_id),
    )[:top_k]

    return [
        RetrievedDocument(
            document=documents[document_id].document,
            score=scores[document_id],
            rank=rank,
            retriever="rrf",
        )
        for rank, document_id in enumerate(ordered_ids, start=1)
    ]
