"""Researcher agent: retrieve evidence from the local corpus.

The researcher owns only the `research_artifact` field of shared state.
It never fabricates sources: every evidence item comes from the retriever.
"""

import hashlib

from ada.retrieval.corpus import get_default_corpus
from ada.retrieval.documents import TechnicalDocument
from ada.retrieval.hybrid import HybridRetriever
from ada.schemas import ResearchArtifact, SourceEvidence


def _content_hash(content: str) -> str:
    """Return a stable short hash of the document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _split_methods(retriever_label: str) -> list[str]:
    """Split a combined retriever label like 'bm25+vector' into parts."""
    return [part for part in retriever_label.split("+") if part]


def retrieve_evidence(
    query: str,
    encoder=None,
    documents: list[TechnicalDocument] | None = None,
    top_k: int = 5,
) -> ResearchArtifact:
    """Retrieve evidence for a research query and build a validated artifact.

    If no evidence is found, the artifact reports the gap explicitly
    instead of fabricating sources.
    """
    if documents is None:
        documents = list(get_default_corpus())

    retriever = HybridRetriever(documents, encoder=encoder)
    results = retriever.search(query, top_k=top_k)

    evidence = [
        SourceEvidence(
            document_id=result.document.document_id,
            chunk_id=f"{result.document.document_id}#full",
            title=result.document.title,
            source_uri=(
                result.document.source
                if result.document.source.startswith("http")
                else None
            ),
            excerpt=result.document.content[:280],
            relevance_score=min(max(result.score, 0.0), 1.0),
            retrieval_method=_split_methods(result.retriever),
            content_hash=_content_hash(result.document.content),
        )
        for result in results
    ]

    covered_topics = sorted({result.document.topic for result in results})
    confidence = (len(evidence) / max(top_k, 1)) if evidence else 0.0

    gaps = []
    if not evidence:
        gaps.append(
            "No evidence found in the local corpus for this query. "
            "The answer will rely on assumptions, not on corpus evidence."
        )

    return ResearchArtifact(
        query=query,
        evidence=evidence,
        covered_topics=covered_topics,
        evidence_gaps=gaps,
        confidence_score=round(min(1.0, max(0.0, confidence)), 6),
    )
