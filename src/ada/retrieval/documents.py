
"""Data contracts for local technical document retrieval."""

from pydantic import Field

from ada.schemas import StrictModel


class TechnicalDocument(StrictModel):
    """A source document stored in the local technical corpus."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    language: str = "es"


class RetrievedDocument(StrictModel):
    """A document returned by a retrieval strategy."""

    document: TechnicalDocument
    score: float
    rank: int = Field(ge=1)
    retriever: str = Field(min_length=1)
