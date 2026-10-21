"""Validated domain contracts exchanged between ADA components."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SystemProfile(StrictModel):
    technology_stack: list[str] = Field(default_factory=list)
    current_architecture: str
    problems: list[str] = Field(min_length=1)
    team_size: int | None = Field(default=None, ge=1)
    constraints: list[str] = Field(default_factory=list)
    workload_profile: str | None = None
    missing_information: list[str] = Field(default_factory=list)


class SourceEvidence(StrictModel):
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    # URI de la fuente como str (no HttpUrl): el estado del grafo se
    # serializa con msgpack en el checkpointer y HttpUrl no es serializable.
    source_uri: str | None = None
    excerpt: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    retrieval_method: list[Literal["bm25", "vector"]] = Field(min_length=1)
    content_hash: str = Field(min_length=1)


class ResearchArtifact(StrictModel):
    query: str = Field(min_length=1)
    evidence: list[SourceEvidence] = Field(default_factory=list)
    covered_topics: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)


class Criterion(StrictModel):
    criterion_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1)


class CriterionScore(StrictModel):
    criterion_id: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    score: int = Field(ge=1, le=5)
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    is_assumption: bool = False


class AlternativeAnalysis(StrictModel):
    alternative_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    scores: list[CriterionScore] = Field(min_length=1)
    weighted_score: float = Field(ge=0.0, le=1.0)
    advantages: list[str] = Field(default_factory=list)
    disadvantages: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class AnalysisArtifact(StrictModel):
    alternatives: list[AlternativeAnalysis] = Field(min_length=2)
    ranking: list[str] = Field(min_length=2)
    data_gaps: list[str] = Field(default_factory=list)
    qualitative_summary: str = Field(min_length=1)


class ValidationResult(StrictModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_evidence_ids: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    # Errores clasificados por el especialista responsable de corregirlos.
    research_errors: list[str] = Field(default_factory=list)
    analysis_errors: list[str] = Field(default_factory=list)
    # Pertinencia semántica: criterios solicitados sin soporte en el corpus.
    unsupported_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_result_has_no_errors(self):
        if self.is_valid and self.errors:
            raise ValueError("A valid result cannot contain validation errors.")
        return self


class RoutingDecision(StrictModel):
    next_agent: Literal[
        "researcher",
        "analyst",
        "validator",
        "synthesizer",
        "limited_finish",
        "END",
    ]
    reason: str = Field(min_length=1)
    task_completed: bool = False


class FinalReport(StrictModel):
    status: Literal["complete", "limited", "needs_human_review"]
    executive_summary: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    risks_and_limitations: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)


def validate_criteria_weights(criteria: list[Criterion], tolerance: float = 1e-6) -> None:
    """Raise ValueError unless criteria weights sum to 1."""
    if not criteria:
        raise ValueError("At least one criterion is required.")

    total = sum(criterion.weight for criterion in criteria)
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"Criterion weights must sum to 1.0; received {total:.6f}.")
