"""Serializable request, policy, candidate, and evidence contracts."""

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.chunk import Chunk


class QueryCharacteristics(BaseModel):
    exact_identifier: bool = False
    natural_language: bool = True
    technical: bool = False
    section_specific: bool = False
    multi_concept: bool = False
    document_specific: bool = False
    version_sensitive: bool = False
    code_related: bool = False


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    versions: list[str] = Field(default_factory=list)
    section_paths: list[list[str]] = Field(default_factory=list)
    collection_ids: list[str] = Field(default_factory=list)

    @field_validator("document_ids", "sources", "source_types", "document_types", "content_types",
                     "languages", "versions", "collection_ids")
    @classmethod
    def strip_values(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("Metadata filter values cannot be empty")
        return list(dict.fromkeys(cleaned))


class QueryAnalysis(BaseModel):
    original_query: str
    normalized_query: str
    characteristics: QueryCharacteristics
    exact_terms: list[str] = Field(default_factory=list)
    extracted_filters: RetrievalFilters
    warnings: list[str] = Field(default_factory=list)


class RetrievalScope(BaseModel):
    """Mandatory caller authorization context; never inferred from query text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_ids: tuple[str, ...]
    tenant_id: str | None = None
    workspace_id: str | None = None
    corpus_generation: str | None = None

    @field_validator("principal_ids")
    @classmethod
    def require_principals(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not cleaned:
            raise ValueError("Retrieval requires at least one authenticated principal")
        return cleaned

    @field_validator("tenant_id", "workspace_id")
    @classmethod
    def validate_optional_scope(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Authorization scope values cannot be blank")
        return value.strip() if value is not None else None


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_limit: int = Field(default=40, gt=0)
    sparse_limit: int = Field(default=40, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    max_evidence: int = Field(default=8, gt=0)
    evidence_token_budget: int = Field(default=3_000, gt=0)
    expansion_token_budget: int = Field(default=1_000, ge=0)
    expansion_max_count: int = Field(default=8, ge=0)
    expansion_seed_limit: int = Field(default=5, ge=0)
    expand_parents: bool = True
    expand_children: bool = False
    expand_neighbours: bool = False
    expand_cross_references: bool = False
    neighbour_distance: int = Field(default=1, ge=1)
    structure_weight: float = Field(default=0.0, ge=0.0)
    reranking_enabled: bool = False
    rerank_limit: int = Field(default=20, gt=0)
    cache_ttl_seconds: int = Field(default=0, ge=0)
    deduplicate_similarity: float = Field(default=0.9, ge=0.5, le=1.0)
    diversity_enabled: bool = True
    dense_retrieval_enabled: bool = True
    sparse_retrieval_enabled: bool = True
    hybrid_fusion_enabled: bool = True
    evidence_selection_enabled: bool = True

    @model_validator(mode="after")
    def validate_retrieval_channels(self):
        if not self.dense_retrieval_enabled and not self.sparse_retrieval_enabled:
            raise ValueError("At least one retrieval channel must remain enabled")
        if self.hybrid_fusion_enabled and not (
            self.dense_retrieval_enabled and self.sparse_retrieval_enabled
        ):
            raise ValueError("Hybrid fusion requires both dense and sparse retrieval")
        return self


class EvidenceAssessmentState(StrEnum):
    STRONG = "strong_relevant_evidence"
    PARTIAL = "partial_evidence"
    WEAK = "weak_or_unrelated_evidence"
    CONFLICTING = "conflicting_evidence"
    INSUFFICIENT = "insufficient_evidence"


class EvidenceAssessment(BaseModel):
    state: EvidenceAssessmentState
    calibrated: bool = False
    signals: dict[str, float | int | bool | str] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class RetrievalEvidence(BaseModel):
    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fusion_rank: int | None = None
    rrf_score: float | None = None
    rrf_dense_contribution: float = 0.0
    rrf_sparse_contribution: float = 0.0
    structure_score: float = 0.0
    ranking_score: float = 0.0
    reranker_score: float | None = None
    expansion_reason: str | None = None
    reranker_rank: int | None = None
    provenance: list[str] = Field(default_factory=list)
    estimated_tokens: int = 0


class RetrievalResult(BaseModel):
    original_query: str
    normalized_query: str
    query_analysis: QueryAnalysis
    applied_filters: RetrievalFilters
    authorization_scope_hash: str
    evidence: list[RetrievalEvidence] = Field(default_factory=list)
    assessment: EvidenceAssessment
    selected_token_estimate: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    retrieval_inspection: list[dict[str, object]] = Field(default_factory=list)
    cost_usd: float | None = Field(default=None, ge=0.0)
    cache_status: str = "disabled"
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class RetrievalCandidate:
    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    rrf_dense_contribution: float = 0.0
    rrf_sparse_contribution: float = 0.0
    rrf_score: float = 0.0
    structure_score: float = 0.0
    ranking_score: float = 0.0
    reranker_score: float | None = None
    expansion_reason: str | None = None
    fusion_rank: int | None = None
    reranker_rank: int | None = None
    provenance: list[str] = field(default_factory=list)

    @property
    def identity(self) -> str:
        return self.chunk.chunk_id
