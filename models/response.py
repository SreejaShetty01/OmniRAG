"""Validated response, citation, and evidence reference contracts."""

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from config.profiles import ProfileName


class VerificationStatus(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class ClaimVerificationStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    NOT_CHECKED = "not_checked"


class Citation(BaseModel):
    model_config = ConfigDict(extra="allow")

    citation_id: str
    source: str
    quote: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    version: str | None = None
    title: str | None = None
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    source_location: str | None = None
    evidence_id: str | None = None


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    evidence_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    version: str | None = None
    source: str | None = None
    title: str | None = None
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    excerpt: str | None = None


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    status: ClaimVerificationStatus = ClaimVerificationStatus.NOT_CHECKED
    methods: list[str] = Field(default_factory=list)
    explanation: str | None = None
    verification_score: float | None = Field(default=None, ge=0.0, le=1.0)
    verifier_model_id: str | None = None
    verifier_model_digest: str | None = None


class OmniRAGResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    profile_used: ProfileName
    verification_status: VerificationStatus = VerificationStatus.NOT_RUN
    abstained: bool = False
    execution_metadata: dict[str, object] = Field(default_factory=dict)
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    original_query: str | None = None
    claims: list[ClaimVerification] = Field(default_factory=list)
    claim_verification_status: VerificationStatus = VerificationStatus.NOT_RUN
    citation_validation_status: VerificationStatus = VerificationStatus.NOT_RUN
    overall_reliability: VerificationStatus = VerificationStatus.NOT_RUN
    evidence_sufficiency: dict[str, object] = Field(default_factory=dict)
    retrieval_inspection: list[dict[str, object]] = Field(default_factory=list)
    abstention_reason: str | None = None
    contradictions: list[dict[str, object]] = Field(default_factory=list)
    generation_metadata: dict[str, object] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, ge=0.0)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
