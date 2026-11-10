"""Initial execution policy defaults for OmniRAG profiles."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProfileName(StrEnum):
    LITE = "lite"
    CORE = "core"
    RESEARCH = "research"


class RetrievalDepth(StrEnum):
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"


class ContextScale(StrEnum):
    CONSTRAINED = "constrained"
    BALANCED = "balanced"
    EXPANDED = "expanded"


class VerificationDepth(StrEnum):
    CITATIONS = "citations_only"
    FULL = "full_claim_checks"
    RESEARCH = "cross_source_research"


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ProfileName
    retrieval_depth: RetrievalDepth
    context_scale: ContextScale
    retrieval_candidates: int | None = Field(default=None, gt=0)
    context_budget_tokens: int | None = Field(default=None, gt=0)
    latency_budget_ms: int | None = Field(default=None, gt=0)
    reranking_enabled: bool = False
    sufficiency_checks_enabled: bool = False
    claim_verification_enabled: bool = False
    multi_query_enabled: bool = False
    iterative_retrieval_steps: int = Field(default=0, ge=0)
    research_planning_enabled: bool = False
    evidence_ledger_enabled: bool = False
    cross_document_verification_enabled: bool = False
    orchestration: str = "sequential"
    dense_candidates: int = Field(default=40, gt=0)
    sparse_candidates: int = Field(default=40, gt=0)
    expansion_max_count: int = Field(default=0, ge=0)
    expansion_token_budget: int = Field(default=0, ge=0)
    expand_parents: bool = False
    expand_children: bool = False
    expand_neighbours: bool = False
    expand_cross_references: bool = False
    verification_depth: VerificationDepth = VerificationDepth.CITATIONS
    generation_max_output_tokens: int = Field(default=800, gt=0)
    generation_style: str = "balanced"
    abstain_on_weak_evidence: bool = True
    research_max_rounds: int = Field(default=1, ge=1)
    research_max_subqueries: int = Field(default=1, ge=1)
    research_max_evidence: int = Field(default=8, gt=0)
    research_max_tokens: int = Field(default=3000, gt=0)
    research_max_time_seconds: float = Field(default=20.0, gt=0)
    defaults_benchmarked: bool = False


PROFILE_CONFIGS: dict[ProfileName, ProfileConfig] = {
    ProfileName.LITE: ProfileConfig(
        name=ProfileName.LITE, retrieval_depth=RetrievalDepth.SHALLOW,
        context_scale=ContextScale.CONSTRAINED,
        retrieval_candidates=12, dense_candidates=12, sparse_candidates=12,
        context_budget_tokens=700, latency_budget_ms=None,
        reranking_enabled=False, sufficiency_checks_enabled=True,
        claim_verification_enabled=False, evidence_ledger_enabled=False,
        expansion_max_count=1, expansion_token_budget=120, expand_parents=True,
        verification_depth=VerificationDepth.CITATIONS,
        generation_max_output_tokens=400, generation_style="concise",
        research_max_rounds=1, research_max_subqueries=1,
        research_max_evidence=4, research_max_tokens=700,
    ),
    ProfileName.CORE: ProfileConfig(
        name=ProfileName.CORE, retrieval_depth=RetrievalDepth.STANDARD,
        context_scale=ContextScale.BALANCED,
        retrieval_candidates=40, dense_candidates=40, sparse_candidates=40,
        context_budget_tokens=3000, reranking_enabled=True, sufficiency_checks_enabled=True,
        claim_verification_enabled=True, expansion_max_count=8, expansion_token_budget=1000,
        expand_parents=True, expand_children=True, expand_neighbours=True,
        verification_depth=VerificationDepth.FULL, generation_max_output_tokens=1200,
        generation_style="balanced", research_max_rounds=1, research_max_subqueries=1,
        research_max_evidence=8, research_max_tokens=3000,
    ),
    ProfileName.RESEARCH: ProfileConfig(
        name=ProfileName.RESEARCH, retrieval_depth=RetrievalDepth.DEEP,
        context_scale=ContextScale.EXPANDED,
        reranking_enabled=True, sufficiency_checks_enabled=True,
        retrieval_candidates=50, dense_candidates=50, sparse_candidates=50,
        context_budget_tokens=6000, claim_verification_enabled=True, multi_query_enabled=True,
        iterative_retrieval_steps=3, research_planning_enabled=True,
        evidence_ledger_enabled=True, cross_document_verification_enabled=True,
        orchestration="langgraph",
        expansion_max_count=12, expansion_token_budget=1800,
        expand_parents=True, expand_children=True, expand_neighbours=True,
        expand_cross_references=True, verification_depth=VerificationDepth.RESEARCH,
        generation_max_output_tokens=1800, generation_style="structured_research",
        research_max_rounds=3, research_max_subqueries=8,
        research_max_evidence=24, research_max_tokens=6000,
        # The live provider timeout is 90s; a 60s end-to-end Research deadline
        # preempts final generation before its configured request timeout.
        research_max_time_seconds=120.0,
    ),
}


def get_profile_config(profile: ProfileName | str) -> ProfileConfig:
    """Resolve a profile name to its immutable configuration."""
    return PROFILE_CONFIGS[ProfileName(profile)]
