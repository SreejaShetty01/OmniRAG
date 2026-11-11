"""LangGraph research state; contains IDs and structured results, not provider secrets."""

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    query: Any
    query_analysis: Any
    research_plan: list[str]
    scope: Any
    retrieval_policy: Any
    limits: Any
    started_at: float
    round_number: int
    pending_queries: list[str]
    executed_queries: list[str]
    round_results: list[dict[str, Any]]
    all_results: list[dict[str, Any]]
    ledger: list[Any]
    merged_retrieval: Any
    unresolved_questions: list[str]
    new_evidence_count: int
    research_termination_reason: str
    contradictions: list[dict[str, Any]]
    response: Any
    warnings: list[str]
    query_expansion_enabled: bool
    verification_enabled: bool
    semantic_verification_enabled: bool
