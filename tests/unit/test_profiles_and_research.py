import time
import pytest

from omnirag.agent.graph import ResearchGraph, ResearchLimits
from omnirag.agent.planner import DeterministicResearchPlanner
from omnirag.config.profiles import PROFILE_CONFIGS, ProfileName
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.models.query import EscalationPreference, Query
from omnirag.retrieval.contracts import (
    EvidenceAssessment, EvidenceAssessmentState, QueryAnalysis, QueryCharacteristics,
    RetrievalEvidence, RetrievalFilters, RetrievalResult,
)
from omnirag.profiles.execution import ConversationProfileStore, EscalationConfig, ProfileExecutionService
from omnirag.llm.llm_client import ChatMessage, ChatRequest, ChatResponse, FallbackChatProvider


class MockRetriever:
    async def retrieve(self, query, *, scope, policy):
        raise AssertionError("not used in these structural tests")


class MockAnswers:
    def answer(self, *args, **kwargs):
        raise AssertionError("not used in these structural tests")


class BrokenProvider:
    def complete(self, request):
        raise RuntimeError("primary down")

    def stream(self, request):
        yield from ()


class WorkingProvider:
    def complete(self, request):
        return ChatResponse("ok", "fallback", request.model, raw_metadata={})

    def stream(self, request):
        yield "ok"


def _retrieval(state=EvidenceAssessmentState.PARTIAL):
    chunk = Chunk(chunk_id="chunk-1", document_id="doc-1", parent_id="doc-1",
                  content="A linked evidence passage.", document_type=DocumentType.TEXT,
                  content_hash="hash", source="source.md", links=["target-section"],
                  access_control=AccessControlMetadata())
    evidence = RetrievalEvidence(chunk=chunk, dense_rank=1, sparse_rank=2, fusion_rank=1,
                                 estimated_tokens=12)
    analysis = QueryAnalysis(original_query="What is alpha and beta?", normalized_query="question",
                             characteristics=QueryCharacteristics(), extracted_filters=RetrievalFilters())
    return RetrievalResult(original_query="What is alpha and beta?", normalized_query="question",
                           query_analysis=analysis, applied_filters=RetrievalFilters(),
                           authorization_scope_hash="scope", evidence=[evidence],
                           assessment=EvidenceAssessment(state=state), selected_token_estimate=12)


def test_profile_policies_have_real_unbenchmarked_differences():
    lite = PROFILE_CONFIGS[ProfileName.LITE]
    core = PROFILE_CONFIGS[ProfileName.CORE]
    research = PROFILE_CONFIGS[ProfileName.RESEARCH]
    assert lite.dense_candidates < core.dense_candidates < research.dense_candidates
    assert lite.context_budget_tokens < core.context_budget_tokens < research.context_budget_tokens
    assert not lite.reranking_enabled and core.reranking_enabled and research.reranking_enabled
    assert lite.verification_depth != core.verification_depth != research.verification_depth
    assert not lite.defaults_benchmarked and not core.defaults_benchmarked and not research.defaults_benchmarked
    assert research.research_max_time_seconds == 120.0


def test_profiles_reuse_shared_retriever_and_answer_pipeline():
    retriever, answer = MockRetriever(), MockAnswers()
    service = ProfileExecutionService(retriever, answer)
    assert service.retriever is retriever
    assert service.answer_pipeline is answer
    assert service.research_graph.retriever is retriever
    assert service.research_graph.answer_pipeline is answer


def test_conversation_profile_persists_and_can_be_switched():
    service = ProfileExecutionService(MockRetriever(), MockAnswers())
    service.conversations.set("conversation-a", ProfileName.LITE)
    inherited = Query(query_text="question", conversation_id="conversation-a")
    assert service._resolve_profile(inherited) == ProfileName.LITE
    changed = Query(query_text="next", conversation_id="conversation-a",
                    selected_profile=ProfileName.RESEARCH)
    assert service._resolve_profile(changed) == ProfileName.RESEARCH
    assert service.conversations.get("conversation-a") == ProfileName.RESEARCH
    other = Query(query_text="independent", conversation_id="conversation-b")
    assert service._resolve_profile(other) == ProfileName.CORE


def test_escalation_is_explicit_budgeted_and_never_by_default():
    service = ProfileExecutionService(MockRetriever(), MockAnswers())
    assert not service._within_escalation_budget(ProfileName.CORE)[0]
    blocked = ProfileExecutionService(MockRetriever(), MockAnswers(), escalation=EscalationConfig(
        enabled=True, max_additional_tokens=100, max_time_seconds=30))
    allowed, reason = blocked._within_escalation_budget(ProfileName.CORE)
    assert not allowed and "token_budget" in reason
    enough = ProfileExecutionService(MockRetriever(), MockAnswers(), escalation=EscalationConfig(
        enabled=True, max_additional_tokens=5000, max_time_seconds=30))
    assert enough._within_escalation_budget(ProfileName.CORE)[0]
    cost_limited = ProfileExecutionService(MockRetriever(), MockAnswers(), escalation=EscalationConfig(
        enabled=True, max_additional_tokens=5000, max_time_seconds=30, max_cost_usd=0.05))
    assert "cost_ceiling" in cost_limited._within_escalation_budget(ProfileName.CORE)[1]
    assert service._next_profile(ProfileName.RESEARCH) is None
    assert Query(query_text="q", escalation_preference=EscalationPreference.NEVER).escalation_preference == "never"


def test_explicit_model_fallback_is_reported():
    request = ChatRequest("primary-model", [ChatMessage("user", "hello")])
    response = FallbackChatProvider(BrokenProvider(), WorkingProvider(), "fallback-model").complete(request)
    assert response.model == "fallback-model"
    assert response.raw_metadata["fallback_used"] is True
    assert response.raw_metadata["primary_failure"] == "RuntimeError"


def test_research_graph_has_distinct_bounded_stages_and_deterministic_planning():
    graph = ResearchGraph(MockRetriever(), MockAnswers())
    nodes = set(graph.graph.get_graph().nodes)
    assert {"analyze", "plan", "retrieve", "assess_and_ledger", "plan_followup", "finalize"} <= nodes
    planner = DeterministicResearchPlanner()
    assert planner.decompose("What is alpha and beta?", 4) == ["What is alpha?", "What is beta?"]
    with pytest.raises(ValueError):
        ResearchLimits(max_rounds=9)
    with pytest.raises(ValueError):
        ResearchLimits(max_correction_attempts=2)


def test_research_ledger_followup_and_evidence_budget_termination():
    graph = ResearchGraph(MockRetriever(), MockAnswers())
    result = _retrieval()
    state = {
        "query": Query(query_text="What is alpha and beta?"),
        "query_analysis": result.query_analysis,
        "limits": ResearchLimits(max_rounds=3, max_subqueries=4, max_evidence=10,
                                  max_token_budget=100, max_output_tokens=20,
                                  max_prompt_overhead_tokens=10, max_time_seconds=50),
        "started_at": time.perf_counter(), "round_number": 1,
        "all_results": [{"sub_question": result.original_query, "result": result}],
        "round_results": [{"sub_question": result.original_query, "result": result}],
        "executed_queries": [result.original_query], "ledger": [], "warnings": [],
    }
    assessed = graph._assess_and_ledger(state)
    state.update(assessed)
    followup = graph._plan_followup(state)
    assert followup["pending_queries"] == ["Find the referenced material for target-section"]
    assert state["ledger"][0].evidence_references[0].chunk_id == "chunk-1"
    assert state["ledger"][0].verification_status == "not_run"

    tight = dict(state)
    tight["limits"] = ResearchLimits(max_rounds=3, max_subqueries=4, max_evidence=1,
                                     max_token_budget=100, max_output_tokens=20,
                                     max_prompt_overhead_tokens=10, max_time_seconds=50)
    assert graph._assess_and_ledger(tight)["research_termination_reason"] == "evidence_budget_exhausted"


def test_research_retrieval_reserves_configured_generation_time():
    graph = ResearchGraph(MockRetriever(), MockAnswers())
    limits = ResearchLimits(max_time_seconds=120.0, finalization_reserve_seconds=90.0)
    state = {"limits": limits, "started_at": time.perf_counter() - 31.0,
             "warnings": [], "pending_queries": ["query"], "executed_queries": []}

    assert graph._retrieval_time_remaining(state) == 0.0
    state["started_at"] = time.perf_counter() - 10.0
    assert 19.0 < graph._retrieval_time_remaining(state) <= 20.0
