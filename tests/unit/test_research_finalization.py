import json
import time

from omnirag.agent.evidence_ledger import LedgerEvidenceReference, ResearchLedgerEntry
from omnirag.agent.graph import ResearchGraph, ResearchLimits
from omnirag.config.profiles import ProfileName
from omnirag.generation.answer_generator import AnswerGenerator
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.llm.llm_client import ChatResponse
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.models.query import Query
from omnirag.models.response import ClaimVerificationStatus
from omnirag.retrieval.contracts import (
    EvidenceAssessment, EvidenceAssessmentState, QueryAnalysis, QueryCharacteristics,
    RetrievalEvidence, RetrievalFilters, RetrievalResult,
)


class InvalidReferenceProvider:
    def complete(self, _request):
        payload = {
            "answer": "The retention period is 30 days [E1].",
            "claims": [{
                "claim": "The retention period is 30 days.",
                # E1 matches the ledger, while E999 forces citation validation
                # to mark the claim unsupported.
                "support_evidence_ids": ["E1", "E999"],
            }],
        }
        return ChatResponse(json.dumps(payload), "test-provider", "test-model")


def test_research_finalization_keeps_citation_status_typed(monkeypatch):
    chunk = Chunk(chunk_id="chunk-a", document_id="doc-a", parent_id="doc-a",
                  content="The retention period is 30 days.",
                  document_type=DocumentType.TEXT, content_hash="hash", source="policy.md",
                  access_control=AccessControlMetadata())
    evidence = RetrievalEvidence(chunk=chunk, dense_rank=1, sparse_rank=1, fusion_rank=1)
    analysis = QueryAnalysis(original_query="How long?", normalized_query="how long",
                             characteristics=QueryCharacteristics(), extracted_filters=RetrievalFilters())
    retrieval = RetrievalResult(
        original_query="How long?", normalized_query="how long", query_analysis=analysis,
        applied_filters=RetrievalFilters(), authorization_scope_hash="scope", evidence=[evidence],
        assessment=EvidenceAssessment(state=EvidenceAssessmentState.STRONG),
    )
    ledger = ResearchLedgerEntry(
        step_id="step-1", round_number=1, sub_question="How long?",
        evidence_references=[LedgerEvidenceReference(
            document_id="doc-a", chunk_id="chunk-a", source="policy.md", excerpt="30 days",
        )],
        evidence_sufficiency_state=EvidenceAssessmentState.STRONG.value,
    )
    pipeline = AnswerPipeline(AnswerGenerator(InvalidReferenceProvider(), "test-model"))
    graph = ResearchGraph(object(), pipeline)
    query = Query(query_text="How long?", selected_profile=ProfileName.RESEARCH)
    state = {
        "query": query, "query_analysis": analysis, "limits": ResearchLimits(
            max_time_seconds=10, finalization_reserve_seconds=1, max_correction_attempts=0,
        ),
        "started_at": time.perf_counter(), "merged_retrieval": retrieval,
        "round_number": 1, "ledger": [ledger], "executed_queries": ["How long?"],
        "unresolved_questions": [], "warnings": [], "verification_enabled": True,
        "semantic_verification_enabled": False,
    }

    class ImmediateAwaitable:
        def __init__(self, value):
            self.value = value

        def __await__(self):
            if False:
                yield None
            return self.value

    monkeypatch.setattr("omnirag.agent.graph.asyncio.to_thread",
                        lambda function, *args, **kwargs: function(*args, **kwargs))
    monkeypatch.setattr("omnirag.agent.graph.asyncio.wait_for",
                        lambda value, timeout: ImmediateAwaitable(value))
    finalization = graph._finalize(state)
    try:
        finalization.send(None)
    except StopIteration as completed:
        result = completed.value["response"]
    else:
        raise AssertionError("finalization unexpectedly yielded outside the mocked provider wait")

    assert result.claims[0].status is ClaimVerificationStatus.UNSUPPORTED
    assert result.execution_metadata["evidence_ledger"][0]["verification_status"] == "unsupported"
    assert not any(warning.startswith("research_finalize_failed:") for warning in result.warnings)
