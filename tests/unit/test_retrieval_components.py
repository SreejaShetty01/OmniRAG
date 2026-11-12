import pytest
from pydantic import ValidationError

from omnirag.models.query import Query
from omnirag.models.chunk import Chunk
from omnirag.models.document import DocumentType
from omnirag.retrieval.cache import RetrievalCache
from omnirag.retrieval.contracts import (
    EvidenceAssessmentState,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalPolicy,
    RetrievalScope,
)
from omnirag.retrieval.deduplication import deduplicate
from omnirag.retrieval.evidence_selection import EvidenceSelector
from omnirag.retrieval.fusion import ReciprocalRankFusion, StructuralScorer
from omnirag.retrieval.query_analyzer import QueryAnalyzer
from omnirag.retrieval.query_filters import FilterBuilder
from omnirag.retrieval.sufficiency import EvidenceSufficiencyAssessor


def candidate(chunk_id: str, *, dense_rank=None, sparse_rank=None, content="some technical evidence"):
    chunk = Chunk(chunk_id=chunk_id, document_id="doc-1", parent_id="doc-1", content=content,
                  document_type=DocumentType.TEXT, content_hash=f"hash-{chunk_id}", source="notes.txt")
    result = RetrievalCandidate(chunk=chunk, dense_rank=dense_rank,
                                sparse_rank=sparse_rank,
                                dense_score=0.7 if dense_rank else None,
                                sparse_score=2.0 if sparse_rank else None)
    result.provenance = [name for name, rank in (("dense", dense_rank), ("sparse", sparse_rank)) if rank]
    return result


def test_query_analysis_preserves_original_and_extracts_explicit_constraints():
    query = Query(query_text='How does fusion work? language:python section:"Retrieval > Fusion" version:v2',
                  filters={"document_id": "doc-1", "content_type": "code"})
    analysis = QueryAnalyzer().analyze(query)
    assert analysis.original_query == query.query_text
    assert analysis.normalized_query == "How does fusion work?"
    assert analysis.characteristics.section_specific
    assert analysis.characteristics.version_sensitive
    assert analysis.extracted_filters.languages == ["python"]
    assert analysis.extracted_filters.section_paths == [["Retrieval", "Fusion"]]
    assert analysis.extracted_filters.document_ids == ["doc-1"]
    assert analysis.extracted_filters.content_types == ["code"]


def test_query_analyzer_does_not_invent_metadata_filters():
    analysis = QueryAnalyzer().analyze(Query(query_text="What is version v2 in file path.py?"))
    assert analysis.extracted_filters.model_dump(exclude_defaults=True) == {}
    assert analysis.characteristics.version_sensitive


def test_authorization_scope_fails_closed_without_principal():
    with pytest.raises(ValidationError):
        RetrievalScope(principal_ids=[])
    scope = RetrievalScope(principal_ids=[" user:alice ", "user:alice"])
    assert scope.principal_ids == ("user:alice",)


def test_filter_builder_requires_active_points_acl_and_tenant():
    from qdrant_client import models
    scope = RetrievalScope(principal_ids=["user:alice"], tenant_id="tenant-a")
    query_filter = FilterBuilder().build(QueryAnalyzer().analyze(Query(query_text="x")).extracted_filters, scope)
    assert any(condition.key == "is_active" for condition in query_filter.must)
    assert any(getattr(condition, "key", None) == "metadata.tenant_id" for condition in query_filter.must)
    assert any(getattr(condition, "should", None) for condition in query_filter.must)
    assert any(isinstance(condition, models.Filter) for condition in query_filter.must)
    assert query_filter.must_not[0].key == "acl_denied_principals"


def test_filter_builder_fails_closed_without_tenant():
    from omnirag.retrieval.query_filters import FilterBuilder
    with pytest.raises(ValueError, match="Tenant scope is required"):
        FilterBuilder().build(RetrievalFilters(), RetrievalScope(principal_ids=("user:alice",)))


def test_rrf_fuses_ranks_without_treating_scores_as_probability():
    dense = [candidate("d1", dense_rank=1), candidate("both", dense_rank=2)]
    sparse = [candidate("both", sparse_rank=1), candidate("s1", sparse_rank=2)]
    ranked = ReciprocalRankFusion().fuse(dense, sparse, rrf_k=10)
    assert ranked[0].identity == "both"
    assert ranked[0].rrf_dense_contribution == pytest.approx(1 / 12)
    assert ranked[0].rrf_sparse_contribution == pytest.approx(1 / 11)
    assert ranked[0].fusion_rank == 1
    assert ranked[0].rrf_score < 1.0


def test_rrf_ties_use_deterministic_chunk_order():
    dense = [candidate("b", dense_rank=1), candidate("a", dense_rank=1)]
    ranked = ReciprocalRankFusion().fuse(dense, [], rrf_k=60)
    assert [item.identity for item in ranked] == ["a", "b"]


def test_structure_scoring_is_observable_and_dedup_is_traceable():
    first = candidate("chunk-1", dense_rank=1, content="Hybrid fusion details and behavior")
    first.chunk.heading = "Hybrid Fusion"
    first.chunk.section_path = ["Retrieval", "Fusion"]
    score = StructuralScorer().score("hybrid fusion", first)
    assert score == 1.0
    second = candidate("chunk-2", sparse_rank=1, content="Hybrid fusion details and behavior")
    result, removed = deduplicate([first, second])
    assert len(result) == 1 and removed == 1
    assert any("duplicate_source" in value for value in result[0].provenance)


def test_evidence_selection_respects_budget_and_preserves_whole_code():
    code = candidate("code", dense_rank=1, content="def answer():\n    return 42")
    docs = candidate("doc", sparse_rank=1, content="short evidence")
    policy = RetrievalPolicy(max_evidence=5, evidence_token_budget=2, diversity_enabled=False)
    evidence, used, skipped = EvidenceSelector().select([code, docs], policy)
    assert len(evidence) == 1
    assert evidence[0].chunk.content == "short evidence"
    assert used <= 2 and skipped == 1


def test_sufficiency_is_heuristic_uncalibrated_and_exposes_conflict_state():
    assessor = EvidenceSufficiencyAssessor(strong_agreement_min=1, strong_source_min=1)
    result = assessor.assess([candidate("both", dense_rank=1, sparse_rank=1)])
    assert result.state is EvidenceAssessmentState.STRONG
    assert result.calibrated is False
    assert result.signals["rrf_scores_are_probabilities"] is False
    conflict = assessor.assess([], conflict_signal=True)
    assert conflict.state is EvidenceAssessmentState.CONFLICTING


def test_retrieval_cache_isolated_by_caller_supplied_key():
    cache = RetrievalCache[str](capacity=2)
    cache.set("alice|generation-a", "alice-result")
    cache.set("bob|generation-a", "bob-result")
    assert cache.get("alice|generation-a", 10) == "alice-result"
    assert cache.get("bob|generation-a", 10) == "bob-result"
    assert cache.get("alice|generation-b", 10) is None
