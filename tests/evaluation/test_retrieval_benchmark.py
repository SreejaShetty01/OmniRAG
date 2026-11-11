from evaluation.retrieval_benchmark import RetrievalGoldCase, aggregate_metrics, case_metrics
from omnirag.retrieval.contracts import EvidenceAssessment, RetrievalEvidence, RetrievalResult
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.retrieval.contracts import QueryAnalysis, QueryCharacteristics, RetrievalFilters


def _chunk(chunk_id, section):
    return Chunk(chunk_id=chunk_id, document_id="doc", parent_id="doc", content="evidence",
                 document_type=DocumentType.TEXT, content_hash=chunk_id, source="guide.md",
                 section_path=[section], access_control=AccessControlMetadata())


def test_retrieval_metrics_and_latency_percentiles():
    case = RetrievalGoldCase("c1", "question", frozenset({"a"}), frozenset({("Intro",)}))
    analysis = QueryAnalysis(original_query="question", normalized_query="question",
                             characteristics=QueryCharacteristics(), extracted_filters=RetrievalFilters())
    result = RetrievalResult(original_query="question", normalized_query="question",
                             query_analysis=analysis, applied_filters=RetrievalFilters(),
                             authorization_scope_hash="scope", assessment=EvidenceAssessment(state="insufficient_evidence"),
                             evidence=[RetrievalEvidence(chunk=_chunk("a", "Intro"), estimated_tokens=4)],
                             selected_token_estimate=4, timings_ms={"total": 12})
    metrics = case_metrics(case, result)
    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.2
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["section_recall@1"] == 1.0
    assert aggregate_metrics([metrics, {**metrics, "latency_ms": 20}])["latency_p95_ms"] == 20


def test_hit_rate_and_precision_handle_empty_retrieval_and_unlabeled_cases():
    case = RetrievalGoldCase("c2", "question", frozenset({"a"}))
    analysis = QueryAnalysis(original_query="question", normalized_query="question",
                             characteristics=QueryCharacteristics(), extracted_filters=RetrievalFilters())
    result = RetrievalResult(original_query="question", normalized_query="question",
                             query_analysis=analysis, applied_filters=RetrievalFilters(),
                             authorization_scope_hash="scope",
                             assessment=EvidenceAssessment(state="insufficient_evidence"))
    metrics = case_metrics(case, result)
    assert metrics["precision@5"] == 0.0
    assert metrics["hit_rate@5"] == 0.0
    unlabeled = RetrievalGoldCase("c3", "question", frozenset())
    assert case_metrics(unlabeled, result)["recall@5"] == 0.0
