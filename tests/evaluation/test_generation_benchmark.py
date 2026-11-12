from eval.generation_benchmark import AnswerGoldCase, evaluate_response
from omnirag.models.response import Citation, EvidenceReference, OmniRAGResponse
from omnirag.config.profiles import ProfileName


def test_generation_benchmark_metrics_are_computed_from_labeled_case():
    case = AnswerGoldCase("case-1", frozenset({"30 days"}), frozenset({"E1"}), {}, False)
    response = OmniRAGResponse(answer="Retention period is 30 days.", profile_used=ProfileName.CORE)
    response.citations = [Citation(citation_id="C1", source="policy.md", evidence_id="E1")]
    response.evidence_references = [EvidenceReference(evidence_id="E1", source="policy.md")]
    metrics = evaluate_response(case, response)
    assert metrics["answer_completeness"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_validity"] == 1.0
    assert metrics["faithfulness"] is None
    assert metrics["answer_relevancy"] is None


def test_citation_validity_requires_a_matching_returned_evidence_reference():
    case = AnswerGoldCase("case-2", frozenset(), frozenset({"E1"}), {}, False)
    response = OmniRAGResponse(answer="A factual answer.", profile_used=ProfileName.CORE)
    response.citations = [Citation(citation_id="C1", source="policy.md", evidence_id="E1",
                                   document_id="doc-a", chunk_id="chunk-a")]
    response.evidence_references = [EvidenceReference(evidence_id="E1", document_id="doc-b",
                                                      chunk_id="chunk-b")]
    metrics = evaluate_response(case, response)
    assert metrics["citation_validity"] == 0.0


def test_unavailable_semantic_metrics_remain_null_when_aggregated():
    case = AnswerGoldCase("case-3", frozenset(), frozenset(), {}, False)
    response = OmniRAGResponse(answer="", profile_used=ProfileName.CORE)
    metrics = evaluate_response(case, response)
    from eval.generation_benchmark import aggregate_answer_metrics
    aggregate = aggregate_answer_metrics([metrics])
    assert aggregate["faithfulness"] is None
    assert aggregate["answer_relevancy"] is None
