"""Metrics for separately maintained, human-labeled answer/evidence evaluation cases."""

from dataclasses import dataclass
import re
import statistics
from typing import Iterable

from omnirag.models.response import (ClaimVerificationStatus, OmniRAGResponse,
                             VerificationStatus)


@dataclass(frozen=True, slots=True)
class AnswerGoldCase:
    case_id: str
    required_answer_facts: frozenset[str]
    relevant_evidence_ids: frozenset[str]
    gold_claim_labels: dict[str, ClaimVerificationStatus]
    should_abstain: bool = False


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def evaluate_response(case: AnswerGoldCase, response: OmniRAGResponse
                      ) -> dict[str, float | int | None]:
    answer = _norm(response.answer)
    fact_hits = sum(_norm(fact) in answer for fact in case.required_answer_facts)
    completeness = fact_hits / len(case.required_answer_facts) if case.required_answer_facts else 1.0
    claims = response.claims
    supported = sum(claim.status == ClaimVerificationStatus.SUPPORTED for claim in claims)
    unsupported = sum(claim.status in {ClaimVerificationStatus.UNSUPPORTED,
                                      ClaimVerificationStatus.CONTRADICTED} for claim in claims)
    context_ids = {reference.evidence_id for reference in response.evidence_references}
    cited = {citation.evidence_id for citation in response.citations if citation.evidence_id}
    context_precision = (len(context_ids & case.relevant_evidence_ids) / len(context_ids)
                         if context_ids else None)
    context_recall = (len(context_ids & case.relevant_evidence_ids) /
                      len(case.relevant_evidence_ids) if case.relevant_evidence_ids else None)
    citation_precision = len(cited & case.relevant_evidence_ids) / len(cited) if cited else 0.0
    citation_recall = (len(cited & case.relevant_evidence_ids) / len(case.relevant_evidence_ids)
                       if case.relevant_evidence_ids else 1.0)
    gold_labels = list(case.gold_claim_labels.items())
    predicted_by_text = {_norm(claim.claim_text): claim.status for claim in claims}
    support_tp = support_fp = support_fn = contradiction_tp = contradiction_fp = contradiction_fn = 0
    unsupported_tp = unsupported_fp = unsupported_fn = 0
    for text, gold in gold_labels:
        predicted = predicted_by_text.get(_norm(text))
        if gold == ClaimVerificationStatus.SUPPORTED:
            support_tp += predicted == ClaimVerificationStatus.SUPPORTED
            support_fn += predicted != ClaimVerificationStatus.SUPPORTED
        else:
            support_fp += predicted == ClaimVerificationStatus.SUPPORTED
        if gold == ClaimVerificationStatus.CONTRADICTED:
            contradiction_tp += predicted == gold
            contradiction_fn += predicted != gold
        elif gold is not None:
            contradiction_fp += predicted == ClaimVerificationStatus.CONTRADICTED
        if gold == ClaimVerificationStatus.UNSUPPORTED:
            unsupported_tp += predicted == ClaimVerificationStatus.UNSUPPORTED
            unsupported_fn += predicted != ClaimVerificationStatus.UNSUPPORTED
        elif gold is not None:
            unsupported_fp += predicted == ClaimVerificationStatus.UNSUPPORTED
    abstention_tp = response.abstained and case.should_abstain
    invalid_citation_warning = any("invalid_" in warning or "fabricated_inline" in warning
                                   for warning in response.warnings)
    references_by_id = {reference.evidence_id: reference
                        for reference in response.evidence_references}
    citations_resolve = all(
        citation.evidence_id in references_by_id and
        (citation.chunk_id is None or references_by_id[citation.evidence_id].chunk_id is None or
         citation.chunk_id == references_by_id[citation.evidence_id].chunk_id) and
        (citation.document_id is None or
         references_by_id[citation.evidence_id].document_id is None or
         citation.document_id == references_by_id[citation.evidence_id].document_id)
        for citation in response.citations
    )
    citation_validity = (None if not response.citations else float(
        citations_resolve and not invalid_citation_warning and
        response.citation_validation_status not in {
            VerificationStatus.FAILED, VerificationStatus.UNSUPPORTED,
            VerificationStatus.CONTRADICTED,
        }))
    metrics: dict[str, float | int | None] = {
        "answer_correctness": float(completeness == 1.0 and completeness > 0),
        "answer_completeness": completeness,
        # These require independent semantic judgments; verifier output is not gold truth.
        "faithfulness": None,
        "answer_relevancy": None,
        "verifier_supported_claim_rate": supported / len(claims) if claims else None,
        "unsupported_claim_rate": unsupported / len(claims) if claims else None,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_validity": citation_validity,
        "claim_to_evidence_alignment": (sum(
            claim.status == ClaimVerificationStatus.SUPPORTED and
            bool(claim.supporting_evidence_ids) and
            set(claim.supporting_evidence_ids).issubset(context_ids)
            for claim in claims) / supported if supported else None),
        "supported_claim_precision": support_tp / (support_tp + support_fp) if support_tp + support_fp else 0.0,
        "supported_claim_recall": support_tp / (support_tp + support_fn) if support_tp + support_fn else 0.0,
        "unsupported_claim_detection_precision": unsupported_tp / (unsupported_tp + unsupported_fp)
        if unsupported_tp + unsupported_fp else 0.0,
        "unsupported_claim_detection_recall": unsupported_tp / (unsupported_tp + unsupported_fn)
        if unsupported_tp + unsupported_fn else 0.0,
        "contradiction_precision": contradiction_tp / (contradiction_tp + contradiction_fp)
        if contradiction_tp + contradiction_fp else 0.0,
        "contradiction_recall": contradiction_tp / (contradiction_tp + contradiction_fn)
        if contradiction_tp + contradiction_fn else 0.0,
        "abstention_precision": float(abstention_tp) / int(response.abstained) if response.abstained else 0.0,
        "abstention_recall": float(abstention_tp) / int(case.should_abstain) if case.should_abstain else 0.0,
        "input_tokens": response.token_usage.get("input_tokens", 0),
        "output_tokens": response.token_usage.get("output_tokens", 0),
        "correction_frequency": float(response.execution_metadata.get("correction_attempts", 0) > 0),
        "verification_overhead_ms": response.timings_ms.get("verification", 0.0),
        "retrieval_diversity": (len({ref.document_id for ref in response.evidence_references
                                     if ref.document_id}) / len(response.evidence_references)
                                if response.evidence_references else 0.0),
        "retrieved_evidence_count": len(response.evidence_references),
        "evidence_context_tokens": float(response.execution_metadata.get("retrieval", {}).get(
            "evidence_context_tokens", 0)) if isinstance(response.execution_metadata.get("retrieval"), dict) else 0.0,
        "retrieval_candidate_count": sum(response.execution_metadata.get("retrieval", {}).get(
            "candidate_counts", {}).values()) if isinstance(response.execution_metadata.get("retrieval"), dict) else 0,
    }
    if "generation" in response.timings_ms:
        metrics["latency_ms"] = response.timings_ms["generation"] + response.timings_ms.get("verification", 0.0)
    if response.cost_usd is not None:
        metrics["cost_usd"] = response.cost_usd
        if metrics["answer_correctness"]:
            metrics["cost_per_successful_answer_usd"] = response.cost_usd
    return metrics


def aggregate_answer_metrics(rows: Iterable[dict[str, float | int | None]]) -> dict[str, float | None]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if value is not None:
                grouped.setdefault(key, []).append(float(value))
            else:
                grouped.setdefault(key, [])
    output: dict[str, float | None] = {}
    for key, values in grouped.items():
        if key == "latency_ms":
            ordered = sorted(values)
            output["latency_p50_ms"] = statistics.median(ordered) if ordered else None
            output["latency_p95_ms"] = (ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
                                         if ordered else None)
        else:
            output[key] = statistics.mean(values) if values else None
    return output
