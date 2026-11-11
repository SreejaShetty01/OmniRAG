"""Shared generation, citation validation, claim verification, and bounded correction."""

import re
import time
import logging

from omnirag.config.profiles import ProfileName
from omnirag.config.settings import get_settings
from omnirag.generation.answer_generator import AnswerGenerator
from omnirag.models.query import Query
from omnirag.models.response import (
    ClaimVerificationStatus, OmniRAGResponse, VerificationStatus,
)
from omnirag.retrieval.contracts import EvidenceAssessmentState, RetrievalResult
from omnirag.verification.abstention import ABSTENTION_TEXT, abstention_reason, unmatched_query_entities
from omnirag.verification.citation_validator import validate_citations
from omnirag.verification.claim_verifier import verify_claims
from omnirag.verification.contradiction_detector import detect_contradictions
from omnirag.infrastructure.provider_budget import BudgetExceeded
from omnirag.verification.nli_verifier import UnconfiguredEntailmentVerifier
from omnirag.experimentation.runtime import record_stage

logger = logging.getLogger(__name__)


class AnswerPipeline:
    def __init__(self, generator: AnswerGenerator | None = None, nli=None):
        self.generator = generator or AnswerGenerator()
        self.nli = nli if nli is not None else self._nli_from_settings()
        self.nli_warning = ("semantic_verification_unconfigured" if isinstance(
            self.nli, UnconfiguredEntailmentVerifier) else None)

    def answer(self, query: Query | str, retrieval: RetrievalResult, *,
               verification_depth: str = "full_claim_checks", generation_style: str = "balanced",
               generation_max_output_tokens: int | None = None,
               allow_correction: bool = True, abstain_on_weak: bool = True,
               semantic_verification: bool = True) -> OmniRAGResponse:
        query_text = query.query_text if isinstance(query, Query) else query
        conversation_context = query.conversation_context if isinstance(query, Query) else []
        profile = query.selected_profile if isinstance(query, Query) else ProfileName.CORE
        response = OmniRAGResponse(answer="", profile_used=profile, original_query=query_text)
        response.evidence_sufficiency = retrieval.assessment.model_dump(mode="json")
        response.retrieval_inspection = retrieval.retrieval_inspection
        response.execution_metadata["evidence_sufficiency"] = response.evidence_sufficiency
        response.execution_metadata["retrieval_scope_hash"] = retrieval.authorization_scope_hash
        response.execution_metadata["retrieval"] = {
            "timings_ms": retrieval.timings_ms,
            "candidate_counts": retrieval.candidate_counts,
            "evidence_context_tokens": retrieval.selected_token_estimate,
            "cache_status": retrieval.cache_status,
            "reranking_used": any(item.reranker_rank is not None for item in retrieval.evidence),
            "reranking_ms": retrieval.timings_ms.get("reranking", 0.0),
        }
        response.contradictions = detect_contradictions(retrieval.evidence)
        response.warnings.extend(retrieval.warnings)
        if self.nli_warning:
            response.warnings.append(self.nli_warning)
        if response.contradictions or retrieval.assessment.state == EvidenceAssessmentState.CONFLICTING:
            response.warnings.append("conflicting_retrieved_evidence")
        reason = abstention_reason(retrieval, abstain_on_weak=abstain_on_weak)
        if reason:
            response.answer = ABSTENTION_TEXT
            response.abstained = True
            response.abstention_reason = reason
            response.verification_status = VerificationStatus.NOT_RUN
            response.overall_reliability = VerificationStatus.NOT_RUN
            response.execution_metadata["abstention_decision"] = reason
            record_stage("context_construction", 0.0, status="not_applicable")
            record_stage("generation", 0.0, status="not_applicable")
            record_stage("citation_validation", 0.0, status="not_applicable")
            record_stage("claim_verification", 0.0, status="not_applicable")
            self._observe(response)
            return response

        missing_query_entities = unmatched_query_entities(
            query_text, [item.chunk.content for item in retrieval.evidence])
        if missing_query_entities:
            reason = "query_entity_not_grounded_in_selected_evidence"
            response.answer = ABSTENTION_TEXT
            response.abstained = True
            response.abstention_reason = reason
            response.verification_status = VerificationStatus.UNCERTAIN
            response.overall_reliability = VerificationStatus.UNCERTAIN
            response.execution_metadata["abstention_decision"] = reason
            response.warnings.append("query_entity_missing_from_selected_evidence")
            record_stage("context_construction", 0.0, status="not_applicable")
            record_stage("generation", 0.0, status="not_applicable")
            record_stage("citation_validation", 0.0, status="not_applicable")
            record_stage("claim_verification", 0.0, status="not_applicable")
            self._observe(response)
            return response

        generation_started = time.perf_counter()
        try:
            draft = self.generator.generate(query_text, retrieval, response_style=generation_style,
                                            max_output_tokens=generation_max_output_tokens,
                                            conversation_context=conversation_context)
        except BudgetExceeded as exc:
            record_stage("generation", self._elapsed(generation_started),
                         status="failure", error_type=type(exc).__name__)
            raise
        except Exception as exc:
            response.answer = ABSTENTION_TEXT
            response.abstained = True
            response.abstention_reason = "generation_provider_or_output_failure"
            response.warnings.append(f"generation_failed:{type(exc).__name__}")
            response.verification_status = VerificationStatus.FAILED
            response.overall_reliability = VerificationStatus.FAILED
            response.timings_ms["generation"] = self._elapsed(generation_started)
            record_stage("generation", response.timings_ms["generation"],
                         status="failure", error_type=type(exc).__name__)
            self._observe(response)
            return response

        generation_latency = draft.latency_ms
        record_stage("generation", generation_latency)
        total_input_tokens = draft.input_tokens or 0
        total_output_tokens = draft.output_tokens or 0
        total_cost = draft.cost_usd or 0.0
        has_cost = draft.cost_usd is not None
        response.warnings.extend(draft.warnings)
        citation_started = time.perf_counter()
        claims, refs, citations, citation_warnings = validate_citations(
            retrieval.evidence, draft.claims, draft.answer)
        record_stage("citation_validation", self._elapsed(citation_started),
                     status="failure" if citation_warnings else "success",
                     metadata={"citation_count": len(citations)})
        final_citation_warnings = citation_warnings
        response.warnings.extend(citation_warnings)
        if verification_depth == "citations_only":
            response.warnings.append("deep_claim_verification_not_performed_by_profile")
            record_stage("claim_verification", 0.0, status="not_applicable")
        else:
            verifier = self.nli
            if not semantic_verification:
                verifier = UnconfiguredEntailmentVerifier()
                if "semantic_verification_unconfigured" not in response.warnings:
                    response.warnings.append("semantic_verification_unconfigured")
            verification_started = time.perf_counter()
            claims, verifier_warnings = verify_claims(claims, retrieval.evidence, verifier)
            record_stage("claim_verification", self._elapsed(verification_started),
                         status="failure" if any("verification_failed" in item for item in verifier_warnings)
                         else "success", metadata={"claim_count": len(claims)})
            response.warnings.extend(verifier_warnings)

        correction_attempts = 0
        invalid_claims = [claim for claim in claims if claim.status in {
            ClaimVerificationStatus.UNSUPPORTED, ClaimVerificationStatus.CONTRADICTED,
        }]
        if invalid_claims and allow_correction and get_settings().generation_max_corrections:
            correction_attempts = 1
            correction_note = (
                "The previous draft contained claims that verification could not support. "
                "Remove those claims or state the evidence limitation; retain only supported facts "
                "and cite supplied evidence IDs. Problematic claims: " +
                " | ".join(claim.claim_text for claim in invalid_claims)
            )
            try:
                correction_started = time.perf_counter()
                corrected = self.generator.generate(
                    query_text, retrieval, correction_note=correction_note, response_style=generation_style,
                    max_output_tokens=generation_max_output_tokens,
                    conversation_context=conversation_context)
                claims, refs, citations, extra_warnings = validate_citations(
                    retrieval.evidence, corrected.claims, corrected.answer)
                final_citation_warnings = extra_warnings
                if verification_depth != "citations_only":
                    verifier = self.nli if semantic_verification else UnconfiguredEntailmentVerifier()
                    claims, verify_warnings = verify_claims(claims, retrieval.evidence, verifier)
                else:
                    verify_warnings = []
                response.warnings.extend(extra_warnings + verify_warnings)
                draft = corrected
                record_stage("generation", corrected.latency_ms)
                generation_latency += corrected.latency_ms
                total_input_tokens += corrected.input_tokens or 0
                total_output_tokens += corrected.output_tokens or 0
                if corrected.cost_usd is not None:
                    total_cost += corrected.cost_usd
                    has_cost = True
            except BudgetExceeded as exc:
                record_stage("generation", self._elapsed(correction_started),
                             status="failure", error_type=type(exc).__name__)
                raise
            except Exception as exc:
                response.warnings.append(f"correction_failed:{type(exc).__name__}")
                record_stage("generation", self._elapsed(correction_started),
                             status="failure", error_type=type(exc).__name__)

        # Strip only fabricated inline markers; actual evidence references are preserved.
        valid_ids = {f"E{index}" for index in range(1, len(retrieval.evidence) + 1)}
        invalid_markers = [ref for ref in re.findall(r"\[(E\d+)\]", draft.answer) if ref not in valid_ids]
        answer_text = re.sub(r"\[(E\d+)\]", lambda match: match.group(0)
                             if match.group(1) in valid_ids else "", draft.answer)
        if invalid_markers:
            response.warnings.append("fabricated_inline_citation_removed")

        if any(claim.status in {ClaimVerificationStatus.UNSUPPORTED, ClaimVerificationStatus.CONTRADICTED}
               for claim in claims):
            response.answer = ABSTENTION_TEXT
            response.abstained = True
            response.abstention_reason = "unsupported_claims_remain_after_bounded_correction"
        else:
            response.answer = answer_text.strip()
        response.claims = claims
        response.evidence_references = refs
        response.citations = citations
        response.generation_metadata = {"provider": draft.provider, "model": draft.model}
        response.execution_metadata["verification_depth"] = verification_depth
        response.execution_metadata["generation_style"] = generation_style
        response.token_usage = {key: value for key, value in {
            "input_tokens": total_input_tokens if total_input_tokens else None,
            "output_tokens": total_output_tokens if total_output_tokens else None,
        }.items() if value is not None}
        response.cost_usd = total_cost if has_cost else None
        response.timings_ms["generation"] = round(generation_latency, 3)
        response.timings_ms["verification"] = max(
            0.0, self._elapsed(generation_started) - round(generation_latency, 3))
        response.execution_metadata.update({
            "correction_attempts": correction_attempts,
            "citation_validation_failures": len([w for w in response.warnings if w.startswith("invalid_")]),
            "unsupported_claim_count": sum(c.status == ClaimVerificationStatus.UNSUPPORTED for c in claims),
            "contradiction_count": len(response.contradictions),
            "verification_methods": sorted({method for claim in claims for method in claim.methods}),
        })
        response.verification_status = self._overall(claims)
        response.claim_verification_status = response.verification_status
        citation_failures = bool(final_citation_warnings or invalid_markers)
        response.citation_validation_status = (
            VerificationStatus.FAILED if citation_failures else
            VerificationStatus.PASSED if citations else
            VerificationStatus.NOT_RUN if not claims else VerificationStatus.UNCERTAIN
        )
        if (response.verification_status == VerificationStatus.PASSED and
                (response.contradictions or response.citation_validation_status != VerificationStatus.PASSED)):
            response.verification_status = VerificationStatus.PARTIAL
        response.overall_reliability = response.verification_status
        if response.abstained:
            response.execution_metadata["abstention_decision"] = response.abstention_reason
        self._observe(response)
        return response

    @staticmethod
    def _observe(response: OmniRAGResponse) -> None:
        logger.info("Answer pipeline completed", extra={
            "response_id": response.response_id,
            "verification_status": response.verification_status.value,
            "claim_count": len(response.claims),
            "unsupported_claim_count": response.execution_metadata.get("unsupported_claim_count", 0),
            "contradiction_count": len(response.contradictions),
            "correction_attempts": response.execution_metadata.get("correction_attempts", 0),
            "abstained": response.abstained,
            "citation_count": len(response.citations),
            "citation_validation_status": response.citation_validation_status.value,
            "provider": response.generation_metadata.get("provider"),
            "model": response.generation_metadata.get("model"),
            "input_tokens": response.token_usage.get("input_tokens", 0),
            "output_tokens": response.token_usage.get("output_tokens", 0),
            "cost_usd": response.cost_usd,
            "abstention_reason": response.abstention_reason,
            "generation_ms": response.timings_ms.get("generation", 0),
            "verification_ms": response.timings_ms.get("verification", 0),
        })

    @staticmethod
    def _overall(claims) -> VerificationStatus:
        statuses = {claim.status for claim in claims}
        if ClaimVerificationStatus.CONTRADICTED in statuses:
            return VerificationStatus.CONTRADICTED
        if ClaimVerificationStatus.UNSUPPORTED in statuses:
            return VerificationStatus.UNSUPPORTED
        if not claims or ClaimVerificationStatus.UNCERTAIN in statuses or ClaimVerificationStatus.NOT_CHECKED in statuses:
            return VerificationStatus.UNCERTAIN
        if ClaimVerificationStatus.PARTIALLY_SUPPORTED in statuses:
            return VerificationStatus.PARTIAL
        return VerificationStatus.PASSED

    @staticmethod
    def _nli_from_settings():
        from omnirag.config.settings import get_settings
        from omnirag.llm.llm_client import provider_from_settings
        from omnirag.verification.nli_verifier import ChatEntailmentVerifier, HttpMiniCheckVerifier
        settings = get_settings()
        if not settings.verification_endpoint or not settings.verification_model_id:
            return UnconfiguredEntailmentVerifier()
        try:
            if settings.verification_provider == "minicheck_http":
                key = (settings.verification_api_key.get_secret_value()
                       if settings.verification_api_key else None)
                return HttpMiniCheckVerifier(
                    settings.verification_endpoint, settings.verification_model_id,
                    settings.verification_model_digest or "", settings.verification_threshold,
                    settings.verification_timeout_seconds, key)
            provider, model = provider_from_settings(verification=True)
            return ChatEntailmentVerifier(provider, model)
        except Exception:
            return UnconfiguredEntailmentVerifier()

    @staticmethod
    def _elapsed(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)
