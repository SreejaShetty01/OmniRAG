"""Modular deterministic and semantic checks for generated factual claims."""

import re

from models.response import ClaimVerification, ClaimVerificationStatus
from retrieval.contracts import RetrievalEvidence
from verification.nli_verifier import EntailmentLabel, EntailmentVerifier
from infrastructure.provider_budget import BudgetExceeded

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])")
_IDENTIFIER = re.compile(
    r"(?:[A-Za-z_$][\w$]*(?:(?:::|[./#])|[_-])[\w$./#:-]*|\b[A-Z][A-Z0-9_-]{2,}\b|\bv\d+(?:\.\d+)*\b)"
)
_IDENTIFIER_TRAILING_PUNCTUATION = ".,:;!?"


def _identifier_tokens(text: str) -> set[str]:
    """Extract identifiers without sentence punctuation captured by the regex.

    Internal punctuation remains significant (for example ``v2.1`` and
    ``ticket_id``); only punctuation at the end of a token is removed.
    """
    return {token.rstrip(_IDENTIFIER_TRAILING_PUNCTUATION)
            for token in _IDENTIFIER.findall(text)
            if token.rstrip(_IDENTIFIER_TRAILING_PUNCTUATION)}


def verify_claims(claims: list[ClaimVerification], evidence: list[RetrievalEvidence],
                  nli: EntailmentVerifier) -> tuple[list[ClaimVerification], list[str]]:
    mapping = {f"E{index}": item for index, item in enumerate(evidence, start=1)}
    warnings: list[str] = []
    for claim in claims:
        if claim.status == ClaimVerificationStatus.UNSUPPORTED:
            continue
        selected = [mapping[item] for item in claim.supporting_evidence_ids if item in mapping]
        if not selected:
            claim.status = ClaimVerificationStatus.UNSUPPORTED
            claim.explanation = "No retrieved evidence was linked to this claim."
            claim.methods.append("citation_validation")
            continue
        source_text = "\n".join(item.chunk.content for item in selected)
        claim_numbers = set(_NUMBER.findall(claim.claim_text))
        evidence_numbers = set(_NUMBER.findall(source_text))
        if claim_numbers - evidence_numbers:
            claim.status = ClaimVerificationStatus.UNSUPPORTED
            claim.methods.append("numerical_check")
            claim.explanation = "Claim contains numeric values absent from its cited evidence."
            continue
        claim.methods.append("numerical_check")
        claim_ids = _identifier_tokens(claim.claim_text)
        evidence_ids = _identifier_tokens(source_text)
        if claim_ids - evidence_ids:
            claim.status = ClaimVerificationStatus.UNSUPPORTED
            claim.methods.append("identifier_check")
            claim.explanation = "Claim contains identifiers absent from its cited evidence."
            continue
        claim.methods.append("identifier_check")
        try:
            verdict = nli.verify(claim.claim_text, [item.chunk.content for item in selected])
        except BudgetExceeded:
            raise
        except Exception as exc:
            claim.status = ClaimVerificationStatus.UNCERTAIN
            claim.methods.append("semantic_verification_failed")
            claim.explanation = "Semantic verification failed; the claim is not marked supported."
            warnings.append(f"semantic_verification_failed:{claim.claim_id}:{type(exc).__name__}")
            continue
        claim.methods.append(verdict.method)
        claim.explanation = verdict.explanation
        claim.verification_score = verdict.raw_score
        claim.verifier_model_id = verdict.model_id
        claim.verifier_model_digest = verdict.model_digest
        claim.status = {
            EntailmentLabel.SUPPORTED: ClaimVerificationStatus.SUPPORTED,
            EntailmentLabel.CONTRADICTED: ClaimVerificationStatus.CONTRADICTED,
            EntailmentLabel.NOT_SUPPORTED: ClaimVerificationStatus.UNSUPPORTED,
            EntailmentLabel.UNCERTAIN: ClaimVerificationStatus.UNCERTAIN,
        }[verdict.label]
    return claims, warnings
