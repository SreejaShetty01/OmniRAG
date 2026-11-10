"""Citation identity and resolution checks, independent of any LLM."""

import re

from generation.citation_builder import build_references, evidence_id_map
from models.response import (
    ClaimVerification,
    ClaimVerificationStatus,
    Citation,
    EvidenceReference,
)
from retrieval.contracts import RetrievalEvidence


def validate_citations(evidence: list[RetrievalEvidence], claims: list[ClaimVerification]
                       , answer: str = "") -> tuple[list[ClaimVerification], list[EvidenceReference], list[Citation], list[str]]:
    evidence_map = evidence_id_map(evidence)
    all_refs, all_citations = build_references(evidence)
    ref_map = {item.evidence_id: item for item in all_refs}
    citation_map = {item.evidence_id: item for item in all_citations}
    referenced: set[str] = set()
    warnings: list[str] = []
    answer_refs = set(re.findall(r"\[(E\d+)\]", answer))
    for ref in sorted(answer_refs):
        if ref not in evidence_map:
            warnings.append("invalid_inline_evidence_reference")
        else:
            referenced.add(ref)
    for claim in claims:
        invalid = [ref for ref in claim.supporting_evidence_ids if ref not in evidence_map]
        if invalid:
            warnings.append(f"invalid_evidence_reference:{claim.claim_id}")
            claim.status = ClaimVerificationStatus.UNSUPPORTED
            claim.methods = [*claim.methods, "citation_validation"]
            claim.explanation = "One or more cited evidence IDs do not exist in the retrieval result."
        else:
            referenced.update(claim.supporting_evidence_ids)
            if claim.supporting_evidence_ids:
                claim.methods = [*claim.methods, "citation_validation"]
            else:
                claim.status = ClaimVerificationStatus.UNSUPPORTED
                claim.explanation = "No supporting evidence reference was provided."
    refs = [ref_map[key] for key in sorted(referenced, key=lambda value: int(value[1:]))]
    citations = [citation_map[key] for key in sorted(referenced, key=lambda value: int(value[1:]))]
    return claims, refs, citations, warnings
