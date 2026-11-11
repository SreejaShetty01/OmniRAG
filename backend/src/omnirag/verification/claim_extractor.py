"""Validated extraction of factual claims supplied by structured generation."""

import re

from omnirag.models.response import ClaimVerification


def extract_claims(payload: object) -> list[ClaimVerification]:
    if not isinstance(payload, dict):
        raise ValueError("Structured generation output must be a JSON object")
    raw = payload.get("claims", [])
    if not isinstance(raw, list):
        raise ValueError("claims must be an array")
    if not raw and isinstance(payload.get("answer"), str):
        return sentence_claims(payload["answer"])
    claims = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            raise ValueError("Each claim requires claim text and evidence IDs")
        text = item["claim"].strip()
        if not text:
            continue
        references = item.get("support_evidence_ids", [])
        if not isinstance(references, list) or any(not isinstance(ref, str) for ref in references):
            raise ValueError("Claim evidence references must be an array of strings")
        claims.append(ClaimVerification(
            claim_id=f"CL{len(claims) + 1}", claim_text=text,
            supporting_evidence_ids=list(dict.fromkeys(references)),
        ))
    return claims


def sentence_claims(answer: str) -> list[ClaimVerification]:
    """Conservative fallback extractor; questions, headings and short discourse are skipped."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)
    claims = []
    for sentence in sentences:
        refs = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", sentence)))
        text = re.sub(r"\[(?:E|C)\d+\]", "", sentence).strip(" -*#\t")
        if len(text.split()) < 4 or text.endswith("?"):
            continue
        if re.match(r"^(however|therefore|in summary|note that)\b", text, re.I) and len(text.split()) < 8:
            continue
        claims.append(ClaimVerification(claim_id=f"CL{len(claims) + 1}", claim_text=text,
                                       supporting_evidence_ids=refs))
    return claims
