"""Provider-independent citation, claim, and contradiction verification."""

from verification.claim_verifier import verify_claims
from verification.citation_validator import validate_citations
from verification.contradiction_detector import detect_contradictions
from verification.nli_verifier import EntailmentLabel, EntailmentResult, EntailmentVerifier

__all__ = ["EntailmentLabel", "EntailmentResult", "EntailmentVerifier", "detect_contradictions",
           "validate_citations", "verify_claims"]
