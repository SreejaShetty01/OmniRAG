"""Provider-independent citation, claim, and contradiction verification."""

from omnirag.verification.claim_verifier import verify_claims
from omnirag.verification.citation_validator import validate_citations
from omnirag.verification.contradiction_detector import detect_contradictions
from omnirag.verification.nli_verifier import EntailmentLabel, EntailmentResult, EntailmentVerifier

__all__ = ["EntailmentLabel", "EntailmentResult", "EntailmentVerifier", "detect_contradictions",
           "validate_citations", "verify_claims"]
