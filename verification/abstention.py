"""Evidence-sufficiency-driven abstention policy, separate from answer correctness."""

import re

from retrieval.contracts import EvidenceAssessmentState, RetrievalResult


ABSTENTION_TEXT = "I don’t have enough retrieved evidence to answer this reliably."
_QUERY_WORD = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")
_ENTITY_CUES = {"of", "for", "from", "at", "in", "into", "about", "by", "to", "called", "named"}
_QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom",
                   "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
                   "will", "should", "please", "tell", "show", "explain", "give", "the", "a", "an",
                   "of", "in", "on", "at", "for", "from", "to", "by", "about", "and", "or"}


def unmatched_query_entities(query: str, evidence_texts: list[str]) -> list[str]:
    """Return explicit proper-name tokens absent from all selected evidence.

    This intentionally narrow lexical guard only recognizes internally mixed-case
    names (for example, ``JPMorgan``) and title-case tokens after the first word.
    It does not claim semantic entailment or replace claim verification.
    """
    matches = list(_QUERY_WORD.finditer(query))
    terms = []
    previous_entity = False
    previous_match = None
    for index, match in enumerate(matches):
        token = match.group(0)
        folded = token.casefold()
        mixed_case = any(char.isupper() for char in token[1:]) and any(char.islower() for char in token)
        previous_token = matches[index - 1].group(0).casefold() if index else ""
        contiguous_name = bool(
            previous_entity and previous_match is not None
            and query[previous_match.end():match.start()].isspace()
        )
        title_case = (token[0].isupper() and token[1:].islower()
                      and (previous_token in _ENTITY_CUES or contiguous_name))
        is_entity = folded not in _QUESTION_WORDS and (mixed_case or title_case)
        if is_entity:
            terms.append(token)
        previous_entity = is_entity
        previous_match = match
    corpus = "\n".join(evidence_texts)
    explicitly_aliased_terms: set[str] = set()
    alias_pattern = re.compile(
        r"(?P<phrase>\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+)\s*"
        r"\(\s*(?P<alias>[A-Z][A-Z0-9]{1,})\s*\)"
    )
    for match in alias_pattern.finditer(query):
        alias = match.group("alias")
        if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", corpus, re.IGNORECASE):
            explicitly_aliased_terms.update(
                token.casefold() for token in _QUERY_WORD.findall(match.group("phrase")))
    return [term for term in dict.fromkeys(terms)
            if term.casefold() not in explicitly_aliased_terms and not re.search(
                rf"(?<!\w){re.escape(term)}(?!\w)", corpus, re.IGNORECASE)]


def abstention_reason(result: RetrievalResult, *, abstain_on_weak: bool = True) -> str | None:
    state = result.assessment.state
    if state == EvidenceAssessmentState.INSUFFICIENT or not result.evidence:
        return "insufficient_retrieved_evidence"
    if state == EvidenceAssessmentState.WEAK and abstain_on_weak:
        return "retrieved_evidence_has_no_cross_channel_support_signal"
    return None
