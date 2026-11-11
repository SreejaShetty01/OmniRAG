"""Conservative contradiction signals, kept separate from retrieval relevance."""

import re
from itertools import combinations

from omnirag.retrieval.contracts import RetrievalEvidence

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?%?(?![\w.])")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def detect_contradictions(evidence: list[RetrievalEvidence]) -> list[dict[str, object]]:
    conflicts = []
    for left, right in combinations(evidence, 2):
        a, b = left.chunk, right.chunk
        nums_a, nums_b = set(_NUMBER.findall(a.content)), set(_NUMBER.findall(b.content))
        words_a = {word.casefold() for word in _WORD.findall(a.content)}
        words_b = {word.casefold() for word in _WORD.findall(b.content)}
        overlap = len(words_a & words_b) / max(1, min(len(words_a), len(words_b)))
        if nums_a and nums_b and nums_a != nums_b and overlap >= 0.35:
            conflicts.append({
                "kind": "possible_numeric_conflict",
                "evidence_ids": [left.chunk.chunk_id, right.chunk.chunk_id],
                "values": [sorted(nums_a), sorted(nums_b)],
                "sources": [a.source, b.source],
                "document_ids": [a.document_id, b.document_id],
                "versions": [a.version, b.version],
                "explanation": "Related passages contain different numeric values; source semantics need review.",
            })
    return conflicts
