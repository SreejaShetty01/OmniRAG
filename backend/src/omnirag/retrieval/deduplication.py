"""Traceable exact and near-duplicate candidate consolidation."""

import hashlib
import re

from omnirag.retrieval.contracts import RetrievalCandidate


def _normalized_content(content: str) -> str:
    return re.sub(r"\s+", " ", content.casefold()).strip()


def _similarity(left: str, right: str) -> float:
    left_terms, right_terms = set(left.split()), set(right.split())
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def deduplicate(candidates: list[RetrievalCandidate], similarity_threshold: float = 0.9
                ) -> tuple[list[RetrievalCandidate], int]:
    ordered = sorted(candidates, key=lambda candidate: (
        -candidate.ranking_score, candidate.chunk.document_id, candidate.chunk.chunk_id))
    selected: list[RetrievalCandidate] = []
    by_id: dict[str, RetrievalCandidate] = {}
    exact_content: dict[str, RetrievalCandidate] = {}
    removed = 0
    for candidate in ordered:
        existing = by_id.get(candidate.identity)
        if existing is not None:
            _merge_provenance(existing, candidate)
            removed += 1
            continue
        normalized = _normalized_content(candidate.chunk.content)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        duplicate = exact_content.get(digest)
        if duplicate is not None:
            _merge_provenance(duplicate, candidate)
            duplicate.provenance.append(f"duplicate_source:{candidate.chunk.document_id}:{candidate.identity}")
            removed += 1
            continue
        near = next((item for item in selected
                     if (item.chunk.document_id == candidate.chunk.document_id
                         or item.chunk.parent_id == candidate.chunk.chunk_id
                         or candidate.chunk.parent_id == item.chunk.chunk_id)
                     and _similarity(_normalized_content(item.chunk.content), normalized) >= similarity_threshold), None)
        if near is not None:
            _merge_provenance(near, candidate)
            near.provenance.append(f"overlap:{candidate.identity}")
            removed += 1
            continue
        by_id[candidate.identity] = candidate
        exact_content[digest] = candidate
        selected.append(candidate)
    return selected, removed


def _merge_provenance(primary: RetrievalCandidate, secondary: RetrievalCandidate) -> None:
    for item in secondary.provenance:
        if item not in primary.provenance:
            primary.provenance.append(item)
