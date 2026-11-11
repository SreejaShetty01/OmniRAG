"""Bounded deterministic research decomposition and one-hop follow-up planning."""

import re
from typing import Protocol

from omnirag.retrieval.contracts import RetrievalResult


class ResearchPlanner(Protocol):
    def decompose(self, query: str, limit: int) -> list[str]: ...

    def followups(self, unresolved: list[str], results: list[RetrievalResult],
                  already_asked: set[str], limit: int) -> list[str]: ...


class DeterministicResearchPlanner:
    """No LLM dependency; decomposes explicit lists and compounds only."""

    def decompose(self, query: str, limit: int) -> list[str]:
        text = query.strip()
        parts = [part.strip(" \t-•0123456789.)") for part in re.split(r"\n+|;", text) if part.strip()]
        if len(parts) == 1:
            match = re.match(r"(?i)^(what|describe|explain|compare)\s+(.+?)\s+and\s+(.+?)[?.]?$", text)
            if match and match.group(1).casefold() != "compare":
                leading = match.group(1)
                auxiliary = ""
                first_topic = match.group(2)
                if leading.casefold() == "what":
                    aux_match = re.match(r"(?i)^(is|are)\s+(.+)$", first_topic)
                    if aux_match:
                        auxiliary, first_topic = aux_match.group(1), aux_match.group(2)
                prefix = f"{leading} {auxiliary}".strip()
                parts = [f"{prefix} {first_topic}?", f"{prefix} {match.group(3)}?"]
        normalized = []
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip()
            if part and part.casefold() not in {x.casefold() for x in normalized}:
                normalized.append(part)
        return (normalized or [text])[:max(1, limit)]

    def followups(self, unresolved: list[str], results: list[RetrievalResult],
                  already_asked: set[str], limit: int) -> list[str]:
        # Use explicit Phase 2 links as leads for the next retrieval hop.
        links = []
        for result in results:
            if result.original_query not in unresolved:
                continue
            for item in result.evidence:
                links.extend(item.chunk.links)
        followups = []
        for link in links:
            question = f"Find the referenced material for {link}"
            if question.casefold() not in already_asked and question not in followups:
                followups.append(question)
            if len(followups) >= limit:
                break
        return followups
