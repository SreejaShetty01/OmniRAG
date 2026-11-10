"""Deterministic document and section reference extraction."""

import re
from urllib.parse import unquote

from ingestion.structure.schema import ContentBlock, ContentKind


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_DOC_REF = re.compile(r"\b([\w./-]+\.(?:pdf|docx|md|markdown|txt|html?))\b", re.IGNORECASE)
_SECTION_REF = re.compile(r"(?:#|section\s+)([\w.-]+)", re.IGNORECASE)


def extract_cross_references(blocks: list[ContentBlock]) -> None:
    """Add normalized link targets to each block without fetching or resolving them."""
    known_headings = {
        heading.casefold(): f"#{heading.casefold().replace(' ', '-')}"
        for block in blocks for heading in block.section_path
    }
    for block in blocks:
        candidates = list(block.links)
        candidates.extend(_MARKDOWN_LINK.findall(block.content))
        candidates.extend(_DOC_REF.findall(block.content))
        for ref in _SECTION_REF.findall(block.content):
            target = known_headings.get(unquote(ref).replace("-", " ").casefold())
            if target:
                candidates.append(target)
        block.links = list(dict.fromkeys(x.strip() for x in candidates if x and x.strip()))
