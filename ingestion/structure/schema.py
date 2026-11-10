"""Parser-independent structural representation shared by ingestion stages."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.document import Document


class ContentKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    COMMENT = "comment"
    IMPORT = "import"
    OTHER = "other"


@dataclass(slots=True)
class ContentBlock:
    block_id: str
    content: str
    content_type: ContentKind = ContentKind.PARAGRAPH
    parent_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    heading: str | None = None
    page_number: int | None = None
    language: str | None = None
    symbol_name: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    links: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NormalizedDocument:
    document: Document
    blocks: list[ContentBlock]
    source_type: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoadedSource:
    source: str
    filename: str
    media_type: str
    data: bytes
    source_type: str
