"""Validated document metadata contract."""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    CODE = "code"
    OTHER = "other"


class Document(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: str | None = None
    document_type: DocumentType
    language: str | None = None
    version: str | None = None
    content_hash: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
