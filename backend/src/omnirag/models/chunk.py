"""Validated chunk and access-control metadata contract."""

from pydantic import BaseModel, ConfigDict, Field

from omnirag.models.document import DocumentType


class AccessControlMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    allowed_principals: list[str] = Field(default_factory=list)
    denied_principals: list[str] = Field(default_factory=list)
    sensitivity: str | None = None


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    parent_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    heading: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    content_type: str = "text"
    language: str | None = None
    content: str = Field(min_length=1)
    document_type: DocumentType
    version: str | None = None
    content_hash: str = Field(min_length=1)
    source: str = Field(min_length=1)
    links: list[str] = Field(default_factory=list)
    access_control: AccessControlMetadata = Field(default_factory=AccessControlMetadata)
    created_at: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
