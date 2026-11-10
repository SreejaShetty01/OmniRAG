"""Shared normalization helpers for every parser implementation."""

import hashlib

from ingestion.structure.schema import ContentBlock, LoadedSource, NormalizedDocument
from models.document import Document, DocumentType


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def make_normalized_document(
    source: LoadedSource,
    title: str,
    document_type: DocumentType,
    blocks: list[ContentBlock],
    language: str | None = None,
) -> NormalizedDocument:
    occurrences: dict[str, int] = {}
    for block in blocks:
        base_id = block.block_id
        occurrence = occurrences.get(base_id, 0)
        occurrences[base_id] = occurrence + 1
        if occurrence:
            block.block_id = f"{base_id}_{occurrence + 1}"
    content_hash = hashlib.sha256(source.data).hexdigest()
    document_id = stable_id("doc", source.source)
    document = Document(
        document_id=document_id,
        title=title.strip() or source.filename,
        source=source.source,
        source_type=source.source_type,
        document_type=document_type,
        language=language,
        content_hash=content_hash,
        metadata={"media_type": source.media_type},
    )
    return NormalizedDocument(document, blocks, source.source_type, source.source)
