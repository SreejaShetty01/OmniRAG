"""Structure-aware chunk packing with word-count size estimation."""

import hashlib
import re
from dataclasses import dataclass

from omnirag.config.settings import get_settings
from omnirag.ingestion.structure.schema import ContentBlock, ContentKind, NormalizedDocument
from omnirag.models.chunk import AccessControlMetadata, Chunk


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_tokens: int = 500
    overlap_tokens: int = 50

    def __post_init__(self) -> None:
        if self.max_tokens < 1 or self.overlap_tokens < 0:
            raise ValueError("Chunk size must be positive and overlap cannot be negative")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("Chunk overlap must be smaller than chunk size")


def estimate_tokens(text: str) -> int:
    """Conservative, deterministic word/ punctuation estimate when no tokenizer is selected."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


class StructureChunker:
    def __init__(self, config: ChunkingConfig | None = None):
        settings = get_settings()
        self.config = config or ChunkingConfig(settings.chunk_max_tokens, settings.chunk_overlap_tokens)

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        groups: list[list[ContentBlock]] = []
        current: list[ContentBlock] = []
        count = 0
        section: tuple[str, ...] = ()
        for block in document.blocks:
            units = self._fit_block(block)
            for unit in units:
                unit_section = tuple(unit.section_path)
                size = estimate_tokens(unit.content)
                section_changed = bool(current) and unit_section != section
                if current and (section_changed or count + size > self.config.max_tokens):
                    groups.append(current)
                    current = [] if section_changed else self._overlap(current, size)
                    count = 0
                    count = sum(estimate_tokens(b.content) for b in current)
                if not current:
                    section = unit_section
                current.append(unit)
                count += size
        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        seen_ids: set[str] = set()
        for group_index, group in enumerate(groups):
            content = "\n\n".join(block.content for block in group).strip()
            if not content:
                continue
            section_path = list(group[-1].section_path)
            structure_key = ":".join(b.block_id for b in group)
            digest = hashlib.sha256(f"{document.document.document_id}:{structure_key}:{content}".encode()).hexdigest()[:24]
            chunk_id = f"chk_{digest}"
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            block = group[0]
            page_numbers = [b.page_number for b in group if b.page_number is not None]
            links = list(dict.fromkeys(link for b in group for link in b.links))
            chunks.append(Chunk(
                chunk_id=chunk_id, document_id=document.document.document_id,
                parent_id=block.parent_id or document.document.document_id,
                section_path=section_path, heading=group[-1].heading,
                page_number=page_numbers[0] if page_numbers else None,
                content_type=block.content_type.value, language=block.language or document.document.language,
                content=content, document_type=document.document.document_type,
                version=document.document.version, content_hash=hashlib.sha256(content.encode()).hexdigest(),
                source=document.document.source, links=links,
                access_control=AccessControlMetadata(),
                metadata={"source_type": document.source_type, "title": document.document.title,
                          "created_at": document.document.created_at.isoformat(),
                          "block_ids": [b.block_id for b in group],
                          "symbols": [b.symbol_name for b in group if b.symbol_name],
                          "start_line": min((b.start_line for b in group if b.start_line), default=None),
                          "end_line": max((b.end_line for b in group if b.end_line), default=None),
                          "chunk_index": group_index},
            ))
        return chunks

    def _fit_block(self, block: ContentBlock) -> list[ContentBlock]:
        if estimate_tokens(block.content) <= self.config.max_tokens:
            return [block]
        # Keep code on line boundaries and prose on sentence/word boundaries.
        pieces = block.content.splitlines() if block.content_type in {ContentKind.CODE, ContentKind.TABLE, ContentKind.LIST} else re.split(r"(?<=[.!?])\s+", block.content)
        result: list[ContentBlock] = []
        buffer: list[str] = []
        size = 0
        for piece in pieces:
            words = piece.split()
            if estimate_tokens(piece) > self.config.max_tokens:
                if buffer:
                    result.append(self._clone(block, "\n".join(buffer))); buffer, size = [], 0
                # Last-resort token-boundary splitting, never character slicing.
                for start in range(0, len(words), self.config.max_tokens):
                    result.append(self._clone(block, " ".join(words[start:start + self.config.max_tokens])))
                continue
            piece_size = estimate_tokens(piece)
            if buffer and size + piece_size > self.config.max_tokens:
                result.append(self._clone(block, "\n".join(buffer)))
                buffer, size = [], 0
            buffer.append(piece)
            size += piece_size
        if buffer:
            result.append(self._clone(block, "\n".join(buffer)))
        return result

    @staticmethod
    def _clone(source: ContentBlock, content: str) -> ContentBlock:
        digest = hashlib.sha256(f"{source.block_id}:{content}".encode()).hexdigest()[:24]
        return ContentBlock(f"{source.block_id}_{digest}", content, source.content_type,
                            parent_id=source.block_id, section_path=source.section_path.copy(),
                            heading=source.heading, page_number=source.page_number,
                            language=source.language, symbol_name=source.symbol_name,
                            start_line=source.start_line, end_line=source.end_line,
                            links=source.links.copy(), metadata=source.metadata.copy())

    def _overlap(self, group: list[ContentBlock], next_size: int) -> list[ContentBlock]:
        if not group or not self.config.overlap_tokens:
            return []
        selected: list[ContentBlock] = []
        size = 0
        limit = min(self.config.overlap_tokens, self.config.max_tokens - next_size)
        for block in reversed(group):
            block_size = estimate_tokens(block.content)
            if size + block_size > limit:
                break
            selected.insert(0, block)
            size += block_size
        return selected
