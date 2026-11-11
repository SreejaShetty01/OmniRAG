"""Deterministic, metadata-grounded context prefixes."""

from omnirag.models.chunk import Chunk


class DeterministicContextualizer:
    def enrich(self, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            parts = [f"Document: {chunk.metadata.get('title')}" ] if chunk.metadata.get("title") else []
            if chunk.section_path:
                parts.append("Section: " + " > ".join(chunk.section_path))
            chunk.metadata["context_prefix"] = " | ".join(parts)
        return chunks
