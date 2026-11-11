"""Repeatable chunking configuration comparisons; reports measurements, not quality claims."""

from dataclasses import dataclass
from enum import StrEnum

from omnirag.ingestion.chunking.contextualizer import DeterministicContextualizer
from omnirag.ingestion.chunking.structure_chunker import ChunkingConfig, StructureChunker, estimate_tokens
from omnirag.ingestion.structure.schema import NormalizedDocument


class ChunkingStrategy(StrEnum):
    STRUCTURE_AWARE = "structure_aware"
    PARAGRAPH_BASELINE = "paragraph_baseline"


@dataclass(frozen=True, slots=True)
class ChunkingBenchmarkCase:
    name: str
    strategy: ChunkingStrategy
    max_tokens: int
    overlap_tokens: int = 0
    contextual_prefixes: bool = False


@dataclass(frozen=True, slots=True)
class ChunkingBenchmarkResult:
    name: str
    strategy: ChunkingStrategy
    max_tokens: int
    overlap_tokens: int
    contextual_prefixes: bool
    chunk_count: int
    mean_estimated_tokens: float
    max_estimated_tokens: int


def _paragraph_baseline(document: NormalizedDocument, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for block in document.blocks:
        for paragraph in block.content.split("\n\n"):
            words = paragraph.split()
            for start in range(0, len(words), max_tokens):
                unit = " ".join(words[start:start + max_tokens])
                unit_size = estimate_tokens(unit)
                if buffer and size + unit_size > max_tokens:
                    chunks.append("\n\n".join(buffer))
                    buffer, size = [], 0
                buffer.append(unit)
                size += unit_size
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def run_chunking_benchmark(
    document: NormalizedDocument,
    cases: list[ChunkingBenchmarkCase],
) -> list[ChunkingBenchmarkResult]:
    """Run explicit chunking cases and return only size/count measurements."""
    results = []
    for case in cases:
        if case.max_tokens < 1 or case.overlap_tokens < 0 or case.overlap_tokens >= case.max_tokens:
            raise ValueError(f"Invalid chunking configuration for case {case.name!r}")
        if case.strategy is ChunkingStrategy.STRUCTURE_AWARE:
            chunks = StructureChunker(ChunkingConfig(case.max_tokens, case.overlap_tokens)).chunk(document)
            if case.contextual_prefixes:
                chunks = DeterministicContextualizer().enrich(chunks)
            contents = [chunk.content for chunk in chunks]
        else:
            contents = _paragraph_baseline(document, case.max_tokens)
        sizes = [estimate_tokens(content) for content in contents]
        results.append(ChunkingBenchmarkResult(
            name=case.name, strategy=case.strategy, max_tokens=case.max_tokens,
            overlap_tokens=case.overlap_tokens, contextual_prefixes=case.contextual_prefixes,
            chunk_count=len(sizes), mean_estimated_tokens=(sum(sizes) / len(sizes) if sizes else 0.0),
            max_estimated_tokens=max(sizes, default=0),
        ))
    return results
