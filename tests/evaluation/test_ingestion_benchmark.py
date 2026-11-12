from eval.ingestion_benchmark import (
    ChunkingBenchmarkCase,
    ChunkingStrategy,
    run_chunking_benchmark,
)
from omnirag.ingestion.parsers.document_parser import DocumentParser
from omnirag.ingestion.structure.schema import LoadedSource


def test_ingestion_benchmark_compares_structure_and_simple_baseline():
    source = LoadedSource("guide.md", "guide.md", "text/markdown",
                          b"# Guide\n\n## One\n\nFirst set of useful words.\n\n## Two\n\nSecond group of useful words.",
                          "document")
    document = DocumentParser().parse(source)
    results = run_chunking_benchmark(document, [
        ChunkingBenchmarkCase("structured-small", ChunkingStrategy.STRUCTURE_AWARE, 8, 0, True),
        ChunkingBenchmarkCase("paragraph-baseline", ChunkingStrategy.PARAGRAPH_BASELINE, 8),
    ])
    assert len(results) == 2
    assert results[0].contextual_prefixes is True
    assert all(result.chunk_count > 0 for result in results)
    assert all(result.max_estimated_tokens <= 8 for result in results)
