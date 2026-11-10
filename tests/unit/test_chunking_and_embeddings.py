from ingestion.chunking.contextualizer import DeterministicContextualizer
from ingestion.chunking.structure_chunker import ChunkingConfig, StructureChunker
from ingestion.parsers.document_parser import DocumentParser
from ingestion.structure.schema import ContentBlock, ContentKind, LoadedSource, NormalizedDocument
from indexing.embeddings import EmbeddingError, EmbeddingService
from indexing.qdrant_indexer import prepare_point
from indexing.sparse_encoder import SparseEncoder
from models.document import Document, DocumentType


class FakeProvider:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


def test_chunking_respects_sections_size_and_parent_relationships():
    source = LoadedSource("x.md", "x.md", "text/markdown", b"", "document")
    doc = Document(document_id="d1", title="Manual", source="x.md", document_type=DocumentType.MARKDOWN, content_hash="h")
    blocks = [ContentBlock(f"b{i}", f"paragraph {i} has words", section_path=["A"], heading="A") for i in range(5)]
    chunks = StructureChunker(ChunkingConfig(max_tokens=7, overlap_tokens=1)).chunk(NormalizedDocument(doc, blocks, "document", "x.md"))
    assert len(chunks) > 1
    assert all(chunk.parent_id == "d1" for chunk in chunks)
    assert all(chunk.section_path == ["A"] for chunk in chunks)
    assert all(len(chunk.content.split()) <= 10 for chunk in chunks)


def test_chunk_ids_remain_stable_when_unrelated_text_is_inserted():
    parser = DocumentParser()
    before = parser.parse(LoadedSource("x.md", "x.md", "text/markdown",
                                       b"# Manual\n\n## Setup\n\nStable instructions.", "document"))
    after = parser.parse(LoadedSource("x.md", "x.md", "text/markdown",
                                      b"# Manual\n\nNew introduction.\n\n## Setup\n\nStable instructions.", "document"))
    config = ChunkingConfig(max_tokens=100, overlap_tokens=0)
    before_chunks = StructureChunker(config).chunk(before)
    after_chunks = StructureChunker(config).chunk(after)
    stable_before = next(chunk for chunk in before_chunks if "Stable instructions" in chunk.content)
    stable_after = next(chunk for chunk in after_chunks if "Stable instructions" in chunk.content)
    assert stable_before.chunk_id == stable_after.chunk_id


def test_oversized_structural_block_creates_child_chunks():
    doc = Document(document_id="d1", title="Code", source="sample.py", document_type=DocumentType.CODE, content_hash="h")
    block = ContentBlock("symbol-1", "alpha beta gamma delta epsilon", ContentKind.CODE,
                         parent_id="class-1", language="python", symbol_name="run")
    chunks = StructureChunker(ChunkingConfig(max_tokens=2, overlap_tokens=0)).chunk(
        NormalizedDocument(doc, [block], "code", "sample.py"))
    assert len(chunks) == 3
    assert all(chunk.parent_id == "symbol-1" for chunk in chunks)


def test_deterministic_context_prefix_is_metadata_and_content_is_preserved():
    source = LoadedSource("x.md", "x.md", "text/markdown", b"# Search\n\nExact source text.", "document")
    normalized = DocumentParser().parse(source)
    chunks = StructureChunker(ChunkingConfig(max_tokens=100, overlap_tokens=0)).chunk(normalized)
    original = chunks[0].content
    enriched = DeterministicContextualizer().enrich(chunks)[0]
    assert enriched.content == original
    assert "Document:" in enriched.metadata["context_prefix"]
    assert "Search" in enriched.metadata["context_prefix"]


def test_embedding_batches_deduplicates_and_enforces_dimensions():
    provider = FakeProvider()
    service = EmbeddingService(provider, batch_size=2, expected_dimensions=2)
    vectors = service.embed(["one", "two", "one"], cache_scope="tenant-a:corpus-v1")
    assert len(vectors) == 3 and vectors[0] == vectors[2]
    assert provider.calls == [["one", "two"]]
    service.embed(["one"], cache_scope="tenant-a:corpus-v1")
    assert len(provider.calls) == 1


def test_embedding_cache_separates_tenants_and_document_versions():
    provider = FakeProvider()
    service = EmbeddingService(provider, expected_dimensions=2)
    service.embed(["shared text"], cache_scope="tenant-a:corpus-v1")
    service.embed(["shared text"], cache_scope="tenant-b:corpus-v1")
    service.embed(["shared text"], cache_scope="tenant-a:corpus-v2")
    assert len(provider.calls) == 3


def test_embedding_rejects_inconsistent_vector_dimensions():
    class BadProvider:
        def embed(self, texts):
            return [[1.0], [1.0, 2.0]]

    import pytest
    with pytest.raises(EmbeddingError, match="dimensions"):
        EmbeddingService(BadProvider(), batch_size=2).embed(["a", "b"])


def test_http_embedding_adapter_uses_configured_model_without_credentials(monkeypatch):
    import json
    import indexing.embeddings as embedding_module
    from indexing.embeddings import HttpEmbeddingProvider

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [
                {"index": 1, "embedding": [0.2, 0.3]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]}).encode()

    requests = []
    monkeypatch.setattr(embedding_module, "urlopen", lambda request, timeout: (requests.append(request) or FakeResponse()))
    vectors = HttpEmbeddingProvider("https://embed.example/v1", "test-model").embed(["first", "second"])
    assert vectors == [[0.1, 0.2], [0.2, 0.3]]
    assert json.loads(requests[0].data)["model"] == "test-model"
    assert "Authorization" not in requests[0].headers


def test_sparse_vector_and_qdrant_payload_are_prepared_consistently():
    sparse = SparseEncoder().encode("search search systems")
    assert len(sparse[0]) == len(sparse[1])
    assert sparse[0] == sorted(sparse[0])
    source = LoadedSource("x", "x.md", "text/markdown", b"", "document")
    doc = Document(document_id="d", title="Title", source="x", document_type="markdown", content_hash="h")
    chunk = StructureChunker(ChunkingConfig(20, 0)).chunk(NormalizedDocument(
        doc, [ContentBlock("b", "hello world", ContentKind.PARAGRAPH)], "document", "x"))[0]
    point = prepare_point(chunk, [0.1, 0.2], sparse, "source-hash", "generation-1")
    assert point.payload["document_id"] == "d"
    assert point.payload["source_hash"] == "source-hash"
    assert point.payload["is_active"] is False
    assert "dense" in point.vectors and "sparse" in point.vectors
