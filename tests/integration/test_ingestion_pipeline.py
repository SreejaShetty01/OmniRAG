from pathlib import Path

from indexing.embeddings import EmbeddingService
from ingestion.pipeline import IngestionPipeline, IngestionStatus
from models.chunk import AccessControlMetadata


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeIndexer:
    def __init__(self):
        self.indexed = []
        self.hashes = {}
        self.deleted = []

    def is_unchanged(self, document_id, source_hash):
        return self.hashes.get(document_id) == source_hash

    def index_document(self, chunks, vectors, source_hash):
        assert len(chunks) == len(vectors)
        self.indexed.append((chunks, vectors, source_hash))
        self.hashes[chunks[0].document_id] = source_hash
        return len(chunks)

    def delete_document(self, document_id):
        self.deleted.append(document_id)
        self.hashes.pop(document_id, None)


def test_complete_local_markdown_pipeline_and_unchanged_short_circuit(tmp_path: Path):
    path = tmp_path / "manual.md"
    path.write_text("# Manual\n\n## Setup\n\nInstall package. See [guide](guide.md).", encoding="utf-8")
    provider, indexer = FakeProvider(), FakeIndexer()
    pipeline = IngestionPipeline(embedder=EmbeddingService(provider, expected_dimensions=2), indexer=indexer)
    first = pipeline.ingest(path)
    second = pipeline.ingest(path)
    assert first.status is IngestionStatus.INDEXED
    assert first.chunk_count > 0
    assert second.status is IngestionStatus.UNCHANGED
    assert provider.calls == first.chunk_count
    chunk = indexer.indexed[0][0][0]
    assert chunk.section_path
    assert chunk.metadata["context_prefix"]


def test_pipeline_returns_document_scoped_failure_status(tmp_path: Path):
    path = tmp_path / "bad.py"
    path.write_text("def bad(:\n", encoding="utf-8")
    pipeline = IngestionPipeline(embedder=EmbeddingService(FakeProvider(), expected_dimensions=2), indexer=FakeIndexer())
    result = pipeline.ingest(path)
    assert result.status is IngestionStatus.FAILED
    assert result.error_type == "SourceLoadError"


def test_pipeline_reingestion_after_update_and_delete(tmp_path: Path):
    path = tmp_path / "manual.txt"
    path.write_text("Original content.", encoding="utf-8")
    indexer = FakeIndexer()
    pipeline = IngestionPipeline(embedder=EmbeddingService(FakeProvider(), expected_dimensions=2), indexer=indexer)
    first = pipeline.ingest(path)
    path.write_text("Updated content.", encoding="utf-8")
    updated = pipeline.ingest(path)
    assert first.document_id == updated.document_id
    assert updated.status is IngestionStatus.INDEXED
    assert len(indexer.indexed) == 2
    pipeline.delete_document(first.document_id)
    assert first.document_id in indexer.deleted


def test_pipeline_preserves_supplied_version_and_access_controls(tmp_path: Path):
    path = tmp_path / "controlled.txt"
    path.write_text("Controlled content.", encoding="utf-8")
    indexer = FakeIndexer()
    pipeline = IngestionPipeline(embedder=EmbeddingService(FakeProvider(), expected_dimensions=2), indexer=indexer)
    result = pipeline.ingest(path, version="rev-7",
                             access_control=AccessControlMetadata(allowed_principals=["team:docs"]))
    assert result.status is IngestionStatus.INDEXED
    chunk = indexer.indexed[0][0][0]
    assert chunk.version == "rev-7"
    assert chunk.access_control.allowed_principals == ["team:docs"]


def test_pipeline_reports_missing_embedding_configuration_per_document(tmp_path: Path, monkeypatch):
    from config.settings import Settings
    import indexing.embeddings as embedding_module

    monkeypatch.delenv("OMNIRAG_EMBEDDING_ENDPOINT", raising=False)
    monkeypatch.delenv("OMNIRAG_EMBEDDING_MODEL_ID", raising=False)
    monkeypatch.setattr(embedding_module, "get_settings", lambda: Settings(_env_file=None))
    path = tmp_path / "notes.txt"
    path.write_text("Text for indexing.", encoding="utf-8")
    result = IngestionPipeline(indexer=FakeIndexer()).ingest(path)
    assert result.status is IngestionStatus.FAILED
    assert result.error_type == "EmbeddingError"
    assert "embedding_endpoint" in result.error_message
