from qdrant_client import QdrantClient

from omnirag.indexing.qdrant_indexer import QdrantIndexer
from omnirag.models.chunk import Chunk


def make_chunk(content: str, chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc-1", parent_id="doc-1",
        content=content, document_type="text", content_hash=f"hash-{chunk_id}", source="notes.txt",
        metadata={"source_type": "document", "title": "Notes"},
    )


def test_qdrant_shared_collection_idempotency_update_and_delete():
    client = QdrantClient(":memory:")
    indexer = QdrantIndexer(client=client, collection="test-omnirag")
    chunk = make_chunk("original text", "chunk-v1")
    assert indexer.index_document([chunk], [[0.1, 0.2, 0.3]], "source-v1") == 1
    assert indexer.is_unchanged("doc-1", "source-v1")
    changed = make_chunk("updated text", "chunk-v2")
    assert indexer.index_document([changed], [[0.2, 0.3, 0.4]], "source-v2") == 1
    points, _ = client.scroll("test-omnirag", limit=10, with_payload=True, with_vectors=False)
    assert len(points) == 1
    assert points[0].payload["content"] == "updated text"
    assert points[0].payload["is_active"] is True
    indexer.delete_document("doc-1")
    points, _ = client.scroll("test-omnirag", limit=10, with_payload=False, with_vectors=False)
    assert points == []
    client.close()


def test_failed_publication_cleans_staged_qdrant_points():
    client = QdrantClient(":memory:")

    class FailingPublisher:
        def __getattr__(self, name):
            return getattr(client, name)

        def set_payload(self, *args, **kwargs):
            raise RuntimeError("simulated activation failure")

    indexer = QdrantIndexer(client=FailingPublisher(), collection="test-cleanup")
    try:
        indexer.index_document([make_chunk("text", "chunk-1")], [[0.1, 0.2]], "hash-1")
    except RuntimeError as exc:
        assert "activation failure" in str(exc)
    else:
        raise AssertionError("expected publication failure")
    points, _ = client.scroll("test-cleanup", limit=10, with_payload=False, with_vectors=False)
    assert points == []
    client.close()
