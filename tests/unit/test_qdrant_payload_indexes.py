from types import SimpleNamespace

from omnirag.indexing.qdrant_indexer import PAYLOAD_INDEX_SCHEMAS, QdrantIndexer


class IndexClient:
    def __init__(self, existing=()):
        self.schema = {name: object() for name in existing}
        self.calls = []

    def collection_exists(self, name):
        return True

    def get_collection(self, name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(payload_schema=self.schema)))

    def create_payload_index(self, **kwargs):
        self.calls.append(kwargs)
        self.schema[kwargs["field_name"]] = object()


def test_payload_indexes_create_missing_fields_once_and_preserve_existing():
    client = IndexClient(existing=["document_id", "metadata.tenant_id"])
    indexer = QdrantIndexer(client=client, collection="shared")
    created = indexer.ensure_payload_indexes()
    assert set(created) == set(PAYLOAD_INDEX_SCHEMAS) - {"document_id", "metadata.tenant_id"}
    assert len(client.calls) == len(created)
    assert indexer.ensure_payload_indexes() == []
    assert len(client.calls) == len(created)
    assert all(call["wait"] is True for call in client.calls)
