from types import SimpleNamespace

import pytest
from qdrant_client import models

from indexing.provenance import (
    EmbeddingProvenanceMismatch, assert_collection_embedder_compatible, new_point_provenance,
)
from indexing.qdrant_indexer import prepare_point
from models.chunk import Chunk


def test_new_point_has_provenance_and_legacy_point_is_tolerated():
    chunk = Chunk(chunk_id="c", document_id="d", parent_id="d", content="text",
                  document_type="text", content_hash="h", source="a.txt")
    provenance = new_point_provenance(provider="p", model="m", dimension=3,
                                      sparse_method="s", sparse_tokenizer_version="t",
                                      chunker_version="c1", enrichment_mode="none",
                                      ingestion_schema_version="1")
    point = prepare_point(chunk, [1, 0, 0], ([1], [1.0]), "h", "g", provenance=provenance)
    assert point.payload["embedder_provider"] == "p"
    assert point.payload["embedder_model"] == "m"
    assert point.payload["embedding_dim"] == 3
    assert all(point.payload[key] for key in ("sparse_method", "sparse_tokenizer_version",
                                               "chunker_version", "enrichment_mode",
                                               "ingestion_schema_version", "ingested_at"))

    class Legacy:
        def collection_exists(self, name): return True
        def scroll(self, **kwargs): return ([SimpleNamespace(payload={"document_id": "old"})], None)

    assert_collection_embedder_compatible(Legacy(), "shared", provider="p", model="m", dimensions=3)


class PointClient:
    def __init__(self, payload): self.payload = payload
    def collection_exists(self, name): return True
    def scroll(self, **kwargs): return ([SimpleNamespace(payload=self.payload)], None)


def test_embedder_guard_blocks_same_dimension_model_swap_and_override_allows():
    client = PointClient({"embedder_provider": "p", "embedder_model": "old-model", "embedding_dim": 3})
    with pytest.raises(EmbeddingProvenanceMismatch, match="configured p/new-model/3, existing points use p/old-model/3"):
        assert_collection_embedder_compatible(client, "shared", provider="p", model="new-model", dimensions=3)
    assert_collection_embedder_compatible(client, "shared", provider="p", model="new-model",
                                          dimensions=3, allow_override=True)
