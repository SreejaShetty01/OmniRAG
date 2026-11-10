from types import SimpleNamespace

import pytest
from qdrant_client import models

from scripts.live_validation.run import (
    _check_qdrant_connectivity,
    _collection_info_after_ingestion,
    _validate_collection_vector_config,
)


class EmptyQdrant:
    def __init__(self):
        self.get_collection_calls = 0

    def get_collections(self):
        return {"collections": []}

    def collection_exists(self, collection):
        raise AssertionError("Connectivity check must not require a collection")

    def get_collection(self, collection):
        self.get_collection_calls += 1
        raise AssertionError("No collection should be read during connectivity check")


def test_qdrant_connectivity_accepts_service_with_no_collections():
    client = EmptyQdrant()

    response = _check_qdrant_connectivity(client)

    assert response == {"collections": []}
    assert client.get_collection_calls == 0


def test_collection_missing_after_ingestion_fails_clearly_before_collection_read():
    class MissingCollectionQdrant:
        def __init__(self):
            self.get_collection_calls = 0

        def collection_exists(self, collection):
            assert collection == "omnirag"
            return False

        def get_collection(self, collection):
            self.get_collection_calls += 1
            raise AssertionError("Must report missing collection before get_collection")

    client = MissingCollectionQdrant()

    with pytest.raises(RuntimeError, match="was not created by smoke-corpus ingestion"):
        _collection_info_after_ingestion(client, "omnirag")

    assert client.get_collection_calls == 0


def test_collection_vector_validation_checks_dense_and_sparse_schema():
    settings = SimpleNamespace(dense_vector_name="dense", sparse_vector_name="sparse")
    info = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
        vectors={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
        sparse_vectors={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )))

    _validate_collection_vector_config(info, settings, expected_dimension=3)
    info.config.params.vectors["dense"] = models.VectorParams(size=4, distance=models.Distance.COSINE)
    with pytest.raises(RuntimeError, match="dimension is 4; expected 3"):
        _validate_collection_vector_config(info, settings, expected_dimension=3)
