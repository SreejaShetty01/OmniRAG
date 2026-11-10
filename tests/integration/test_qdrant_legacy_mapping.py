from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models

from scripts.map_legacy_tenant import run_mapping


def test_legacy_qdrant_mapping_dry_run_apply_and_idempotence(tmp_path):
    client = QdrantClient(":memory:")
    collection = f"legacy-map-{uuid4().hex[:8]}"
    point_id = str(uuid4())
    client.create_collection(collection, vectors_config={"dense": models.VectorParams(
        size=2, distance=models.Distance.COSINE)})
    client.upsert(collection, points=[models.PointStruct(
        id=point_id, vector={"dense": [1.0, 0.0]},
        payload={"document_id": "doc-1", "is_active": True,
                 "acl_allowed_principals": [], "acl_denied_principals": [],
                 "access_control": {"allowed_principals": [], "denied_principals": []},
                 "metadata": {"title": "preserved"}},
    )], wait=True)
    mapping = {"qdrant": [{"document_id": "doc-1", "tenant_id": "tenant-a", "owner_id": "owner-a"}]}
    audit = tmp_path / "audit.jsonl"
    try:
        dry = run_mapping(mapping, sqlite_path=Path("unused"), qdrant_client=client,
                           collection=collection, apply=False, audit_path=audit)
        point = client.retrieve(collection, ids=[point_id], with_payload=True)[0]
        assert dry[0]["action"] == "map_qdrant"
        assert "tenant_id" not in point.payload["metadata"]

        run_mapping(mapping, sqlite_path=Path("unused"), qdrant_client=client,
                    collection=collection, apply=True, audit_path=audit)
        point = client.retrieve(collection, ids=[point_id], with_payload=True)[0]
        assert point.payload["metadata"] == {"title": "preserved", "tenant_id": "tenant-a"}
        assert point.payload["acl_allowed_principals"] == ["user:owner-a"]
        assert point.payload["access_control"]["allowed_principals"] == ["user:owner-a"]

        rerun = run_mapping(mapping, sqlite_path=Path("unused"), qdrant_client=client,
                            collection=collection, apply=True, audit_path=audit)
        assert rerun[0]["action"] == "already_mapped"
    finally:
        client.delete_collection(collection)
        client.close()
