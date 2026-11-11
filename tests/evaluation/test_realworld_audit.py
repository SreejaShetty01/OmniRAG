import hashlib
from types import SimpleNamespace

from omnirag.indexing.qdrant_indexer import PAYLOAD_INDEX_SCHEMAS
from scripts.realworld import audit_corpus


def _fake_collection_client(payload_overrides=None):
    payload = {"source_hash": "a" * 64, "document_id": "api-doc-1", "version": "b" * 40,
        "is_active": True, "acl_allowed_principals": ["user:corpus-reader"],
        "acl_denied_principals": [], "metadata": {"tenant_id": "realworld"},
        "embedder_provider": "openai_compatible", "embedder_model": "nomic-embed-text",
        "embedding_dim": 768}
    payload.update(payload_overrides or {})
    info = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(
            vectors={"dense": SimpleNamespace(size=768, distance=SimpleNamespace(value="Cosine"))},
            sparse_vectors={"sparse": SimpleNamespace(modifier=SimpleNamespace(value="idf"))})),
        payload_schema={name: object() for name in PAYLOAD_INDEX_SCHEMAS},
        disk_data_size=10_000, ram_data_size=5_000)
    point = SimpleNamespace(id="point-1", payload=payload)
    class Client:
        def get_collection(self, collection_name): return info
        def scroll(self, **kwargs): return [point], None
    return Client()


def _audit_inputs():
    source_hash = "a" * 64
    manifest = {"target_collection": "omnirag_eval_test_v1", "content_set_sha256": "set-hash",
                "sources": [{"sha256": source_hash, "repository_commit": "b" * 40}]}
    progress = {"status": "completed", "collection": manifest["target_collection"],
        "content_set_sha256": "set-hash", "schedule": [{"source_sha256": source_hash,
            "document_id": "manifest-doc", "api_document_id": "api-doc-1", "status": "indexed",
            "chunk_count": 1}]}
    return manifest, progress


def test_corpus_audit_checks_points_schema_acl_tenant_version_and_provenance(monkeypatch, tmp_path):
    manifest, progress = _audit_inputs()
    monkeypatch.setattr(audit_corpus, "validate_corpus", lambda *args: {"valid": True, "errors": []})
    report = audit_corpus.audit_collection(_fake_collection_client(), manifest, tmp_path,
        progress=progress, user_id="corpus-reader", tenant_id="realworld")
    assert report["valid"] is True
    assert report["source_file_count"] == report["source_hashes_accounted"] == 1
    assert report["actual_active_point_count"] == report["expected_chunk_count"] == 1
    assert report["acl_points_checked"] == report["tenant_points_checked"] == 1
    assert report["version_points_checked"] == report["provenance_points_checked"] == 1
    assert report["missing_payload_indexes"] == []


def test_corpus_audit_fails_closed_on_wrong_tenant_and_provenance(monkeypatch, tmp_path):
    manifest, progress = _audit_inputs()
    monkeypatch.setattr(audit_corpus, "validate_corpus", lambda *args: {"valid": True, "errors": []})
    client = _fake_collection_client({"metadata": {"tenant_id": "other-tenant"},
                                      "embedder_model": "wrong-model"})
    report = audit_corpus.audit_collection(client, manifest, tmp_path, progress=progress,
        user_id="corpus-reader", tenant_id="realworld")
    assert report["valid"] is False
    assert any("tenant payload" in error for error in report["errors"])
    assert any("embedding provenance" in error for error in report["errors"])


def test_snapshot_creation_refuses_overwrite_and_records_sha256(monkeypatch, tmp_path):
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return b"snapshot-bytes"
    monkeypatch.setattr(audit_corpus, "urlopen", lambda *args, **kwargs: Response())
    client = SimpleNamespace(create_snapshot=lambda **kwargs: SimpleNamespace(name="snapshot-1"))
    path = tmp_path / "corpus.snapshot"
    result = audit_corpus.create_snapshot_and_hash(client, collection="omnirag_eval_test_v1",
        qdrant_url="http://localhost:6333", output_path=path)
    assert result["snapshot_sha256"] == hashlib.sha256(b"snapshot-bytes").hexdigest()
    assert result["snapshot_bytes"] == len(b"snapshot-bytes")


def test_protected_collection_fingerprint_detects_changes_without_writing_collections(tmp_path):
    points = {
        "omnirag": [SimpleNamespace(id="p1", payload={"document_id": "d1", "chunk_id": "c1",
            "source_hash": "s1", "content_hash": "h1", "generation": "g1", "is_active": True})],
        "omnirag_eval_phase8_20261104": [SimpleNamespace(id="p2", payload={"document_id": "d2",
            "chunk_id": "c2", "source_hash": "s2", "content_hash": "h2", "generation": "g2",
            "is_active": True})],
    }
    class Client:
        def collection_exists(self, collection): return collection in points
        def scroll(self, collection_name, **kwargs): return points[collection_name], None
    client = Client()
    baseline_path = tmp_path / "protected.json"
    audit_corpus.record_protected_baselines(client, baseline_path)
    assert audit_corpus.compare_protected_baselines(client, baseline_path)["preserved"] is True
    points["omnirag"][0].payload["content_hash"] = "changed"
    result = audit_corpus.compare_protected_baselines(client, baseline_path)
    assert result["preserved"] is False
    assert result["changed_collections"] == ["omnirag"]
