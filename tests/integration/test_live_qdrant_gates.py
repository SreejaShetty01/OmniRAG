"""Hard-gate probes intended for CI with a real Qdrant server endpoint."""

import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.live_qdrant


def _client_or_skip():
    endpoint = os.getenv("OMNIRAG_7A_QDRANT_URL")
    if not endpoint:
        pytest.skip("OMNIRAG_7A_QDRANT_URL is not configured; live Qdrant gate not run")
    from qdrant_client import QdrantClient

    api_key = os.getenv("OMNIRAG_7A_QDRANT_API_KEY")
    return QdrantClient(url=endpoint, api_key=api_key)


def test_live_qdrant_tenant_and_acl_filters_reject_other_principals():
    from qdrant_client import models

    from omnirag.config.settings import get_settings
    from omnirag.retrieval.contracts import RetrievalFilters, RetrievalScope
    from omnirag.retrieval.query_filters import FilterBuilder

    client = _client_or_skip()
    name = f"omnirag_7a_gate_{uuid4().hex[:12]}"
    settings = get_settings()
    try:
        client.create_collection(
            name,
            vectors_config={settings.dense_vector_name: models.VectorParams(size=2, distance=models.Distance.COSINE)},
            sparse_vectors_config={settings.sparse_vector_name: models.SparseVectorParams()},
        )
        client.upsert(name, points=[
            models.PointStruct(id=str(uuid4()), vector={settings.dense_vector_name: [1.0, 0.0],
                                                         settings.sparse_vector_name: models.SparseVector(indices=[1], values=[1.0])},
                               payload={"document_id": "restricted", "is_active": True,
                                        "acl_allowed_principals": ["user:phase7a-user-a"],
                                        "acl_denied_principals": [], "metadata": {"tenant_id": "tenant-a"}}),
            models.PointStruct(id=str(uuid4()), vector={settings.dense_vector_name: [0.0, 1.0],
                                                         settings.sparse_vector_name: models.SparseVector(indices=[2], values=[1.0])},
                               payload={"document_id": "other-tenant", "is_active": True,
                                        "acl_allowed_principals": ["user:phase7a-user-b"],
                                        "acl_denied_principals": [], "metadata": {"tenant_id": "tenant-b"}}),
            models.PointStruct(id=str(uuid4()), vector={settings.dense_vector_name: [0.5, 0.5],
                                                         settings.sparse_vector_name: models.SparseVector(indices=[3], values=[1.0])},
                               payload={"document_id": "unattributed", "is_active": True,
                                        "acl_allowed_principals": [], "acl_denied_principals": [],
                                        "metadata": {"title": "legacy point without tenant"}}),
        ], wait=True)
        scope = RetrievalScope(principal_ids=("user:phase7a-user-b",), tenant_id="tenant-b")
        points, _ = client.scroll(name, scroll_filter=FilterBuilder().build(RetrievalFilters(), scope),
                                  limit=10, with_payload=True, with_vectors=False)
        assert [point.payload["document_id"] for point in points] == ["other-tenant"]
    finally:
        client.delete_collection(name)
        client.close()


def test_live_retrieval_cache_key_separates_tenants_and_corpus_versions():
    from omnirag.retrieval.contracts import RetrievalFilters, RetrievalPolicy, RetrievalScope
    from omnirag.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(dense_retriever=object(), sparse_retriever=object(),
                                expander=object(), collection="phase7a-live")
    policy = RetrievalPolicy(cache_ttl_seconds=60)
    filters = RetrievalFilters()
    tenant_a = RetrievalScope(principal_ids=("user:shared",), tenant_id="tenant-a", corpus_generation="g1")
    tenant_b = RetrievalScope(principal_ids=("user:shared",), tenant_id="tenant-b", corpus_generation="g1")
    updated_a = RetrievalScope(principal_ids=("user:shared",), tenant_id="tenant-a", corpus_generation="g2")
    key_a = retriever._cache_key("shared query", filters, tenant_a, policy)
    assert key_a != retriever._cache_key("shared query", filters, tenant_b, policy)
    assert key_a != retriever._cache_key("shared query", filters, updated_a, policy)
