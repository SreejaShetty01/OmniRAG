from qdrant_client import QdrantClient

from omnirag.indexing.qdrant_indexer import QdrantIndexer
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.retrieval.context_expander import ContextExpander
from omnirag.retrieval.contracts import RetrievalPolicy, RetrievalScope
from omnirag.retrieval.contracts import RetrievalFilters
from omnirag.retrieval.dense_retriever import DenseRetriever
from omnirag.retrieval.cache import RetrievalCache
from omnirag.retrieval.query_filters import FilterBuilder
from omnirag.retrieval.sparse_retriever import SparseRetriever


def chunk(chunk_id: str, content: str, *, parent_id="doc-1", allowed=(), denied=(), version="v1",
          metadata=None, links=()):
    return Chunk(
        chunk_id=chunk_id, document_id="doc-1", parent_id=parent_id, content=content,
        document_type=DocumentType.CODE if chunk_id.startswith("code") else DocumentType.TEXT,
        content_hash=f"hash-{chunk_id}", source="guide.md", version=version, links=list(links),
        access_control=AccessControlMetadata(allowed_principals=list(allowed),
                                             denied_principals=list(denied)),
        metadata={"source_type": "document", "title": "Guide", "tenant_id": "tenant-a",
                  **(metadata or {})},
    )


def test_dense_sparse_hybrid_acl_version_scope_and_cache_isolation():
    client = QdrantClient(":memory:")
    indexer = QdrantIndexer(client=client, collection="hybrid-test")
    chunks = [
        chunk("public", "ZX-81 protocol overview", metadata={"tenant_id": "tenant-a"}),
        chunk("private", "ZX-81 protocol confidential note", allowed=["user:bob"],
              metadata={"tenant_id": "tenant-a"}, version="v2"),
        chunk("denied", "ZX-81 protocol denied note", allowed=["user:alice"],
              denied=["user:alice"], metadata={"tenant_id": "tenant-a"}),
        chunk("other-tenant", "ZX-81 protocol other tenant", metadata={"tenant_id": "tenant-b"}),
    ]
    for position, item in enumerate(chunks):
        indexer.index_document([item], [[1.0, position / 10]], "generation-1")

    filters = FilterBuilder()
    alice = RetrievalScope(principal_ids=["user:alice"], tenant_id="tenant-a")
    alice_filter = filters.build(RetrievalFilters(), alice)
    dense = DenseRetriever(client, collection="hybrid-test").search([1.0, 0.0], alice_filter, 10)
    sparse = SparseRetriever(client, collection="hybrid-test").search("ZX-81 protocol", alice_filter, 10)
    assert {item.identity for item in dense} == {"public"}
    assert {item.identity for item in sparse} == {"public"}

    bob = RetrievalScope(principal_ids=["user:bob"], tenant_id="tenant-a")
    bob_filter = filters.build(RetrievalFilters(versions=["v2"]), bob)
    assert {item.identity for item in SparseRetriever(client, collection="hybrid-test").search(
        "ZX-81 protocol", bob_filter, 10)} == {"private"}
    cache = RetrievalCache()
    cache.set("alice-generation-1", "result-a")
    assert cache.get("alice-generation-1", 30) == "result-a"
    assert cache.get("bob-generation-1", 30) is None
    assert cache.get("alice-generation-2", 30) is None
    client.close()


def test_parent_child_neighbour_and_cross_reference_expansion_reapply_acl():
    client = QdrantClient(":memory:")
    indexer = QdrantIndexer(client=client, collection="expansion-test")
    chunks = [
        chunk("root", "class Service", metadata={"block_ids": ["class-symbol"], "chunk_index": 0},
              links=["target"]),
        chunk("child", "def secret_method(): return 1", parent_id="class-symbol",
              allowed=["user:bob"], metadata={"block_ids": ["method-symbol"], "chunk_index": 8}),
        chunk("neighbour", "adjacent public explanation", metadata={"chunk_index": 1}),
        chunk("target", "cross referenced public definition", metadata={"chunk_index": 3}),
    ]
    for item in chunks:
        indexer.index_document([item], [[0.5, 0.5]], "generation-1")
    base_filters = FilterBuilder()
    alice_scope = RetrievalScope(principal_ids=["user:alice"], tenant_id="tenant-a")
    alice_filter = base_filters.build(RetrievalFilters(), alice_scope)
    expander = ContextExpander(client, collection="expansion-test")
    seed = next(item for item in DenseRetriever(client, collection="expansion-test").search(
        [0.5, 0.5], alice_filter, 10) if item.identity == "root")
    policy = RetrievalPolicy(expand_parents=False, expand_children=True, expand_neighbours=True,
                             expand_cross_references=True, expansion_token_budget=300,
                             expansion_max_count=10)
    expanded, _ = expander.expand([seed], alice_filter, policy)
    expanded_ids = {item.identity for item in expanded}
    assert "child" not in expanded_ids
    assert "neighbour" in expanded_ids and "target" in expanded_ids

    bob_scope = RetrievalScope(principal_ids=["user:bob"], tenant_id="tenant-a")
    bob_filter = base_filters.build(RetrievalFilters(), bob_scope)
    child = SparseRetriever(client, collection="expansion-test").search("secret_method", bob_filter, 10)[0]
    parent_expansion, _ = expander.expand([child], bob_filter,
                                          RetrievalPolicy(expand_parents=True, expand_children=False,
                                                          expand_neighbours=False, expand_cross_references=False))
    assert "root" in {item.identity for item in parent_expansion}
    client.close()
