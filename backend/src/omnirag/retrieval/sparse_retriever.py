"""Sparse lexical retrieval using the Phase 2 hashed-term encoder and named vector."""

from qdrant_client import models

from omnirag.config.settings import get_settings
from omnirag.indexing.sparse_encoder import SparseEncoder
from omnirag.infrastructure.qdrant_client import get_qdrant_client
from omnirag.retrieval.contracts import RetrievalCandidate
from omnirag.retrieval.qdrant_utils import candidate_from_point


class SparseRetriever:
    def __init__(self, client=None, *, collection: str | None = None, vector_name: str | None = None,
                 encoder: SparseEncoder | None = None):
        settings = get_settings()
        self.client = client or get_qdrant_client()
        self.collection = collection or settings.qdrant_collection
        self.vector_name = vector_name or settings.sparse_vector_name
        self.encoder = encoder or SparseEncoder()

    def search(self, query: str, query_filter: models.Filter, limit: int) -> list[RetrievalCandidate]:
        if not query.strip() or limit < 1:
            return []
        indices, values = self.encoder.encode(query)
        if not indices:
            return []
        result = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=indices, values=values),
            using=self.vector_name, query_filter=query_filter, limit=limit,
            with_payload=True, with_vectors=False,
        )
        return [candidate_from_point(point, channel="sparse", rank=rank)
                for rank, point in enumerate(result.points, start=1)]
