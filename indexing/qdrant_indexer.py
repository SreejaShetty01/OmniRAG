"""Shared dense+sparse Qdrant collection operations and payload preparation."""

import logging
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from config.settings import get_settings
from indexing.provenance import (assert_collection_embedder_compatible,
                                 configured_embedder, new_point_provenance)
from indexing.sparse_encoder import SparseEncoder
from models.chunk import Chunk

logger = logging.getLogger(__name__)

# Every payload path used by retrieval, expansion, idempotency, and lifecycle filters.
PAYLOAD_INDEX_SCHEMAS = {
    "document_id": "keyword", "chunk_id": "keyword", "parent_id": "keyword",
    "source": "keyword", "source_type": "keyword", "document_type": "keyword",
    "content_type": "keyword", "language": "keyword", "version": "keyword",
    "section_path": "keyword", "generation": "keyword", "source_hash": "keyword",
    "is_active": "bool", "acl_allowed_principals": "keyword",
    "acl_denied_principals": "keyword", "metadata.tenant_id": "keyword",
    "metadata.workspace_id": "keyword", "metadata.collection_id": "keyword",
    "metadata.block_ids": "keyword", "metadata.chunk_index": "integer",
}


@dataclass(frozen=True, slots=True)
class PreparedPoint:
    point_id: str
    vectors: dict[str, object]
    payload: dict[str, object]


def prepare_point(chunk: Chunk, dense: list[float], sparse: tuple[list[int], list[float]],
                  source_hash: str, generation: str, *, dense_name: str | None = None,
                  sparse_name: str | None = None,
                  provenance: dict[str, object] | None = None) -> PreparedPoint:
    if not dense:
        raise ValueError("Dense vector cannot be empty")
    indices, values = sparse
    if len(indices) != len(values):
        raise ValueError("Sparse vector indices and values must have equal lengths")
    if any(index < 0 for index in indices):
        raise ValueError("Sparse vector indices must be non-negative")
    identity = f"{chunk.document_id}:{generation}:{chunk.chunk_id}"
    point_id = str(uuid5(NAMESPACE_URL, identity))
    payload: dict[str, object] = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "source": chunk.source,
        "source_type": chunk.metadata.get("source_type"),
        "document_type": chunk.document_type.value,
        "title": chunk.metadata.get("title"),
        "section_path": chunk.section_path,
        "heading": chunk.heading,
        "page_number": chunk.page_number,
        "content_type": chunk.content_type,
        "language": chunk.language,
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "source_hash": source_hash,
        "generation": generation,
        "version": chunk.version,
        "created_at": chunk.metadata.get("created_at"),
        "links": chunk.links,
        "metadata": chunk.metadata,
        "access_control": chunk.access_control.model_dump(mode="json"),
        "acl_allowed_principals": chunk.access_control.allowed_principals,
        "acl_denied_principals": chunk.access_control.denied_principals,
        "is_active": False,
    }
    settings = get_settings()
    if provenance:
        payload.update(provenance)
    vectors: dict[str, object] = {
        dense_name or settings.dense_vector_name: dense,
        sparse_name or settings.sparse_vector_name: {"indices": indices, "values": values},
    }
    return PreparedPoint(point_id, vectors, payload)


class QdrantIndexer:
    def __init__(self, client=None, *, collection: str | None = None,
                 dense_name: str | None = None, sparse_name: str | None = None,
                 sparse_encoder: SparseEncoder | None = None):
        settings = get_settings()
        if client is None:
            from infrastructure.qdrant_client import get_qdrant_client
            client = get_qdrant_client()
        self.client = client
        self.collection = collection or settings.qdrant_collection
        self.dense_name = dense_name or settings.dense_vector_name
        self.sparse_name = sparse_name or settings.sparse_vector_name
        self.sparse_encoder = sparse_encoder or SparseEncoder()

    def prepare(self, chunk: Chunk, dense: list[float], source_hash: str,
                generation: str | None = None,
                provenance: dict[str, object] | None = None) -> PreparedPoint:
        generation = generation or source_hash
        return prepare_point(chunk, dense, self.sparse_encoder.encode(chunk.content), source_hash,
                             generation, dense_name=self.dense_name, sparse_name=self.sparse_name,
                             provenance=provenance)

    def ensure_collection(self, dimensions: int) -> None:
        from qdrant_client import models
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            current = vectors.get(self.dense_name) if isinstance(vectors, dict) else None
            if current is None:
                raise ValueError(f"Collection is missing named dense vector {self.dense_name!r}")
            if current.size != dimensions:
                raise ValueError(f"Collection dense dimensions are {current.size}; configured provider returned {dimensions}")
            sparse_vectors = info.config.params.sparse_vectors or {}
            if self.sparse_name not in sparse_vectors:
                raise ValueError(f"Collection is missing named sparse vector {self.sparse_name!r}")
            self.ensure_payload_indexes()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={self.dense_name: models.VectorParams(size=dimensions, distance=models.Distance.COSINE)},
            sparse_vectors_config={self.sparse_name: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> list[str]:
        """Create only missing filter indexes; does not recreate or mutate vectors."""
        if not self.client.collection_exists(self.collection):
            return []
        # qdrant-client's embedded local engine warns that payload indexes have no effect.
        # Server startup/collection creation performs the real migration.
        if type(getattr(self.client, "_client", None)).__name__ == "QdrantLocal":
            return []
        from qdrant_client import models
        info = self.client.get_collection(self.collection)
        existing = getattr(info.config.params, "payload_schema", None) or {}
        created = []
        schemas = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "bool": models.PayloadSchemaType.BOOL,
            "integer": models.PayloadSchemaType.INTEGER,
        }
        for field_name, schema in PAYLOAD_INDEX_SCHEMAS.items():
            if field_name in existing:
                continue
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field_name,
                field_schema=schemas[schema], wait=True,
            )
            created.append(field_name)
        return created

    def is_unchanged(self, document_id: str, source_hash: str) -> bool:
        if not self.client.collection_exists(self.collection):
            return False
        from qdrant_client import models
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                models.FieldCondition(key="source_hash", match=models.MatchValue(value=source_hash)),
                models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
            ]), limit=1, with_payload=False, with_vectors=False,
        )
        unchanged = bool(records)
        if unchanged:
            # Retry stale-generation cleanup after a previous publication cleanup error.
            self._retire_old_generations(document_id, source_hash)
        return unchanged

    def index_document(self, chunks: list[Chunk], vectors: list[list[float]], source_hash: str,
                       *, provenance: dict[str, object] | None = None) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one dense vector")
        if not chunks:
            return 0
        if len({chunk.document_id for chunk in chunks}) != 1:
            raise ValueError("One Qdrant indexing call must contain chunks from exactly one document")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Dense vector dimensions are inconsistent")
        settings = get_settings()
        provider, model, configured_dimensions = configured_embedder(settings)
        expected_provider = (provenance or {}).get("embedder_provider", provider)
        expected_model = (provenance or {}).get("embedder_model", model)
        assert_collection_embedder_compatible(
            self.client, self.collection, provider=str(expected_provider), model=str(expected_model),
            dimensions=dimensions or configured_dimensions,
            allow_override=settings.embedding_provenance_override,
        )
        self.ensure_collection(dimensions)
        if provenance is None:
            provenance = new_point_provenance(
                provider=provider, model=model, dimension=dimensions,
                sparse_method=self.sparse_encoder.method,
                sparse_tokenizer_version=self.sparse_encoder.tokenizer_version,
                chunker_version=settings.chunker_version,
                enrichment_mode="unknown",
                ingestion_schema_version=settings.ingestion_schema_version,
            )
        from qdrant_client import models
        prepared = [self.prepare(chunk, vector, source_hash, provenance=provenance)
                    for chunk, vector in zip(chunks, vectors, strict=True)]
        points = []
        for prepared_point in prepared:
            vectors_for_point = dict(prepared_point.vectors)
            sparse_value = vectors_for_point.get(self.sparse_name)
            if isinstance(sparse_value, dict):
                vectors_for_point[self.sparse_name] = models.SparseVector(**sparse_value)
            points.append(models.PointStruct(id=prepared_point.point_id,
                                             vector=vectors_for_point,
                                             payload=prepared_point.payload))
        staged_ids = [p.point_id for p in prepared]
        document_id = chunks[0].document_id
        try:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            self.client.set_payload(
                collection_name=self.collection, payload={"is_active": True},
                points=staged_ids, wait=True,
            )
        except Exception:
            # New generation IDs are disjoint from the currently published generation.
            try:
                self.client.delete(collection_name=self.collection,
                                   points_selector=models.PointIdsList(points=staged_ids), wait=True)
            except Exception:
                logger.error("Failed to clean staged Qdrant points",
                             extra={"document_id": document_id, "error_type": "cleanup_failed"})
            raise
        # Retire stale versions only after the new generation is visible.
        # Propagate cleanup errors so the caller can report the document as needing attention.
        self._retire_old_generations(document_id, source_hash)
        return len(points)

    def _retire_old_generations(self, document_id: str, generation: str) -> None:
        from qdrant_client import models
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))],
                must_not=[models.FieldCondition(key="generation", match=models.MatchValue(value=generation))],
            )),
            wait=True,
        )

    def delete_document(self, document_id: str) -> None:
        from qdrant_client import models
        self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id))])), wait=True)

    def delete_chunk(self, chunk_id: str) -> None:
        from qdrant_client import models
        self.client.delete(collection_name=self.collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(
                key="chunk_id", match=models.MatchValue(value=chunk_id))])), wait=True)
