"""Point embedder provenance and same-dimension model mismatch checks."""

from __future__ import annotations

from datetime import datetime, timezone


class EmbeddingProvenanceMismatch(ValueError):
    pass


def configured_embedder(settings) -> tuple[str, str, int | None]:
    provider = settings.embedding_provider or "openai_compatible"
    return provider, settings.embedding_model_id or "unknown", settings.embedding_dimensions


def assert_collection_embedder_compatible(client, collection: str, *, provider: str,
                                          model: str, dimensions: int | None,
                                          allow_override: bool = False) -> None:
    """Sample active point provenance; missing legacy fields are explicitly unknown."""
    if not client.collection_exists(collection):
        return
    from qdrant_client import models

    records, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(must=[models.FieldCondition(
            key="is_active", match=models.MatchValue(value=True))]),
        limit=128, with_payload=True, with_vectors=False,
    )
    for point in records:
        payload = point.payload or {}
        existing_provider = payload.get("embedder_provider")
        existing_model = payload.get("embedder_model")
        existing_dimensions = payload.get("embedding_dim")
        if not existing_provider or not existing_model or existing_dimensions is None:
            continue
        mismatch = (existing_provider != provider or existing_model != model or
                    (dimensions is not None and int(existing_dimensions) != dimensions))
        if mismatch and not allow_override:
            expected = f"{provider}/{model}/{dimensions or 'collection-dimension-unknown'}"
            found = f"{existing_provider}/{existing_model}/{existing_dimensions}"
            raise EmbeddingProvenanceMismatch(
                f"Embedding configuration mismatch for collection {collection!r}: "
                f"configured {expected}, existing points use {found}. "
                "Reindex with a compatible model or enable the explicit override."
            )


def new_point_provenance(*, provider: str, model: str, dimension: int,
                         sparse_method: str, sparse_tokenizer_version: str,
                         chunker_version: str, enrichment_mode: str,
                         ingestion_schema_version: str) -> dict[str, object]:
    return {
        "embedder_provider": provider,
        "embedder_model": model,
        "embedding_dim": dimension,
        "sparse_method": sparse_method,
        "sparse_tokenizer_version": sparse_tokenizer_version,
        "chunker_version": chunker_version,
        "enrichment_mode": enrichment_mode,
        "ingestion_schema_version": ingestion_schema_version,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
