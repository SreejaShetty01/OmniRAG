"""Rank-fusion strategy contract and deterministic Reciprocal Rank Fusion baseline."""

from typing import Protocol
from dataclasses import dataclass

from qdrant_client import models

from omnirag.config.settings import get_settings

from omnirag.retrieval.contracts import RetrievalCandidate


class FusionStrategy(Protocol):
    def fuse(self, dense: list[RetrievalCandidate], sparse: list[RetrievalCandidate], *,
             rrf_k: int = 60) -> list[RetrievalCandidate]: ...


class ReciprocalRankFusion:
    """RRF scores are ordering signals only; they are not probabilities or confidence."""

    def fuse(self, dense: list[RetrievalCandidate], sparse: list[RetrievalCandidate], *,
             rrf_k: int = 60) -> list[RetrievalCandidate]:
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        merged: dict[str, RetrievalCandidate] = {}
        for source, channel in ((dense, "dense"), (sparse, "sparse")):
            for candidate in source:
                existing = merged.get(candidate.identity)
                if existing is None:
                    existing = RetrievalCandidate(chunk=candidate.chunk.model_copy(deep=True))
                    merged[candidate.identity] = existing
                if channel == "dense" and candidate.dense_rank is not None:
                    if existing.dense_rank is None or candidate.dense_rank < existing.dense_rank:
                        existing.dense_rank = candidate.dense_rank
                        existing.dense_score = candidate.dense_score
                if channel == "sparse" and candidate.sparse_rank is not None:
                    if existing.sparse_rank is None or candidate.sparse_rank < existing.sparse_rank:
                        existing.sparse_rank = candidate.sparse_rank
                        existing.sparse_score = candidate.sparse_score
                for provenance in candidate.provenance:
                    if provenance not in existing.provenance:
                        existing.provenance.append(provenance)

        for candidate in merged.values():
            candidate.rrf_dense_contribution = (1 / (rrf_k + candidate.dense_rank)
                                                if candidate.dense_rank is not None else 0.0)
            candidate.rrf_sparse_contribution = (1 / (rrf_k + candidate.sparse_rank)
                                                 if candidate.sparse_rank is not None else 0.0)
            candidate.rrf_score = candidate.rrf_dense_contribution + candidate.rrf_sparse_contribution
            candidate.ranking_score = candidate.rrf_score
        results = sorted(merged.values(), key=lambda c: (
            -c.ranking_score,
            min(c.dense_rank or 10**9, c.sparse_rank or 10**9),
            c.dense_rank or 10**9,
            c.sparse_rank or 10**9,
            c.chunk.document_id,
            c.chunk.chunk_id,
        ))
        for rank, candidate in enumerate(results, start=1):
            candidate.fusion_rank = rank
        return results


class StructuralScorer:
    """Expose interpretable metadata overlap; callers choose any nonzero ranking weight."""

    def score(self, normalized_query: str, candidate: RetrievalCandidate) -> float:
        query_terms = {term.casefold() for term in normalized_query.split() if len(term) > 1}
        structural_text = " ".join([
            candidate.chunk.metadata.get("title", ""), candidate.chunk.heading or "",
            " ".join(candidate.chunk.section_path), candidate.chunk.source,
            " ".join(candidate.chunk.metadata.get("symbols", [])),
            candidate.chunk.language or "",
        ])
        structure_terms = {term.casefold() for term in structural_text.split() if len(term) > 1}
        if not query_terms:
            return 0.0
        return len(query_terms & structure_terms) / len(query_terms)


@dataclass(frozen=True, slots=True)
class DBSFExperimentHit:
    point_id: str
    score: float
    rank: int


class QdrantDBSFExperiment:
    """Isolated experiment adapter for Qdrant-native DBSF; never selected by the baseline engine."""

    @staticmethod
    def supported() -> bool:
        return hasattr(models.Fusion, "DBSF")

    def search(self, client, dense_query: list[float], sparse_query: models.SparseVector,
               query_filter: models.Filter, *, collection: str | None = None,
               dense_vector: str | None = None, sparse_vector: str | None = None,
               prefetch_limit: int = 50, result_limit: int = 20) -> list[DBSFExperimentHit]:
        if not self.supported():
            raise RuntimeError("Installed Qdrant client does not expose DBSF fusion")
        settings = get_settings()
        response = client.query_points(
            collection_name=collection or settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=dense_query, using=dense_vector or settings.dense_vector_name,
                                filter=query_filter, limit=prefetch_limit),
                models.Prefetch(query=sparse_query, using=sparse_vector or settings.sparse_vector_name,
                                filter=query_filter, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.DBSF),
            query_filter=query_filter, limit=result_limit, with_payload=False, with_vectors=False,
        )
        return [DBSFExperimentHit(str(point.id), float(point.score), rank)
                for rank, point in enumerate(response.points, start=1)]
