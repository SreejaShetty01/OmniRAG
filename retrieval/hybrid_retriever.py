"""Shared async hybrid retrieval orchestrator."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from config.settings import get_settings
from indexing.embeddings import EmbeddingService, EmbeddingProvider, provider_from_config
from indexing.provenance import assert_collection_embedder_compatible, configured_embedder
from infrastructure.provider_budget import BudgetExceeded
from infrastructure.qdrant_client import get_qdrant_client
from models.query import Query
from retrieval.cache import RetrievalCache
from retrieval.context_expander import ContextExpander
from retrieval.contracts import (
    RetrievalCandidate,
    RetrievalPolicy,
    RetrievalResult,
    RetrievalScope,
)
from retrieval.deduplication import deduplicate
from retrieval.dense_retriever import DenseRetriever
from retrieval.evidence_selection import EvidenceSelector
from retrieval.fusion import FusionStrategy, ReciprocalRankFusion, StructuralScorer
from retrieval.query_analyzer import QueryAnalyzer
from retrieval.query_filters import FilterBuilder
from retrieval.reranker import Reranker, reranker_from_settings
from retrieval.sparse_retriever import SparseRetriever
from retrieval.sufficiency import EvidenceSufficiencyAssessor
from experiments.runtime import record_stage

logger = logging.getLogger(__name__)


class HybridRetriever:
    """One retrieval engine with injected, independently testable stage interfaces."""

    def __init__(self, *, client=None, embedding_service: EmbeddingService | None = None,
                 embedding_provider: EmbeddingProvider | None = None,
                 dense_retriever: DenseRetriever | None = None,
                 sparse_retriever: SparseRetriever | None = None,
                 analyzer: QueryAnalyzer | None = None,
                 filter_builder: FilterBuilder | None = None,
                 fusion: FusionStrategy | None = None,
                 expander: ContextExpander | None = None,
                 reranker: Reranker | None = None,
                 selector: EvidenceSelector | None = None,
                 assessor: EvidenceSufficiencyAssessor | None = None,
                 cache: RetrievalCache[RetrievalResult] | None = None,
                 collection: str | None = None):
        self.client = client if client is not None else (get_qdrant_client() if any(
            item is None for item in (dense_retriever, sparse_retriever, expander)) else None)
        self.embedding_service = (embedding_service if embedding_service is not None else
                                  EmbeddingService(embedding_provider) if embedding_provider is not None else None)
        self.dense_retriever = dense_retriever or DenseRetriever(self.client, collection=collection)
        self.sparse_retriever = sparse_retriever or SparseRetriever(self.client, collection=collection)
        self.analyzer = analyzer or QueryAnalyzer()
        self.filter_builder = filter_builder or FilterBuilder()
        self.fusion = fusion or ReciprocalRankFusion()
        self.expander = expander or ContextExpander(self.client, collection=collection)
        self.reranker = reranker
        self.selector = selector or EvidenceSelector()
        self.assessor = assessor or EvidenceSufficiencyAssessor()
        self.cache = cache or RetrievalCache()
        self.collection = collection or get_settings().qdrant_collection

    async def retrieve(self, query: Query | str, *, scope: RetrievalScope,
                       policy: RetrievalPolicy | None = None) -> RetrievalResult:
        started = time.perf_counter()
        policy = policy or RetrievalPolicy()
        query_model = query if isinstance(query, Query) else Query(query_text=query)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        counts: dict[str, int] = {}

        stage = time.perf_counter()
        try:
            analysis = self.analyzer.analyze(query_model)
            applied_filters = analysis.extracted_filters
            qdrant_filter = self.filter_builder.build(applied_filters, scope)
        except Exception as exc:
            record_stage("query_processing", self._elapsed(stage), status="failure",
                         error_type=type(exc).__name__)
            raise
        timings["query_analysis"] = self._elapsed(stage)
        cache_status = "disabled"
        cache_key = self._cache_key(analysis.normalized_query, applied_filters, scope, policy)
        if policy.cache_ttl_seconds > 0:
            if scope.corpus_generation:
                cached = self.cache.get(cache_key, policy.cache_ttl_seconds)
                if cached is not None:
                    result = cached.model_copy(deep=True)
                    result.cache_status = "hit"
                    result.timings_ms["cache_lookup"] = self._elapsed(started)
                    result.timings_ms["total"] = self._elapsed(started)
                    return result
                cache_status = "miss"
            else:
                cache_status = "bypass_missing_corpus_generation"

        dense_candidates: list[RetrievalCandidate] = []
        sparse_candidates: list[RetrievalCandidate] = []
        dense_task: Any

        async def dense_search():
            call_started = time.perf_counter()
            try:
                result = await asyncio.to_thread(self.dense_retriever.search, query_vector,
                                                 qdrant_filter, policy.dense_limit)
            except Exception as exc:
                record_stage("dense_retrieval", self._elapsed(call_started), status="failure",
                             error_type=type(exc).__name__)
                raise
            record_stage("dense_retrieval", self._elapsed(call_started))
            return result

        # Query vector generation is shared and performed only once for the dense path.
        stage = time.perf_counter()
        query_vector = []
        if policy.dense_retrieval_enabled:
            self._assert_embedder_compatible(scope)
            try:
                if self.embedding_service is None:
                    self.embedding_service = EmbeddingService(provider_from_config())
                embedded = await asyncio.to_thread(self.embedding_service.embed,
                                                   [analysis.normalized_query],
                                                   cache_scope=self._embedding_cache_scope(scope))
                query_vector = embedded[0]
            except BudgetExceeded:
                raise
            except Exception as exc:
                warnings.append(f"dense_embedding_failed:{type(exc).__name__}")
        timings["query_embedding"] = self._elapsed(stage)
        if not policy.dense_retrieval_enabled:
            record_stage("embedding", timings["query_embedding"], status="not_applicable")
            record_stage("dense_retrieval", 0.0, status="not_applicable")
        if not policy.sparse_retrieval_enabled:
            record_stage("sparse_retrieval", 0.0, status="not_applicable")
        stage = time.perf_counter()
        local_qdrant = self.client is not None and type(getattr(self.client, "_client", None)).__name__ == "QdrantLocal"
        dense_task = asyncio.create_task(dense_search()) if query_vector and not local_qdrant else None
        sparse_call = lambda: asyncio.to_thread(
            self.sparse_retriever.search, analysis.normalized_query, qdrant_filter, policy.sparse_limit)
        async def sparse_search():
            call_started = time.perf_counter()
            try:
                result = await sparse_call()
            except Exception as exc:
                record_stage("sparse_retrieval", self._elapsed(call_started), status="failure",
                             error_type=type(exc).__name__)
                raise
            record_stage("sparse_retrieval", self._elapsed(call_started))
            return result
        # Qdrant's embedded in-memory client is not safe for concurrent access.
        # Production/server clients still run both independent retrieval paths concurrently.
        if local_qdrant:
            # Keep embedded client calls on the owning event-loop thread; its local
            # storage lock can otherwise deadlock when moved across worker threads.
            results = []
            if query_vector:
                call_started = time.perf_counter()
                try:
                    results.append(self.dense_retriever.search(query_vector, qdrant_filter, policy.dense_limit))
                    record_stage("dense_retrieval", self._elapsed(call_started))
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    results.append(exc)
                    record_stage("dense_retrieval", self._elapsed(call_started), status="failure",
                                 error_type=type(exc).__name__)
            if policy.sparse_retrieval_enabled:
                call_started = time.perf_counter()
                try:
                    results.append(self.sparse_retriever.search(
                        analysis.normalized_query, qdrant_filter, policy.sparse_limit))
                    record_stage("sparse_retrieval", self._elapsed(call_started))
                except Exception as exc:
                    results.append(exc)
                    record_stage("sparse_retrieval", self._elapsed(call_started), status="failure",
                                 error_type=type(exc).__name__)
        else:
            sparse_task = asyncio.create_task(sparse_search()) if policy.sparse_retrieval_enabled else None
            tasks = [task for task in (dense_task, sparse_task) if task is not None]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        timings["dense_sparse_retrieval"] = self._elapsed(stage)
        offset = 0
        if query_vector:
            value = results[offset]
            offset += 1
            if isinstance(value, BaseException):
                warnings.append(f"dense_retrieval_failed:{type(value).__name__}")
            else:
                dense_candidates = value
        if policy.sparse_retrieval_enabled:
            sparse_value = results[offset]
            if isinstance(sparse_value, BaseException):
                warnings.append(f"sparse_retrieval_failed:{type(sparse_value).__name__}")
            else:
                sparse_candidates = sparse_value
        counts["dense"] = len(dense_candidates)
        counts["sparse"] = len(sparse_candidates)

        stage = time.perf_counter()
        try:
            if policy.hybrid_fusion_enabled and policy.dense_retrieval_enabled and policy.sparse_retrieval_enabled:
                fused = self.fusion.fuse(dense_candidates, sparse_candidates, rrf_k=policy.rrf_k)
                for candidate in fused:
                    candidate.structure_score = StructuralScorer().score(analysis.normalized_query, candidate)
                    candidate.ranking_score = candidate.rrf_score + policy.structure_weight * candidate.structure_score
                fused.sort(key=lambda c: (-c.ranking_score, c.fusion_rank or 10**9,
                                         c.chunk.document_id, c.chunk.chunk_id))
            else:
                # No-fusion ablation preserves retrieval channel order (dense, then sparse).
                fused = [*dense_candidates, *sparse_candidates]
                for rank, candidate in enumerate(fused, start=1):
                    candidate.fusion_rank = rank
                    candidate.ranking_score = -float(rank)
        except Exception as exc:
            record_stage("fusion", self._elapsed(stage), status="failure", error_type=type(exc).__name__)
            raise
        counts["fused"] = len(fused)
        timings["fusion_and_structure"] = self._elapsed(stage)

        fused, duplicate_count = deduplicate(fused, policy.deduplicate_similarity)
        counts["deduplicated"] = len(fused)
        if duplicate_count:
            warnings.append(f"deduplicated_candidates:{duplicate_count}")

        stage = time.perf_counter()
        if policy.reranking_enabled and fused:
            reranker = self.reranker
            if reranker is None:
                try:
                    reranker = reranker_from_settings()
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    warnings.append(f"reranker_unavailable:{type(exc).__name__}")
            if reranker is not None:
                try:
                    target = fused[:policy.rerank_limit]
                    scores = await asyncio.to_thread(reranker.rerank, analysis.normalized_query, target)
                    by_id = {item.candidate_id: item.score for item in scores}
                    ordered_scores = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))
                    ranks = {candidate_id: rank for rank, (candidate_id, _) in enumerate(ordered_scores, start=1)}
                    for candidate in fused:
                        if candidate.identity in by_id:
                            candidate.reranker_score = by_id[candidate.identity]
                            candidate.reranker_rank = ranks[candidate.identity]
                    fused.sort(key=lambda c: (
                        c.reranker_rank is None,
                        c.reranker_rank if c.reranker_rank is not None else 10**9,
                        -c.ranking_score, c.chunk.document_id, c.chunk.chunk_id,
                    ))
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    warnings.append(f"reranker_failed_fallback_to_fusion:{type(exc).__name__}")
        timings["reranking"] = self._elapsed(stage)

        stage = time.perf_counter()
        expanded: list[RetrievalCandidate] = []
        expansion_tokens = 0
        if any((policy.expand_parents, policy.expand_children,
                policy.expand_neighbours, policy.expand_cross_references)):
            try:
                expanded, expansion_tokens = await asyncio.to_thread(
                    self.expander.expand, fused, qdrant_filter, policy)
            except BudgetExceeded:
                raise
            except Exception as exc:
                warnings.append(f"context_expansion_failed:{type(exc).__name__}")
        counts["expanded"] = len(expanded)
        timings["expansion"] = self._elapsed(stage)
        all_candidates = fused + expanded
        all_candidates, expansion_duplicates = deduplicate(all_candidates, policy.deduplicate_similarity)
        if expansion_duplicates:
            warnings.append(f"deduplicated_expansion_candidates:{expansion_duplicates}")

        stage = time.perf_counter()
        selection_policy = policy if policy.evidence_selection_enabled else policy.model_copy(
            update={"diversity_enabled": False})
        try:
            evidence, token_estimate, skipped_count = self.selector.select(all_candidates, selection_policy)
        except Exception as exc:
            record_stage("evidence_selection", self._elapsed(stage), status="failure",
                         error_type=type(exc).__name__)
            raise
        timings["evidence_selection"] = self._elapsed(stage)
        if skipped_count:
            warnings.append(f"evidence_over_budget_skipped:{skipped_count}")
        selected_ids = {item.chunk.chunk_id for item in evidence}
        selected_candidates = [candidate for candidate in all_candidates if candidate.identity in selected_ids]
        assessment = self.assessor.assess(selected_candidates)
        counts["selected"] = len(evidence)
        counts["expansion_tokens"] = expansion_tokens
        selected_candidate_ids = {candidate.identity for candidate in selected_candidates}
        inspection = [{
            "chunk_id": candidate.chunk.chunk_id,
            "document_id": candidate.chunk.document_id,
            "source": candidate.chunk.source,
            "section_path": candidate.chunk.section_path,
            "dense_rank": candidate.dense_rank,
            "dense_score": candidate.dense_score,
            "sparse_rank": candidate.sparse_rank,
            "sparse_score": candidate.sparse_score,
            "fusion_rank": candidate.fusion_rank,
            "rrf_score": candidate.rrf_score,
            "rrf_score_is_probability": False,
            "structure_score": candidate.structure_score,
            "reranker_rank": candidate.reranker_rank,
            "reranker_score": candidate.reranker_score,
            "expansion_reason": candidate.expansion_reason,
            "provenance": candidate.provenance,
            "selected": candidate.identity in selected_candidate_ids,
        } for candidate in all_candidates]

        authorization_scope_hash = self._scope_hash(scope)
        timings["total"] = self._elapsed(started)
        warning_names = tuple(warnings)
        trace_stages = {
            "query_processing": "query_analysis", "embedding": "query_embedding",
            "fusion": "fusion_and_structure", "reranking": "reranking",
            "evidence_selection": "evidence_selection",
        }
        for trace_name, timing_name in trace_stages.items():
            applicable = {
                "embedding": policy.dense_retrieval_enabled,
                "fusion": policy.hybrid_fusion_enabled and policy.dense_retrieval_enabled and policy.sparse_retrieval_enabled,
                "reranking": policy.reranking_enabled,
                "evidence_selection": True,
            }.get(trace_name, True)
            failed = any(name in warning for warning in warning_names for name in {
                "query_processing": ("query_analysis_failed",),
                "embedding": ("dense_embedding_failed",),
                "dense_retrieval": ("dense_retrieval_failed",),
                "sparse_retrieval": ("sparse_retrieval_failed",),
                "reranking": ("reranker_failed", "reranker_unavailable"),
            }.get(trace_name, ()))
            record_stage(trace_name, timings.get(timing_name),
                         status="not_applicable" if not applicable else "failure" if failed else "success",
                         metadata={"candidate_count": counts.get({"dense_retrieval": "dense",
                             "sparse_retrieval": "sparse", "fusion": "fused",
                             "evidence_selection": "selected"}.get(trace_name, "selected"), 0)})
        result = RetrievalResult(
            original_query=query_model.query_text,
            normalized_query=analysis.normalized_query,
            query_analysis=analysis,
            applied_filters=applied_filters,
            authorization_scope_hash=authorization_scope_hash,
            evidence=evidence,
            assessment=assessment,
            selected_token_estimate=token_estimate,
            retrieval_inspection=inspection,
            timings_ms=timings,
            candidate_counts=counts,
            cache_status=cache_status,
            warnings=warnings,
        )
        logger.info("Retrieval completed", extra={"candidate_counts": counts,
                                                  "selected_count": len(evidence),
                                                  "selected_token_estimate": token_estimate,
                                                  "cache_status": cache_status,
                                                  "assessment": assessment.state.value})
        if cache_status == "miss":
            self.cache.set(cache_key, result.model_copy(deep=True))
        return result

    def _cache_key(self, query: str, filters, scope: RetrievalScope,
                   policy: RetrievalPolicy) -> str:
        settings = get_settings()
        payload = {
            "query": query,
            "filters": filters.model_dump(mode="json"),
            "scope": {"principals": sorted(scope.principal_ids), "tenant": scope.tenant_id,
                      "workspace": scope.workspace_id, "corpus_generation": scope.corpus_generation},
            "policy": policy.model_dump(mode="json"),
            "collection": self.collection,
            "dense_vector": settings.dense_vector_name,
            "sparse_vector": settings.sparse_vector_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model_id,
            "embedding_dimensions": settings.embedding_dimensions,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _embedding_cache_scope(self, scope: RetrievalScope) -> str | None:
        if not scope.corpus_generation:
            return None
        settings = get_settings()
        payload = {
            "tenant": scope.tenant_id,
            "principals": sorted(scope.principal_ids),
            "workspace": scope.workspace_id,
            "corpus_generation": scope.corpus_generation,
            "collection": self.collection,
            "provider": settings.embedding_provider,
            "model": settings.embedding_model_id,
            "dimensions": settings.embedding_dimensions,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _assert_embedder_compatible(self, scope: RetrievalScope) -> None:
        if self.client is None:
            return
        settings = get_settings()
        provider, model, dimensions = configured_embedder(settings)
        if dimensions is None and self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            named = vectors.get(settings.dense_vector_name) if isinstance(vectors, dict) else None
            dimensions = named.size if named is not None else None
        assert_collection_embedder_compatible(
            self.client, self.collection, provider=provider, model=model, dimensions=dimensions,
            allow_override=settings.embedding_provenance_override,
        )

    @staticmethod
    def _scope_hash(scope: RetrievalScope) -> str:
        payload = {"principals": sorted(scope.principal_ids), "tenant": scope.tenant_id,
                   "workspace": scope.workspace_id, "corpus_generation": scope.corpus_generation}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _elapsed(start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 3)
