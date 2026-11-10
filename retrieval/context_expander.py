"""One-hop, ACL-filtered structural and cross-reference context expansion."""

from qdrant_client import models

from config.settings import get_settings
from ingestion.chunking.structure_chunker import estimate_tokens
from retrieval.contracts import RetrievalCandidate, RetrievalPolicy
from retrieval.qdrant_utils import candidate_from_point


class ContextExpander:
    def __init__(self, client=None, *, collection: str | None = None):
        settings = get_settings()
        if client is None:
            from infrastructure.qdrant_client import get_qdrant_client
            client = get_qdrant_client()
        self.client = client
        self.collection = collection or settings.qdrant_collection

    def expand(self, seeds: list[RetrievalCandidate], base_filter: models.Filter,
               policy: RetrievalPolicy) -> tuple[list[RetrievalCandidate], int]:
        if policy.expansion_max_count == 0 or policy.expansion_token_budget == 0:
            return [], 0
        existing = {candidate.identity for candidate in seeds}
        expanded: list[RetrievalCandidate] = []
        used_tokens = 0
        operations = ("parent", "child", "neighbour", "cross_reference")
        enabled = {"parent": policy.expand_parents, "child": policy.expand_children,
                   "neighbour": policy.expand_neighbours,
                   "cross_reference": policy.expand_cross_references}
        for seed in seeds[:policy.expansion_seed_limit]:
            for operation in operations:
                if len(expanded) >= policy.expansion_max_count or used_tokens >= policy.expansion_token_budget:
                    return expanded, used_tokens
                if not enabled[operation]:
                    continue
                points = self._fetch(seed, operation, base_filter, policy)
                for point in points:
                    candidate = candidate_from_point(point, channel="expansion", rank=0)
                    if candidate.identity in existing:
                        continue
                    size = estimate_tokens(candidate.chunk.content)
                    if used_tokens + size > policy.expansion_token_budget:
                        continue
                    candidate.expansion_reason = operation
                    candidate.ranking_score = 0.0
                    candidate.provenance = [f"expanded:{operation}"]
                    existing.add(candidate.identity)
                    expanded.append(candidate)
                    used_tokens += size
                    if len(expanded) >= policy.expansion_max_count:
                        break
        return expanded, used_tokens

    def _fetch(self, seed: RetrievalCandidate, operation: str,
               base_filter: models.Filter, policy: RetrievalPolicy):
        c = seed.chunk
        must = list(base_filter.must or [])
        must_not = list(base_filter.must_not or [])
        should = list(base_filter.should or [])
        min_should = base_filter.min_should
        if operation == "parent":
            if not c.parent_id or c.parent_id == c.document_id:
                return []
            must.extend([
                models.FieldCondition(key="document_id", match=models.MatchValue(value=c.document_id)),
                models.FieldCondition(key="metadata.block_ids", match=models.MatchValue(value=c.parent_id)),
            ])
        elif operation == "child":
            block_ids = c.metadata.get("block_ids", [])
            if not block_ids:
                return []
            must.extend([
                models.FieldCondition(key="document_id", match=models.MatchValue(value=c.document_id)),
                models.FieldCondition(key="parent_id", match=models.MatchAny(any=block_ids)),
            ])
        elif operation == "neighbour":
            chunk_index = c.metadata.get("chunk_index")
            if not isinstance(chunk_index, int):
                return []
            indices = [chunk_index - policy.neighbour_distance, chunk_index + policy.neighbour_distance]
            indices = [index for index in indices if index >= 0]
            if not indices:
                return []
            must.extend([
                models.FieldCondition(key="document_id", match=models.MatchValue(value=c.document_id)),
                models.FieldCondition(key="metadata.chunk_index", match=models.MatchAny(any=indices)),
            ])
        elif operation == "cross_reference":
            links = c.links[:policy.expansion_max_count]
            if not links:
                return []
            should.extend([
                models.FieldCondition(key="chunk_id", match=models.MatchAny(any=links)),
                models.FieldCondition(key="source", match=models.MatchAny(any=links)),
            ])
            min_should = models.MinShould(conditions=should, min_count=1)
        query_filter = models.Filter(must=must, must_not=must_not, should=should or None,
                                     min_should=min_should)
        records, _ = self.client.scroll(
            collection_name=self.collection, scroll_filter=query_filter,
            limit=min(policy.expansion_max_count, 16), with_payload=True, with_vectors=False,
        )
        return records
