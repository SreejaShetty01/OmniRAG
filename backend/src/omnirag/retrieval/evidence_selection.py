"""Diversity-aware, token-budgeted evidence selection without destructive truncation."""

from omnirag.ingestion.chunking.structure_chunker import estimate_tokens
from omnirag.retrieval.contracts import RetrievalCandidate, RetrievalEvidence, RetrievalPolicy


class EvidenceSelector:
    def select(self, candidates: list[RetrievalCandidate], policy: RetrievalPolicy
               ) -> tuple[list[RetrievalEvidence], int, int]:
        ordered = sorted(candidates, key=lambda c: (
            c.expansion_reason is not None,
            c.reranker_rank is None,
            c.reranker_rank if c.reranker_rank is not None else 10**9,
            -c.ranking_score,
            min(c.dense_rank or 10**9, c.sparse_rank or 10**9),
            c.chunk.document_id,
            c.chunk.chunk_id,
        ))
        selected: list[RetrievalEvidence] = []
        selected_candidates: set[str] = set()
        token_count = skipped = 0
        seen_documents: set[str] = set()
        seen_sections: set[tuple[str, ...]] = set()
        remaining = list(ordered)
        while remaining and len(selected) < policy.max_evidence:
            index = self._choose_next(remaining, seen_documents, seen_sections, policy.diversity_enabled)
            candidate = remaining.pop(index)
            if candidate.identity in selected_candidates:
                continue
            size = estimate_tokens(candidate.chunk.content)
            if token_count + size > policy.evidence_token_budget:
                skipped += 1
                continue
            selected_candidates.add(candidate.identity)
            token_count += size
            seen_documents.add(candidate.chunk.document_id)
            seen_sections.add((candidate.chunk.document_id, *candidate.chunk.section_path))
            selected.append(RetrievalEvidence(
                chunk=candidate.chunk, dense_rank=candidate.dense_rank, dense_score=candidate.dense_score,
                sparse_rank=candidate.sparse_rank, sparse_score=candidate.sparse_score,
                rrf_score=candidate.rrf_score if candidate.rrf_score else None,
                rrf_dense_contribution=candidate.rrf_dense_contribution,
                rrf_sparse_contribution=candidate.rrf_sparse_contribution,
                structure_score=candidate.structure_score, ranking_score=candidate.ranking_score,
                reranker_score=candidate.reranker_score, expansion_reason=candidate.expansion_reason,
                fusion_rank=candidate.fusion_rank, reranker_rank=candidate.reranker_rank,
                provenance=candidate.provenance.copy(), estimated_tokens=size,
            ))
        return selected, token_count, skipped

    @staticmethod
    def _choose_next(candidates: list[RetrievalCandidate], documents: set[str],
                     sections: set[tuple[str, ...]], diversity: bool) -> int:
        if not diversity:
            return 0
        for index, candidate in enumerate(candidates):
            doc_new = candidate.chunk.document_id not in documents
            section_key = (candidate.chunk.document_id, *candidate.chunk.section_path)
            section_new = section_key not in sections
            if doc_new or section_new:
                return index
        return 0
