"""Uncalibrated evidence sufficiency signals for downstream generation policy."""

from collections.abc import Sequence

from omnirag.retrieval.contracts import EvidenceAssessment, EvidenceAssessmentState, RetrievalCandidate


class EvidenceSufficiencyAssessor:
    def __init__(self, *, strong_agreement_min: int = 2, strong_source_min: int = 2):
        if strong_agreement_min < 1 or strong_source_min < 1:
            raise ValueError("Sufficiency signal thresholds must be positive")
        self.strong_agreement_min = strong_agreement_min
        self.strong_source_min = strong_source_min

    def assess(self, candidates: Sequence[RetrievalCandidate], *, conflict_signal: bool = False) -> EvidenceAssessment:
        if conflict_signal:
            return EvidenceAssessment(
                state=EvidenceAssessmentState.CONFLICTING, calibrated=False,
                signals={"conflict_signal": True, "candidate_count": len(candidates)},
                rationale=["A conflict signal was supplied by an upstream evidence comparison component."],
            )
        if not candidates:
            return EvidenceAssessment(
                state=EvidenceAssessmentState.INSUFFICIENT, calibrated=False,
                signals={"candidate_count": 0, "channel_agreement_count": 0},
                rationale=["No evidence survived retrieval and evidence selection."],
            )
        agreement_count = sum(c.dense_rank is not None and c.sparse_rank is not None for c in candidates)
        source_count = len({c.chunk.document_id for c in candidates})
        signals: dict[str, float | int | bool | str] = {
            "candidate_count": len(candidates),
            "channel_agreement_count": agreement_count,
            "distinct_document_count": source_count,
            "mean_rrf_score": sum(c.rrf_score for c in candidates) / len(candidates),
            "rrf_scores_are_probabilities": False,
        }
        if agreement_count >= self.strong_agreement_min and source_count >= self.strong_source_min:
            state = EvidenceAssessmentState.STRONG
            rationale = ["Dense and sparse retrieval agree across the configured number of distinct sources."]
        elif agreement_count:
            state = EvidenceAssessmentState.PARTIAL
            rationale = ["At least one selected item was independently found by dense and sparse retrieval."]
        else:
            state = EvidenceAssessmentState.WEAK
            rationale = ["Evidence was retrieved, but selected evidence has no dense/sparse rank agreement."]
        rationale.append("This heuristic is uncalibrated and is not a probability of answer correctness.")
        return EvidenceAssessment(state=state, calibrated=False, signals=signals, rationale=rationale)
