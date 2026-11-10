"""Shared Qdrant candidate conversion and retrieval errors."""

from qdrant_client import models

from models.chunk import Chunk
from retrieval.contracts import RetrievalCandidate


class RetrievalError(RuntimeError):
    """Safe, caller-facing retrieval operation failure."""


def candidate_from_point(point, *, channel: str, rank: int) -> RetrievalCandidate:
    payload = point.payload or {}
    try:
        chunk = Chunk.model_validate(payload)
    except Exception as exc:
        raise RetrievalError("Qdrant returned a point without a valid OmniRAG chunk payload") from exc
    candidate = RetrievalCandidate(chunk=chunk, provenance=[channel])
    score = getattr(point, "score", None)
    if channel == "dense":
        candidate.dense_rank = rank
        candidate.dense_score = float(score) if score is not None else None
    elif channel == "sparse":
        candidate.sparse_rank = rank
        candidate.sparse_score = float(score) if score is not None else None
    return candidate


def make_query_filter(base_filter: models.Filter, **conditions) -> models.Filter:
    must = list(base_filter.must or [])
    must_not = list(base_filter.must_not or [])
    should = list(base_filter.should or [])
    min_should = base_filter.min_should
    for key, value in conditions.items():
        if value is None:
            continue
        if isinstance(value, list):
            if value:
                must.append(models.FieldCondition(key=key, match=models.MatchAny(any=value)))
        else:
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must, must_not=must_not, should=should, min_should=min_should)
