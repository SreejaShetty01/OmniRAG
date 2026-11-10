"""Reproducible metrics and runner for separately maintained retrieval gold sets."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from retrieval.contracts import RetrievalPolicy, RetrievalResult, RetrievalScope


@dataclass(frozen=True, slots=True)
class RetrievalGoldCase:
    case_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    relevant_sections: frozenset[tuple[str, ...]] = frozenset()
    scope: RetrievalScope = field(default_factory=lambda: RetrievalScope(
        principal_ids=("benchmark",), tenant_id="benchmark-tenant"))


@dataclass(frozen=True, slots=True)
class BenchmarkVariant:
    name: str
    policy: RetrievalPolicy
    retrieve: Callable[[str, RetrievalScope, RetrievalPolicy], RetrievalResult]


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def case_metrics(case: RetrievalGoldCase, result: RetrievalResult) -> dict[str, float | int]:
    ids = [item.chunk.chunk_id for item in result.evidence]
    relevant = case.relevant_chunk_ids
    metrics: dict[str, float | int] = {}
    for k in (5, 10, 20, 50):
        top_k = set(ids[:k])
        metrics[f"recall@{k}"] = (len(top_k & relevant) / len(relevant)) if relevant else 0.0
        metrics[f"precision@{k}"] = len(top_k & relevant) / k
        metrics[f"hit_rate@{k}"] = float(bool(top_k & relevant))
    reciprocal = next((1.0 / rank for rank, chunk_id in enumerate(ids[:10], 1)
                       if chunk_id in relevant), 0.0)
    metrics["mrr@10"] = reciprocal
    gains = [int(chunk_id in relevant) for chunk_id in ids[:10]]
    ideal = [1] * min(len(relevant), 10)
    metrics["ndcg@10"] = _dcg(gains) / _dcg(ideal) if ideal else 0.0
    section_hits = [tuple(item.chunk.section_path) in case.relevant_sections for item in result.evidence]
    for k in (1, 3):
        metrics[f"section_recall@{k}"] = float(any(section_hits[:k])) if case.relevant_sections else 0.0
    selected = set(ids)
    metrics["evidence_precision"] = len(selected & relevant) / len(selected) if selected else 0.0
    metrics["evidence_coverage"] = len(selected & relevant) / len(relevant) if relevant else 0.0
    metrics["duplicate_evidence_rate"] = (len(ids) - len(set(ids))) / len(ids) if ids else 0.0
    metrics["context_tokens"] = result.selected_token_estimate
    metrics["latency_ms"] = result.timings_ms.get("total", 0.0)
    metrics["candidate_count"] = sum(result.candidate_counts.values())
    metrics["reranking_overhead_ms"] = result.timings_ms.get("reranking", 0.0)
    if result.cost_usd is not None:
        metrics["cost_usd"] = result.cost_usd
    return metrics


def aggregate_metrics(rows: Iterable[Mapping[str, float | int]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            grouped.setdefault(key, []).append(float(value))
    aggregated: dict[str, float] = {}
    for key, values in grouped.items():
        if key == "latency_ms":
            ordered = sorted(values)
            aggregated["latency_p50_ms"] = statistics.median(ordered)
            aggregated["latency_p95_ms"] = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
        else:
            aggregated[key] = statistics.mean(values)
    return aggregated


def run_benchmark(cases: Sequence[RetrievalGoldCase], variants: Sequence[BenchmarkVariant]
                  ) -> dict[str, dict[str, float]]:
    """Run injected shared-engine configurations; does not invent or load gold data."""
    reports: dict[str, dict[str, float]] = {}
    for variant in variants:
        per_case = []
        for case in cases:
            result = variant.retrieve(case.query, case.scope, variant.policy)
            per_case.append(case_metrics(case, result))
        reports[variant.name] = aggregate_metrics(per_case)
    return reports
