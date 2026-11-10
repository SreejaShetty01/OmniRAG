"""Deterministic offline scoring of safe Phase 8 benchmark artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evaluation.phase8_runner import FAILURE_CODE_PREFIXES

SCORER_VERSION = "phase8-offline-2"
TERMINAL_FAILURES = {"failed", "timed_out", "indeterminate_after_interruption"}


def _dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def _expected_chunks(case: dict[str, Any], snapshot: dict[str, Any]) -> set[str]:
    mapped: set[str] = set()
    wanted = {(item["source_document"], item["locator"]) for item in case.get("expected_evidence", [])}
    for entry in snapshot.get("evidence_to_chunks", []):
        if (entry.get("case_id") == case.get("case_id") and
                (entry.get("source_document"), entry.get("locator")) in wanted):
            mapped.update(str(item["chunk_id"]) for item in entry.get("indexed_chunks", [])
                          if item.get("chunk_id"))
    return mapped


def _score_unit(case: dict[str, Any], scheduled: dict[str, Any],
                artifact: dict[str, Any] | None, snapshot: dict[str, Any]) -> dict[str, Any]:
    status = scheduled.get("status", "missing")
    expected = _expected_chunks(case, snapshot)
    metric_expected = expected
    if case.get("category") == "unsupported" or (
            case.get("category") == "acl_tenant_isolation" and
            (case.get("acl_context") or {}).get("expected_access") == "deny"):
        # These cases are evaluated through abstention/authorization outcomes;
        # their expected evidence is either intentionally inaccessible or not
        # a positive relevance label for an answerable retrieval task.
        metric_expected = set()
    if artifact is None:
        return {"case_id": case["case_id"], "profile": scheduled.get("profile"),
                "repetition": scheduled.get("repetition"), "status": status,
                "execution_success": False, "expected_chunk_count": len(expected),
                **{f"recall_at_{k}": None for k in (5, 10, 20, 50)},
                **{f"precision_at_{k}": None for k in (5, 10, 20, 50)},
                "retrieval_recall_at_10": None, "mrr_at_10": None, "ndcg_at_10": None,
                "evidence_precision": None, "citation_precision": None,
                "citation_recall": None, "exact_answer_match": None,
                "abstention_correct": None, "acl_audit_pass": None,
                "unsupported_claim_count": None, "unsupported_claim_rate": None,
                "supported_claim_rate": None, "partially_supported_claim_rate": None,
                "latency_ms": None, "input_tokens": None, "output_tokens": None,
                "cost_usd": None}
    raw_codes = artifact.get("warning_codes") or artifact.get("failure_codes", [])
    actual_failures = sorted(code for code in raw_codes if code in FAILURE_CODE_PREFIXES)
    if status in {"completed", "completed_with_warnings", "completed_with_errors"}:
        status = "completed_with_errors" if actual_failures else (
            "completed_with_warnings" if raw_codes else "completed")
    ranked = [str(item.get("chunk_id")) for item in artifact.get("retrieval_candidates", [])
              if item.get("chunk_id")]
    evidence_ids = [str(item.get("chunk_id")) for item in artifact.get("evidence_references", [])
                    if item.get("chunk_id")]
    citation_ids = [str(item.get("chunk_id")) for item in artifact.get("citations", [])
                    if item.get("chunk_id")]
    hits = [int(chunk_id in metric_expected) for chunk_id in ranked[:10]]
    first_rank = next((index for index, hit in enumerate(hits, 1) if hit), None)
    ideal = [1] * min(len(metric_expected), 10)
    ndcg = _dcg(hits) / _dcg(ideal) if ideal else None
    recall = len(set(ranked[:10]) & metric_expected) / len(metric_expected) if metric_expected else None
    retrieval_metrics = {}
    for k in (5, 10, 20, 50):
        top_k = ranked[:k]
        retrieval_metrics[f"recall_at_{k}"] = (
            len(set(top_k) & metric_expected) / len(metric_expected) if metric_expected else None)
        retrieval_metrics[f"precision_at_{k}"] = (
            len(set(top_k) & metric_expected) / k if metric_expected else None)
        retrieval_metrics[f"hit_rate_at_{k}"] = (
            float(bool(set(top_k) & metric_expected)) if metric_expected else None)
    evidence_precision = (len(set(evidence_ids) & metric_expected) / len(set(evidence_ids))
                          if evidence_ids and metric_expected else None)
    citation_precision = (len(set(citation_ids) & metric_expected) / len(set(citation_ids))
                          if citation_ids and metric_expected else None)
    citation_recall = (len(set(citation_ids) & metric_expected) / len(metric_expected)
                       if metric_expected else None)
    exact = artifact.get("expected_answer_match") if case.get("answer_match") == "exact" else None
    abstention = (bool(artifact.get("abstained")) == bool(case["expected_abstention"])
                  if case.get("expected_abstention") is not None else None)
    auth = artifact.get("authorization_audit") or {}
    points = {str(point.get("chunk_id")): point for point in snapshot.get("points", [])
              if point.get("chunk_id")}
    acl_context = case.get("acl_context") or {}
    tenant_context = case.get("tenant_context") or {}
    principal = acl_context.get("request_principal", "user:phase8-evaluation-user")
    tenant = tenant_context.get("request_tenant_id", "phase8-evaluation-tenant")
    posthoc_violation = bool(auth.get("violation") or auth.get("unknown_evidence_mapping"))
    for item in [*artifact.get("retrieval_candidates", []), *artifact.get("evidence_references", []),
                 *artifact.get("citations", [])]:
        point = points.get(str(item.get("chunk_id")))
        if point is None or point.get("document_id") != item.get("document_id"):
            posthoc_violation = True
            continue
        allowed = point.get("acl_allowed_principals") or []
        denied = point.get("acl_denied_principals") or []
        if (point.get("tenant_id") != tenant or principal in denied or
                (allowed and principal not in allowed)):
            posthoc_violation = True
    denied_case = (case.get("category") == "acl_tenant_isolation" and
                   acl_context.get("expected_access") == "deny")
    denied_expected_found = bool(set(ranked) & expected or set(evidence_ids) & expected or
                                 set(citation_ids) & expected)
    if denied_case:
        acl_pass = (not posthoc_violation and not denied_expected_found and
                    not auth.get("restricted_marker_leaked"))
    elif case.get("category") == "acl_tenant_isolation":
        acl_pass = not posthoc_violation
    else:
        acl_pass = None
    claim_counts = artifact.get("claim_status_counts") or {}
    claim_total = sum(value for value in claim_counts.values() if isinstance(value, int))
    unsupported_count = claim_counts.get("unsupported", 0)
    supported_rate = (claim_counts.get("supported", 0) / claim_total if claim_total else None)
    unsupported_rate = (unsupported_count / claim_total if claim_total else None)
    partial_rate = (claim_counts.get("partially_supported", 0) / claim_total if claim_total else None)
    tokens = artifact.get("token_usage") or {}
    return {
        "case_id": case["case_id"], "profile": scheduled.get("profile"),
        "repetition": scheduled.get("repetition"), "status": status,
        "execution_success": status in {"completed", "completed_with_warnings", "completed_with_errors"},
        "failure_codes": actual_failures,
        "warning_codes": sorted(set(raw_codes)),
        "expected_chunk_count": len(expected),
        **retrieval_metrics,
        "retrieval_recall_at_10": recall,
        "evidence_coverage": (len(set(evidence_ids) & metric_expected) / len(metric_expected)
                              if metric_expected else None),
        "mrr_at_10": (1.0 / first_rank if first_rank else 0.0) if metric_expected else None,
        "ndcg_at_10": ndcg,
        "evidence_precision": evidence_precision,
        "citation_precision": citation_precision, "citation_recall": citation_recall,
        "exact_answer_match": exact, "abstention_correct": abstention,
        "acl_audit_pass": acl_pass,
        "claim_count": claim_total,
        "unsupported_claim_count": unsupported_count,
        "unsupported_claim_rate": unsupported_rate,
        "supported_claim_rate": supported_rate,
        "partially_supported_claim_rate": partial_rate,
        "latency_ms": artifact.get("elapsed_ms"),
        "input_tokens": tokens.get("input_tokens"), "output_tokens": tokens.get("output_tokens"),
        "cost_usd": artifact.get("cost_usd"),
    }


def score_artifacts(dataset: dict[str, Any], snapshot: dict[str, Any], progress: dict[str, Any],
                    attempts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Score scheduled units without network access or raw answer reconstruction."""
    cases = {case["case_id"]: case for case in dataset["cases"]}
    if progress.get("warmup"):
        counts: dict[str, int] = {}
        for scheduled in progress.get("schedule", []):
            status = scheduled.get("status", "missing")
            counts[status] = counts.get(status, 0) + 1
        return {"scorer_version": SCORER_VERSION, "benchmark_id": progress.get("benchmark_id"),
                "dataset": progress.get("dataset"), "snapshot_sha256": progress.get("snapshot_sha256"),
                "warmup": True, "summary": {"scheduled_units": 0,
                                               "warmup_units_excluded": len(progress.get("schedule", [])),
                                               "warmup_status_counts": counts},
                "units": []}
    rows = []
    for scheduled in progress.get("schedule", []):
        case = cases.get(scheduled.get("case_id"))
        if case is None:
            raise ValueError(f"Scheduled case is absent from dataset: {scheduled.get('case_id')}")
        rows.append(_score_unit(case, scheduled,
                                attempts.get(scheduled.get("attempt_key", "")), snapshot))
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1

    def rate(key: str) -> dict[str, int | float | None]:
        eligible = [row[key] for row in rows if isinstance(row.get(key), bool)]
        return {"numerator": sum(eligible), "denominator": len(eligible),
                "rate": sum(eligible) / len(eligible) if eligible else None}

    latency = sorted(row["latency_ms"] for row in rows if isinstance(row.get("latency_ms"), (int, float)))
    summary = {
        "scheduled_units": len(rows), "execution_status_counts": statuses,
        "execution_success_rate": (sum(row["execution_success"] for row in rows) / len(rows)
                                   if rows else None),
        "exact_answer_match": rate("exact_answer_match"),
        "abstention_correct": rate("abstention_correct"),
        "acl_audit_pass": rate("acl_audit_pass"),
        "latency_p50_ms": latency[(len(latency) - 1) // 2] if latency else None,
        "latency_p95_ms": latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
        "token_counts_available": sum(row["input_tokens"] is not None and
                                       row["output_tokens"] is not None for row in rows),
        "cost_values_available": sum(row["cost_usd"] is not None for row in rows),
    }
    return {"scorer_version": SCORER_VERSION,
            "benchmark_id": progress.get("benchmark_id"),
            "dataset": progress.get("dataset"), "snapshot_sha256": progress.get("snapshot_sha256"),
            "summary": summary, "units": rows}


def score_directory(dataset_path: str | Path, snapshot_path: str | Path,
                    benchmark_dir: str | Path) -> dict[str, Any]:
    root = Path(benchmark_dir)
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    attempts = {}
    for path in sorted((root / "attempts").glob("*.json")):
        attempts[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return score_artifacts(json.loads(Path(dataset_path).read_text(encoding="utf-8")),
                           json.loads(Path(snapshot_path).read_text(encoding="utf-8")),
                           progress, attempts)
