"""Independent diagnostic evaluation for the Phase 8 verified dataset.

Dry run is network-free. Live requests require an explicit ``--execute`` flag;
this task intentionally runs only ``--dry-run``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import os
import platform
import subprocess
from pathlib import Path
import statistics
import time
from typing import Any
from uuid import uuid4
from urllib.error import URLError

from evaluation.dataset import EvaluationDataset, EvaluationCase, CORPUS_RELATIVE_ROOT, validate_dataset
from evaluation.phase8_scorer import _expected_chunks, score_artifacts
from evaluation.phase8_runner import _collect_ids, _identity_for, sign_identity, _http_post, _http_get

ROOT = Path(__file__).resolve().parents[2]
PINNED_DATASET_HASH_PREFIX = "7b44b8f6"
LOCAL_TOKEN_CAP = 400_000
LOCAL_CALL_CAP = 300
REQUEST_TIMEOUT_SECONDS = 150
PROFILES = ("lite", "core")
METRICS = ("recall_at_5", "recall_at_10", "precision_at_5", "hit_rate_at_5",
           "hit_rate_at_10", "mrr_at_10", "ndcg_at_10")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot_integrity(snapshot: dict[str, Any]) -> str:
    """Validate snapshot self-consistency without enforcing Phase 8 dataset pins."""
    core_keys = ("dataset_id", "dataset_version", "dataset_sha256", "corpus_id", "corpus_version",
                 "collection", "vector_schema", "required_payload_indexes", "documents", "points",
                 "evidence_to_chunks", "known_exclusion")
    if not isinstance(snapshot, dict) or any(key not in snapshot for key in core_keys):
        raise ValueError("Diagnostic snapshot lacks required evidence mapping fields")
    if not str(snapshot.get("collection", "")).startswith("omnirag_eval_"):
        raise ValueError("Diagnostic snapshot must reference an isolated evaluation collection")
    core = {key: snapshot[key] for key in core_keys}
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != snapshot.get("snapshot_sha256"):
        raise ValueError("Diagnostic snapshot self-checksum mismatch")
    if snapshot.get("point_count") != len(snapshot.get("points", [])):
        raise ValueError("Snapshot point count mismatch")
    if snapshot.get("document_count") != len(snapshot.get("documents", [])):
        raise ValueError("Snapshot document count mismatch")
    return actual


def diagnostic_stamp(dataset_sha256: str) -> str:
    return (f"DIAGNOSTIC — dataset hash {dataset_sha256} != pinned "
            f"{PINNED_DATASET_HASH_PREFIX}…")


def classify_http_outcome(http_status: int, elapsed_ms: float) -> tuple[str, bool]:
    """Keep HTTP completion separate from the request latency budget."""
    timed_out = http_status in {408, 504}
    status = "timed_out" if timed_out else "completed" if 200 <= http_status < 300 else "failed"
    return status, elapsed_ms >= REQUEST_TIMEOUT_SECONDS * 1000


def redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, list):
        return [redact_secret(item, secret) for item in value]
    if isinstance(value, dict):
        return {key: redact_secret(item, secret) for key, item in value.items()}
    return value


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054
                    ) -> list[float] | None:
    if trials <= 0:
        return None
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(relevances, 1))


def retrieval_metric_values(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float | None]:
    if not relevant_ids:
        return {metric: None for metric in METRICS}
    output: dict[str, float | None] = {}
    for k in (5, 10):
        top = ranked_ids[:k]
        hits = len(set(top) & relevant_ids)
        output[f"recall_at_{k}"] = hits / len(relevant_ids)
        if k == 5:
            output["precision_at_5"] = hits / k
        output[f"hit_rate_at_{k}"] = float(hits > 0)
    first = next((rank for rank, chunk_id in enumerate(ranked_ids[:10], 1)
                  if chunk_id in relevant_ids), None)
    output["mrr_at_10"] = 1 / first if first else 0.0
    gains = [int(chunk_id in relevant_ids) for chunk_id in ranked_ids[:10]]
    ideal = [1] * min(len(relevant_ids), 10)
    output["ndcg_at_10"] = _dcg(gains) / _dcg(ideal) if ideal else None
    return output


def _ranked_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    _collect_ids(response.get("retrieval_inspection", []), items)
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique.setdefault(item["chunk_id"], item)
    return sorted(unique.values(), key=lambda item: (
        item.get("rank") if isinstance(item.get("rank"), int) else 10**9,
        item.get("chunk_id", "")))


def _reference_ids(items: Any) -> list[str]:
    found: list[dict[str, Any]] = []
    _collect_ids(items if isinstance(items, (dict, list)) else [], found)
    return list(dict.fromkeys(item["chunk_id"] for item in found))


def _restricted_ids(snapshot: dict[str, Any], case: EvaluationCase) -> set[str]:
    principal_user, tenant = _identity_for(case)
    principal = f"user:{principal_user}"
    restricted = set()
    for point in snapshot.get("points", []):
        allowed = point.get("acl_allowed_principals") or []
        denied = point.get("acl_denied_principals") or []
        if (point.get("tenant_id") != tenant or principal in denied or
                (allowed and principal not in allowed)):
            if point.get("chunk_id"):
                restricted.add(str(point["chunk_id"]))
    return restricted


def _source_authorized(snapshot: dict[str, Any], case: EvaluationCase,
                       source_document: str) -> bool | None:
    """Return source ACL decision for this case identity, or None if unmapped."""
    principal_user, tenant = _identity_for(case)
    principal = f"user:{principal_user}"
    documents = [row for row in snapshot.get("documents", [])
                 if row.get("filename") == source_document]
    if not documents:
        return None
    doc = documents[0]
    allowed = doc.get("acl_allowed_principals") or []
    denied = doc.get("acl_denied_principals") or []
    return (doc.get("tenant_id") == tenant and principal not in denied and
            (not allowed or principal in allowed))


def _future_run_provenance(dataset_hash: str, snapshot: dict[str, Any], runtime_settings: dict[str, Any],
                           base_url: str, api_container_name: str | None) -> dict[str, Any]:
    """Capture non-secret identity/configuration for new runs; never retrofit old artifacts."""
    def git_value(*args: str) -> str | None:
        try:
            value = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                                   text=True, timeout=5).stdout.strip()
            return value or None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain", "--untracked-files=all")
    diff_hash = None
    try:
        diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True,
                              capture_output=True, timeout=20).stdout
        diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    image_id = None
    repo_digests = None
    if api_container_name:
        try:
            inspected = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", api_container_name],
                                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=10)
            image_id = inspected.stdout.strip() or None
            if image_id:
                image_info = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}",
                                             image_id], cwd=ROOT, check=True, capture_output=True,
                                            text=True, timeout=10)
                repo_digests = json.loads(image_info.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        except (TypeError, ValueError):
            repo_digests = None
    try:
        import psutil
        total_memory = psutil.virtual_memory().total
    except (ImportError, AttributeError):
        total_memory = None
    embedding = runtime_settings.get("embedding_space") or {}
    models = runtime_settings.get("models") or {}
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "tracked_diff_sha256": diff_hash,
        "docker_image_id": image_id,
        "docker_image_repo_digests": repo_digests,
        "docker_container_name": api_container_name,
        "dataset_sha256": dataset_hash,
        "corpus_snapshot_sha256": snapshot.get("snapshot_sha256"),
        "qdrant_collection": snapshot.get("collection"),
        "qdrant_point_count": snapshot.get("point_count"),
        "qdrant_point_count_source": "pinned_corpus_snapshot_manifest; live count not queried",
        "embedding": {key: embedding.get(key) for key in ("alias", "provider", "model_id", "model_digest", "dimensions")},
        "generation_model": {key: (models.get("generation") or {}).get(key)
                             for key in ("provider", "model_id", "configured")},
        "runtime_limits": runtime_settings.get("limits"),
        "base_url": base_url,
        "hardware": {"platform": platform.platform(), "processor": platform.processor() or None,
                     "cpu_count": os.cpu_count(), "total_ram_bytes": total_memory},
    }


def score_response(case: EvaluationCase, response: dict[str, Any], snapshot: dict[str, Any],
                   elapsed_ms: float | None = None) -> dict[str, Any]:
    ranked = _ranked_items(response)
    ranked_ids = [item["chunk_id"] for item in ranked]
    stage_candidates = {
        "dense": [item["chunk_id"] for item in sorted(
            (row for row in ranked if isinstance(row.get("dense_rank"), int)),
            key=lambda row: row["dense_rank"])],
        "sparse": [item["chunk_id"] for item in sorted(
            (row for row in ranked if isinstance(row.get("sparse_rank"), int)),
            key=lambda row: row["sparse_rank"])],
        "fusion": ranked_ids,
        "reranked": [item["chunk_id"] for item in sorted(
            (row for row in ranked if isinstance(row.get("reranker_rank"), int)),
            key=lambda row: row["reranker_rank"])],
    }
    expected = _expected_chunks(case.model_dump(mode="json"), snapshot)
    if case.category == "unsupported" or (
            case.category == "acl_tenant_isolation" and
            (case.acl_context or {}).get("expected_access") == "deny"):
        expected = set()
    metrics = retrieval_metric_values(ranked_ids, expected)
    evidence_ids = _reference_ids(response.get("evidence_references"))
    citation_ids = _reference_ids(response.get("citations"))
    evidence_precision = (len(set(evidence_ids) & expected) / len(set(evidence_ids))
                          if evidence_ids and expected else None)
    evidence_coverage = (len(set(evidence_ids) & expected) / len(expected) if expected else None)
    citation_precision = (len(set(citation_ids) & expected) / len(set(citation_ids))
                          if citation_ids and expected else None)
    citation_recall = (len(set(citation_ids) & expected) / len(expected) if expected else None)
    answer = response.get("answer") if isinstance(response.get("answer"), str) else ""
    abstained = bool(response.get("abstained"))
    if case.category == "unsupported":
        # Unsupported-case abstention is reported as policy correctness, not answer accuracy.
        correctness = None
    elif case.answer_match == "exact" and case.expected_answer and not abstained:
        normalized = " ".join(answer.casefold().split())
        expected_values = [case.expected_answer, *case.acceptable_answer_variants]
        correctness = any(" ".join(value.casefold().split()) in normalized
                          for value in expected_values if value)
    else:
        correctness = None

    restricted = _restricted_ids(snapshot, case)
    citations_present = set(citation_ids)
    final_context = set(evidence_ids)
    candidates = set(ranked_ids)
    marker_sources = [e.source_document for e in case.expected_evidence
                      if e.source_document and re.search(r"synthetic phrase:\s*([^.]*)", e.quote, re.I)]
    marker_results = []
    for source in marker_sources:
        evidence = next((e for e in case.expected_evidence if e.source_document == source), None)
        match = re.search(r"synthetic phrase:\s*([^.]*)", evidence.quote, re.I) if evidence else None
        phrase = match.group(1).strip() if match else ""
        if phrase and phrase.casefold() in answer.casefold():
            marker_results.append({"source_document": source,
                                   "authorized": _source_authorized(snapshot, case, source)})
    answer_marker = bool(marker_results)
    unauthorized_answer_marker = any(row["authorized"] is False for row in marker_results)
    unknown_answer_marker = any(row["authorized"] is None for row in marker_results)
    expected_abstention = case.expected_abstention
    abstention_correct = (abstained == bool(expected_abstention)
                          if expected_abstention is not None else None)
    restricted_leaks = bool((candidates | final_context | citations_present) & restricted) or \
        unauthorized_answer_marker or unknown_answer_marker
    acl_pass = (not restricted_leaks if case.category == "acl_tenant_isolation" else None)
    return {
        "case_id": case.case_id, "category": case.category,
        "expected_access": (case.acl_context or {}).get("expected_access"),
        "query": case.query, "expected_answer": case.expected_answer,
        "profile": response.get("profile_used"), "answer": answer,
        "abstained": abstained, "answer_correct_raw": correctness,
        "expected_abstention": case.expected_abstention,
        "answer_match_type": case.answer_match,
        "abstention_correct": abstention_correct, "acl_outcome_pass": acl_pass,
        "manual_adjudication": None,
        "ranked_candidate_ids": ranked_ids,
        "candidate_ids_by_stage": stage_candidates,
        "retrieval_metrics": metrics,
        "evidence_precision": evidence_precision, "evidence_coverage": evidence_coverage,
        "citation_precision": citation_precision, "citation_recall": citation_recall,
        "latency_ms": elapsed_ms,
        "token_usage": response.get("token_usage") or {},
        "security_case": case.category in {"acl_tenant_isolation", "prompt_injection"},
        "restricted_chunk_leakage": {
            "candidates": sorted(candidates & restricted),
            "final_context": sorted(final_context & restricted),
            "citations": sorted(citations_present & restricted),
            "answer_restricted_marker": answer_marker,
            "answer_restricted_marker_authorized": any(row["authorized"] is True for row in marker_results),
            "answer_restricted_marker_unauthorized": unauthorized_answer_marker,
            "answer_restricted_marker_acl_unknown": unknown_answer_marker,
        },
        "response": response,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = {}
    profiles_seen = list(dict.fromkeys(
        [profile for profile in PROFILES if any(row.get("profile") == profile for row in records)] +
        [str(row.get("profile")) for row in records if row.get("profile") and row.get("profile") not in PROFILES]
    ))
    for profile in profiles_seen:
        rows = [row for row in records if row.get("profile") == profile]
        latency = sorted(float(row["latency_ms"]) for row in rows
                         if isinstance(row.get("latency_ms"), (int, float)))
        answerable = [row for row in rows if row.get("answer_match_type") == "exact" and
                      row.get("expected_abstention") is False and not row.get("abstained")]
        correctness = [row["answer_correct_raw"] for row in answerable
                       if isinstance(row.get("answer_correct_raw"), bool)]
        abstention_values = [row["abstention_correct"] for row in rows
                             if isinstance(row.get("abstention_correct"), bool)]
        acl_values = [row["acl_outcome_pass"] for row in rows
                      if isinstance(row.get("acl_outcome_pass"), bool)]
        metrics = {}
        for key in METRICS:
            values = [row["retrieval_metrics"].get(key) for row in rows
                      if row.get("retrieval_metrics", {}).get(key) is not None]
            metric_summary = {"n": len(values), "mean": statistics.mean(values) if values else None}
            if key.startswith("hit_rate_"):
                successes = sum(value == 1.0 for value in values)
                metric_summary.update({"successes": successes,
                                       "wilson_95": wilson_interval(successes, len(values))})
            metrics[key] = metric_summary
        stage_samples: dict[str, list[float]] = {}
        for row in rows:
            response = row.get("response") or {}
            timings = response.get("timings_ms") or {}
            for stage, value in timings.items():
                if isinstance(value, (int, float)):
                    stage_samples.setdefault(stage, []).append(float(value))
        stage_latency = {stage: {"n": len(values), "p50": statistics.median(values),
                                 "p95": sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)],
                                 "max": max(values)}
                         for stage, values in stage_samples.items() if values}
        providers = sorted({f"{(row.get('response') or {}).get('generation_metadata', {}).get('provider')}"
                            f"/{(row.get('response') or {}).get('generation_metadata', {}).get('model')}"
                            for row in rows if (row.get("response") or {}).get("generation_metadata")})
        evidence_metrics = {}
        for key in ("evidence_precision", "evidence_coverage", "citation_precision", "citation_recall"):
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            evidence_metrics[key] = {"n": len(values), "mean": statistics.mean(values) if values else None}
        claim_statuses: dict[str, int] = {}
        warning_codes: dict[str, int] = {}
        for row in rows:
            response = row.get("response") or {}
            for claim in (response.get("claims") or []):
                if isinstance(claim, dict) and isinstance(claim.get("status"), str):
                    claim_statuses[claim["status"]] = claim_statuses.get(claim["status"], 0) + 1
            for warning in response.get("warnings") or []:
                if isinstance(warning, str):
                    code = warning.split(":", 1)[0]
                    warning_codes[code] = warning_codes.get(code, 0) + 1
        answerable_rows = [row for row in rows if row.get("expected_abstention") is False]
        unanswerable_rows = [row for row in rows if row.get("expected_abstention") is True]
        unanswerable_answered = [row for row in unanswerable_rows if not row.get("abstained")]
        unanswerable_sufficiency = [((row.get("response") or {}).get("evidence_sufficiency") or {}).get("state")
                                    for row in unanswerable_answered]
        budget_ledgers = []
        for row in rows:
            header = row.get("budget_usage_header")
            if isinstance(header, str):
                try:
                    parsed = json.loads(header)
                    if isinstance(parsed, dict):
                        budget_ledgers.append((row.get("profile"), parsed))
                except (TypeError, ValueError):
                    pass
        confusion = {
            "answerable_correct_exact": sum(row.get("answer_correct_raw") is True for row in answerable_rows),
            "answerable_incorrect_exact": sum(row.get("answer_correct_raw") is False and not row.get("abstained")
                                               for row in answerable_rows),
            "answerable_incorrectly_abstained": sum(bool(row.get("abstained")) for row in answerable_rows),
            "answerable_answered_unscored": sum(not row.get("abstained") and row.get("answer_correct_raw") is None
                                                 for row in answerable_rows),
            "unanswerable_correctly_abstained": sum(bool(row.get("abstained")) for row in unanswerable_rows),
            "unanswerable_answered": len(unanswerable_answered),
            "unanswerable_answered_insufficient_heuristic": sum(value == "insufficient" for value in unanswerable_sufficiency),
            "unanswerable_answered_partial_or_strong_heuristic": sum(value in {"partial_evidence", "strong_relevant_evidence"}
                                                                      for value in unanswerable_sufficiency),
            "unanswerable_answered_sufficiency_unknown": sum(value not in {"insufficient", "partial_evidence",
                                                                           "strong_relevant_evidence"}
                                                               for value in unanswerable_sufficiency),
            "answerable_n": len(answerable_rows), "unanswerable_n": len(unanswerable_rows),
        }
        profiles[profile] = {
            "units": len(rows), "latency_ms": {
                "n": len(latency), "p50": statistics.median(latency) if latency else None,
                "p95": latency[max(0, math.ceil(.95 * len(latency)) - 1)] if latency else None,
                "max": max(latency) if latency else None,
            },
            "status_counts": {status: sum(row.get("status") == status for row in rows)
                              for status in sorted({row.get("status", "unknown") for row in rows})},
            "timeout_count": sum(row.get("status") == "timed_out" or row.get("http_status") in {408, 504}
                                 for row in rows),
            "latency_budget_violation_count": sum(bool(row.get("latency_budget_exceeded")) for row in rows),
            "failure_count": sum(row.get("status") not in {None, "completed"} or
                                  (row.get("http_status") is not None and not 200 <= row["http_status"] < 300)
                                  for row in rows),
            "stage_latency_ms": stage_latency,
            "provider_models": providers,
            "retrieval_metrics": metrics,
            "evidence_metrics": evidence_metrics,
            "claim_status_counts_machine": claim_statuses,
            "warning_code_counts": warning_codes,
            "answer_correctness_raw": {
                "numerator": sum(correctness), "denominator": len(correctness),
                "rate": sum(correctness) / len(correctness) if correctness else None,
                "wilson_95": wilson_interval(sum(correctness), len(correctness)),
            },
            "answer_confusion": confusion,
            "false_abstention_rate": {
                "numerator": confusion["answerable_incorrectly_abstained"],
                "denominator": confusion["answerable_n"],
                "rate": (confusion["answerable_incorrectly_abstained"] / confusion["answerable_n"]
                         if confusion["answerable_n"] else None),
                "wilson_95": wilson_interval(confusion["answerable_incorrectly_abstained"],
                                              confusion["answerable_n"]),
            },
            "correct_unanswerable_abstention_rate": {
                "numerator": confusion["unanswerable_correctly_abstained"],
                "denominator": confusion["unanswerable_n"],
                "rate": (confusion["unanswerable_correctly_abstained"] / confusion["unanswerable_n"]
                         if confusion["unanswerable_n"] else None),
                "wilson_95": wilson_interval(confusion["unanswerable_correctly_abstained"],
                                              confusion["unanswerable_n"]),
            },
            "abstention_correctness": {
                "numerator": sum(abstention_values), "denominator": len(abstention_values),
                "rate": sum(abstention_values) / len(abstention_values) if abstention_values else None,
                "wilson_95": wilson_interval(sum(abstention_values), len(abstention_values)),
            },
            "acl_outcome_pass": {
                "numerator": sum(acl_values), "denominator": len(acl_values),
                "rate": sum(acl_values) / len(acl_values) if acl_values else None,
                "wilson_95": wilson_interval(sum(acl_values), len(acl_values)),
            },
            "token_usage": {
                "available": sum(bool(row.get("token_usage")) for row in rows),
                "input_total": sum((row.get("token_usage") or {}).get("input_tokens", 0) for row in rows),
                "output_total": sum((row.get("token_usage") or {}).get("output_tokens", 0) for row in rows),
            },
            "budget_ledger_observations": {
                "header_count": len(budget_ledgers),
                "run_ids": sorted({str(item.get("run_id")) for _, item in budget_ledgers if item.get("run_id")}),
                "max_cumulative_calls_seen": max((item.get("calls", 0) for _, item in budget_ledgers), default=None),
                "max_cumulative_tokens_seen": max((item.get("total_tokens", 0) for _, item in budget_ledgers), default=None),
                "max_cumulative_cost_usd_seen": max((item.get("estimated_cost_usd", 0) for _, item in budget_ledgers), default=None),
                "profile_incremental_calls": None,
                "profile_incremental_tokens": None,
                "scope_note": "headers are cumulative by run_id; profile-attributed deltas are not reconstructed here",
            },
        }
    security = [row for row in records if row.get("security_case")]
    fields = ("candidates", "final_context", "citations")
    leakage = {}
    for field in fields:
        leaking = sum(bool(row["restricted_chunk_leakage"][field]) for row in security)
        leaked_chunk_count = sum(len(row["restricted_chunk_leakage"][field]) for row in security)
        leakage[field] = {"leaks": leaking, "n": len(security),
                          "restricted_chunk_count": leaked_chunk_count,
                          "wilson_95": wilson_interval(leaking, len(security)),
                          "rule_of_three_upper_95": min(1.0, 3 / len(security)) if security else None}
    markers = sum(bool(row["restricted_chunk_leakage"]["answer_restricted_marker"])
                  for row in security)
    unauthorized_markers = sum(bool(row["restricted_chunk_leakage"].get(
        "answer_restricted_marker_unauthorized")) or bool(row["restricted_chunk_leakage"].get(
        "answer_restricted_marker_acl_unknown")) for row in security)
    authorized_markers = sum(bool(row["restricted_chunk_leakage"].get(
        "answer_restricted_marker_authorized")) for row in security)
    leakage["answer_markers"] = {"occurrences": markers, "authorized_occurrences": authorized_markers,
                                  "unauthorized_or_unknown_leaks": unauthorized_markers,
                                  "n": len(security),
                                  "occurrence_wilson_95": wilson_interval(markers, len(security)),
                                  "leak_wilson_95": wilson_interval(unauthorized_markers, len(security)),
                                  "rule_of_three_upper_95": min(1.0, 3 / len(security)) if security else None}
    leakage["unauthorized_answer_markers"] = {"leaks": unauthorized_markers, "n": len(security),
                                               "wilson_95": wilson_interval(unauthorized_markers, len(security)),
                                               "rule_of_three_upper_95": min(1.0, 3 / len(security)) if security else None}
    security_groups = {}
    for profile in profiles_seen:
        categories = sorted({row.get("category", "unknown") for row in security
                             if row.get("profile") == profile})
        for category in categories:
            for access in ("allow", "deny", "unknown"):
                group = [row for row in security if row.get("profile") == profile and
                         row.get("category", "unknown") == category and
                         ((row.get("expected_access") or "unknown") == access)]
                if group:
                    security_groups[f"{profile}:{category}:{access}"] = {
                        "n": len(group),
                        "acl_outcome_pass": sum(row.get("acl_outcome_pass") is True for row in group),
                        "acl_outcome_fail": sum(row.get("acl_outcome_pass") is False for row in group),
                        "candidate_leak_cases": sum(bool(row["restricted_chunk_leakage"]["candidates"]) for row in group),
                        "context_leak_cases": sum(bool(row["restricted_chunk_leakage"]["final_context"]) for row in group),
                        "citation_leak_cases": sum(bool(row["restricted_chunk_leakage"]["citations"]) for row in group),
                        "unauthorized_or_unknown_answer_marker_cases": sum(
                            bool(row["restricted_chunk_leakage"].get("answer_restricted_marker_unauthorized")) or
                            bool(row["restricted_chunk_leakage"].get("answer_restricted_marker_acl_unknown"))
                            for row in group),
                    }
    return {"profiles": profiles, "security_case_count": len(security),
            "security_by_profile_and_expected_access": security_groups, "leakage": leakage,
            "answer_correctness_mismatches": [
                {"case_id": row["case_id"], "profile": row.get("profile"),
                 "expected_answer": row.get("expected_answer"), "answer": row.get("answer")}
                for row in records if row.get("answer_correct_raw") is False
            ],
            "answer_correctness_adjudicated": {
                "n": sum(isinstance(row.get("manual_adjudication"), bool) for row in records),
                "numerator": sum(row.get("manual_adjudication") is True for row in records),
                "rate": (sum(row.get("manual_adjudication") is True for row in records) /
                         sum(isinstance(row.get("manual_adjudication"), bool) for row in records)
                         if any(isinstance(row.get("manual_adjudication"), bool) for row in records) else None),
            }}


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Diagnostic Core Metrics", "",
             f"- Run ID: `{report.get('run_id')}`",
             f"- Dataset SHA-256: `{report.get('dataset_sha256')}`",
             f"- Stamp: {report.get('stamp')}",
             "- Mode: independent diagnostic; not the Phase 8 official runner", "",
             "| Profile | Units | Latency n | p50 ms | p95 ms | max ms | Timeouts |", "|---|---:|---:|---:|---:|---:|---:|"]
    summary = report.get("summary", {})
    for profile in summary.get("profiles", {}):
        row = (summary.get("profiles") or {}).get(profile, {})
        latency = row.get("latency_ms", {})
        lines.append(f"| {profile} | {row.get('units', 0)} | {latency.get('n', 0)} | "
                     f"{latency.get('p50')} | {latency.get('p95')} | {latency.get('max')} | "
                     f"{row.get('timeout_count', 0)} |")
    lines += ["", "## Retrieval metrics", "", "Metric means are computed only over eligible cases with mapped relevant chunks.", ""]
    lines += ["| Profile | Metric | n | Mean | Wilson 95% (binary hit rates only) |",
              "|---|---|---:|---:|---|"]
    for profile in summary.get("profiles", {}):
        for metric, values in ((summary.get("profiles") or {}).get(profile, {})
                               .get("retrieval_metrics", {}).items()):
            lines.append(f"| {profile} | {metric} | {values.get('n')} | {values.get('mean')} | "
                         f"{values.get('wilson_95')} |")
    lines += ["", "## Answer and abstention", "",
              "Exact-answer accuracy excludes unsupported/evidence-based cases and is separate from abstention correctness.", "",
              "| Profile | Exact correct / answered exact | False abstentions / answerable | Correct unsupported abstentions / unsupported |",
              "|---|---:|---:|---:|"]
    for profile, values in summary.get("profiles", {}).items():
        accuracy = values.get("answer_correctness_raw", {})
        false_abstention = values.get("false_abstention_rate", {})
        safe_abstention = values.get("correct_unanswerable_abstention_rate", {})
        lines.append(f"| {profile} | {accuracy.get('numerator')}/{accuracy.get('denominator')} | "
                     f"{false_abstention.get('numerator')}/{false_abstention.get('denominator')} | "
                     f"{safe_abstention.get('numerator')}/{safe_abstention.get('denominator')} |")
    lines += ["", "## ACL/security", "", "Answer-marker occurrence is distinguished from unauthorized or unknown-source disclosure.", "",
              "| Check | Observed / security attempts |"]
    for name, value in summary.get("leakage", {}).items():
        count = value.get("unauthorized_or_unknown_leaks", value.get("leaks"))
        lines.append(f"| {name} | {count}/{value.get('n')} |")
    lines += ["", "No result is statistically generalized beyond this run's case sample.", ""]
    return "\n".join(lines)


def _human_rows(records: list[dict[str, Any]], snapshot: dict[str, Any], limit: int = 30
                ) -> list[dict[str, str]]:
    points = {str(p.get("chunk_id")): p for p in snapshot.get("points", []) if p.get("chunk_id")}
    answered = [r for r in records if r.get("answer")]
    selected = []
    categories_seen: set[tuple[str, str]] = set()
    for row in sorted(answered, key=lambda r: (r.get("profile", ""), r.get("category", ""), r.get("case_id", ""))):
        key = (row.get("profile", ""), row.get("category", ""))
        if key not in categories_seen:
            selected.append(row)
            categories_seen.add(key)
    for row in answered:
        if row not in selected and len(selected) < limit:
            selected.append(row)
    selected = selected[:limit]
    output = []
    for row in selected:
        response = row.get("response") or {}
        evidence_items = response.get("evidence_references") or []
        citation_ids = {str(item.get("chunk_id")) for item in response.get("citations", [])
                        if isinstance(item, dict) and item.get("chunk_id")}
        cited = [points.get(str(item.get("chunk_id")), {}) for item in evidence_items
                 if isinstance(item, dict) and str(item.get("chunk_id")) in citation_ids]
        texts = [str(item.get("excerpt") or item.get("text") or item.get("content") or "") for item in evidence_items
                 if isinstance(item, dict)]
        citation_texts = [str(item.get("quote") or "") for item in response.get("citations", [])
                          if isinstance(item, dict)]
        claims = response.get("claims") or []
        claim_texts = [str(item.get("claim_text", "")) for item in claims if isinstance(item, dict)] or [""]
        for index, claim in enumerate(claim_texts):
            output.append({"audit_item_id": f"{row['case_id']}:{row.get('profile')}:{index}",
                           "review_copy": "1", "case_id": row["case_id"],
                           "profile": str(row.get("profile", "")), "category": row["category"],
                           "question": str(row.get("query", "")),
                           "answer": str(row.get("answer", "")),
                           "cited_chunk_text": "\n\n".join(
                               [str(p.get("text") or p.get("content") or "") for p in cited] + citation_texts),
                           "retrieved_evidence_text": "\n\n".join(texts), "claim": claim,
                           "supported_by_evidence": "", "cited_chunk_supports_claim": "",
                           "answer_correct": "", "reviewer": ""})
    return output


def export_human_audit(records: list[dict[str, Any]], snapshot: dict[str, Any], path: Path) -> int:
    rows = _human_rows(records, snapshot)
    duplicates = []
    for row in rows[:10]:
        duplicate = dict(row)
        duplicate["review_copy"] = "2"
        duplicate["reviewer"] = ""
        duplicates.append(duplicate)
    fields = ["audit_item_id", "review_copy", "case_id", "profile", "category", "question", "answer",
              "cited_chunk_text", "retrieved_evidence_text", "claim", "supported_by_evidence",
              "cited_chunk_supports_claim", "answer_correct", "reviewer"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + duplicates)
    return len(rows) + len(duplicates)


def score_human_audit(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    def rate(column: str) -> dict[str, Any]:
        labels = [row[column].strip().upper() for row in rows if row.get(column, "").strip().upper() in {"Y", "N"}]
        yes = sum(value == "Y" for value in labels)
        return {"numerator": yes, "denominator": len(labels), "rate": yes / len(labels) if labels else None,
                "wilson_95": wilson_interval(yes, len(labels))}
    grouped: dict[str, dict[str, str]] = {}
    agrees = total = 0
    for row in rows:
        if row.get("reviewer") and row.get("review_copy") in {"1", "2"}:
            grouped.setdefault(row["audit_item_id"], {})[row["review_copy"]] = "/".join(
                row.get(col, "").strip().upper() for col in ("supported_by_evidence", "cited_chunk_supports_claim"))
    for ratings in grouped.values():
        if "1" in ratings and "2" in ratings:
            total += 1
            agrees += ratings["1"] == ratings["2"]
    return {"faithfulness_supported_claim_rate": rate("supported_by_evidence"),
            "citation_accuracy_rate": rate("cited_chunk_supports_claim"),
            "answer_correct_adjudicated": rate("answer_correct"),
            "reviewer_agreement": {"agreements": agrees, "paired_rows": total,
                                   "rate": agrees / total if total else None}}


def dry_run(dataset_path: Path, snapshot_path: Path, historical_run: Path | None = None) -> dict[str, Any]:
    raw_hash = sha256_file(dataset_path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.model_validate(raw)
    validate_dataset(raw, corpus_root=ROOT / CORPUS_RELATIVE_ROOT, repository_root=ROOT)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_hash = verify_snapshot_integrity(snapshot)
    eligible = {profile: sum(profile in case.profile_coverage for case in dataset.cases)
                for profile in PROFILES}
    toy = retrieval_metric_values(["c1", "c2", "c3"], {"c2"})
    toy_leakage = bool({"restricted"} & {"restricted"})
    fixture_note = "not supplied"
    if historical_run:
        progress = json.loads((historical_run / "progress.json").read_text(encoding="utf-8"))
        attempt_files = list((historical_run / "attempts").glob("*.json"))
        attempts = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in attempt_files}
        historical_score = score_artifacts(raw, snapshot, progress, attempts)
        attempt_count = len(attempt_files)
        fixture_note = (f"historical-only scorer fixture inspected: {attempt_count} attempt files; "
                        f"run={progress.get('benchmark_id')}; scorer produced {len(historical_score['units'])} "
                        "fixture rows for offline validation only; not included as diagnostic results")
    return {
        "mode": "dry_run_no_api_requests", "dataset_path": str(dataset_path),
        "dataset_raw_sha256": raw_hash, "dataset_stamp": diagnostic_stamp(raw_hash),
        "schema_valid": True, "dataset_case_count": len(dataset.cases),
        "eligible_case_count_by_profile": eligible,
        "snapshot_sha256": snapshot_hash,
        "toy_metric_check": toy, "toy_leakage_check_detected": toy_leakage,
        "historical_fixture": fixture_note,
        "api_requests": 0,
        "live_execution_not_authorized": True,
    }


def execute(dataset_path: Path, snapshot_path: Path, output_dir: Path, base_url: str,
            operator_confirmed_local_ollama: bool = False,
            api_container_name: str | None = None) -> dict[str, Any]:
    """Serial live execution. Caller must obtain explicit approval before --execute."""
    from omnirag.config.settings import get_settings
    from evaluation.phase8_runner import schedule_units

    dataset_raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.model_validate(dataset_raw)
    validate_dataset(dataset_raw, corpus_root=ROOT / CORPUS_RELATIVE_ROOT, repository_root=ROOT)
    dataset_hash = sha256_file(dataset_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    verify_snapshot_integrity(snapshot)
    settings = get_settings()
    secret = settings.auth_shared_secret.get_secret_value() if settings.auth_shared_secret else ""
    if not secret:
        raise RuntimeError("Signed API identity secret is unavailable")
    if not operator_confirmed_local_ollama:
        raise RuntimeError("Live diagnostic requires explicit confirmation that the API uses local Ollama")
    first_case = dataset.cases[0]
    settings_user, settings_tenant = _identity_for(first_case)
    settings_headers = sign_identity(settings_user, settings_tenant, secret)
    settings_status, _, runtime_settings = _http_get(
        f"{base_url.rstrip('/')}/api/v1/settings", settings_headers, timeout=10)
    if settings_status != 200 or not isinstance(runtime_settings, dict):
        raise RuntimeError("Signed runtime settings preflight failed")
    limits = runtime_settings.get("limits") or {}
    if (limits.get("max_cost_usd") != 0 or
            not isinstance(limits.get("max_total_tokens"), int) or
            limits["max_total_tokens"] > 4_200_000 or
            not isinstance(limits.get("max_calls"), int) or limits["max_calls"] > 3_000):
        raise RuntimeError("Runtime budget settings are incompatible with approved diagnostic caps")
    if not str(runtime_settings.get("physical_collection", "")).startswith("omnirag_eval_"):
        raise RuntimeError("Runtime collection is not isolated under omnirag_eval_ prefix")
    if runtime_settings.get("physical_collection") != snapshot.get("collection"):
        raise RuntimeError("Runtime physical collection does not match the pinned corpus snapshot")
    generation_model = (runtime_settings.get("models") or {}).get("generation") or {}
    if (not generation_model.get("configured") or generation_model.get("provider") != "openai_compatible" or
            not str(generation_model.get("model_id", "")).casefold().startswith("llama")):
        raise RuntimeError("Runtime generation model is not the expected local Ollama-compatible model")
    accepted = {(row.get("user_id"), row.get("tenant_id"))
                for row in runtime_settings.get("accepted_identities", []) if isinstance(row, dict)}
    if accepted:
        needed = {_identity_for(case) for case in dataset.cases
                  if case.acl_context and case.tenant_context}
        if not needed.issubset(accepted):
            raise RuntimeError("Runtime signed-identity allowlist does not cover dataset principals")
    run_id = "diag-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"run_id": run_id, "mode": "live_diagnostic", "dataset_sha256": dataset_hash,
              "stamp": diagnostic_stamp(dataset_hash), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "profiles_order": list(PROFILES), "order_seed": 20261106, "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
              "local_budget_limits": {"tokens": LOCAL_TOKEN_CAP, "calls": LOCAL_CALL_CAP},
              "server_budget_limits_not_modified": {"cost_usd": 0, "tokens": 4_200_000, "calls": 3000},
              "cache_coldness": "profile cache policy is TTL-zero; provider/model cold state is uncontrolled",
              "provenance": _future_run_provenance(dataset_hash, snapshot, runtime_settings,
                                                    base_url, api_container_name),
              "attempts": []}
    try:
        all_units = {profile: schedule_units([case for case in dataset.cases if profile in case.profile_coverage],
                                             profiles=(profile,), repetitions=1, seed=20261106)
                     for profile in PROFILES}
        cumulative_calls = cumulative_tokens = 0
        final_budget_usage = None
        for profile in PROFILES:
            for unit in all_units[profile]:
                user, tenant = _identity_for(unit.case)
                headers = sign_identity(user, tenant, secret)
                request_id = str(uuid4())
                headers.update({"Content-Type": "application/json", "X-Request-ID": request_id,
                                "X-OmniRAG-Run-ID": run_id})
                body = {"query_text": unit.case.query, "selected_profile": profile,
                        "escalation_preference": "never"}
                started = time.monotonic()
                try:
                    status, response_headers, payload = _http_post(
                        f"{base_url.rstrip('/')}/api/v1/chat", headers,
                        json.dumps(body, ensure_ascii=False).encode("utf-8"), REQUEST_TIMEOUT_SECONDS)
                    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
                except (URLError, TimeoutError, OSError) as exc:
                    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
                    record = {"case_id": unit.case.case_id, "profile": profile,
                              "query": unit.case.query, "status": "timed_out" if isinstance(exc, TimeoutError) else "failed",
                              "failure_classification": type(exc).__name__, "request_id": request_id,
                              "elapsed_ms": elapsed_ms, "token_usage": {}, "response": {}}
                    report["attempts"].append(record)
                    (run_dir / f"{unit.case.case_id}-{profile}.json").write_text(
                        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    report["stop_reason"] = "transport_failure_budget_state_unknown"
                    break
                response = payload.get("response", {}) if isinstance(payload, dict) else {}
                record = score_response(unit.case, response, snapshot, elapsed_ms)
                record = redact_secret(record, secret)
                attempt_status, latency_violation = classify_http_outcome(status, elapsed_ms)
                record.update({"request_id": request_id, "http_status": status,
                               "status": attempt_status,
                               "latency_budget_exceeded": latency_violation,
                               "http_payload": payload,
                               "budget_usage_header": response_headers.get("X-OmniRAG-Budget-Usage") or
                               response_headers.get("x-omnirag-budget-usage")})
                report["attempts"].append(record)
                (run_dir / f"{unit.case.case_id}-{profile}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                ledger_header = record.get("budget_usage_header")
                try:
                    ledger = json.loads(ledger_header) if ledger_header else None
                except (ValueError, TypeError):
                    ledger = None
                if isinstance(ledger, dict):
                    final_budget_usage = ledger
                    cumulative_calls = int(ledger.get("calls", cumulative_calls))
                    cumulative_tokens = int(ledger.get("total_tokens", cumulative_tokens))
                    if ledger.get("run_id") != run_id:
                        report["stop_reason"] = "budget_run_id_mismatch"
                    elif cumulative_calls >= LOCAL_CALL_CAP or cumulative_tokens >= LOCAL_TOKEN_CAP:
                        report["stop_reason"] = "local_budget_cap"
                else:
                    report["stop_reason"] = "budget_state_unknown"
                if not 200 <= status < 300:
                    report["stop_reason"] = "http_failure"
                if report.get("stop_reason"):
                    break
            if report.get("stop_reason"):
                break
        report["summary"] = summarize(report["attempts"])
        report["budget_usage_final"] = final_budget_usage
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        audit_rows = export_human_audit(report["attempts"], snapshot, run_dir / "human_audit.csv")
        report["human_audit_rows"] = audit_rows
        (run_dir / "run.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (run_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        return {"run_id": run_id, "run_dir": str(run_dir), "summary": report["summary"]}
    finally:
        secret = ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/audit/phase8_eval_corpus_manifest.json")
    parser.add_argument("--historical-run", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/diagnostic")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--api-container-name", default=None,
                        help="Optional local Docker container name for future-run image-ID provenance")
    parser.add_argument("--operator-confirmed-local-ollama", action="store_true",
                        help="Attest that the API's hidden generation endpoint is local Ollama")
    parser.add_argument("--score-human-audit", type=Path, default=None,
                        help="Score a human-completed audit CSV without making API requests")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.score_human_audit:
        print(json.dumps(score_human_audit(args.score_human_audit), indent=2))
        return 0
    if not args.dry_run and not args.execute:
        parser.error("choose --dry-run or --execute")
    if args.dry_run:
        result = dry_run(args.dataset, args.snapshot, args.historical_run)
        run_id = "dryrun-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        directory = args.output_root / run_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "dry_run.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_id": run_id, "output_dir": str(directory), **result}, indent=2))
        return 0
    output_dir = args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(execute(args.dataset, args.snapshot, output_dir, args.base_url,
                             args.operator_confirmed_local_ollama, args.api_container_name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
