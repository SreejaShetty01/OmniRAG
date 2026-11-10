"""Serial, resumable Phase 8 HTTP benchmark collection runner.

Raw case queries and API answers are transient values only. Persisted benchmark
attempts contain allowlisted metadata and derived evaluation features.
"""

from __future__ import annotations

import hashlib
import hmac
import errno
import json
import random
import re
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4
from utils.atomic_file import atomic_replace

from evaluation.dataset import DATASET_ID, DATASET_VERSION, EvaluationCase, EvaluationDataset, dataset_checksum
from evaluation.phase8_safety import require_evaluation_collection, verify_snapshot_manifest


EXPECTED_DATASET_SHA256 = "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 150.0
DEFAULT_RUN_TIMEOUT_SECONDS = 108_000.0
MAX_COST_USD = 0.0
MAX_TOTAL_TOKENS = 4_200_000
MAX_PROVIDER_CALLS = 3_000
FAILURE_CODE_PREFIXES = (
    "generation_failed", "research_finalize_failed", "dense_embedding_failed",
    "dense_retrieval_failed", "sparse_retrieval_failed",
    "reranker_failed", "reranker_failed_fallback_to_fusion", "correction_failed",
    "semantic_verification_failed", "research_retrieval_failed",
    "research_subquery_timed_out", "provider_budget_exceeded",
)
WARMUP_CASE_ID = "P7C-AF-001"
WARMUP_PROFILES = {"lite", "core", "research"}


@dataclass(frozen=True)
class ScheduledUnit:
    case: EvaluationCase
    profile: str
    repetition: int
    attempt: int = 0

    @property
    def attempt_key(self) -> str:
        return f"{self.case.case_id}-{self.profile}-r{self.repetition:02d}-a{self.attempt:02d}"


def sign_identity(user_id: str, tenant_id: str, secret: str, *, timestamp: int | None = None
                  ) -> dict[str, str]:
    stamp = str(int(time.time()) if timestamp is None else timestamp)
    signed = f"{user_id}\n{tenant_id}\n{stamp}".encode()
    signature = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return {"X-OmniRAG-User-ID": user_id, "X-OmniRAG-Tenant-ID": tenant_id,
            "X-OmniRAG-Auth-Timestamp": stamp, "X-OmniRAG-Auth-Signature": signature}


def schedule_units(cases: list[EvaluationCase], *, profiles: tuple[str, ...] = ("lite", "core", "research"),
                   repetitions: int = 1, seed: int = 20261104) -> list[ScheduledUnit]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if len(set(profiles)) != len(profiles):
        raise ValueError("profiles must be unique")
    base = [ScheduledUnit(case, profile, 1)
            for case in cases for profile in profiles if profile in case.profile_coverage]
    random.Random(seed).shuffle(base)
    ordered: list[ScheduledUnit] = []
    for repetition in range(1, repetitions + 1):
        offset = ((repetition - 1) * max(1, len(base) // repetitions)) % len(base) if base else 0
        rotated = base[offset:] + base[:offset]
        ordered.extend(ScheduledUnit(item.case, item.profile, repetition) for item in rotated)
    return ordered


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    atomic_replace(temp, path)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _safe_name(source: str | None) -> str | None:
    if not source:
        return None
    return Path(source.replace("\\", "/")).name


def _identity_for(case: EvaluationCase) -> tuple[str, str]:
    if case.acl_context and case.tenant_context:
        principal = str(case.acl_context["request_principal"])
        user_id = principal.removeprefix("user:")
        tenant = str(case.tenant_context["request_tenant_id"])
        return user_id, tenant
    return "phase8-evaluation-user", "phase8-evaluation-tenant"


def build_chat_request(unit: ScheduledUnit, *, secret: str, benchmark_id: str,
                       request_id: str | None = None, timestamp: int | None = None
                       ) -> tuple[dict[str, Any], dict[str, str], str]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", benchmark_id):
        raise ValueError("benchmark_id is not a valid budget run ID")
    user_id, tenant_id = _identity_for(unit.case)
    headers = sign_identity(user_id, tenant_id, secret, timestamp=timestamp)
    req_id = request_id or str(uuid4())
    headers.update({"Content-Type": "application/json", "X-Request-ID": req_id,
                    "X-OmniRAG-Run-ID": benchmark_id})
    body = {
        "query_text": unit.case.query,
        "selected_profile": unit.profile,
        "escalation_preference": "never",
        "experiment": {"dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
                       "research_escalation": False},
    }
    return body, headers, req_id


def _http_post(url: str, headers: dict[str, str], body: bytes, timeout: float):
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, dict(response.headers), json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return exc.code, dict(exc.headers), payload


def _http_get(url: str, headers: dict[str, str], timeout: float):
    """Single signed JSON GET using the same urllib transport as benchmark POSTs."""
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, dict(response.headers), json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return exc.code, dict(exc.headers), payload


def _target_fields(url: str) -> dict[str, Any]:
    parts = urlsplit(url)
    return {"scheme": parts.scheme, "host": parts.hostname,
            "port": parts.port or (443 if parts.scheme.lower() == "https" else 80),
            "path": parts.path or "/"}


def _safe_reason_repr(reason: Any, secret: str) -> str:
    value = repr(reason)
    if secret:
        value = value.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)


def _transport_failure_details(url: str, exc: Exception, elapsed_ms: float,
                               secret: str) -> dict[str, Any]:
    reason = getattr(exc, "reason", exc)
    error_number = getattr(reason, "errno", None)
    winerror = getattr(reason, "winerror", None)
    definitely_preconnect = isinstance(
        reason, (PermissionError, ConnectionRefusedError, socket.gaierror))
    if error_number in {errno.EACCES, errno.EPERM, errno.ECONNREFUSED,
                        errno.EHOSTUNREACH, errno.ENETUNREACH}:
        definitely_preconnect = True
    return {
        "exception_class": f"{type(exc).__module__}.{type(exc).__name__}",
        "reason_repr": _safe_reason_repr(reason, secret),
        "errno": error_number,
        "winerror": winerror,
        "target": _target_fields(url),
        "elapsed_ms": elapsed_ms,
        # False only for errors that establish failure before a connection;
        # otherwise the transport cannot establish whether request bytes left.
        "request_likely_sent": None if not definitely_preconnect else False,
    }


def _validated_base_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except (TypeError, ValueError) as exc:
        raise ValueError("Benchmark base URL must be an absolute HTTP(S) URL with a valid port") from exc
    if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname or
            any(character.isspace() for character in value) or
            parts.username is not None or parts.password is not None or
            parts.query or parts.fragment or (port is not None and not 1 <= port <= 65535)):
        raise ValueError("Benchmark base URL must be an absolute HTTP(S) URL with a valid host and port")
    return value.rstrip("/")


def _collect_ids(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if value.get("chunk_id") and value.get("document_id"):
            output.append({
                "chunk_id": str(value["chunk_id"]), "document_id": str(value["document_id"]),
                "evidence_id": str(value["evidence_id"]) if value.get("evidence_id") else None,
                "source": _safe_name(value.get("source")),
                "version": value.get("version") if isinstance(value.get("version"), str) else None,
                "page_number": value.get("page_number") if isinstance(value.get("page_number"), int) else None,
                "section_path": [x for x in value.get("section_path", []) if isinstance(x, str)]
                if isinstance(value.get("section_path"), list) else [],
                "rank": value.get("fusion_rank") if isinstance(value.get("fusion_rank"), int) else None,
                "selected": value.get("selected") if isinstance(value.get("selected"), bool) else None,
            })
        for child in value.values():
            _collect_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_ids(child, output)


def _safe_response(case: EvaluationCase, payload: dict[str, Any], snapshot: dict[str, Any],
                   response_headers: dict[str, str], request_id: str, http_status: int,
                   elapsed_ms: float, benchmark_id: str) -> dict[str, Any]:
    response = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response, dict):
        return {"status": "failed", "failure_classification": "invalid_api_response",
                "http_status": http_status, "request_id": request_id, "elapsed_ms": elapsed_ms}

    points = {str(item.get("chunk_id")): item for item in snapshot.get("points", [])
              if item.get("chunk_id")}
    request_user, request_tenant = _identity_for(case)
    principal = f"user:{request_user}"
    raw_items = []
    _collect_ids(response.get("retrieval_inspection", []), raw_items)
    _collect_ids(response.get("evidence_references", []), raw_items)
    _collect_ids(response.get("citations", []), raw_items)
    unique = {}
    for item in raw_items:
        unique[item["chunk_id"]] = item

    unauthorized = False
    unknown_mapping = False
    safe_items = []
    for item in unique.values():
        point = points.get(item["chunk_id"])
        if point is None or point.get("document_id") != item["document_id"]:
            unknown_mapping = True
            continue
        allowed = point.get("acl_allowed_principals") or []
        denied = point.get("acl_denied_principals") or []
        authorized = (point.get("tenant_id") == request_tenant and principal not in denied and
                      (not allowed or principal in allowed))
        if not authorized:
            unauthorized = True
            continue
        safe_items.append(item)

    answer = response.get("answer") if isinstance(response.get("answer"), str) else ""
    restricted_quote = next((e.quote for e in case.expected_evidence
                             if e.source_document == "12_restricted_team.txt"), "")
    restricted_marker = re.search(r"cobalt\s+lantern\s+731", restricted_quote, re.IGNORECASE)
    marker_leak = bool(restricted_marker and restricted_marker.group(0).casefold() in answer.casefold())
    denied = case.category == "acl_tenant_isolation" and case.acl_context.get("expected_access") == "deny"
    acl_violation = bool(unauthorized or (denied and marker_leak))

    citations = response.get("citations") if isinstance(response.get("citations"), list) else []
    evidence = response.get("evidence_references") if isinstance(response.get("evidence_references"), list) else []
    safe_chunk_ids = {item["chunk_id"] for item in safe_items}
    safe_citations = []
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("chunk_id") not in safe_chunk_ids:
            continue
        safe_citations.append({"citation_id": citation.get("citation_id"),
                               "evidence_id": citation.get("evidence_id"),
                               "document_id": citation.get("document_id"),
                               "chunk_id": citation.get("chunk_id"),
                               "source": _safe_name(citation.get("source")),
                               "version": citation.get("version"),
                               "page_number": citation.get("page_number"),
                               "section_path": citation.get("section_path", [])})
    safe_evidence = []
    for reference in evidence:
        if not isinstance(reference, dict) or reference.get("chunk_id") not in safe_chunk_ids:
            continue
        safe_evidence.append({"evidence_id": reference.get("evidence_id"),
                              "document_id": reference.get("document_id"),
                              "chunk_id": reference.get("chunk_id"),
                              "source": _safe_name(reference.get("source")),
                              "version": reference.get("version"),
                              "page_number": reference.get("page_number"),
                              "section_path": reference.get("section_path", [])})
    claims = response.get("claims") if isinstance(response.get("claims"), list) else []
    claim_status_counts: dict[str, int] = {}
    for claim in claims:
        status = claim.get("status") if isinstance(claim, dict) else None
        if isinstance(status, str):
            claim_status_counts[status] = claim_status_counts.get(status, 0) + 1
    timings = response.get("timings_ms") if isinstance(response.get("timings_ms"), dict) else {}
    usage = response.get("token_usage") if isinstance(response.get("token_usage"), dict) else {}
    generation = response.get("generation_metadata") if isinstance(response.get("generation_metadata"), dict) else {}
    execution = response.get("execution_metadata") if isinstance(response.get("execution_metadata"), dict) else {}
    warnings = response.get("warnings") if isinstance(response.get("warnings"), list) else []
    warning_codes = sorted({str(item).split(":", 1)[0] for item in warnings
                            if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.-]+(?::.*)?", item)})
    failure_codes = sorted(code for code in warning_codes
                           if code in FAILURE_CODE_PREFIXES)
    expected = [case.expected_answer, *case.acceptable_answer_variants]
    exact_match = (any(_normalized(value) in _normalized(answer) for value in expected if value)
                   if case.answer_match == "exact" else None)
    budget_header = response_headers.get("X-OmniRAG-Budget-Usage") or response_headers.get("x-omnirag-budget-usage")
    try:
        budget_usage = json.loads(budget_header) if budget_header else None
    except (json.JSONDecodeError, TypeError):
        budget_usage = None
    status = "completed_with_errors" if failure_codes else (
        "completed_with_warnings" if warning_codes else "completed")
    return {
        "status": status, "http_status": http_status,
        "request_id": request_id,
        "experiment_run_id": execution.get("experiment_run_id"),
        "profile": response.get("profile_used"), "elapsed_ms": elapsed_ms,
        "abstained": bool(response.get("abstained")),
        "abstention_reason": response.get("abstention_reason"),
        "verification_status": response.get("verification_status"),
        "claim_verification_status": response.get("claim_verification_status"),
        "citation_validation_status": response.get("citation_validation_status"),
        "overall_reliability": response.get("overall_reliability"),
        "claim_count": len(claims), "claim_status_counts": claim_status_counts,
        "generation_metadata": {k: generation.get(k) for k in ("provider", "model")},
        "token_usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens")
                        if usage.get(k) is not None},
        "cost_usd": response.get("cost_usd"), "timings_ms": timings,
        "failure_codes": failure_codes,
        "warning_codes": warning_codes,
        "expected_answer_match": exact_match,
        "evidence_references": safe_evidence, "citations": safe_citations,
        "retrieval_candidates": safe_items,
        "authorization_audit": {"principal": request_user, "tenant_id": request_tenant,
                                 "denied_case": denied, "violation": acl_violation,
                                 "unknown_evidence_mapping": unknown_mapping,
                                 "restricted_marker_leaked": marker_leak if denied else False},
        "budget_usage": budget_usage,
        "budget_run_id": benchmark_id,
    }


class Phase8BenchmarkRunner:
    def __init__(self, *, base_url: str, secret: str, snapshot_manifest: dict[str, Any],
                 output_root: str | Path, request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
                 run_timeout: float = DEFAULT_RUN_TIMEOUT_SECONDS,
                 max_cost_usd: float = MAX_COST_USD, max_total_tokens: int = MAX_TOTAL_TOKENS,
                 max_provider_calls: int = MAX_PROVIDER_CALLS,
                 post: Callable = _http_post):
        snapshot_manifest = verify_snapshot_manifest(snapshot_manifest)
        collection = require_evaluation_collection(snapshot_manifest.get("collection"))
        if snapshot_manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
            raise ValueError("Snapshot dataset checksum is not the approved immutable release")
        if request_timeout <= 0 or run_timeout <= 0:
            raise ValueError("Request and run timeouts must be finite positive values")
        if max_cost_usd != 0 or max_total_tokens <= 0 or max_provider_calls <= 0:
            raise ValueError("Phase 8 requires USD cap zero plus positive token and call caps")
        self.base_url = _validated_base_url(base_url)
        self.secret = secret
        self.snapshot = snapshot_manifest
        self.collection = collection
        self.output_root = Path(output_root)
        self.request_timeout = request_timeout
        self.run_timeout = run_timeout
        self.max_cost_usd = max_cost_usd
        self.max_total_tokens = max_total_tokens
        self.max_provider_calls = max_provider_calls
        self.post = post

    def preflight(self, benchmark_id: str) -> dict[str, Any]:
        """Verify signed access and the active budget ledger before any chat request."""
        endpoint = f"{self.base_url}/api/v1/settings"
        headers = sign_identity("phase7a-user-a", "phase7a-tenant-a", self.secret)
        headers["X-OmniRAG-Run-ID"] = benchmark_id
        started = time.monotonic()
        try:
            status, response_headers, payload = _http_get(endpoint, headers, min(10.0, self.request_timeout))
        except json.JSONDecodeError:
            raise RuntimeError("Runner API self-preflight returned malformed JSON") from None
        except (URLError, TimeoutError, OSError) as exc:
            details = _transport_failure_details(
                endpoint, exc, round((time.monotonic() - started) * 1000, 3), self.secret)
            raise RuntimeError(
                "Runner API self-preflight transport failure: " +
                json.dumps(details, sort_keys=True)
            ) from None
        if status != 200 or not isinstance(payload, dict) or payload.get("user_id") != "phase7a-user-a":
            raise RuntimeError(
                f"Runner API self-preflight failed: HTTP {status} or signed identity did not validate "
                f"at {json.dumps(_target_fields(endpoint), sort_keys=True)}"
            )
        raw_budget = response_headers.get("X-OmniRAG-Budget-Usage") or response_headers.get(
            "x-omnirag-budget-usage")
        try:
            budget = json.loads(raw_budget) if raw_budget else None
        except (json.JSONDecodeError, TypeError):
            budget = None
        expected_caps = {"max_cost_usd": self.max_cost_usd,
                         "max_total_tokens": self.max_total_tokens,
                         "max_calls": self.max_provider_calls}
        if (not isinstance(budget, dict) or budget.get("run_id") != benchmark_id or
                any(budget.get(key) != value for key, value in expected_caps.items()) or
                not isinstance(budget.get("calls"), int) or
                not isinstance(budget.get("total_tokens"), int) or
                budget.get("calls", -1) < 0 or budget.get("total_tokens", -1) < 0 or
                not isinstance(budget.get("estimated_cost_usd"), (int, float)) or
                budget.get("estimated_cost_usd", -1) < 0):
            raise RuntimeError("Runner API self-preflight failed: budget ledger is missing or mismatched")
        return {"http_status": status, "endpoint": _target_fields(endpoint),
                "budget_usage": budget}

    def run(self, units: list[ScheduledUnit], *, benchmark_id: str, seed: int,
            resume: bool = False, warmup: bool = False) -> dict[str, Any]:
        if not units:
            raise ValueError("At least one scheduled evaluation unit is required")
        if warmup and (len(units) != 3 or
                       {unit.case.case_id for unit in units} != {WARMUP_CASE_ID} or
                       {unit.profile for unit in units} != WARMUP_PROFILES or
                       any(unit.repetition != 1 for unit in units)):
            raise ValueError("Warm-up must contain exactly one P7C-AF-001 unit for each profile")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", benchmark_id):
            raise ValueError("Invalid benchmark ID")
        directory = self.output_root / benchmark_id
        progress_path = directory / "progress.json"
        schedule = [{"attempt_key": unit.attempt_key, "case_id": unit.case.case_id,
                     "category": unit.case.category, "profile": unit.profile,
                     "repetition": unit.repetition, "attempt": unit.attempt,
                     "status": "pending"} for unit in units]
        schedule_hash = hashlib.sha256(json.dumps(schedule, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
        if directory.exists():
            if not resume or not progress_path.is_file():
                raise FileExistsError(f"Benchmark directory already exists: {directory}")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("schedule_sha256") != schedule_hash or progress.get("seed") != seed:
                raise ValueError("Resume schedule or seed differs from original benchmark")
            if progress.get("snapshot_sha256") != self.snapshot.get("snapshot_sha256"):
                raise ValueError("Resume corpus snapshot differs from original benchmark")
            if progress.get("run_timeout_seconds") != self.run_timeout:
                raise ValueError("Resume run timeout differs from original benchmark")
            if progress.get("request_timeout_seconds") != self.request_timeout:
                raise ValueError("Resume request timeout differs from original benchmark")
            expected_caps = {"max_cost_usd": self.max_cost_usd,
                             "max_total_tokens": self.max_total_tokens,
                             "max_provider_calls": self.max_provider_calls}
            if progress.get("budget_caps") != expected_caps:
                raise ValueError("Resume budget caps differ from original benchmark")
            if bool(progress.get("warmup")) != warmup:
                raise ValueError("Resume warm-up mode differs from original benchmark")
            if progress.get("status") != "running":
                return progress
        else:
            directory.mkdir(parents=True, exist_ok=False)
            progress = {
                "schema_version": 1, "benchmark_id": benchmark_id,
                "dataset": {"id": DATASET_ID, "version": DATASET_VERSION,
                            "sha256": EXPECTED_DATASET_SHA256},
                "snapshot_sha256": self.snapshot["snapshot_sha256"],
                "collection": self.collection, "seed": seed,
                "concurrency": 1, "application_retrieval_cache_policy": "disabled_by_profile_policy_ttl_zero",
                "request_timeout_seconds": self.request_timeout,
                "run_timeout_seconds": self.run_timeout,
                "deadline_unix": time.time() + self.run_timeout,
                "budget_caps": {"max_cost_usd": self.max_cost_usd,
                                "max_total_tokens": self.max_total_tokens,
                                "max_provider_calls": self.max_provider_calls},
                "schedule_sha256": schedule_hash, "warmup": warmup,
                "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "schedule": schedule, "budget_usage": None,
            }
            _atomic_json(progress_path, progress)

        units_by_key = {unit.attempt_key: unit for unit in units}
        for row in progress["schedule"]:
            if row["status"] == "in_progress":
                interrupted_path = directory / "attempts" / f"{row['attempt_key']}.json"
                persisted = None
                if interrupted_path.is_file():
                    try:
                        persisted = json.loads(interrupted_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        persisted = None
                if isinstance(persisted, dict) and persisted.get("status") in {
                        "completed", "completed_with_warnings", "completed_with_errors",
                        "failed", "timed_out", "indeterminate_after_interruption"}:
                    row["status"] = persisted["status"]
                    if persisted.get("experiment_run_id"):
                        row["experiment_run_id"] = persisted["experiment_run_id"]
                else:
                    row["status"] = "indeterminate_after_interruption"
                    _atomic_json(interrupted_path, {
                        "status": "indeterminate_after_interruption",
                        "failure_classification": "interrupted_before_result_persisted",
                        "request_id": row.get("request_id"),
                    })
                _atomic_json(progress_path, progress)
            elif row["status"] == "pending":
                orphan_path = directory / "attempts" / f"{row['attempt_key']}.json"
                if orphan_path.exists():
                    row["status"] = "indeterminate_after_interruption"
                    _atomic_json(progress_path, progress)
        deadline_unix = progress.get("deadline_unix")
        if not isinstance(deadline_unix, (int, float)):
            raise ValueError("Progress record lacks the cumulative run deadline")
        stop_reason: str | None = None
        for row in progress["schedule"]:
            if row["status"] != "pending":
                continue
            remaining_seconds = deadline_unix - time.time()
            if remaining_seconds <= 0:
                row["status"] = "not_run_run_timeout"
                continue
            budget = progress.get("budget_usage")
            if budget and (budget.get("estimated_cost_usd", 0) > self.max_cost_usd or
                           budget.get("total_tokens", 0) >= self.max_total_tokens or
                           budget.get("calls", 0) >= self.max_provider_calls):
                row["status"] = "not_run_budget_cap"
                stopped_for_budget = True
                continue
            if row["status"] != "pending":
                continue
            unit = units_by_key[row["attempt_key"]]
            request_id = str(uuid4())
            body, headers, request_id = build_chat_request(unit, secret=self.secret,
                                                           benchmark_id=benchmark_id,
                                                           request_id=request_id)
            request_url = f"{self.base_url}/api/v1/chat"
            row["status"] = "in_progress"
            row["request_id"] = request_id
            row["request_started_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_json(progress_path, progress)
            started = time.monotonic()
            attempt_path = directory / "attempts" / f"{unit.attempt_key}.json"
            try:
                bounded_timeout = min(self.request_timeout, max(0.1, remaining_seconds))
                status_code, response_headers, payload = self.post(
                    request_url, headers,
                    json.dumps(body, ensure_ascii=False).encode(), bounded_timeout)
                elapsed = round((time.monotonic() - started) * 1000, 3)
                budget_header = response_headers.get("X-OmniRAG-Budget-Usage") or response_headers.get(
                    "x-omnirag-budget-usage")
                budget_header_missing = not bool(budget_header)
                latest = None
                if budget_header:
                    try:
                        latest = json.loads(budget_header)
                    except (json.JSONDecodeError, TypeError):
                        latest = None
                    required_ledger_fields = ("run_id", "calls", "total_tokens", "estimated_cost_usd")
                    valid_ledger = (isinstance(latest, dict) and
                                    all(key in latest for key in required_ledger_fields) and
                                    latest.get("run_id") == benchmark_id and
                                    isinstance(latest.get("calls"), int) and
                                    isinstance(latest.get("total_tokens"), int) and
                                    isinstance(latest.get("estimated_cost_usd"), (int, float)) and
                                    latest.get("estimated_cost_usd") >= 0 and
                                    latest.get("calls") >= 0 and latest.get("total_tokens") >= 0)
                    if valid_ledger:
                        old = progress.get("budget_usage") or {}
                        if any(latest.get(k, 0) < old.get(k, 0) for k in ("calls", "total_tokens")):
                            row["status"] = "failed"
                            _atomic_json(attempt_path, {"status": "failed", "http_status": status_code,
                                                       "failure_classification": "budget_ledger_reset"})
                            stop_reason = "budget_state_reset"
                            _atomic_json(progress_path, progress)
                            break
                        progress["budget_usage"] = latest
                    else:
                        stop_reason = "budget_state_unknown"
                if stop_reason == "budget_state_unknown":
                    row["status"] = "failed"
                    _atomic_json(attempt_path, {"status": "failed", "http_status": status_code,
                                               "failure_classification": "budget_usage_header_invalid",
                                               "request_id": request_id, "elapsed_ms": elapsed})
                elif status_code < 200 or status_code >= 300:
                    err = payload.get("error", {}) if isinstance(payload, dict) else {}
                    classification = (err.get("code") if isinstance(err, dict) and
                                      isinstance(err.get("code"), str) else f"http_{status_code}")
                    row["status"] = "failed"
                    _atomic_json(attempt_path, {"status": "failed", "http_status": status_code,
                                               "failure_classification": classification,
                                               "request_id": request_id, "elapsed_ms": elapsed})
                    if classification == "provider_budget_exceeded" or (
                            latest and latest.get("estimated_cost_usd", 0) > self.max_cost_usd):
                        stop_reason = "budget_cap"
                    elif budget_header_missing or not isinstance(latest, dict):
                        stop_reason = "budget_state_unknown"
                    else:
                        stop_reason = "api_failure"
                else:
                    safe = _safe_response(unit.case, payload or {}, self.snapshot,
                                          response_headers, request_id, status_code, elapsed, benchmark_id)
                    if (safe["budget_usage"] is None or
                            safe["budget_usage"].get("run_id") != benchmark_id):
                        row["status"] = "failed"
                        safe.update({"status": "failed", "failure_classification": "budget_usage_header_missing"})
                        stop_reason = "budget_state_unknown"
                    else:
                        row["status"] = safe["status"]
                    row["experiment_run_id"] = safe.get("experiment_run_id")
                    _atomic_json(attempt_path, safe)
                    usage = safe.get("budget_usage") or {}
                    if safe.get("authorization_audit", {}).get("violation"):
                        stop_reason = "security_violation"
                    elif safe.get("authorization_audit", {}).get("unknown_evidence_mapping"):
                        stop_reason = "unknown_evidence_mapping"
                    elif usage.get("estimated_cost_usd", 0) > self.max_cost_usd:
                        stop_reason = "budget_cap"
                    elif (usage.get("total_tokens", 0) >= self.max_total_tokens or
                          usage.get("calls", 0) >= self.max_provider_calls):
                        stop_reason = "budget_cap"
                    elif safe.get("status") == "completed_with_errors":
                        stop_reason = "provider_failure"
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.casefold()
                row["status"] = "timed_out" if timed_out else "failed"
                elapsed_ms = round((time.monotonic() - started) * 1000, 3)
                failure = {"status": row["status"],
                           "failure_classification": type(exc).__name__,
                           "request_id": request_id, "elapsed_ms": elapsed_ms}
                if isinstance(exc, (URLError, TimeoutError, OSError)):
                    failure["transport_failure"] = _transport_failure_details(
                        request_url, exc, elapsed_ms, self.secret)
                _atomic_json(attempt_path, failure)
                stop_reason = ("request_timeout_budget_state_unknown" if timed_out else
                               "runner_failure_budget_state_unknown")
            _atomic_json(progress_path, progress)
            if stop_reason:
                break
        progress["stop_reason"] = stop_reason
        for row in progress["schedule"]:
            if row["status"] == "pending" and stop_reason:
                row["status"] = {
                    "budget_cap": "not_run_budget_cap",
                    "security_violation": "not_run_security_violation",
                    "unknown_evidence_mapping": "not_run_unknown_evidence_mapping",
                    "provider_failure": "not_run_provider_failure",
                    "api_failure": "not_run_api_failure",
                }.get(stop_reason, "not_run_budget_state_unknown")
        terminal = all(row["status"] != "pending" and row["status"] != "in_progress"
                       for row in progress["schedule"])
        progress["status"] = "stopped_by_budget" if stop_reason == "budget_cap" else (
            "stopped_security_violation" if stop_reason == "security_violation" else
            "stopped_unknown_evidence_mapping" if stop_reason == "unknown_evidence_mapping" else
            "stopped_provider_failure" if stop_reason == "provider_failure" else
            "stopped_api_failure" if stop_reason == "api_failure" else
            "stopped_request_timeout_budget_state_unknown" if stop_reason == "request_timeout_budget_state_unknown" else
            "stopped_runner_failure_budget_state_unknown" if stop_reason == "runner_failure_budget_state_unknown" else
            "stopped_budget_state_unknown" if stop_reason else "completed" if terminal and all(
            row["status"] in {"completed", "completed_with_warnings", "completed_with_errors"}
            for row in progress["schedule"]
        ) else "stopped_run_timeout" if any(row["status"] == "not_run_run_timeout"
                                             for row in progress["schedule"])
            else "incomplete" if not terminal else "completed_with_failures")
        progress["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(progress_path, progress)
        return progress
