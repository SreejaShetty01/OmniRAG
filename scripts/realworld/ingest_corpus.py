"""Serial, resumable M1 corpus ingestion through the existing authenticated API.

Dry-run remains the default. Upload requires the explicit --execute flag and a
signed runtime proof that the API is isolated to the manifest collection.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from uuid import uuid4
from api.trusted_proxy import verify_proxy_identity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.realworld.prepare_corpus import CORPUS_ROOT, MANIFEST, validate_corpus
from scripts.realworld.isolation import inspect_dedicated_runtime, validate_dedicated_qdrant

PROTECTED = {"omnirag", "omnirag_eval_phase8_20261104"}
MAX_COST_USD = 0.0
MAX_TOTAL_TOKENS = 4_200_000
MAX_PROVIDER_CALLS = 3_000
DEFAULT_USER_ID = "omnirag-m1p2-ingest"
DEFAULT_TENANT_ID = "omnirag-realworld"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    delays = (0.05, 0.15, 0.4)
    try:
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def sign_identity_headers(user_id: str, tenant_id: str, secret: str,
                          *, timestamp: int | None = None) -> dict[str, str]:
    stamp = str(int(time.time()) if timestamp is None else timestamp)
    message = f"{user_id}\n{tenant_id}\n{stamp}".encode()
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    headers = {"X-OmniRAG-User-ID": user_id, "X-OmniRAG-Tenant-ID": tenant_id,
               "X-OmniRAG-Auth-Timestamp": stamp, "X-OmniRAG-Auth-Signature": signature}
    verified = verify_proxy_identity(user_id, tenant_id, stamp, signature, secret,
                                     now=float(stamp))
    if verified != {"user_id": user_id, "tenant_id": tenant_id}:
        raise RuntimeError("Generated trusted-proxy identity headers failed local verification")
    return headers


def _header(headers: dict, name: str):
    return next((value for key, value in headers.items() if str(key).casefold() == name.casefold()), None)


def _safe_error_details(payload: object) -> dict | None:
    """Keep bounded diagnostic fields only; never persist arbitrary response bodies."""
    if not isinstance(payload, dict):
        return None
    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    detail = payload.get("detail")
    message = error.get("message") or (detail if isinstance(detail, str) else None)
    result = {}
    for key, value in (("code", error.get("code")), ("message", message),
                       ("request_id", error.get("request_id") or payload.get("request_id")),
                       ("run_id", error.get("run_id") or payload.get("run_id"))):
        if isinstance(value, str) and value:
            result[key] = " ".join(value.split())[:500]
    return result or None


def _safe_exception_details(exc: Exception, *, secret: str | None, upload_url: str | None,
                            stage: str) -> dict:
    message = "Permission denied" if isinstance(exc, PermissionError) else " ".join(str(exc).split())[:500]
    if secret:
        message = message.replace(secret, "[REDACTED]")
    if upload_url:
        parts = urlsplit(upload_url)
        safe_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        message = message.replace(upload_url, safe_url)
    result = {"exception_class": type(exc).__name__, "message": message, "stage": stage,
              "likely_sent": False if stage in {"source_read", "request_signing"} else None}
    for name in ("errno", "winerror"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            result[name] = value
    return result


def _http_json(url: str, method: str, headers: dict[str, str], body: bytes | None,
               timeout: float) -> tuple[int, dict[str, str], dict | None]:
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            payload = json.loads(raw) if raw else None
            return response.status, dict(response.headers), payload
    except HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        return exc.code, dict(exc.headers), payload


def _budget(headers: dict[str, str], run_id: str) -> dict:
    raw = _header(headers, "X-OmniRAG-Budget-Usage")
    try:
        value = json.loads(raw) if raw else None
    except (TypeError, json.JSONDecodeError):
        value = None
    if (not isinstance(value, dict) or value.get("run_id") != run_id or
            value.get("max_cost_usd") != MAX_COST_USD or
            value.get("max_total_tokens") != MAX_TOTAL_TOKENS or
            value.get("max_calls") != MAX_PROVIDER_CALLS or
            not isinstance(value.get("calls"), int) or
            not isinstance(value.get("total_tokens"), int) or
            not isinstance(value.get("estimated_cost_usd"), (int, float)) or
            value["calls"] < 0 or value["total_tokens"] < 0 or
            value["estimated_cost_usd"] < 0):
        raise RuntimeError("API budget proof is missing, invalid, or differs from ingestion caps")
    if (value["calls"] > MAX_PROVIDER_CALLS or value["total_tokens"] > MAX_TOTAL_TOKENS or
            value["estimated_cost_usd"] > MAX_COST_USD):
        raise RuntimeError("API budget cap is exhausted")
    return value


def _combined_budget(carry: dict | None, current: dict) -> dict:
    carry = carry or {}
    total = {
        "calls": int(carry.get("calls", 0)) + int(current.get("calls", 0)),
        "total_tokens": int(carry.get("total_tokens", 0)) + int(current.get("total_tokens", 0)),
        "estimated_cost_usd": float(carry.get("estimated_cost_usd", 0.0)) +
                              float(current.get("estimated_cost_usd", 0.0)),
    }
    if (total["calls"] > MAX_PROVIDER_CALLS or total["total_tokens"] > MAX_TOTAL_TOKENS or
            total["estimated_cost_usd"] > MAX_COST_USD):
        raise RuntimeError("Cumulative ingestion budget cap exceeded across prior and continuation runs")
    return total


def verify_api(base_url: str, *, secret: str, user_id: str, tenant_id: str,
               run_id: str, expected_collection: str, embedding_alias: str,
               timeout: float, request_fn=_http_json) -> tuple[dict, dict]:
    parts = urlsplit(base_url)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or
            parts.password or parts.query or parts.fragment):
        raise ValueError("API base URL must be an absolute HTTP(S) URL without credentials/query/fragment")
    base = urlunsplit(parts._replace(path=parts.path.rstrip("/"), query="", fragment="")).rstrip("/")
    headers = sign_identity_headers(user_id, tenant_id, secret)
    headers["X-OmniRAG-Run-ID"] = run_id
    health_status, health_headers, health = request_fn(
        f"{base}/health/ready", "GET", headers, None, timeout)
    if health_status != 200 or not isinstance(health, dict) or health.get("status") != "ready":
        raise RuntimeError(f"Isolated API readiness failed (HTTP {health_status})")
    settings_status, settings_headers, settings = request_fn(
        f"{base}/api/v1/settings", "GET", headers, None, timeout)
    if settings_status != 200 or not isinstance(settings, dict):
        raise RuntimeError(f"Signed isolated API settings proof failed (HTTP {settings_status})")
    if (settings.get("user_id"), settings.get("tenant_id")) != (user_id, tenant_id):
        raise RuntimeError("Signed API identity did not match the selected corpus principal/tenant")
    if settings.get("physical_collection") != expected_collection or not expected_collection.startswith("omnirag_eval_"):
        raise RuntimeError("API physical collection does not match the isolated corpus target")
    embed = settings.get("embedding_space") or {}
    if (embed.get("alias") != embedding_alias or embed.get("provider") != "openai_compatible" or
            embed.get("model_id") != "nomic-embed-text" or embed.get("dimensions") != 768):
        raise RuntimeError("API embedding space does not match the corpus configuration")
    allowed = settings.get("accepted_identities") or []
    if not any(item.get("user_id") == user_id and item.get("tenant_id") == tenant_id
               for item in allowed if isinstance(item, dict)):
        raise RuntimeError("Selected principal/tenant is absent from the API identity allowlist")
    limits = settings.get("limits") or {}
    if (limits.get("max_cost_usd") != MAX_COST_USD or
            limits.get("max_total_tokens") != MAX_TOTAL_TOKENS or
            limits.get("max_calls") != MAX_PROVIDER_CALLS):
        raise RuntimeError("API effective budgets do not match the approved ingestion caps")
    usage = _budget(settings_headers, run_id)
    return settings, usage


def _upload_filename(row: dict) -> str:
    suffix = Path(row["original_path"]).suffix.lower()
    if len(suffix) > 12 or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        raise ValueError("Unsupported manifest filename suffix")
    repo = re.sub(r"[^a-zA-Z0-9_-]", "_", row["document_id"].split(":", 1)[0])[:32]
    stable = hashlib.sha256(row["document_id"].encode()).hexdigest()[:24]
    return f"m1p2_{repo}_{stable}{suffix}"


def _new_progress(manifest: dict, manifest_path: Path, *, run_id: str, user_id: str,
                  tenant_id: str, request_timeout: float, run_timeout: float,
                  initial_budget: dict) -> dict:
    return {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at_unix": time.time(),
        "deadline_unix": time.time() + run_timeout,
        "collection": manifest["target_collection"], "corpus_id": manifest["corpus_id"],
        "content_set_sha256": manifest["content_set_sha256"],
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "identity": {"user_id": user_id, "tenant_id": tenant_id},
        "request_timeout_seconds": request_timeout, "run_timeout_seconds": run_timeout,
        "budget_caps": {"max_cost_usd": MAX_COST_USD, "max_total_tokens": MAX_TOTAL_TOKENS,
                        "max_provider_calls": MAX_PROVIDER_CALLS},
        "budget_usage": initial_budget, "schedule": [
            {"document_id": row["document_id"], "source_path": row["local_path"],
             "source_sha256": row["sha256"], "version": row["repository_commit"],
             "status": "pending"} for row in manifest["sources"]],
    }


def _record_measurements(progress: dict) -> None:
    elapsed = max(0.001, time.time() - progress["started_at_unix"])
    completed = [row for row in progress["schedule"] if row.get("status") in {"indexed", "unchanged"}]
    newly_completed = [row for row in completed if not row.get("reconciled_from_run_id")]
    known_chunks = [row.get("chunk_count") for row in completed
                    if isinstance(row.get("chunk_count"), int) and row["chunk_count"] > 0]
    newly_known_chunks = [row.get("chunk_count") for row in newly_completed
                          if isinstance(row.get("chunk_count"), int) and row["chunk_count"] > 0]
    total_chunks = sum(known_chunks)
    newly_total_chunks = sum(newly_known_chunks)
    progress["measurements"] = {
        "elapsed_seconds": round(elapsed, 3), "completed_documents": len(completed),
        "newly_completed_documents": len(newly_completed),
        "completed_docs_per_second": round(len(newly_completed) / elapsed, 6),
        "known_chunks": total_chunks,
        "newly_completed_chunks": newly_total_chunks,
        "chunks_per_second": round(newly_total_chunks / elapsed, 6) if len(newly_known_chunks) == len(newly_completed) else None,
        "chunk_count_unknown_documents": len(completed) - len(known_chunks),
        "failure_count": sum(row.get("status") in {"failed", "indeterminate"}
                              for row in progress["schedule"]),
        "fallback_count": None, "fallback_count_status": "not_exposed_by_upload_response",
    }


def _read_collection_points(qdrant_url: str, collection: str) -> list:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=qdrant_url)
    try:
        points = []
        offset = None
        while True:
            batch, offset = client.scroll(collection_name=collection, offset=offset,
                                          limit=512, with_payload=True, with_vectors=False)
            points.extend(batch)
            if offset is None:
                return points
    finally:
        client.close()


def _read_api_documents(api_base_url: str, *, user_id: str, tenant_id: str, secret: str,
                        run_id: str, timeout: float, request_fn=_http_json) -> list[dict]:
    headers = sign_identity_headers(user_id, tenant_id, secret)
    headers["X-OmniRAG-Run-ID"] = run_id
    status_code, _, payload = request_fn(f"{api_base_url.rstrip('/')}/api/v1/documents",
                                         "GET", headers, None, timeout)
    if status_code != 200 or not isinstance(payload, list):
        raise RuntimeError(f"Signed document reconciliation failed (HTTP {status_code})")
    return [row for row in payload if isinstance(row, dict)]


def reconcile_prior_run(prior_progress_path: Path, *, manifest: dict, manifest_path: Path,
                        current_isolation_proof: dict, current_api_documents: list[dict],
                        points: list, user_id: str, tenant_id: str,
                        prior_http_status: int | None, retry_failed_after_identity_check: bool,
                        retry_proven_absent_indeterminate: bool = False,
                        retry_unsupported_extension_after_fix: bool = False) -> dict:
    """Reconcile completed source rows to exact SQLite and Qdrant identities."""
    prior = json.loads(prior_progress_path.read_text(encoding="utf-8"))
    if (prior.get("collection") != manifest.get("target_collection") or
            prior.get("manifest_sha256") != _sha256(manifest_path.read_bytes()) or
            prior.get("content_set_sha256") != manifest.get("content_set_sha256") or
            prior.get("identity") != {"user_id": user_id, "tenant_id": tenant_id} or
            prior.get("guard_mode") != "dedicated_qdrant_storage_isolation"):
        raise RuntimeError("Prior run does not match the current isolated collection, identity, or manifest")
    if prior.get("status") != "stopped_first_error":
        raise RuntimeError("Prior run is not a stopped first-error run eligible for continuation")
    prior_proof = prior.get("isolation_proof") or {}
    prior_qdrant = prior_proof.get("qdrant") or {}
    current_qdrant = current_isolation_proof.get("qdrant") or {}
    if (prior_qdrant.get("id") != current_qdrant.get("id") or
            prior_proof.get("qdrant_volume") != current_isolation_proof.get("qdrant_volume")):
        raise RuntimeError("Prior points are not on the currently verified dedicated Qdrant volume")

    source_rows = {row["document_id"]: row for row in manifest.get("sources", [])}
    old_rows = prior.get("schedule") or []
    if len(old_rows) != len(source_rows) or {row.get("document_id") for row in old_rows} != set(source_rows):
        raise RuntimeError("Prior run schedule does not exactly match the corpus manifest")
    by_hash: dict[str, list] = {}
    point_ids: set[str] = set()
    for point in points:
        point_id = str(getattr(point, "id", ""))
        if not point_id or point_id in point_ids:
            raise RuntimeError("Qdrant returned a missing or duplicate point identity")
        point_ids.add(point_id)
        payload = getattr(point, "payload", {}) or {}
        by_hash.setdefault(str(payload.get("source_hash")), []).append(point)

    api_by_hash: dict[str, list[dict]] = {}
    for row in current_api_documents:
        if row.get("source_hash"):
            api_by_hash.setdefault(str(row["source_hash"]), []).append(row)
    reconciled: dict[str, dict] = {}
    failed_rows = []
    allowed_principal = [f"user:{user_id}"]
    for old in old_rows:
        doc_id = old["document_id"]
        source = source_rows[doc_id]
        source_hash = source["sha256"]
        old_status = old.get("status")
        qdrant_rows = by_hash.get(source_hash, [])
        api_rows = api_by_hash.get(source_hash, [])
        if old_status in {"indexed", "unchanged"}:
            expected_api_id = old.get("api_document_id")
            chunk_count = old.get("chunk_count")
            if not expected_api_id or not isinstance(chunk_count, int) or chunk_count <= 0 or len(qdrant_rows) != chunk_count:
                raise RuntimeError(f"Completed prior source failed exact point-count reconciliation: {doc_id}")
            if len(api_rows) != 1:
                raise RuntimeError(f"Completed prior source lacks a unique SQLite document record: {doc_id}")
            api_doc = api_rows[0]
            if (api_doc.get("document_id") != expected_api_id or api_doc.get("status") not in {"indexed", "unchanged"} or
                    api_doc.get("version") != source["repository_commit"] or api_doc.get("chunk_count") != chunk_count):
                raise RuntimeError(f"SQLite document record differs from prior progress: {doc_id}")
            chunk_ids = set()
            for point in qdrant_rows:
                payload = getattr(point, "payload", {}) or {}
                metadata = payload.get("metadata") or {}
                if (payload.get("document_id") != expected_api_id or
                        payload.get("version") != source["repository_commit"] or
                        payload.get("acl_allowed_principals") != allowed_principal or
                        payload.get("acl_denied_principals") or metadata.get("tenant_id") != tenant_id or
                        payload.get("embedder_provider") != "openai_compatible" or
                        payload.get("embedder_model") != "nomic-embed-text" or
                        payload.get("embedding_dim") != 768 or not payload.get("is_active")):
                    raise RuntimeError(f"Prior point ACL, tenant, version, or provenance mismatch: {doc_id}")
                chunk_id = payload.get("chunk_id")
                if not chunk_id or chunk_id in chunk_ids:
                    raise RuntimeError(f"Prior source has missing or duplicate chunk identity: {doc_id}")
                chunk_ids.add(chunk_id)
            ids_digest = hashlib.sha256("\n".join(sorted(str(getattr(p, "id", "")) for p in qdrant_rows)).encode()).hexdigest()
            reconciled[doc_id] = {**old, "point_ids_sha256": ids_digest, "point_count": len(qdrant_rows)}
        elif old_status == "failed":
            retryable_auth_503 = retry_failed_after_identity_check and prior_http_status == 503
            retryable_type_415 = retry_unsupported_extension_after_fix and prior_http_status == 415
            if not (retryable_auth_503 or retryable_type_415):
                raise RuntimeError("Prior failed source requires its specific confirmed HTTP status and explicit correction gate")
            if qdrant_rows or api_rows:
                raise RuntimeError(f"Failed source already has stored API/Qdrant state; refusing retry: {doc_id}")
            failed_rows.append(old)
        elif old_status == "indeterminate":
            if not retry_proven_absent_indeterminate:
                raise RuntimeError("Indeterminate prior source requires explicit safe-retry authorization")
            if qdrant_rows or api_rows:
                raise RuntimeError(f"Indeterminate source has persisted state; refusing retry: {doc_id}")
            failed_rows.append({**old, "safe_retry_reason":
                "signed SQLite listing and Qdrant scroll both confirm no persisted source state"})
        elif old_status == "pending":
            if qdrant_rows or api_rows:
                raise RuntimeError(f"Pending source already has API/Qdrant state; refusing duplicate ingestion: {doc_id}")
        else:
            raise RuntimeError(f"Prior source status is unsafe to continue: {doc_id} ({old_status})")
    if set(by_hash) - {row["sha256"] for row in manifest["sources"]}:
        raise RuntimeError("Qdrant contains points from sources outside the corpus manifest")
    if len(points) != sum(item["point_count"] for item in reconciled.values()):
        raise RuntimeError("Qdrant point total does not equal reconciled completed source points")
    if len(failed_rows) != 1:
        raise RuntimeError("Continuation requires exactly one verified failed source")
    return {"prior_run_id": prior["run_id"], "completed": reconciled,
            "failed_rows": failed_rows, "point_count": len(points),
            "budget_usage": prior.get("cumulative_budget_usage") or _combined_budget(
                prior.get("carried_budget_usage"), prior.get("budget_usage") or {})}


def execute_ingestion(manifest: dict, corpus_root: Path, manifest_path: Path, *,
                      api_base_url: str, secret: str, user_id: str, tenant_id: str,
                      run_id: str, output_dir: Path, request_timeout: float,
                      run_timeout: float, protected_baseline_path: Path | None = None,
                      qdrant_url: str = "http://localhost:6333",
                      protected_baseline_check=None, resume: bool = False,
                      isolated_qdrant: bool = False,
                      api_container_name: str = "omnirag-api-eval-rw",
                      qdrant_container_name: str = "omnirag-qdrant-rw",
                      runtime_inspector=inspect_dedicated_runtime,
                      continue_from_path: Path | None = None,
                      retry_failed_after_identity_check: bool = False,
                      retry_proven_absent_indeterminate: bool = False,
                      retry_unsupported_extension_after_fix: bool = False,
                      prior_http_status: int | None = None,
                      request_fn=_http_json) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("Invalid ingestion run ID")
    if request_timeout <= 0 or run_timeout <= 0:
        raise ValueError("Request and run timeouts must be finite positive values")
    report = validate_corpus(manifest, corpus_root, manifest_path)
    if not report["valid"]:
        raise RuntimeError("Corpus integrity validation failed; refusing upload")
    if manifest.get("target_collection") in PROTECTED or not manifest.get("target_collection", "").startswith("omnirag_eval_"):
        raise RuntimeError("Manifest collection is protected or lacks the evaluation prefix")
    settings, initial_budget = verify_api(
        api_base_url, secret=secret, user_id=user_id, tenant_id=tenant_id, run_id=run_id,
        expected_collection=manifest["target_collection"], embedding_alias=manifest["embedding_space"],
        timeout=min(request_timeout, 15), request_fn=request_fn)
    isolation_proof = None
    preservation = None
    if isolated_qdrant:
        evidence = runtime_inspector(api_container_name, qdrant_container_name)
        isolation_proof = validate_dedicated_qdrant(
            **evidence, api_settings=settings, expected_collection=manifest["target_collection"],
            qdrant_host_url=qdrant_url, user_id=user_id, tenant_id=tenant_id)
        protected_baseline_hash = None
    else:
        if protected_baseline_path is None or not protected_baseline_path.is_file():
            raise RuntimeError("Pre-ingestion dev/eval collection baseline is missing")
        if protected_baseline_check is None:
            from qdrant_client import QdrantClient
            from scripts.realworld.audit_corpus import compare_protected_baselines
            qdrant_client = QdrantClient(url=qdrant_url)
            try:
                preservation = compare_protected_baselines(qdrant_client, protected_baseline_path)
            finally:
                qdrant_client.close()
        else:
            preservation = protected_baseline_check(qdrant_url, protected_baseline_path)
        if not preservation.get("preserved"):
            raise RuntimeError("Development or Phase 8 collection baseline changed before ingestion")
        protected_baseline_hash = _sha256(protected_baseline_path.read_bytes())
    if manifest["sources"] and max(row["file_size_bytes"] for row in manifest["sources"]) > settings["limits"]["max_upload_bytes"]:
        raise RuntimeError("At least one corpus file exceeds the API upload byte limit")

    continuation = None
    if continue_from_path is not None:
        if not isolated_qdrant:
            raise RuntimeError("Prior-run reconciliation requires the dedicated Qdrant isolation guard")
        points = _read_collection_points(qdrant_url, manifest["target_collection"])
        api_documents = _read_api_documents(api_base_url, user_id=user_id, tenant_id=tenant_id,
            secret=secret, run_id=run_id, timeout=min(request_timeout, 15), request_fn=request_fn)
        continuation = reconcile_prior_run(
            continue_from_path, manifest=manifest, manifest_path=manifest_path,
            current_isolation_proof=isolation_proof, current_api_documents=api_documents,
            points=points, user_id=user_id, tenant_id=tenant_id,
            prior_http_status=prior_http_status or 0,
            retry_failed_after_identity_check=retry_failed_after_identity_check,
            retry_proven_absent_indeterminate=retry_proven_absent_indeterminate,
            retry_unsupported_extension_after_fix=retry_unsupported_extension_after_fix)

    progress_path = output_dir / "progress.json"
    if output_dir.exists():
        if not resume or not progress_path.is_file():
            raise FileExistsError(f"Ingestion run directory already exists: {output_dir}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected_manifest_hash = _sha256(manifest_path.read_bytes())
        if (progress.get("run_id") != run_id or progress.get("manifest_sha256") != expected_manifest_hash or
                progress.get("collection") != manifest["target_collection"] or
                progress.get("identity") != {"user_id": user_id, "tenant_id": tenant_id} or
                progress.get("request_timeout_seconds") != request_timeout or
                progress.get("run_timeout_seconds") != run_timeout or
                progress.get("protected_baseline_sha256") != protected_baseline_hash or
                progress.get("guard_mode") != ("dedicated_qdrant_storage_isolation" if isolated_qdrant else "shared_qdrant_fingerprints") or
                progress.get("isolation_proof_sha256") != (isolation_proof or {}).get("proof_sha256") or
                progress.get("budget_caps") != {"max_cost_usd": MAX_COST_USD,
                    "max_total_tokens": MAX_TOTAL_TOKENS, "max_provider_calls": MAX_PROVIDER_CALLS}):
            raise RuntimeError("Resume configuration differs from the recorded ingestion run")
        previous_budget = progress.get("budget_usage") or {}
        if (not isinstance(previous_budget, dict) or previous_budget.get("run_id") != run_id or
                not isinstance(previous_budget.get("calls"), int) or
                not isinstance(previous_budget.get("total_tokens"), int)):
            raise RuntimeError("Resume progress lacks a valid provider budget ledger")
        if not isinstance(progress.get("deadline_unix"), (int, float)):
            raise RuntimeError("Resume record lacks the cumulative ingestion deadline")
        if progress["deadline_unix"] <= time.time():
            progress["status"] = "stopped_run_timeout"
            progress["stop_reason"] = "cumulative_deadline_expired"
            _atomic_json(progress_path, progress)
            raise RuntimeError("Cumulative ingestion deadline expired; refusing to resume")
        if (initial_budget.get("calls", -1) < previous_budget.get("calls", 0) or
                initial_budget.get("total_tokens", -1) < previous_budget.get("total_tokens", 0)):
            raise RuntimeError("API budget ledger reset/regressed; refusing to resume")
        expected_rows = {row["document_id"]: row for row in manifest["sources"]}
        recorded_ids = [row.get("document_id") for row in progress.get("schedule", [])]
        if len(recorded_ids) != len(set(recorded_ids)) or set(recorded_ids) != set(expected_rows):
            raise RuntimeError("Resume schedule does not exactly match the source manifest")
        for row in progress.get("schedule", []):
            source = expected_rows.get(row.get("document_id"))
            if source is None or source.get("sha256") != row.get("source_sha256"):
                raise RuntimeError("Resume schedule differs from the verified source manifest")
            if row.get("status") in {"failed", "indeterminate", "in_progress", "not_run_run_timeout"}:
                if row.get("status") == "in_progress":
                    row["status"] = "indeterminate"
                    row["failure_classification"] = "process_interrupted_during_upload"
                    _atomic_json(progress_path, progress)
                raise RuntimeError("Prior failed/indeterminate upload requires manual audit; not retrying")
            if row.get("status") in {"indexed", "unchanged"} and not row.get("api_document_id"):
                raise RuntimeError("Completed resume row lacks its API document ID; refusing to skip")
        if progress.get("status") == "completed":
            return progress
        progress["status"] = "running"
        progress["budget_usage"] = initial_budget
        progress["cumulative_budget_usage"] = _combined_budget(
            progress.get("carried_budget_usage"), initial_budget)
    else:
        progress = _new_progress(manifest, manifest_path, run_id=run_id, user_id=user_id,
                                  tenant_id=tenant_id, request_timeout=request_timeout,
                                  run_timeout=run_timeout, initial_budget=initial_budget)
        progress["protected_baseline_sha256"] = protected_baseline_hash
        progress["guard_mode"] = "dedicated_qdrant_storage_isolation" if isolated_qdrant else "shared_qdrant_fingerprints"
        progress["isolation_proof"] = isolation_proof
        progress["isolation_proof_sha256"] = (isolation_proof or {}).get("proof_sha256")
        progress["protected_collections_before"] = (preservation or {}).get("current")
        progress["carried_budget_usage"] = (continuation or {}).get("budget_usage") or {
            "calls": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
        progress["cumulative_budget_usage"] = _combined_budget(
            progress["carried_budget_usage"], initial_budget)
        if continuation:
            failed_by_id = {row["document_id"]: row for row in continuation["failed_rows"]}
            for row in progress["schedule"]:
                prior = continuation["completed"].get(row["document_id"])
                if prior:
                    row.update({"status": prior["status"], "api_document_id": prior["api_document_id"],
                                "chunk_count": prior["chunk_count"],
                                "reconciled_from_run_id": continuation["prior_run_id"],
                                "reconciled_point_count": prior["point_count"],
                                "reconciled_point_ids_sha256": prior["point_ids_sha256"]})
                elif row["document_id"] in failed_by_id:
                    prior = failed_by_id[row["document_id"]]
                    row["prior_failure"] = {"run_id": continuation["prior_run_id"],
                        "request_id": prior.get("request_id"),
                        "failure_classification": prior.get("failure_classification"),
                        "confirmed_http_status": prior_http_status,
                        "safe_retry_reason": prior.get("safe_retry_reason"),
                        "retry_gate": "fresh signed identity settings proof passed"}
            progress["continuation"] = {"prior_run_id": continuation["prior_run_id"],
                "reconciled_completed_files": len(continuation["completed"]),
                "reconciled_point_count": continuation["point_count"],
                "confirmed_prior_http_status": prior_http_status,
                "retry_failed_after_identity_check": retry_failed_after_identity_check}
        if isolated_qdrant:
            progress["historical_protected_collection_references"] = [
                {"path": "docs/audit/phase8_dev_collection_baseline.json",
                 "recorded_date": "2026-11-04", "interpretation": "historical_reference_not_current_measurement"},
                {"path": "docs/audit/phase8_eval_corpus_manifest.json",
                 "snapshot_sha256": "2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847",
                 "interpretation": "historical_reference_not_current_measurement"},
            ]
        output_dir.mkdir(parents=True, exist_ok=False)

    source_rows = {row["document_id"]: row for row in manifest["sources"]}
    for status_row in progress["schedule"]:
        if status_row["status"] in {"indexed", "unchanged"}:
            continue
        remaining = progress["deadline_unix"] - time.time()
        if remaining <= 0:
            status_row["status"] = "not_run_run_timeout"
            progress["status"] = "stopped_run_timeout"
            progress["stop_reason"] = "run_timeout"
            break
        source = source_rows[status_row["document_id"]]
        stage = "source_read"
        source_path = (corpus_root / source["local_path"]).resolve()
        if corpus_root.resolve() not in source_path.parents:
            raise RuntimeError("Manifest source path escaped the corpus root")
        body = source_path.read_bytes()
        if _sha256(body) != source["sha256"] or len(body) != source["file_size_bytes"]:
            raise RuntimeError(f"Source integrity changed before upload: {source['document_id']}")
        query = urlencode({"filename": _upload_filename(source), "version": source["repository_commit"]})
        base = api_base_url.rstrip("/")
        url = f"{base}/api/v1/documents/upload?{query}"
        upload_started = time.monotonic()
        stage = "request_signing"
        try:
            attempt_records = []
            max_attempts = 2 if status_row.get("prior_failure") else 1
            final_status = None
            final_headers = {}
            response = None
            for attempt_number in range(1, max_attempts + 1):
                if attempt_number > 1:
                    # A prior exact authentication-dependency 503 is retried once only
                    # after a fresh signed settings request proves the same identity.
                    stage = "identity_recheck"
                    verify_api(api_base_url, secret=secret, user_id=user_id, tenant_id=tenant_id,
                        run_id=run_id, expected_collection=manifest["target_collection"],
                        embedding_alias=manifest["embedding_space"],
                        timeout=min(request_timeout, 15), request_fn=request_fn)
                request_id = str(uuid4())
                stage = "request_signing"
                headers = sign_identity_headers(user_id, tenant_id, secret)
                headers.update({"Content-Type": "application/octet-stream", "X-Request-ID": request_id,
                                "X-OmniRAG-Run-ID": run_id})
                status_row.update({"status": "in_progress", "request_id": request_id,
                                   "attempt_number": attempt_number})
                _atomic_json(progress_path, progress)
                attempt_started = time.monotonic()
                stage = "http_request"
                try:
                    status_code, response_headers, response = request_fn(
                        url, "PUT", headers, body, min(request_timeout, remaining))
                except Exception as request_exc:
                    attempt_record = {"attempt": attempt_number, "request_id": request_id,
                        "transport_exception": type(request_exc).__name__,
                        "elapsed_ms": round((time.monotonic() - attempt_started) * 1000, 3)}
                    attempt_records.append(attempt_record)
                    status_row["attempts"] = attempt_records
                    raise
                final_status, final_headers = status_code, response_headers
                error_details = _safe_error_details(response)
                attempt_record = {"attempt": attempt_number, "request_id": request_id,
                    "http_status": status_code,
                    "elapsed_ms": round((time.monotonic() - attempt_started) * 1000, 3)}
                if error_details:
                    attempt_record["error"] = error_details
                attempt_records.append(attempt_record)
                status_row["attempts"] = attempt_records
                # Retry only the known dependency failure, and only on the failed
                # document explicitly gated by continuation reconciliation.
                is_auth_dependency_503 = (status_code == 503 and error_details and
                    error_details.get("message") == "A trusted authentication provider is not configured")
                if not (attempt_number == 1 and max_attempts == 2 and is_auth_dependency_503):
                    break
            if final_status is None:
                raise RuntimeError("Upload attempt did not produce an HTTP response")
            stage = "budget_validation"
            if _header(final_headers, "X-OmniRAG-Budget-Usage") is not None:
                usage = _budget(final_headers, run_id)
                previous = progress.get("budget_usage") or {}
                if (usage["calls"] < previous.get("calls", 0) or
                        usage["total_tokens"] < previous.get("total_tokens", 0)):
                    raise RuntimeError("API budget ledger reset/regressed during ingestion")
                progress["budget_usage"] = usage
                progress["cumulative_budget_usage"] = _combined_budget(
                    progress.get("carried_budget_usage"), usage)
            elif final_status in {200, 201}:
                raise RuntimeError("Successful upload omitted the API budget proof")
            status_code = final_status
            stage = "response_validation"
            if status_code not in {200, 201} or not isinstance(response, dict):
                status_row["http_status"] = status_code
                status_row["error_details"] = _safe_error_details(response)
                raise RuntimeError(f"Upload returned unexpected HTTP status {status_code}")
            if (response.get("source_hash") != source["sha256"] or
                    response.get("version") != source["repository_commit"] or
                    response.get("status") not in {"indexed", "unchanged"} or
                    not response.get("document_id")):
                raise RuntimeError("Upload response did not verify source hash, version, and ingestion status")
            status_row.update({"status": response["status"], "api_document_id": response["document_id"],
                               "chunk_count": response.get("chunk_count"), "http_status": status_code,
                               "elapsed_ms": round((time.monotonic() - upload_started) * 1000, 3),
                               "budget_usage": progress.get("budget_usage"),
                               "cumulative_budget_usage": progress.get("cumulative_budget_usage"),
                               "attempts": attempt_records})
            _atomic_json(progress_path, progress)
        except Exception as exc:
            status_row.update({"status": "indeterminate" if isinstance(
                exc, (URLError, TimeoutError, OSError)) else "failed",
                "failure_classification": type(exc).__name__,
                "elapsed_ms": round((time.monotonic() - upload_started) * 1000, 3)})
            status_row["exception"] = _safe_exception_details(
                exc, secret=secret, upload_url=url, stage=stage)
            if "attempt_records" in locals():
                status_row["attempts"] = attempt_records
            if 'final_status' in locals() and final_status is not None:
                status_row["http_status"] = final_status
                status_row["error_details"] = _safe_error_details(response)
                status_row["attempts"] = attempt_records
                status_row["request_id"] = attempt_records[-1].get("request_id") if attempt_records else None
            if status_row.get("prior_failure"):
                progress["budget_usage"] = progress.get("budget_usage") or initial_budget
                progress["cumulative_budget_usage"] = _combined_budget(
                    progress.get("carried_budget_usage"), progress["budget_usage"])
            progress["status"] = "stopped_first_error"
            progress["stop_reason"] = type(exc).__name__
            _record_measurements(progress)
            _atomic_json(progress_path, progress)
            raise RuntimeError(f"Ingestion stopped at {source['document_id']} ({type(exc).__name__}); see progress.json") from None

    if isolated_qdrant:
        try:
            final_evidence = runtime_inspector(api_container_name, qdrant_container_name)
            final_proof = validate_dedicated_qdrant(
                **final_evidence, api_settings=settings, expected_collection=manifest["target_collection"],
                qdrant_host_url=qdrant_url, user_id=user_id, tenant_id=tenant_id)
            if final_proof.get("proof_sha256") != progress.get("isolation_proof_sha256"):
                raise RuntimeError("dedicated runtime identity or storage changed")
            progress["isolation_proof_after"] = final_proof
        except Exception as exc:
            progress["status"] = "stopped_isolation_guard"
            progress["stop_reason"] = type(exc).__name__
            _record_measurements(progress)
            _atomic_json(progress_path, progress)
            raise RuntimeError("Post-ingestion dedicated isolation proof failed; see progress.json") from None
    if all(row["status"] in {"indexed", "unchanged"} for row in progress["schedule"]):
        progress["status"] = "completed"
        progress["stop_reason"] = None
    _record_measurements(progress)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(progress_path, progress)
    return progress


def build_dry_run(manifest: dict, corpus_root: Path, evaluation_manifest: Path = MANIFEST,
                  *, api_settings: dict | None = None, isolation_proof: dict | None = None,
                  require_isolation: bool = False,
                  continuation_summary: dict | None = None) -> dict:
    integrity = validate_corpus(manifest, corpus_root, evaluation_manifest)
    collection = manifest.get("target_collection", "")
    collision = collection in PROTECTED
    files = [{"document_id": row.get("document_id"), "source_sha256": row.get("sha256"),
              "version": row.get("repository_commit"), "status": "planned_not_started"}
             for row in manifest.get("sources", [])]
    blockers = []
    if not integrity["valid"]:
        blockers.append("corpus integrity validation failed")
    if collision:
        blockers.append("target collection collides with a protected collection")
    if not collection.startswith("omnirag_eval_"):
        blockers.append("target collection lacks required evaluation prefix")
    api_verified = False
    if api_settings is None:
        blockers.append("no signed /api/v1/settings proof from the isolated API instance was supplied")
    elif api_settings.get("physical_collection") != collection:
        blockers.append("signed API settings physical collection does not match the manifest target")
    else:
        api_verified = True
        max_upload = (api_settings.get("limits") or {}).get("max_upload_bytes")
        if isinstance(max_upload, int) and any(row.get("file_size_bytes", 0) > max_upload
                                                for row in manifest.get("sources", [])):
            blockers.append("a manifest file exceeds the API upload byte limit")
    if require_isolation and not (isolation_proof or {}).get("verified"):
        blockers.append("dedicated Qdrant storage isolation is not proven")
    if continuation_summary is not None and continuation_summary.get("verified") is not True:
        blockers.append("prior-run reconciliation did not pass")
    return {"mode": "dry_run_only", "collection": collection,
            "source_count": len(files), "per_file_status": "planned_not_started",
            "planned_file_count": len(files), "idempotency_key_fields": ["document_id", "repository_commit", "sha256"],
            "first_error_policy": "stop_before_next_file", "resume_policy": "skip only hash-verified completed files",
            "identity": "runtime proof required" if not api_verified else "verified by signed API settings",
            "integrity": integrity, "api_physical_collection_verified": api_verified,
            "isolation_proof": isolation_proof,
            "continuation_reconciliation": continuation_summary,
            "execute_supported": True, "ready": not blockers, "blockers": blockers,
            "files": files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.phase8-rw")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument("--protected-baseline", type=Path, default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--isolated-qdrant", action="store_true",
                        help="prove the dedicated qdrant-rw Docker boundary instead of shared fingerprints")
    parser.add_argument("--api-container-name", default="omnirag-api-eval-rw")
    parser.add_argument("--qdrant-container-name", default="omnirag-qdrant-rw")
    parser.add_argument("--principal-user-id", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path, default=None,
                        help="reconcile a prior stopped run into a new unique run; never edits prior artifacts")
    parser.add_argument("--retry-failed-after-identity-check", action="store_true",
                        help="permit the one prior failed source retry after fresh signed settings verification")
    parser.add_argument("--retry-proven-absent-indeterminate", action="store_true",
                        help="permit retry only after signed SQLite listing and Qdrant scroll show no source state")
    parser.add_argument("--retry-confirmed-415-after-extension-fix", action="store_true",
                        help="permit retry of a confirmed 415 only after the extension support fix is deployed")
    parser.add_argument("--confirmed-prior-http-status", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="perform serial uploads; requires isolated API proof")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    api_settings = None
    isolation_proof = None
    env_values = None
    user_id, tenant_id, secret = None, None, None
    run_id = args.run_id
    if args.api_base_url:
        env_values = read_env_file(args.env_file)
        secret = env_values.get("OMNIRAG_AUTH_SHARED_SECRET")
        try:
            configured = json.loads(env_values.get("OMNIRAG_ACCEPTED_IDENTITIES_JSON", "[]"))
        except json.JSONDecodeError as exc:
            raise SystemExit("Isolated env file has invalid accepted-identities JSON") from exc
        selected = next((item for item in configured if
                         (args.principal_user_id is None or item.get("user_id") == args.principal_user_id) and
                         (args.tenant_id is None or item.get("tenant_id") == args.tenant_id)), None)
        if not secret or not selected:
            raise SystemExit("Isolated env file lacks signing secret or selected accepted identity")
        user_id, tenant_id = selected["user_id"], selected["tenant_id"]
        run_id = run_id or f"m1p2-proof-{uuid4()}"
        try:
            api_settings, _ = verify_api(args.api_base_url, secret=secret, user_id=user_id,
                tenant_id=tenant_id, run_id=run_id, expected_collection=manifest["target_collection"],
                embedding_alias=manifest["embedding_space"], timeout=min(args.request_timeout, 15))
        except Exception as exc:
            if args.execute:
                raise SystemExit(f"Isolated API proof failed: {type(exc).__name__}") from None
            print(json.dumps({"mode": "dry_run_only", "ready": False,
                              "api_proof_error_type": type(exc).__name__}, indent=2))
            return 2
        if args.isolated_qdrant:
            try:
                evidence = inspect_dedicated_runtime(args.api_container_name, args.qdrant_container_name)
                isolation_proof = validate_dedicated_qdrant(
                    **evidence, api_settings=api_settings, expected_collection=manifest["target_collection"],
                    qdrant_host_url=args.qdrant_url, user_id=user_id, tenant_id=tenant_id)
            except Exception as exc:
                if args.execute:
                    raise SystemExit(f"Dedicated Qdrant isolation proof failed: {type(exc).__name__}") from None
                print(json.dumps({"mode": "dry_run_only", "ready": False,
                                  "isolation_proof_error_type": type(exc).__name__}, indent=2))
                return 2
    continuation_summary = None
    if args.continue_from is not None:
        valid_retry_gate = (args.retry_failed_after_identity_check and args.confirmed_prior_http_status == 503) or (
            args.retry_proven_absent_indeterminate and args.confirmed_prior_http_status is None) or (
            args.retry_confirmed_415_after_extension_fix and args.confirmed_prior_http_status == 415)
        if not (args.isolated_qdrant and valid_retry_gate and args.api_base_url and args.run_id):
            raise SystemExit("--continue-from requires isolated Qdrant, a new run id, and either confirmed HTTP 503 retry or a proven-absent indeterminate retry")
        try:
            points = _read_collection_points(args.qdrant_url, manifest["target_collection"])
            api_documents = _read_api_documents(args.api_base_url, user_id=user_id, tenant_id=tenant_id,
                secret=secret, run_id=args.run_id, timeout=min(args.request_timeout, 15))
            reconciled = reconcile_prior_run(args.continue_from, manifest=manifest,
                manifest_path=args.manifest, current_isolation_proof=isolation_proof,
                current_api_documents=api_documents, points=points, user_id=user_id,
                tenant_id=tenant_id, prior_http_status=args.confirmed_prior_http_status,
                retry_failed_after_identity_check=args.retry_failed_after_identity_check,
                retry_proven_absent_indeterminate=args.retry_proven_absent_indeterminate,
                retry_unsupported_extension_after_fix=args.retry_confirmed_415_after_extension_fix)
            continuation_summary = {"verified": True, "prior_run_id": reconciled["prior_run_id"],
                "reconciled_files": len(reconciled["completed"]),
                "reconciled_points": reconciled["point_count"],
                "failed_files_to_retry": len(reconciled["failed_rows"]),
                "pending_files": len(manifest["sources"]) - len(reconciled["completed"]) - len(reconciled["failed_rows"])}
        except Exception as exc:
            if args.execute:
                raise SystemExit(f"Prior-run reconciliation failed safely: {type(exc).__name__}") from None
            print(json.dumps({"mode": "dry_run_only", "ready": False,
                              "continuation_error_type": type(exc).__name__}, indent=2))
            return 2
    if args.execute:
        missing_shared_guard = not args.isolated_qdrant and args.protected_baseline is None
        if (not args.api_base_url or not args.run_id or not args.output_dir or args.run_timeout is None or
                missing_shared_guard):
            raise SystemExit("--execute requires API URL, run ID, output dir, finite run timeout, and the applicable isolation or protected-baseline guard")
        try:
            result = execute_ingestion(manifest, args.corpus_root, args.manifest,
                api_base_url=args.api_base_url, secret=secret, user_id=user_id, tenant_id=tenant_id,
                run_id=args.run_id, output_dir=args.output_dir, request_timeout=args.request_timeout,
                run_timeout=args.run_timeout, protected_baseline_path=args.protected_baseline,
                qdrant_url=args.qdrant_url, resume=args.resume, isolated_qdrant=args.isolated_qdrant,
                api_container_name=args.api_container_name, qdrant_container_name=args.qdrant_container_name,
                continue_from_path=args.continue_from,
                retry_failed_after_identity_check=args.retry_failed_after_identity_check,
                retry_proven_absent_indeterminate=args.retry_proven_absent_indeterminate,
                retry_unsupported_extension_after_fix=args.retry_confirmed_415_after_extension_fix,
                prior_http_status=args.confirmed_prior_http_status)
        except Exception as exc:
            raise SystemExit(f"Ingestion failed safely: {type(exc).__name__}; inspect the run progress record") from None
        print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                          "completed_files": sum(row["status"] in {"indexed", "unchanged"}
                                                 for row in result["schedule"]),
                          "total_files": len(result["schedule"]),
                          "progress_path": str(args.output_dir / "progress.json")}, indent=2))
        return 0 if result["status"] == "completed" else 2
    report = build_dry_run(manifest, args.corpus_root, args.manifest, api_settings=api_settings,
                           isolation_proof=isolation_proof, require_isolation=args.isolated_qdrant,
                           continuation_summary=continuation_summary)
    if not args.list_files:
        report.pop("files")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
