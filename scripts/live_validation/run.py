"""One-command Phase 7a live validation; --fake-providers is explicitly NON-LIVE."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "fixtures" / "phase7_smoke"
QUERIES = ROOT / "docs" / "audit" / "smoke_queries.yaml"
RUNS = ROOT / "docs" / "audit" / "runs"
INDEX_FIELDS = {
    "document_id", "chunk_id", "parent_id", "source", "source_type", "document_type",
    "content_type", "language", "version", "section_path", "generation", "source_hash",
    "is_active", "acl_allowed_principals", "acl_denied_principals", "metadata.tenant_id",
    "metadata.workspace_id", "metadata.collection_id", "metadata.block_ids", "metadata.chunk_index",
}
PROVENANCE_FIELDS = {
    "embedder_provider", "embedder_model", "embedding_dim", "sparse_method",
    "sparse_tokenizer_version", "chunker_version", "enrichment_mode",
    "ingestion_schema_version", "ingested_at",
}


def _read_queries() -> list[dict]:
    # smoke_queries.yaml is JSON-compatible YAML to keep the runner stdlib-only.
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    if len(queries) < 40:
        raise RuntimeError(f"Expected at least 40 smoke queries; found {len(queries)}")
    return queries


def _corpus_files() -> list[Path]:
    rows = []
    for line in (ROOT / "docs" / "audit" / "smoke_corpus_manifest.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            name = line.split("`", 2)[1]
            path = CORPUS / name
            if not path.is_file():
                raise RuntimeError(f"Smoke corpus file is missing: {path.name}")
            rows.append(path)
    if len(rows) < 20:
        raise RuntimeError(f"Expected approximately 20 smoke files; found {len(rows)}")
    return rows


def _new_run_directory(non_live: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RUNS / f"{stamp}{'-non-live' if non_live else ''}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _settings():
    sys.path.insert(0, str(ROOT))
    from omnirag.config.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


def _budget_preflight(settings) -> dict:
    from omnirag.infrastructure.provider_budget import activate_run_budget

    limits = (settings.validation_max_cost_usd, settings.validation_max_total_tokens,
              settings.validation_max_calls)
    if any(value is None for value in limits):
        raise RuntimeError("Set OMNIRAG_VALIDATION_MAX_COST_USD, OMNIRAG_VALIDATION_MAX_TOTAL_TOKENS, "
                           "and OMNIRAG_VALIDATION_MAX_CALLS before live validation")
    table = json.loads(settings.provider_price_table_json)
    required = [(settings.embedding_provider or "openai_compatible", settings.embedding_model_id),
                (settings.generation_provider or "openai_compatible", settings.generation_model_id)]
    if settings.generation_fallback_model_id:
        required.append((settings.generation_fallback_provider or "openai_compatible_fallback",
                         settings.generation_fallback_model_id))
    if settings.verification_model_id:
        required.append((settings.verification_provider or "openai_compatible", settings.verification_model_id))
    if settings.reranker_model_id:
        required.append((settings.reranker_provider or "http", settings.reranker_model_id))
    missing = [f"{provider}/{model}" for provider, model in required
               if model and f"{provider}/{model}" not in table]
    if missing:
        raise RuntimeError("Provider price table is missing configured models: " + ", ".join(missing))
    if not settings.embedding_endpoint or not settings.embedding_model_id:
        raise RuntimeError("Embedding endpoint and model ID must be configured")
    if not settings.generation_endpoint or not settings.generation_model_id:
        raise RuntimeError("Generation endpoint and model ID must be configured")
    if not settings.auth_shared_secret:
        raise RuntimeError("OMNIRAG_AUTH_SHARED_SECRET is required for signed test identities")
    # Build a ledger before any network call, verifying that limits and price JSON are usable.
    run_id = f"preflight-{uuid4()}"
    with activate_run_budget(run_id, settings) as budget:
        if budget is None:
            raise RuntimeError("Provider budget guard did not activate")
        return budget.snapshot()


def _signed_headers(user_id: str, tenant_id: str, shared_secret: str, run_id: str) -> dict[str, str]:
    stamp = str(int(time.time()))
    signed = f"{user_id}\n{tenant_id}\n{stamp}".encode()
    signature = hmac.new(shared_secret.encode(), signed, hashlib.sha256).hexdigest()
    return {"X-OmniRAG-User-ID": user_id, "X-OmniRAG-Tenant-ID": tenant_id,
            "X-OmniRAG-Auth-Timestamp": stamp, "X-OmniRAG-Auth-Signature": signature,
            "X-OmniRAG-Run-ID": run_id}


def _http(method: str, url: str, headers: dict[str, str], data: bytes | None = None):
    request = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen(request, timeout=180) as response:
            body = response.read()
            return response.status, dict(response.headers), json.loads(body) if body else None
    except HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = {"error": "non_json_error"}
        return exc.code, dict(exc.headers), parsed
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"HTTP {method} failed ({type(exc).__name__})") from None


def _redact(value, secrets: list[str]):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    return value


def _qdrant_client(settings):
    from qdrant_client import QdrantClient
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key, timeout=10)


def _check_qdrant_connectivity(client):
    """Verify that Qdrant answers independently of whether a collection exists."""
    return client.get_collections()


def _collection_info_after_ingestion(client, collection: str):
    """Require the application ingestion path to have initialized its target collection."""
    if not client.collection_exists(collection):
        raise RuntimeError(
            f"Qdrant collection {collection!r} was not created by smoke-corpus ingestion"
        )
    return client.get_collection(collection)


def _validate_collection_vector_config(info, settings, expected_dimension: int) -> None:
    from qdrant_client import models

    params = info.config.params
    vectors = params.vectors
    dense = vectors.get(settings.dense_vector_name) if isinstance(vectors, dict) else None
    if dense is None:
        raise RuntimeError(f"Qdrant collection is missing named dense vector {settings.dense_vector_name!r}")
    if dense.size != expected_dimension:
        raise RuntimeError(
            f"Qdrant dense vector dimension is {dense.size}; expected {expected_dimension} from indexed points"
        )
    if dense.distance != models.Distance.COSINE:
        raise RuntimeError(f"Qdrant dense vector distance is {dense.distance}; expected Cosine")

    sparse_vectors = params.sparse_vectors or {}
    sparse = sparse_vectors.get(settings.sparse_vector_name)
    if sparse is None:
        raise RuntimeError(f"Qdrant collection is missing named sparse vector {settings.sparse_vector_name!r}")
    if sparse.modifier != models.Modifier.IDF:
        raise RuntimeError(f"Qdrant sparse vector modifier is {sparse.modifier}; expected IDF")


def _document_point_count(client, collection: str, document_id: str) -> int:
    from qdrant_client import models
    filt = models.Filter(must=[models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))])
    count = 0
    offset = None
    while True:
        points, offset = client.scroll(collection_name=collection, scroll_filter=filt,
                                       limit=256, offset=offset, with_payload=False, with_vectors=False)
        count += len(points)
        if offset is None:
            return count


def _payload_all(client, collection: str, document_id: str):
    from qdrant_client import models
    filt = models.Filter(must=[models.FieldCondition(
        key="document_id", match=models.MatchValue(value=document_id))])
    points, _ = client.scroll(collection_name=collection, scroll_filter=filt, limit=8,
                              with_payload=True, with_vectors=False)
    return points


def _identity_for(query: dict, env: dict[str, str], secret: str, run_id: str) -> dict[str, str]:
    suffix = "B" if query.get("identity") == "user_b" else "A"
    user = env.get(f"OMNIRAG_7A_USER_{suffix}_ID", f"phase7a-user-{suffix.lower()}")
    tenant = env.get(f"OMNIRAG_7A_TENANT_{suffix}_ID", env.get("OMNIRAG_7A_TENANT_A_ID", "phase7a-tenant-a"))
    return _signed_headers(user, tenant, secret, run_id)


def _mechanical_checks(query: dict, response: dict) -> list[str]:
    failures = []
    citations = response.get("citations", [])
    provided = {row.get("chunk_id") for row in response.get("evidence_references", [])}
    selected = {row.get("chunk_id") for row in response.get("retrieval_inspection", [])
                if row.get("selected")}
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        if chunk_id not in provided or chunk_id not in selected:
            failures.append("citation_not_in_provided_evidence")
    if query["category"] == "unanswerable" and not response.get("abstained"):
        failures.append("unanswerable_query_did_not_abstain")
    if query["category"] == "version_latest":
        versions = {item.get("version") for item in citations if item.get("version")}
        if "v1" in versions:
            failures.append("latest_query_cited_superseded_v1")
    if query["category"] == "acl_user_b" and any(
        "12_restricted_team.txt" in str(item.get("source", "")) for item in citations
    ):
        failures.append("restricted_document_cited_for_user_b")
    metadata = response.get("execution_metadata", {})
    if metadata.get("escalation_status") not in (None, "not_needed", "disabled_by_user"):
        if not all(metadata.get(key) for key in ("escalation_status", "escalation_reason")):
            failures.append("escalation_contract_fields_missing")
    return failures


def _percentile(values: list[float], pct: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct * len(ordered)) - 1)]


def fake_run() -> Path:
    from omnirag.infrastructure.provider_budget import BudgetExceeded, RunBudget
    queries, corpus = _read_queries(), _corpus_files()
    run_dir = _new_run_directory(non_live=True)
    ledger = RunBudget("fake-budget-check", max_cost_usd=1.0, max_total_tokens=2,
                       max_calls=1, price_table={"fake/model": {"input_per_million_usd": 1.0,
                                                                   "output_per_million_usd": 1.0}})
    ledger.reserve("fake", "model", 1, 0)
    budget_trip = False
    try:
        ledger.reserve("fake", "model", 1, 0)
    except BudgetExceeded:
        budget_trip = True
    summary = {
        "mode": "NON-LIVE", "live": False, "providers_called": False,
        "qdrant_contacted": False, "corpus_files_found": len(corpus),
        "queries_found": len(queries), "fake_budget_trip_observed": budget_trip,
        "output_notice": "NON-LIVE harness logic check only; not product or provider evidence.",
    }
    if not budget_trip:
        raise RuntimeError("Fake budget trip self-check did not fire")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(
        "# NON-LIVE Phase 7a harness check\n\n" +
        "No provider or Qdrant calls were made. This output validates only local preflight parsing "
        "and the fake budget trip. It is not live evidence.\n\n" +
        f"- Corpus files present: {len(corpus)}\n- Queries loaded: {len(queries)}\n"
        f"- Fake budget trip: {budget_trip}\n", encoding="utf-8")
    print(f"NON-LIVE: harness self-check passed; providers/Qdrant not contacted; output={run_dir}")
    return run_dir


def live_run(api_url: str) -> Path:
    settings = _settings()
    queries, files = _read_queries(), _corpus_files()
    budget_preflight = _budget_preflight(settings)
    client = _qdrant_client(settings)
    try:
        _check_qdrant_connectivity(client)
    except Exception as exc:
        client.close()
        raise RuntimeError(f"Qdrant preflight failed ({type(exc).__name__})") from None
    status, _, ready = _http("GET", api_url.rstrip("/") + "/health/ready", {}, None)
    if status != 200:
        client.close()
        raise RuntimeError(f"API readiness preflight failed with HTTP {status}")

    run_id = f"phase7a-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = _new_run_directory(non_live=False)
    secret = settings.auth_shared_secret.get_secret_value()
    known_secrets = [secret]
    for key_name in ("embedding_api_key", "generation_api_key", "generation_fallback_api_key",
                     "verification_api_key", "verification_fallback_api_key",
                     "reranker_api_key", "qdrant_api_key"):
        configured = getattr(settings, key_name, None)
        if configured:
            known_secrets.append(configured.get_secret_value())
    user_a = os.environ.get("OMNIRAG_7A_USER_A_ID", "phase7a-user-a")
    tenant_a = os.environ.get("OMNIRAG_7A_TENANT_A_ID", "phase7a-tenant-a")
    ingestion_rows = []
    document_ids = {}
    budget_usage = {}
    upload_probes = []
    try:
        probe_headers = {**_signed_headers(user_a, tenant_a, secret, run_id),
                         "Content-Type": "application/octet-stream"}
        invalid_url = api_url.rstrip("/") + "/api/v1/documents/upload?filename=phase7a_invalid.exe"
        invalid_code, _, _ = _http("PUT", invalid_url, probe_headers, b"not an allowed document")
        upload_probes.append({"case": "unsupported_extension", "http_status": invalid_code,
                              "passed": invalid_code == 415})
        oversize_url = api_url.rstrip("/") + "/api/v1/documents/upload?filename=phase7a_oversize.txt"
        oversize_code, _, _ = _http("PUT", oversize_url, probe_headers,
                                    b"x" * (settings.max_upload_bytes + 1))
        upload_probes.append({"case": "streaming_upload_limit", "http_status": oversize_code,
                              "passed": oversize_code == 413})
        if any(not row["passed"] for row in upload_probes):
            raise RuntimeError("Upload validation probes did not return expected 415/413 statuses")
        for path in files:
            version = "v1" if "rate_guide_v1" in path.name else "v2" if "rate_guide_v2" in path.name else None
            params = {"filename": path.name}
            if version:
                params["version"] = version
            url = api_url.rstrip("/") + "/api/v1/documents/upload?" + urlencode(params)
            headers = _signed_headers(user_a, tenant_a, secret, run_id)
            headers["Content-Type"] = "application/octet-stream"
            content = path.read_bytes()
            started = time.perf_counter()
            code, response_headers, body = _http("PUT", url, headers, content)
            elapsed = (time.perf_counter() - started) * 1000
            if code not in (200, 201):
                raise RuntimeError(f"Upload failed for {path.name} with HTTP {code}: "
                                   f"{json.dumps(body, ensure_ascii=False)[:400]}")
            document = body if isinstance(body, dict) else {}
            document_ids[path.name] = document.get("document_id")
            if response_headers.get("X-OmniRAG-Budget-Usage"):
                budget_usage = json.loads(response_headers["X-OmniRAG-Budget-Usage"])
            ingestion_rows.append({"filename": path.name, "status": document.get("status"),
                                   "document_id": document.get("document_id"),
                                   "chunk_count": document.get("chunk_count"),
                                   "elapsed_ms": round(elapsed, 3), "fallbacks": "not exposed by upload response"})

        # The initial Qdrant collection is expected to be absent. The API's normal
        # ingestion/indexing flow creates it; verify that transition before scrolling.
        _collection_info_after_ingestion(client, settings.qdrant_collection)
        initial_counts = {name: _document_point_count(client, settings.qdrant_collection, document_id)
                          for name, document_id in document_ids.items() if document_id}
        identical_rows = []
        for path in files:
            params = {"filename": path.name}
            version = "v1" if "rate_guide_v1" in path.name else "v2" if "rate_guide_v2" in path.name else None
            if version:
                params["version"] = version
            code, _, body = _http("PUT", api_url.rstrip("/") + "/api/v1/documents/upload?" + urlencode(params),
                                  {**_signed_headers(user_a, tenant_a, secret, run_id),
                                   "Content-Type": "application/octet-stream"}, path.read_bytes())
            if code not in (200, 201):
                raise RuntimeError(f"Idempotency upload failed for {path.name} with HTTP {code}")
            doc_id = document_ids.get(path.name)
            after_count = _document_point_count(client, settings.qdrant_collection, doc_id) if doc_id else 0
            identical_rows.append({"filename": path.name, "status": body.get("status"),
                                   "before_points": initial_counts.get(path.name, 0),
                                   "after_points": after_count,
                                   "duplicate_point_delta": after_count - initial_counts.get(path.name, 0)})

        mutate = next(path for path in files if path.name == "01_network_manual.md")
        modified = mutate.read_bytes() + b"\n\nSynthetic remediation mutation: violet bridge has 6 spans.\n"
        params = {"filename": mutate.name, "version": "phase7a-modified"}
        code, _, modified_body = _http(
            "PUT", api_url.rstrip("/") + "/api/v1/documents/upload?" + urlencode(params),
            {**_signed_headers(user_a, tenant_a, secret, run_id), "Content-Type": "application/octet-stream"}, modified)
        if code not in (200, 201):
            raise RuntimeError(f"Modified-content reingest failed with HTTP {code}")

        info = _collection_info_after_ingestion(client, settings.qdrant_collection)
        schema = getattr(info, "payload_schema", None) or {}
        missing_indexes = sorted(INDEX_FIELDS - set(schema))
        sample_ids = [document_ids.get("01_network_manual.md")]
        sample_points = [point for doc_id in sample_ids if doc_id
                         for point in _payload_all(client, settings.qdrant_collection, doc_id)]
        provenance_missing = sorted(PROVENANCE_FIELDS - set(sample_points[0].payload or {})) if sample_points else sorted(PROVENANCE_FIELDS)
        if not sample_points:
            raise RuntimeError("Smoke-corpus ingestion created no points for provenance/vector validation")
        indexed_dimension = (sample_points[0].payload or {}).get("embedding_dim")
        if not isinstance(indexed_dimension, int) or indexed_dimension <= 0:
            raise RuntimeError("Smoke-corpus point has no valid embedding_dim provenance")
        if settings.embedding_dimensions is not None and indexed_dimension != settings.embedding_dimensions:
            raise RuntimeError(
                f"Indexed embedding dimension is {indexed_dimension}; configured dimension is "
                f"{settings.embedding_dimensions}"
            )
        _validate_collection_vector_config(info, settings, indexed_dimension)
        if missing_indexes:
            raise RuntimeError("Qdrant payload indexes missing: " + ", ".join(missing_indexes))
        if provenance_missing:
            raise RuntimeError("Point provenance missing: " + ", ".join(provenance_missing))
        payload_samples = [{key: point.payload.get(key) for key in (
            "document_id", "chunk_id", "version", "metadata", *sorted(PROVENANCE_FIELDS))}
            for point in sample_points[:3]]

        # Run server-dependent ACL checks plus local mechanical hard-gate tests.
        os.environ["OMNIRAG_7A_QDRANT_URL"] = settings.qdrant_url
        if settings.qdrant_api_key:
            os.environ["OMNIRAG_7A_QDRANT_API_KEY"] = settings.qdrant_api_key.get_secret_value()
        import pytest
        gate_exit = pytest.main(["-q", "tests/integration/test_live_qdrant_gates.py",
                                 "tests/unit/test_legacy_tenant_mapping.py",
                                 "tests/integration/test_qdrant_legacy_mapping.py",
                                 "tests/unit/test_provider_budget.py", "tests/unit/test_research_budget.py",
                                 "tests/unit/test_generation_verification.py", "tests/unit/test_api.py"])
        if int(gate_exit) != 0:
            raise RuntimeError(f"Hard-gate pytest run failed with exit code {int(gate_exit)}")

        query_results = []
        latency_values: list[float] = []
        input_tokens = output_tokens = 0
        prompt_injection_observation = None
        for query in queries:
            for profile in query["profiles"]:
                identity_headers = _identity_for(query, os.environ, secret, run_id)
                payload = {"query_text": query["text"], "selected_profile": profile,
                           "escalation_preference": "when_needed"}
                started = time.perf_counter()
                code, response_headers, body = _http(
                    "POST", api_url.rstrip("/") + "/api/v1/chat",
                    {**identity_headers, "Content-Type": "application/json"},
                    json.dumps(payload).encode("utf-8"))
                elapsed = (time.perf_counter() - started) * 1000
                if code == 429 and isinstance(body, dict) and body.get("error", {}).get("code") == "provider_budget_exceeded":
                    raise RuntimeError("Provider run budget tripped; remaining query requests were aborted")
                if code != 200:
                    result = {"query_id": query["id"], "category": query["category"],
                              "profile": profile, "http_status": code, "error": body}
                    (run_dir / f"{query['id']}-{profile}.json").write_text(
                        json.dumps(_redact(result, known_secrets), indent=2, ensure_ascii=False), encoding="utf-8")
                    query_results.append(result)
                    continue
                response = body.get("response", {})
                failures = _mechanical_checks(query, response)
                timing = response.get("timings_ms", {})
                latency_values.extend(value for value in timing.values() if isinstance(value, (int, float)))
                usage = response.get("token_usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                if query["category"] == "prompt_injection":
                    answer = response.get("answer", "").casefold()
                    prompt_injection_observation = {"query_id": query["id"],
                        "response_mentions_system_prompt": "system prompt" in answer,
                        "abstained": response.get("abstained", False)}
                if response_headers.get("X-OmniRAG-Budget-Usage"):
                    budget_usage = json.loads(response_headers["X-OmniRAG-Budget-Usage"])
                result = {"query_id": query["id"], "category": query["category"],
                          "profile": profile, "http_status": code, "elapsed_ms": round(elapsed, 3),
                          "mechanical_check_failures": failures, "response": body}
                (run_dir / f"{query['id']}-{profile}.json").write_text(
                    json.dumps(_redact(result, known_secrets), indent=2, ensure_ascii=False), encoding="utf-8")
                query_results.append({key: value for key, value in result.items() if key != "response"})

        qdrant_snapshot = {"points_count": info.points_count, "indexed_vectors_count": info.indexed_vectors_count,
                           "vector_config": str(info.config.params.vectors),
                           "sparse_vector_config": str(info.config.params.sparse_vectors),
                           "vector_config_validated": True,
                           "payload_schema_fields": sorted(schema), "missing_indexes": missing_indexes,
                           "provenance_missing": provenance_missing, "payload_samples": payload_samples}
        latencies = [float(value) for value in latency_values]
        mechanical_failures = [row for row in query_results if row.get("mechanical_check_failures")]
        summary = {
            "live": True, "run_id": run_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "api_ready": ready, "corpus_files": len(files), "ingestion": ingestion_rows,
            "upload_probes": upload_probes,
            "identical_reingestion": identical_rows,
            "modified_reingestion": {"filename": mutate.name, "status": modified_body.get("status"),
                                     "chunk_count": modified_body.get("chunk_count")},
            "qdrant": qdrant_snapshot, "query_count": len(query_results),
            "mechanical_check_failures": mechanical_failures,
            "latency_ms": {"sample_count": len(latencies), "p50": _percentile(latencies, .50),
                           "p95": _percentile(latencies, .95)},
            "token_totals": {"input": input_tokens, "output": output_tokens,
                             "total": input_tokens + output_tokens},
            "budget_usage": budget_usage, "prompt_injection_observation": prompt_injection_observation,
            "quality_conclusion": "Sample is too small for conclusions about answer quality.",
        }
        (run_dir / "summary.json").write_text(
            json.dumps(_redact(summary, known_secrets), indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "summary.md").write_text(
            "# Phase 7a live validation\n\n"
            f"- Run ID: `{run_id}`\n- Files ingested: {len(files)}\n- Query/profile responses: {len(query_results)}\n"
            f"- Mechanical failure rows: {len(mechanical_failures)}\n- Latency samples: {len(latencies)}; "
            f"P50={_percentile(latencies,.50)} ms; P95={_percentile(latencies,.95)} ms\n"
            f"- Token totals: input={input_tokens}; output={output_tokens}\n"
            "- The sample is too small for conclusions about answer quality.\n", encoding="utf-8")
        print(f"LIVE validation outputs written to {run_dir}")
        print(json.dumps({"run_id": run_id, "query_count": len(query_results),
                          "mechanical_failures": len(mechanical_failures),
                          "budget_usage": budget_usage}, indent=2))
        return run_dir
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.environ.get("OMNIRAG_7A_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--fake-providers", action="store_true",
                        help="Run local harness checks only; labels every output NON-LIVE")
    args = parser.parse_args()
    try:
        if args.fake_providers:
            fake_run()
        else:
            live_run(args.api_url)
    except Exception as exc:
        print(f"Validation preflight/run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
