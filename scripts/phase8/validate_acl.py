"""Run the four Phase 7c ACL/tenant cases as isolated B2 API security checks."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omnirag.config.settings import get_settings


DATASET = ROOT / "evaluation/datasets/omnirag-phase8-eval-1.0.1.json"
SNAPSHOT = ROOT / "docs/audit/phase8_eval_corpus_manifest.json"
MARKER = re.compile(r"cobalt\s+lantern\s+731", re.IGNORECASE)


def _contains(value, needle: str) -> bool:
    if isinstance(value, str):
        return needle.casefold() in value.casefold()
    if isinstance(value, dict):
        return any(_contains(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_contains(v, needle) for v in value)
    return False


def main():
    settings = get_settings()
    if settings.auth_shared_secret is None:
        raise RuntimeError("Trusted proxy secret is not configured")
    secret = settings.auth_shared_secret.get_secret_value()
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in dataset["cases"]
             if case["category"] == "acl_tenant_isolation"}
    required = ["P7C-AT-001", "P7C-AT-002", "P7C-AT-003", "P7C-AT-004"]
    if any(case_id not in cases for case_id in required):
        raise RuntimeError("All four ACL/tenant cases are required")
    restricted_points = {item["chunk_id"] for item in snapshot["points"]
                         if item["source_basename"] == "12_restricted_team.txt"}
    restricted_documents = {item["document_id"] for item in snapshot["points"]
                            if item["source_basename"] == "12_restricted_team.txt"}
    restricted_quote = next(e["quote"] for e in cases[required[0]]["expected_evidence"]
                            if e["source_document"] == "12_restricted_team.txt")
    marker_match = MARKER.search(restricted_quote)
    if not restricted_points or not restricted_documents or not marker_match:
        raise RuntimeError("Audited restricted corpus metadata is incomplete")
    marker = marker_match.group(0)

    summaries = []
    budget_run_id = "phase8-b2-acl-smoke-20261104"
    for case_id in required:
        case = cases[case_id]
        user_id = case["acl_context"]["request_principal"].removeprefix("user:")
        tenant_id = case["tenant_context"]["request_tenant_id"]
        stamp = str(int(time.time()))
        signed = f"{user_id}\n{tenant_id}\n{stamp}".encode()
        signature = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        request_id = str(uuid4())
        headers = {
            "Content-Type": "application/json", "X-Request-ID": request_id,
            "X-OmniRAG-Run-ID": budget_run_id,
            "X-OmniRAG-User-ID": user_id, "X-OmniRAG-Tenant-ID": tenant_id,
            "X-OmniRAG-Auth-Timestamp": stamp, "X-OmniRAG-Auth-Signature": signature,
        }
        request_body = {
            "query_text": case["query"], "selected_profile": "lite",
            "escalation_preference": "never",
            "experiment": {"dataset_id": dataset["dataset_id"],
                           "dataset_version": dataset["dataset_version"],
                           "research_escalation": False},
        }
        req = Request("http://localhost:8000/api/v1/chat",
                      data=json.dumps(request_body, ensure_ascii=False).encode(),
                      headers=headers, method="POST")
        with urlopen(req, timeout=150) as response:
            body = json.loads(response.read())
            if response.status != 200:
                raise RuntimeError(f"{case_id}: unexpected API status {response.status}")
            request_id_echo = response.headers.get("X-Request-ID")
            budget_raw = response.headers.get("X-OmniRAG-Budget-Usage")
            budget = json.loads(budget_raw) if budget_raw else None
        result = body.get("response", {})
        if not isinstance(result, dict):
            raise RuntimeError(f"{case_id}: response schema missing")
        refs = result.get("evidence_references", [])
        citations = result.get("citations", [])
        inspection = result.get("retrieval_inspection", [])
        all_candidate_ids = {item.get("chunk_id") for item in inspection if isinstance(item, dict)}
        ref_ids = {item.get("chunk_id") for item in refs if isinstance(item, dict)}
        citation_ids = {item.get("chunk_id") for item in citations if isinstance(item, dict)}
        observed_restricted = bool((all_candidate_ids | ref_ids | citation_ids) & restricted_points)
        answer = result.get("answer") if isinstance(result.get("answer"), str) else ""
        answer_marker = bool(MARKER.search(answer))
        allowed = case["acl_context"]["expected_access"] == "allow"
        if allowed:
            passed = (observed_restricted and answer_marker and bool(refs) and bool(citations))
        else:
            passed = (not observed_restricted and not answer_marker and not refs and not citations and
                      not _contains(body, marker))
        if not passed:
            raise RuntimeError(f"{case_id}: ACL/tenant outcome failed")

        experiment_run_id = result.get("execution_metadata", {}).get("experiment_run_id")
        if not experiment_run_id:
            raise RuntimeError(f"{case_id}: Phase 7b run ID missing")
        run_dir = settings.data_dir / "experiments" / "runs" / experiment_run_id
        files = [run_dir / name for name in ("manifest.json", "trace.json", "result.json")]
        if not all(path.is_file() for path in files):
            raise RuntimeError(f"{case_id}: experiment artifacts are incomplete")
        artifact_leak = any(_contains(path.read_text(encoding="utf-8"), marker) or
                            any(point_id in path.read_text(encoding="utf-8")
                                for point_id in restricted_points | restricted_documents)
                            for path in files)
        if artifact_leak:
            raise RuntimeError(f"{case_id}: restricted evidence leaked into experiment artifacts")
        if not budget or budget.get("run_id") != budget_run_id:
            raise RuntimeError(f"{case_id}: global budget ledger missing")
        if (budget["calls"] > settings.validation_max_calls or
                budget["total_tokens"] > settings.validation_max_total_tokens or
                budget["estimated_cost_usd"] > settings.validation_max_cost_usd):
            raise RuntimeError(f"{case_id}: budget cap exceeded")
        summaries.append({
            "case_id": case_id, "expected_access": case["acl_context"]["expected_access"],
            "tenant_match_expected": case["tenant_context"]["expected_tenant_match"],
            "http_status": 200, "request_id": request_id_echo,
            "experiment_run_id": experiment_run_id,
            "abstained": bool(result.get("abstained")),
            "evidence_count": len(refs), "citation_count": len(citations),
            "restricted_evidence_observed": observed_restricted,
            "restricted_answer_marker_observed": answer_marker,
            "experiment_artifact_leak": False, "status": "passed",
            "budget_usage": budget,
        })

    audit = {"date": "2026-11-04", "kind": "B2 ACL/tenant security preflight; not counted as measured pilot",
             "dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"],
             "dataset_sha256": "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1",
             "collection": snapshot["collection"], "snapshot_sha256": snapshot["snapshot_sha256"],
             "concurrency": 1, "inference_profile": "lite", "cases": summaries,
             "budget_final": summaries[-1]["budget_usage"], "status": "passed"}
    target = settings.data_dir / "phase8" / "audits" / "acl-smoke-20261104.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "case_count": len(summaries),
                      "audit_path": str(target), "budget_final": audit["budget_final"]}, sort_keys=True))


if __name__ == "__main__":
    main()
