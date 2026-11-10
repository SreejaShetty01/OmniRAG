import json
import hashlib
import pytest
from urllib.error import URLError

import evaluation.phase8_runner as runner_module
from evaluation.dataset import EvaluationCase
from evaluation.phase8_runner import (
    DEFAULT_RUN_TIMEOUT_SECONDS,
    EXPECTED_DATASET_SHA256,
    MAX_PROVIDER_CALLS,
    MAX_TOTAL_TOKENS,
    Phase8BenchmarkRunner,
    ScheduledUnit,
    build_chat_request,
    schedule_units,
)


def case():
    return EvaluationCase(
        case_id="P7C-AF-001", category="answerable_factual",
        query="Synthetic private query must not be persisted.", expected_answer="Northstar",
        answer_match="exact", expected_source_documents=["01_network_manual.md"],
        expected_evidence=[{"source_document": "01_network_manual.md", "locator": "line:4",
                            "quote": "The synthetic gateway is named Northstar."}],
        expected_citations=["01_network_manual.md"], expected_behavior="Answer factually.",
        expected_abstention=False, profile_coverage=["lite", "core", "research"],
        difficulty="easy", ground_truth_status="verified", verification_note="test fixture",
    )


def snapshot():
    core = {
        "dataset_id": "omnirag-phase8-eval", "dataset_version": "1.0.1",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "corpus_id": "phase7a-smoke-corpus", "corpus_version": "phase7a-synthetic-v1",
        "collection": "omnirag_eval_phase8_test",
        "vector_schema": {"dense_name": "dense", "dense_size": 768, "sparse_name": "sparse"},
        "required_payload_indexes": [],
        "documents": [{"filename": "01_network_manual.md"}],
        "points": [{"point_id": "p1", "document_id": "d1", "chunk_id": "c1",
                    "source_basename": "01_network_manual.md", "source": "fixture/01_network_manual.md",
                    "tenant_id": "phase8-evaluation-tenant",
                    "acl_allowed_principals": ["user:phase8-evaluation-user"],
                    "acl_denied_principals": []}],
        "evidence_to_chunks": [], "known_exclusion": "test",
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    return {**core, "snapshot_sha256": digest, "audit_status": "passed",
            "point_count": 1, "document_count": 1}


def response_body():
    ref = {"evidence_id": "E1", "document_id": "d1", "chunk_id": "c1",
           "source": "/private/path/01_network_manual.md", "version": "v1", "excerpt": "private evidence"}
    return {"response": {
        "answer": "The synthetic gateway is Northstar.", "profile_used": "lite", "abstained": False,
        "abstention_reason": None, "verification_status": "passed", "claim_verification_status": "passed",
        "citation_validation_status": "passed", "overall_reliability": "passed", "claims": [],
        "generation_metadata": {"provider": "openai_compatible", "model": "local-test"},
        "token_usage": {"input_tokens": 20, "output_tokens": 5}, "cost_usd": None,
        "timings_ms": {"generation": 14.0}, "warnings": [],
        "execution_metadata": {"experiment_run_id": "experiment-run-1"},
        "retrieval_inspection": [{"chunk_id": "c1", "document_id": "d1", "source": "fixture/01_network_manual.md",
                                   "fusion_rank": 1, "selected": True}],
        "evidence_references": [ref],
        "citations": [{"citation_id": "C1", "evidence_id": "E1", "document_id": "d1",
                       "chunk_id": "c1", "source": "fixture/01_network_manual.md", "version": "v1"}],
    }}


def budget_header(run_id="phase8-test"):
    return {"X-OmniRAG-Budget-Usage": json.dumps({"run_id": run_id, "calls": 2,
                                                   "total_tokens": 25, "input_tokens": 20,
                                                   "output_tokens": 5, "estimated_cost_usd": 0.0})}


def test_schedule_is_deterministic_rotated_and_complete():
    cases = [case().model_copy(update={"case_id": f"P7C-AF-{i:03d}"}) for i in range(1, 4)]
    first = schedule_units(cases, repetitions=2, seed=42)
    second = schedule_units(cases, repetitions=2, seed=42)
    assert [u.attempt_key for u in first] == [u.attempt_key for u in second]
    assert len(first) == 18
    assert {u.profile for u in first} == {"lite", "core", "research"}
    assert [u.attempt_key for u in first[:9]] != [u.attempt_key for u in first[9:]]


def test_b7_default_caps_and_deadline_match_approved_plan():
    assert MAX_TOTAL_TOKENS == 4_200_000
    assert MAX_PROVIDER_CALLS == 3_000
    assert runner_module.MAX_COST_USD == 0.0
    assert DEFAULT_RUN_TIMEOUT_SECONDS == 108_000.0


def test_base_url_rejects_malformed_or_non_http_targets():
    for target in ("localhost:8001", "ftp://localhost:8001", "http://localhost:notaport"):
        with pytest.raises(ValueError, match=r"absolute HTTP\(S\) URL"):
            Phase8BenchmarkRunner(base_url=target, secret="test-secret",
                                 snapshot_manifest=snapshot(), output_root="unused")


def test_signed_self_preflight_uses_settings_endpoint_and_checks_budget(monkeypatch, tmp_path):
    seen = {}
    budget = {"run_id": "phase8-preflight", "calls": 0, "total_tokens": 0,
              "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
              "max_cost_usd": 0.0, "max_total_tokens": MAX_TOTAL_TOKENS,
              "max_calls": MAX_PROVIDER_CALLS}

    def fake_get(url, headers, timeout):
        seen.update(url=url, headers=headers, timeout=timeout)
        return 200, {"X-OmniRAG-Budget-Usage": json.dumps(budget)}, {"user_id": "phase7a-user-a"}

    monkeypatch.setattr(runner_module, "_http_get", fake_get)
    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path)
    result = runner.preflight("phase8-preflight")
    assert seen["url"] == "http://localhost:8001/api/v1/settings"
    assert seen["headers"]["X-OmniRAG-Run-ID"] == "phase8-preflight"
    assert seen["headers"]["X-OmniRAG-Auth-Signature"]
    assert seen["timeout"] == 10.0
    assert result["http_status"] == 200
    assert result["budget_usage"]["calls"] == 0
    assert not (tmp_path / "phase8-preflight").exists()


def test_signed_self_preflight_fails_clearly_for_unreachable_target(monkeypatch, tmp_path):
    def fail_get(*args):
        raise URLError(ConnectionRefusedError(10061, "connection refused"))

    monkeypatch.setattr(runner_module, "_http_get", fail_get)
    runner = Phase8BenchmarkRunner(base_url="http://127.0.0.1:65534", secret="private-test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path)
    with pytest.raises(RuntimeError, match="self-preflight transport failure") as error:
        runner.preflight("phase8-unreachable")
    message = str(error.value)
    assert "URLError" in message and "ConnectionRefusedError" in message
    assert "127.0.0.1" in message and "65534" in message and "/api/v1/settings" in message
    assert "private-test-secret" not in message
    assert "errno" in message
    assert not (tmp_path / "phase8-unreachable").exists()


def test_transport_failure_is_recorded_and_still_stops_without_retry(tmp_path):
    calls = []

    def fail_post(url, headers, body, timeout):
        calls.append(url)
        raise URLError(PermissionError(13, "socket blocked", None, 10013, None))

    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="private-test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path, post=fail_post)
    unit = ScheduledUnit(case(), "core", 1)
    result = runner.run([unit], benchmark_id="phase8-transport", seed=15)
    attempt_path = tmp_path / "phase8-transport" / "attempts" / f"{unit.attempt_key}.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    detail = attempt["transport_failure"]
    assert result["status"] == "stopped_runner_failure_budget_state_unknown"
    assert result["stop_reason"] == "runner_failure_budget_state_unknown"
    assert len(calls) == 1
    assert detail["exception_class"] == "urllib.error.URLError"
    assert "PermissionError" in detail["reason_repr"] and detail["errno"] == 13
    assert detail["winerror"] == 10013
    assert detail["target"] == {"scheme": "http", "host": "localhost", "port": 8001,
                                 "path": "/api/v1/chat"}
    assert detail["request_likely_sent"] is False
    assert detail["elapsed_ms"] >= 0
    assert "private-test-secret" not in attempt_path.read_text(encoding="utf-8")


def test_empty_generation_usage_for_safe_abstention_does_not_trigger_budget_stop(tmp_path):
    denied = case().model_copy(update={
        "category": "acl_tenant_isolation",
        "acl_context": {"request_principal": "user:other-user", "expected_access": "deny"},
        "tenant_context": {"request_tenant_id": "phase8-evaluation-tenant",
                            "expected_tenant_match": True},
    })
    payload = response_body()
    response = payload["response"]
    response.update({"answer": "I do not have enough evidence.", "abstained": True,
                     "abstention_reason": "insufficient_retrieved_evidence",
                     "generation_metadata": {"provider": None, "model": None},
                     "token_usage": {}, "timings_ms": {}, "warnings": ["semantic_verification_unconfigured"],
                     "retrieval_inspection": [], "evidence_references": [], "citations": []})
    ledger = {"run_id": "phase8-empty-usage", "calls": 1, "total_tokens": 9,
              "input_tokens": 9, "output_tokens": 0, "estimated_cost_usd": 0.0,
              "max_calls": MAX_PROVIDER_CALLS, "max_total_tokens": MAX_TOTAL_TOKENS,
              "max_cost_usd": 0.0}
    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path,
        post=lambda *args: (200, {"X-OmniRAG-Budget-Usage": json.dumps(ledger)}, payload),
    )
    unit = ScheduledUnit(denied, "lite", 1)
    result = runner.run([unit], benchmark_id="phase8-empty-usage", seed=16)
    attempt = json.loads((tmp_path / "phase8-empty-usage" / "attempts" /
                          f"{unit.attempt_key}.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["budget_usage"]["calls"] == 1
    assert attempt["token_usage"] == {}
    assert attempt["abstained"] is True
    assert attempt["status"] == "completed_with_warnings"


def test_request_signs_identity_and_forces_no_escalation():
    from api.trusted_proxy import verify_proxy_identity

    unit = ScheduledUnit(case(), "lite", 1)
    body, headers, request_id = build_chat_request(
        unit, secret="test-secret", benchmark_id="phase8-test", request_id="request-1", timestamp=1700000000)
    signed_identity = verify_proxy_identity(
        headers["X-OmniRAG-User-ID"], headers["X-OmniRAG-Tenant-ID"],
        headers["X-OmniRAG-Auth-Timestamp"], headers["X-OmniRAG-Auth-Signature"],
        "test-secret", now=1700000000)
    assert signed_identity == {"user_id": "phase8-evaluation-user", "tenant_id": "phase8-evaluation-tenant"}
    assert request_id == "request-1"
    assert headers["X-OmniRAG-Run-ID"] == "phase8-test"
    assert body["escalation_preference"] == "never"
    assert body["experiment"]["research_escalation"] is False
    assert body["experiment"]["dataset_version"] == "1.0.1"


def test_runner_writes_safe_attempt_and_resume_does_not_retry(tmp_path):
    calls = []

    def fake_post(url, headers, body, timeout):
        calls.append((url, headers, body, timeout))
        return 200, budget_header(), response_body()

    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path,
                                   request_timeout=150, run_timeout=1800, post=fake_post)
    units = [ScheduledUnit(case(), "lite", 1)]
    result = runner.run(units, benchmark_id="phase8-test", seed=42)
    assert result["status"] == "completed"
    assert result["budget_usage"]["calls"] == 2
    attempt = tmp_path / "phase8-test" / "attempts" / f"{units[0].attempt_key}.json"
    text = attempt.read_text(encoding="utf-8")
    assert "Northstar" not in text
    assert "Synthetic private query" not in text
    assert "private evidence" not in text
    assert "c1" in text and "experiment-run-1" in text
    resumed = runner.run(units, benchmark_id="phase8-test", seed=42, resume=True)
    assert resumed["status"] == "completed"
    assert len(calls) == 1
    changed_caps_runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path, request_timeout=150, run_timeout=1800,
        max_total_tokens=MAX_TOTAL_TOKENS - 1, post=fake_post)
    with pytest.raises(ValueError, match="budget caps differ"):
        changed_caps_runner.run(units, benchmark_id="phase8-test", seed=42, resume=True)
    assert len(calls) == 1


def test_runner_records_timeout_and_does_not_retry_on_resume(tmp_path):
    calls = []

    def timeout_post(*args):
        calls.append(args)
        raise TimeoutError("synthetic timeout detail must not be stored")

    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path,
                                   post=timeout_post)
    unit = ScheduledUnit(case(), "core", 1)
    result = runner.run([unit], benchmark_id="phase8-timeout", seed=7)
    assert result["status"] == "stopped_request_timeout_budget_state_unknown"
    attempt = tmp_path / "phase8-timeout" / "attempts" / f"{unit.attempt_key}.json"
    detail = json.loads(attempt.read_text(encoding="utf-8"))
    assert detail["status"] == "timed_out"
    assert "synthetic timeout detail" in detail["transport_failure"]["reason_repr"]
    assert detail["transport_failure"]["request_likely_sent"] is None
    runner.run([unit], benchmark_id="phase8-timeout", seed=7, resume=True)
    assert len(calls) == 1


def test_runner_stops_after_authorization_violation(tmp_path):
    calls = []
    denied_case = case().model_copy(update={
        "category": "acl_tenant_isolation",
        "acl_context": {"request_principal": "user:other-user", "expected_access": "deny"},
        "tenant_context": {"request_tenant_id": "phase8-evaluation-tenant",
                            "expected_tenant_match": True},
    })

    def fake_post(url, headers, body, timeout):
        calls.append(url)
        return 200, budget_header("phase8-security-stop"), response_body()

    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path,
                                   post=fake_post)
    units = [ScheduledUnit(denied_case, "lite", 1), ScheduledUnit(case(), "core", 1)]
    result = runner.run(units, benchmark_id="phase8-security-stop", seed=13)
    assert result["status"] == "stopped_security_violation"
    assert [row["status"] for row in result["schedule"]] == [
        "completed", "not_run_security_violation"]
    assert len(calls) == 1
    attempt = tmp_path / "phase8-security-stop" / "attempts" / f"{units[0].attempt_key}.json"
    saved = json.loads(attempt.read_text(encoding="utf-8"))
    assert saved["authorization_audit"]["violation"] is True
    assert saved["retrieval_candidates"] == []


def test_runner_keeps_nonfatal_warnings_separate_from_failures(tmp_path):
    payload = response_body()
    payload["response"]["warnings"] = ["semantic_verification_unconfigured",
                                         "conflicting_retrieved_evidence"]

    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path,
        post=lambda *args: (200, budget_header("phase8-warning-status"), payload),
    )
    unit = ScheduledUnit(case(), "lite", 1)
    result = runner.run([unit], benchmark_id="phase8-warning-status", seed=14)
    assert result["status"] == "completed"
    attempt = tmp_path / "phase8-warning-status" / "attempts" / f"{unit.attempt_key}.json"
    saved = json.loads(attempt.read_text(encoding="utf-8"))
    assert saved["status"] == "completed_with_warnings"
    assert saved["failure_codes"] == []
    assert saved["warning_codes"] == ["conflicting_retrieved_evidence",
                                       "semantic_verification_unconfigured"]


def test_unavailable_reranker_with_fusion_fallback_is_nonfatal(tmp_path):
    calls = []
    payload = response_body()
    payload["response"]["warnings"] = ["reranker_unavailable:ValueError"]
    units = [ScheduledUnit(case(), "core", 1), ScheduledUnit(case(), "lite", 1)]

    def fake_post(*args):
        calls.append(args)
        return 200, budget_header("phase8-reranker-unavailable"), payload

    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path, post=fake_post)
    result = runner.run(units, benchmark_id="phase8-reranker-unavailable", seed=141)

    assert result["status"] == "completed"
    assert [row["status"] for row in result["schedule"]] == ["completed_with_warnings"] * 2
    assert len(calls) == 2
    saved = json.loads((tmp_path / "phase8-reranker-unavailable" / "attempts" /
                        f"{units[0].attempt_key}.json").read_text(encoding="utf-8"))
    assert saved["failure_codes"] == []
    assert saved["warning_codes"] == ["reranker_unavailable"]


def test_reranker_execution_failure_with_fusion_fallback_remains_fatal(tmp_path):
    calls = []
    payload = response_body()
    payload["response"]["warnings"] = ["reranker_failed_fallback_to_fusion:RerankerError"]
    units = [ScheduledUnit(case(), "core", 1), ScheduledUnit(case(), "lite", 1)]

    def fake_post(*args):
        calls.append(args)
        return 200, budget_header("phase8-reranker-failure"), payload

    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path, post=fake_post)
    result = runner.run(units, benchmark_id="phase8-reranker-failure", seed=142)

    assert result["status"] == "stopped_provider_failure"
    assert [row["status"] for row in result["schedule"]] == [
        "completed_with_errors", "not_run_provider_failure"]
    assert len(calls) == 1
    saved = json.loads((tmp_path / "phase8-reranker-failure" / "attempts" /
                        f"{units[0].attempt_key}.json").read_text(encoding="utf-8"))
    assert saved["failure_codes"] == ["reranker_failed_fallback_to_fusion"]


def test_runner_stops_on_provider_failure_and_preserves_failure_record(tmp_path):
    calls = []
    payload = response_body()
    payload["response"]["warnings"] = ["generation_failed:ProviderError"]
    units = [ScheduledUnit(case(), "lite", 1), ScheduledUnit(case(), "core", 1)]

    def fake_post(*args):
        calls.append(args)
        return 200, budget_header("phase8-provider-stop"), payload

    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path, post=fake_post)
    result = runner.run(units, benchmark_id="phase8-provider-stop", seed=15)
    assert result["status"] == "stopped_provider_failure"
    assert result["stop_reason"] == "provider_failure"
    assert [row["status"] for row in result["schedule"]] == [
        "completed_with_errors", "not_run_provider_failure"]
    assert len(calls) == 1
    attempt = tmp_path / "phase8-provider-stop" / "attempts" / f"{units[0].attempt_key}.json"
    saved = json.loads(attempt.read_text(encoding="utf-8"))
    assert saved["status"] == "completed_with_errors"
    assert saved["failure_codes"] == ["generation_failed"]


def test_zero_cost_tripwire_stops_and_preserves_attempt(tmp_path):
    calls = []
    ledger = json.loads(budget_header("phase8-cost-stop")["X-OmniRAG-Budget-Usage"])
    ledger["estimated_cost_usd"] = 0.001
    headers = {"X-OmniRAG-Budget-Usage": json.dumps(ledger)}
    units = [ScheduledUnit(case(), "lite", 1), ScheduledUnit(case(), "core", 1)]
    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path, post=lambda *args: (calls.append(args) or (200, headers, response_body())))
    result = runner.run(units, benchmark_id="phase8-cost-stop", seed=16)
    assert result["status"] == "stopped_by_budget"
    assert len(calls) == 1
    assert result["budget_usage"]["estimated_cost_usd"] == 0.001
    saved = json.loads((tmp_path / "phase8-cost-stop" / "attempts" /
                        f"{units[0].attempt_key}.json").read_text(encoding="utf-8"))
    assert saved["status"] == "completed"


def test_token_cap_stops_before_next_unit_and_preserves_first_attempt(tmp_path):
    calls = []
    units = [ScheduledUnit(case(), "lite", 1), ScheduledUnit(case(), "core", 1)]
    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path, max_total_tokens=25,
        post=lambda *args: (calls.append(args) or (200, budget_header("phase8-token-stop"), response_body())))
    result = runner.run(units, benchmark_id="phase8-token-stop", seed=17)
    assert result["status"] == "stopped_by_budget"
    assert [row["status"] for row in result["schedule"]] == ["completed", "not_run_budget_cap"]
    assert len(calls) == 1


def test_warmup_requires_three_profiles_and_is_marked_unscored(tmp_path):
    calls = []
    units = [ScheduledUnit(case(), profile, 1) for profile in ("lite", "core", "research")]
    def fake_post(*args):
        calls.append(args)
        ledger = {"run_id": "phase8-warmup", "calls": len(calls) * 2,
                  "total_tokens": len(calls) * 25, "input_tokens": len(calls) * 20,
                  "output_tokens": len(calls) * 5, "estimated_cost_usd": 0.0}
        return 200, {"X-OmniRAG-Budget-Usage": json.dumps(ledger)}, response_body()
    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path, post=fake_post)
    result = runner.run(units, benchmark_id="phase8-warmup", seed=18, warmup=True)
    assert result["status"] == "completed"
    assert result["warmup"] is True
    assert len(calls) == 3
    assert {row["profile"] for row in result["schedule"]} == {"lite", "core", "research"}
    with pytest.raises(ValueError, match="exactly one P7C-AF-001"):
        runner.run([units[0]], benchmark_id="invalid-warmup", seed=19, warmup=True)


def test_cumulative_deadline_survives_resume_and_completed_attempt_is_not_repeated(tmp_path, monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(runner_module.time, "time", lambda: now[0])
    calls = []
    units = [ScheduledUnit(case(), profile, 1) for profile in ("lite", "core", "research")]
    def fake_post(*args):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return 200, budget_header("phase8-deadline"), response_body()
    runner = Phase8BenchmarkRunner(base_url="http://localhost:8001", secret="test-secret",
                                   snapshot_manifest=snapshot(), output_root=tmp_path,
                                   run_timeout=10, post=fake_post)
    with pytest.raises(KeyboardInterrupt):
        runner.run(units, benchmark_id="phase8-deadline", seed=20)
    progress_path = tmp_path / "phase8-deadline" / "progress.json"
    interrupted = json.loads(progress_path.read_text(encoding="utf-8"))
    assert interrupted["deadline_unix"] == 1_010.0
    first_attempt_path = tmp_path / "phase8-deadline" / "attempts" / f"{units[0].attempt_key}.json"
    assert json.loads(first_attempt_path.read_text(encoding="utf-8"))["status"] == "completed"
    now[0] = 1_011.0
    resumed = runner.run(units, benchmark_id="phase8-deadline", seed=20, resume=True)
    assert resumed["status"] == "stopped_run_timeout"
    assert len(calls) == 2
    assert [row["status"] for row in resumed["schedule"]] == [
        "completed", "indeterminate_after_interruption", "not_run_run_timeout"]
    interruption_path = tmp_path / "phase8-deadline" / "attempts" / f"{units[1].attempt_key}.json"
    assert json.loads(interruption_path.read_text(encoding="utf-8"))["status"] == "indeterminate_after_interruption"


def test_http_failure_stops_and_preserves_failed_attempt(tmp_path):
    calls = []
    units = [ScheduledUnit(case(), "lite", 1), ScheduledUnit(case(), "core", 1)]
    runner = Phase8BenchmarkRunner(
        base_url="http://localhost:8001", secret="test-secret", snapshot_manifest=snapshot(),
        output_root=tmp_path,
        post=lambda *args: (calls.append(args) or (503, budget_header("phase8-http-stop"), {"error": {"code": "unavailable"}})))
    result = runner.run(units, benchmark_id="phase8-http-stop", seed=21)
    assert result["status"] == "stopped_api_failure"
    assert result["schedule"][0]["status"] == "failed"
    assert result["schedule"][1]["status"] == "not_run_api_failure"
    assert len(calls) == 1
    saved = json.loads((tmp_path / "phase8-http-stop" / "attempts" /
                        f"{units[0].attempt_key}.json").read_text(encoding="utf-8"))
    assert saved["failure_classification"] == "unavailable"
