import pytest
from pydantic import ValidationError
import hashlib
import json
from pathlib import Path

from eval.realworld_annotations import (HumanAnnotation, convert_human_approved_annotations,
                                              transition_annotation)
from scripts.realworld.prepare_corpus import (content_set_sha256, leakage_candidates,
                                              validate_corpus, validate_manifest)
from scripts.realworld.acquire_selected import scan_file_notices
from scripts.realworld.ingest_corpus import (build_dry_run, execute_ingestion,
                                             sign_identity_headers)
from omnirag.api.trusted_proxy import verify_proxy_identity


def _integrity_fixture(tmp_path):
    root = tmp_path / "corpus"
    doc = b"A technical document.\n"
    license_bytes = b"MIT License\n"
    rel = "sources/test/README.md"
    source_path = root / rel
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(doc)
    license_rel = "metadata/licenses/test/LICENSE"
    license_path = root / license_rel
    license_path.parent.mkdir(parents=True)
    license_path.write_bytes(license_bytes)
    repo = {"repository_id": "test", "repository_url": "https://github.com/example/test",
            "commit": "a" * 40, "release_tag": "v1", "license_file_path": license_rel,
            "license": "MIT", "source_inventory": ["README.md"], "acquisition_status": "acquired"}
    license_entry = {"repository": "test", "path": license_rel,
                     "source_url": "https://github.com/example/test/blob/" + "a" * 40 + "/LICENSE",
                     "license": "MIT", "sha256": hashlib.sha256(license_bytes).hexdigest(),
                     "file_size_bytes": len(license_bytes), "retrieved_at": "2026-11-04T00:00:00Z"}
    source = {"document_id": "test:README.md", "source_id": "test:README.md",
              "source_repository": "https://github.com/example/test",
              "source_url": "https://github.com/example/test/blob/" + "a" * 40 + "/README.md",
              "retrieval_url": "https://raw.githubusercontent.com/example/test/" + "a" * 40 + "/README.md",
              "original_path": "README.md", "local_path": rel, "repository_commit": "a" * 40,
              "version": "v1", "publisher": "Example", "license": "MIT", "license_version": "MIT",
              "license_evidence": [license_entry["source_url"]], "license_evidence_url": license_entry["source_url"],
              "cleared_for_reuse": True, "intended_use_clearance": "CLEARED",
              "retrieved_at": "2026-11-04T00:00:00Z", "retrieval_date": "2026-11-04",
              "sha256": hashlib.sha256(doc).hexdigest(), "file_size_bytes": len(doc),
              "file_format": "md", "category": "overview", "status": "CLEARED",
              "source_url_verification": "VERIFIED_OFFICIAL_PINNED_REVISION"}
    manifest = {"corpus_id": "omnirag-realtech-m1p2", "target_collection": "omnirag_eval_test_v1",
                "embedding_space": "nomic-embed-text-768",
                "source_count": 1, "sources": [source], "repositories": [repo],
                "license_files": [license_entry]}
    manifest["content_set_sha256"] = content_set_sha256(manifest["sources"])
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata/source_licences.json").write_text(json.dumps({"repositories": [repo],
        "license_files": [license_entry]}), encoding="utf-8")
    (root / "metadata/repository_revisions.json").write_text(json.dumps({"repositories": [repo]}), encoding="utf-8")
    (root / "metadata/acquisition_report.json").write_text(json.dumps({"document_count": 1,
        "content_set_sha256": manifest["content_set_sha256"]}), encoding="utf-8")
    eval_manifest = tmp_path / "evaluation_manifest.json"
    eval_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest, eval_manifest, source_path, license_path


def test_integrity_validator_accepts_matching_files_and_hashes(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    report = validate_corpus(manifest, root, eval_manifest)
    assert report["valid"] is True
    assert report["sha256_verified_count"] == 1
    assert report["document_count"] == 1


def test_notice_scan_reports_candidates_for_human_review():
    hits = scan_file_notices(b"# SPDX-License-Identifier: MIT\n:copyright: Example\n")
    assert len(hits) == 2
    assert scan_file_notices(b"plain technical content\n") == []


def test_ingestion_dry_run_validates_files_and_refuses_unsafe_api_contract(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    report = build_dry_run(manifest, root, eval_manifest)
    assert report["integrity"]["valid"] is True
    assert report["api_physical_collection_verified"] is False
    assert report["execute_supported"] is True
    assert report["ready"] is False


def test_dry_run_can_be_ready_only_with_matching_signed_runtime_settings(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    settings = {"physical_collection": manifest["target_collection"],
                "limits": {"max_upload_bytes": 2_000_000}}
    report = build_dry_run(manifest, root, eval_manifest, api_settings=settings)
    assert report["ready"] is True
    assert report["api_physical_collection_verified"] is True


def _mock_ingestion_api(manifest, *, fail_upload=False, auth_503_once=False,
                        auth_503_message="A trusted authentication provider is not configured"):
    calls = {"uploads": 0, "provider_calls": 0, "identity_checks": 0}
    def budget(run_id):
        return {"run_id": run_id, "calls": calls["provider_calls"],
                "total_tokens": calls["provider_calls"] * 10, "estimated_cost_usd": 0.0,
                "max_cost_usd": 0.0, "max_total_tokens": 4_200_000, "max_calls": 3_000}
    def request(url, method, headers, body, timeout):
        if url.endswith("/health/ready"):
            return 200, {}, {"status": "ready", "service": "OmniRAG"}
        if url.endswith("/api/v1/settings"):
            calls["identity_checks"] += 1
            run_id = headers["X-OmniRAG-Run-ID"]
            return 200, {"X-OmniRAG-Budget-Usage": json.dumps(budget(run_id))}, {
                "user_id": "corpus-reader", "tenant_id": "realworld",
                "physical_collection": manifest["target_collection"],
                "embedding_space": {"alias": "nomic-embed-text-768", "provider": "openai_compatible",
                                    "model_id": "nomic-embed-text", "dimensions": 768},
                "accepted_identities": [{"user_id": "corpus-reader", "tenant_id": "realworld"}],
                "limits": {"max_cost_usd": 0, "max_total_tokens": 4_200_000,
                           "max_calls": 3_000, "max_upload_bytes": 2_000_000},
            }
        if url.endswith("/api/v1/documents") and method == "GET":
            return 200, {}, []
        if method == "PUT":
            calls["uploads"] += 1
            run_id = headers["X-OmniRAG-Run-ID"]
            if auth_503_once and calls["uploads"] == 1:
                return 503, {"X-OmniRAG-Budget-Usage": json.dumps(budget(run_id))}, {
                    "detail": auth_503_message, "request_id": headers["X-Request-ID"]}
            calls["provider_calls"] += 1
            source = manifest["sources"][0]
            payload = {"source_hash": source["sha256"], "version": source["repository_commit"],
                       "status": "failed" if fail_upload else "indexed",
                       "document_id": "api-doc-1", "chunk_count": 2, "size_bytes": len(body)}
            return 201, {"X-OmniRAG-Budget-Usage": json.dumps(budget(run_id))}, payload
        raise AssertionError(f"unexpected mocked request {method} {url}")
    return calls, request


def test_runner_identity_headers_match_api_trusted_proxy_verifier():
    headers = sign_identity_headers("corpus-reader", "realworld", "unit-test-secret", timestamp=1000)
    identity = verify_proxy_identity(headers["X-OmniRAG-User-ID"], headers["X-OmniRAG-Tenant-ID"],
        headers["X-OmniRAG-Auth-Timestamp"], headers["X-OmniRAG-Auth-Signature"],
        "unit-test-secret", now=1000)
    assert identity == {"user_id": "corpus-reader", "tenant_id": "realworld"}


def test_atomic_progress_replace_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    import scripts.realworld.ingest_corpus as ingestion
    real_replace = ingestion.os.replace
    calls = {"count": 0}
    def transient_replace(source, destination):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError(5, "temporarily locked")
        return real_replace(source, destination)
    monkeypatch.setattr(ingestion.os, "replace", transient_replace)
    path = tmp_path / "progress.json"
    ingestion._atomic_json(path, {"status": "running"})
    assert calls["count"] == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "running"}


def _prior_failed_run(tmp_path, manifest, eval_manifest):
    from scripts.realworld.ingest_corpus import _sha256
    prior = {"run_id": "old-ingest", "status": "stopped_first_error",
        "collection": manifest["target_collection"],
        "manifest_sha256": _sha256(eval_manifest.read_bytes()),
        "content_set_sha256": manifest["content_set_sha256"],
        "identity": {"user_id": "corpus-reader", "tenant_id": "realworld"},
        "guard_mode": "dedicated_qdrant_storage_isolation",
        "isolation_proof": {"qdrant": {"id": "qdrant-id"}, "qdrant_volume": "qdrant-volume"},
        "budget_usage": {"calls": 18, "total_tokens": 1000, "estimated_cost_usd": 0.0},
        "schedule": [{"document_id": manifest["sources"][0]["document_id"],
            "source_sha256": manifest["sources"][0]["sha256"], "status": "failed",
            "request_id": "old-request-id", "failure_classification": "RuntimeError"}]}
    path = tmp_path / "old-progress.json"
    path.write_text(json.dumps(prior), encoding="utf-8")
    return path


@pytest.mark.parametrize("failure_message,expect_retry", [
    ("A trusted authentication provider is not configured", True),
    ("Different service failure", False),
])
def test_continuation_retries_only_known_auth_503_after_fresh_signed_probe(
        tmp_path, monkeypatch, failure_message, expect_retry):
    import scripts.realworld.ingest_corpus as ingestion
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest, auth_503_once=True,
                                         auth_503_message=failure_message)
    proof = {"verified": True, "proof_sha256": "same-proof",
             "qdrant": {"id": "qdrant-id"}, "qdrant_volume": "qdrant-volume"}
    monkeypatch.setattr(ingestion, "_read_collection_points", lambda *_: [])
    monkeypatch.setattr(ingestion, "validate_dedicated_qdrant", lambda **_: proof)
    prior_path = _prior_failed_run(tmp_path, manifest, eval_manifest)
    run_dir = tmp_path / "runs" / "new-unique-run"
    args = dict(api_base_url="http://localhost:8002", secret="unit-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="new-unique-run",
        output_dir=run_dir, request_timeout=30, run_timeout=60,
        qdrant_url="http://localhost:6335", isolated_qdrant=True,
        runtime_inspector=lambda *_: {}, continue_from_path=prior_path,
        retry_failed_after_identity_check=True, prior_http_status=503, request_fn=request)
    if expect_retry:
        result = execute_ingestion(manifest, root, eval_manifest, **args)
        row = result["schedule"][0]
        assert result["status"] == "completed"
        assert calls["uploads"] == 2
        assert calls["identity_checks"] >= 2  # initial proof plus fresh pre-retry proof
        assert len(row["attempts"]) == 2
        assert row["attempts"][0]["http_status"] == 503
        assert row["attempts"][0]["error"]["message"] == failure_message
        assert row["attempts"][1]["http_status"] == 201
        assert result["cumulative_budget_usage"]["calls"] >= 19
    else:
        with pytest.raises(RuntimeError, match="Ingestion stopped at"):
            execute_ingestion(manifest, root, eval_manifest, **args)
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        assert calls["uploads"] == 1
        assert progress["schedule"][0]["http_status"] == 503
        assert progress["schedule"][0]["error_details"]["message"] == failure_message


def test_continuation_retries_indeterminate_only_when_live_state_proves_absence(tmp_path, monkeypatch):
    import scripts.realworld.ingest_corpus as ingestion
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    proof = {"verified": True, "proof_sha256": "same-proof",
             "qdrant": {"id": "qdrant-id"}, "qdrant_volume": "qdrant-volume"}
    monkeypatch.setattr(ingestion, "_read_collection_points", lambda *_: [])
    monkeypatch.setattr(ingestion, "validate_dedicated_qdrant", lambda **_: proof)
    prior_path = _prior_failed_run(tmp_path, manifest, eval_manifest)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior["schedule"][0]["status"] = "indeterminate"
    prior["schedule"][0]["failure_classification"] = "PermissionError"
    prior["budget_usage"] = {"calls": 4, "total_tokens": 200, "estimated_cost_usd": 0.0}
    prior["carried_budget_usage"] = {"calls": 18, "total_tokens": 1000, "estimated_cost_usd": 0.0}
    prior_path.write_text(json.dumps(prior), encoding="utf-8")
    result = execute_ingestion(manifest, root, eval_manifest,
        api_base_url="http://localhost:8002", secret="unit-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="new-after-indeterminate",
        output_dir=tmp_path / "runs" / "indeterminate-retry", request_timeout=30, run_timeout=60,
        qdrant_url="http://localhost:6335", isolated_qdrant=True,
        runtime_inspector=lambda *_: {}, continue_from_path=prior_path,
        retry_proven_absent_indeterminate=True, prior_http_status=None, request_fn=request)
    assert result["status"] == "completed"
    assert result["schedule"][0]["prior_failure"]["safe_retry_reason"].startswith(
        "signed SQLite listing and Qdrant scroll")
    assert result["cumulative_budget_usage"]["calls"] == 23
    assert calls["uploads"] == 1


def test_confirmed_415_requires_explicit_extension_fix_gate(tmp_path, monkeypatch):
    import scripts.realworld.ingest_corpus as ingestion
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    proof = {"verified": True, "proof_sha256": "same-proof",
             "qdrant": {"id": "qdrant-id"}, "qdrant_volume": "qdrant-volume"}
    monkeypatch.setattr(ingestion, "_read_collection_points", lambda *_: [])
    monkeypatch.setattr(ingestion, "validate_dedicated_qdrant", lambda **_: proof)
    prior_path = _prior_failed_run(tmp_path, manifest, eval_manifest)
    with pytest.raises(RuntimeError, match="explicit correction gate"):
        ingestion.reconcile_prior_run(prior_path, manifest=manifest, manifest_path=eval_manifest,
            current_isolation_proof=proof, current_api_documents=[], points=[],
            user_id="corpus-reader", tenant_id="realworld", prior_http_status=415,
            retry_failed_after_identity_check=False)
    result = execute_ingestion(manifest, root, eval_manifest,
        api_base_url="http://localhost:8002", secret="unit-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="new-after-415-fix",
        output_dir=tmp_path / "runs" / "retry-after-415", request_timeout=30, run_timeout=60,
        qdrant_url="http://localhost:6335", isolated_qdrant=True,
        runtime_inspector=lambda *_: {}, continue_from_path=prior_path,
        retry_unsupported_extension_after_fix=True, prior_http_status=415, request_fn=request)
    assert result["status"] == "completed"
    assert calls["uploads"] == 1


def test_transport_exception_details_and_request_id_are_preserved(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    def failing_request(url, method, headers, body, timeout):
        if method == "PUT":
            raise PermissionError(13, "Permission denied")
        return request(url, method, headers, body, timeout)
    baseline_path = tmp_path / "protected.json"
    baseline_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "runs" / "transport-failure"
    with pytest.raises(RuntimeError, match="Ingestion stopped at"):
        execute_ingestion(manifest, root, eval_manifest, api_base_url="http://localhost:8002",
            secret="unit-test-secret", user_id="corpus-reader", tenant_id="realworld",
            run_id="transport-failure", output_dir=run_dir, request_timeout=30, run_timeout=60,
            protected_baseline_path=baseline_path,
            protected_baseline_check=lambda *_: {"preserved": True, "current": []},
            request_fn=failing_request)
    row = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))["schedule"][0]
    assert row["status"] == "indeterminate"
    assert row["exception"]["exception_class"] == "PermissionError"
    assert row["exception"]["errno"] == 13
    assert row["exception"]["stage"] == "http_request"
    assert row["attempts"][0]["request_id"] == row["request_id"]


def test_serial_ingestion_executor_records_file_and_is_idempotent_on_resume(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    run_dir = tmp_path / "runs" / "m1p2-test-run"
    baseline_path = tmp_path / "protected.json"
    baseline_path.write_text("{}", encoding="utf-8")
    args = dict(api_base_url="http://localhost:8002", secret="not-printed-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="m1p2-test-run",
        output_dir=run_dir, request_timeout=30, run_timeout=60,
        protected_baseline_path=baseline_path,
        protected_baseline_check=lambda *_: {"preserved": True, "current": []}, request_fn=request)
    result = execute_ingestion(manifest, root, eval_manifest, **args)
    assert result["status"] == "completed"
    assert result["schedule"][0]["status"] == "indexed"
    assert result["schedule"][0]["chunk_count"] == 2
    assert calls["uploads"] == 1

    resumed = execute_ingestion(manifest, root, eval_manifest, resume=True, **args)
    assert resumed["status"] == "completed"
    assert calls["uploads"] == 1


def test_ingestion_executor_aborts_and_preserves_first_failed_attempt(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest, fail_upload=True)
    run_dir = tmp_path / "runs" / "m1p2-test-run"
    baseline_path = tmp_path / "protected.json"
    baseline_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Ingestion stopped at"):
        execute_ingestion(manifest, root, eval_manifest, api_base_url="http://localhost:8002",
            secret="not-printed-test-secret", user_id="corpus-reader", tenant_id="realworld",
            run_id="m1p2-test-run", output_dir=run_dir, request_timeout=30, run_timeout=60,
            protected_baseline_path=baseline_path,
            protected_baseline_check=lambda *_: {"preserved": True, "current": []},
            request_fn=request)
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "stopped_first_error"
    assert progress["schedule"][0]["status"] == "failed"
    assert calls["uploads"] == 1


def test_ingestion_resume_rejects_a_reset_budget_ledger(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    run_dir = tmp_path / "runs" / "m1p2-test-run"
    baseline_path = tmp_path / "protected.json"
    baseline_path.write_text("{}", encoding="utf-8")
    args = dict(api_base_url="http://localhost:8002", secret="not-printed-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="m1p2-test-run",
        output_dir=run_dir, request_timeout=30, run_timeout=60,
        protected_baseline_path=baseline_path,
        protected_baseline_check=lambda *_: {"preserved": True, "current": []}, request_fn=request)
    execute_ingestion(manifest, root, eval_manifest, **args)
    calls["provider_calls"] = 0
    with pytest.raises(RuntimeError, match="ledger reset/regressed"):
        execute_ingestion(manifest, root, eval_manifest, resume=True, **args)


def test_ingestion_executor_refuses_to_upload_without_preservation_baseline(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    with pytest.raises(RuntimeError, match="baseline is missing"):
        execute_ingestion(manifest, root, eval_manifest, api_base_url="http://localhost:8002",
            secret="not-printed-test-secret", user_id="corpus-reader", tenant_id="realworld",
            run_id="m1p2-test-run", output_dir=tmp_path / "runs" / "missing-baseline",
            request_timeout=30, run_timeout=60,
            protected_baseline_path=tmp_path / "does-not-exist.json", request_fn=request)
    assert calls["uploads"] == 0


def test_shared_qdrant_fingerprint_guard_remains_required(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    baseline_path = tmp_path / "protected.json"
    baseline_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="baseline changed before ingestion"):
        execute_ingestion(manifest, root, eval_manifest, api_base_url="http://localhost:8002",
            secret="not-printed-test-secret", user_id="corpus-reader", tenant_id="realworld",
            run_id="m1p2-test-run", output_dir=tmp_path / "runs" / "shared",
            request_timeout=30, run_timeout=60, protected_baseline_path=baseline_path,
            protected_baseline_check=lambda *_: {"preserved": False, "current": []},
            request_fn=request)
    assert calls["uploads"] == 0


def test_dedicated_qdrant_guard_uses_isolation_proof_without_legacy_fingerprints(tmp_path, monkeypatch):
    import scripts.realworld.ingest_corpus as ingestion

    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    calls, request = _mock_ingestion_api(manifest)
    proof_checks = []
    monkeypatch.setattr(ingestion, "validate_dedicated_qdrant", lambda **kwargs: (
        proof_checks.append(kwargs) or {"verified": True, "proof_sha256": "isolated-proof"}))
    run_dir = tmp_path / "runs" / "isolated"
    result = execute_ingestion(manifest, root, eval_manifest,
        api_base_url="http://localhost:8002", secret="not-printed-test-secret",
        user_id="corpus-reader", tenant_id="realworld", run_id="m1p2-test-run",
        output_dir=run_dir, request_timeout=30, run_timeout=60,
        qdrant_url="http://localhost:6335", isolated_qdrant=True,
        runtime_inspector=lambda *_: {"runtime": "docker-inspection"}, request_fn=request)
    assert result["status"] == "completed"
    assert result["guard_mode"] == "dedicated_qdrant_storage_isolation"
    assert result["protected_baseline_sha256"] is None
    assert result["historical_protected_collection_references"][0]["interpretation"] == "historical_reference_not_current_measurement"
    assert len(proof_checks) == 2  # before and after the upload
    assert calls["uploads"] == 1


def test_integrity_validator_rejects_missing_file(tmp_path):
    root, manifest, eval_manifest, source_path, _ = _integrity_fixture(tmp_path)
    source_path.unlink()
    report = validate_corpus(manifest, root, eval_manifest)
    assert any("source file missing" in error for error in report["errors"])


def test_integrity_validator_rejects_altered_file(tmp_path):
    root, manifest, eval_manifest, source_path, _ = _integrity_fixture(tmp_path)
    source_path.write_bytes(b"changed content")
    report = validate_corpus(manifest, root, eval_manifest)
    assert any("SHA-256 mismatch" in error for error in report["errors"])
    assert any("file size mismatch" in error for error in report["errors"])


def test_integrity_validator_rejects_unmanifested_file(tmp_path):
    root, manifest, eval_manifest, _, _ = _integrity_fixture(tmp_path)
    (root / "sources/test/extra.md").write_text("untracked", encoding="utf-8")
    report = validate_corpus(manifest, root, eval_manifest)
    assert any("unmanifested source file" in error for error in report["errors"])


def test_manifest_rejects_unapproved_licence_and_missing_hash(tmp_path):
    _, manifest, _, _, _ = _integrity_fixture(tmp_path)
    row = manifest["sources"][0]
    row["status"] = "BLOCKED"
    row["cleared_for_reuse"] = False
    row["sha256"] = None
    errors = validate_manifest(manifest)
    assert any("not cleared for reuse" in item for item in errors)
    assert any("invalid SHA-256" in item for item in errors)


def test_manifest_rejects_abbreviated_repository_commit(tmp_path):
    _, manifest, _, _, _ = _integrity_fixture(tmp_path)
    manifest["repositories"][0]["commit"] = "abc1234"
    manifest["sources"][0]["repository_commit"] = "abc1234"
    errors = validate_manifest(manifest)
    assert any("full Git object ID" in item for item in errors)


def test_integrity_validator_rejects_duplicate_document_hashes(tmp_path):
    root, manifest, eval_manifest, source_path, _ = _integrity_fixture(tmp_path)
    duplicate = dict(manifest["sources"][0], document_id="test:copy.md", source_id="test:copy.md",
                     local_path="sources/test/copy.md", original_path="copy.md")
    copy_path = root / duplicate["local_path"]
    copy_path.write_bytes(source_path.read_bytes())
    manifest["sources"].append(duplicate)
    manifest["source_count"] = 2
    manifest["content_set_sha256"] = content_set_sha256(manifest["sources"])
    repo = manifest["repositories"][0]
    repo["source_inventory"].append("copy.md")
    (root / "metadata/source_licences.json").write_text(json.dumps({"repositories": manifest["repositories"],
        "license_files": manifest["license_files"]}), encoding="utf-8")
    (root / "metadata/repository_revisions.json").write_text(json.dumps({"repositories": manifest["repositories"]}), encoding="utf-8")
    (root / "metadata/acquisition_report.json").write_text(json.dumps({"document_count": 2,
        "content_set_sha256": manifest["content_set_sha256"]}), encoding="utf-8")
    eval_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_corpus(manifest, root, eval_manifest)
    assert report["duplicate_content_hashes"] == 1


def test_integrity_validator_rejects_license_hash_mismatch(tmp_path):
    root, manifest, eval_manifest, _, license_path = _integrity_fixture(tmp_path)
    license_path.write_bytes(b"changed license")
    report = validate_corpus(manifest, root, eval_manifest)
    assert any("license evidence SHA-256 mismatch" in error for error in report["errors"])


def test_manifest_rejects_unexpected_file_type(tmp_path):
    _, manifest, _, _, _ = _integrity_fixture(tmp_path)
    manifest["sources"][0]["local_path"] = "sources/test/document.exe"
    manifest["sources"][0]["file_format"] = "exe"
    errors = validate_manifest(manifest)
    assert any("unexpected file type" in error for error in errors)


def test_manifest_rejects_duplicate_paths(tmp_path):
    _, manifest, _, _, _ = _integrity_fixture(tmp_path)
    duplicate = dict(manifest["sources"][0], document_id="test:copy.md", source_id="test:copy.md")
    manifest["sources"].append(duplicate)
    errors = validate_manifest(manifest)
    assert any("duplicate local_path" in error for error in errors)


def _verified_annotation(**changes):
    row = {"case_id": "P7C-RW-001", "query": "What is the gateway identifier?",
           "category": "exact_identifier", "difficulty": "easy",
           "expected_answerability": "answerable", "expected_answer": "OMNI-AX7",
           "evidence_spans": [{"source_document": "gateway.txt", "document_version": "v1",
                               "start_offset": 15, "end_offset": 23}],
           "author_id": "human-author", "reviewer_id": "human-reviewer", "status": "VERIFIED",
           "review_outcome": "approved", "review_notes": "Evidence checked against source.",
           "expected_behavior": "Answer with the cited identifier."}
    row.update(changes)
    return row


def _convert(rows, corpus):
    return convert_human_approved_annotations(
        rows, dataset_id="omnirag-realworld-sealed", dataset_version="0.1.0",
        corpus_id="omnirag-realtech-m1p2", corpus_version="v1",
        corpus_manifest="manifest.json", corpus_root=corpus,
        document_versions={"gateway.txt": "v1"})


def test_duplicate_case_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate annotation case_id"):
        _convert([_verified_annotation(), _verified_annotation()], tmp_path)


def test_duplicate_normalized_queries_rejected(tmp_path):
    second = _verified_annotation(case_id="P7C-RW-002", query="  what is the gateway identifier? ")
    with pytest.raises(ValueError, match="duplicate annotation query"):
        _convert([_verified_annotation(), second], tmp_path)


def test_invalid_evidence_span_rejected(tmp_path):
    (tmp_path / "gateway.txt").write_text("The gateway is OMNI-AX7.", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid evidence offsets"):
        _convert([_verified_annotation(evidence_spans=[{"source_document": "gateway.txt",
                  "document_version": "v1", "start_offset": 100, "end_offset": 120}])], tmp_path)


def test_cross_line_evidence_span_rejected(tmp_path):
    (tmp_path / "gateway.txt").write_text("alpha evidence\nbeta evidence\n", encoding="utf-8")
    row = _verified_annotation(evidence_spans=[{"source_document": "gateway.txt", "document_version": "v1",
                                                "start_offset": 3, "end_offset": 20}])
    with pytest.raises(ValueError, match="one existing evaluator line/page locator"):
        _convert([row], tmp_path)


def test_evidence_document_version_must_match_manifest(tmp_path):
    (tmp_path / "gateway.txt").write_text("The gateway is OMNI-AX7.\n", encoding="utf-8")
    row = _verified_annotation(evidence_spans=[{"source_document": "gateway.txt", "document_version": "v2",
                                                "start_offset": 15, "end_offset": 23}])
    with pytest.raises(ValueError, match="version mismatch"):
        _convert([row], tmp_path)


def test_unreviewed_annotation_cannot_convert(tmp_path):
    (tmp_path / "gateway.txt").write_text("The gateway is OMNI-AX7.", encoding="utf-8")
    row = _verified_annotation(status="REVIEWED")
    with pytest.raises(ValueError, match="not human-approved and VERIFIED"):
        _convert([row], tmp_path)


def test_incomplete_answerable_annotation_rejected():
    with pytest.raises(ValidationError, match="human answer and evidence spans"):
        HumanAnnotation.model_validate(_verified_annotation(expected_answer=None, evidence_spans=[]))


def test_annotation_lifecycle_requires_distinct_human_author_and_reviewer():
    row = HumanAnnotation.model_validate({**_verified_annotation(), "status": "DRAFT",
        "author_id": None, "reviewer_id": None, "review_outcome": None, "review_notes": None})
    annotated = transition_annotation(row, "ANNOTATED", actor_id="author")
    with pytest.raises(ValueError, match="distinct"):
        transition_annotation(annotated, "REVIEWED", actor_id="author",
                             review_outcome="approved", review_notes="reviewed")
    reviewed = transition_annotation(annotated, "REVIEWED", actor_id="reviewer",
                                     review_outcome="approved", review_notes="reviewed")
    verified = transition_annotation(reviewed, "VERIFIED", actor_id="reviewer")
    assert verified.status == "VERIFIED"


def test_human_approved_annotation_converts_through_existing_schema(tmp_path):
    (tmp_path / "gateway.txt").write_text("The gateway is OMNI-AX7.\n", encoding="utf-8")
    dataset = _convert([_verified_annotation()], tmp_path)
    assert dataset.cases[0].expected_answer == "OMNI-AX7"
    assert dataset.cases[0].expected_evidence[0].locator == "line:1"
    assert dataset.cases[0].ground_truth_status == "verified"


def test_human_approved_unanswerable_case_maps_to_existing_unsupported_schema(tmp_path):
    text = "The corpus documents the gateway identifier only.\n"
    (tmp_path / "gateway.txt").write_text(text, encoding="utf-8")
    row = _verified_annotation(category="unanswerable", expected_answerability="unanswerable",
        expected_answer=None, evidence_spans=[{"source_document": "gateway.txt", "document_version": "v1",
                                               "start_offset": 0, "end_offset": len(text.rstrip())}],
        expected_behavior="Abstain because the requested token cap is not in the cited corpus.")
    dataset = _convert([row], tmp_path)
    case = dataset.cases[0]
    assert case.category == "unsupported"
    assert case.expected_abstention is True
    assert case.expected_answer is None
    assert case.answer_match == "not_applicable"


def test_leakage_checker_reports_without_discarding(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("The gateway identifier is OMNI-AX7.", encoding="utf-8")
    findings = leakage_candidates({"cases": [
        {"case_id": "M1-001", "query": "What is OMNI AX7?"},
        {"case_id": "M1-002", "query": "The gateway identifier is OMNI-AX7."},
    ]},
                                  {"cases": [{"case_id": "P7C-US-001", "query": "What is OMNI AX7?"}]}, corpus)
    assert {item["kind"] for item in findings} == {"development_query_overlap", "corpus_wording_overlap"}
    assert all(item["review_required"] for item in findings)
