import json

from evaluation.phase8_scorer import score_artifacts


def test_offline_scorer_is_deterministic_and_keeps_unavailable_null():
    dataset = {"cases": [{
        "case_id": "P7C-AF-001", "category": "answerable_factual",
        "answer_match": "exact", "expected_answer": "Northstar",
        "expected_abstention": False,
        "expected_evidence": [{"source_document": "guide.md", "locator": "line:4"}],
    }]}
    snapshot = {"snapshot_sha256": "snapshot", "evidence_to_chunks": [{
        "case_id": "P7C-AF-001", "source_document": "guide.md", "locator": "line:4",
        "indexed_chunks": [{"chunk_id": "c1"}],
    }]}
    progress = {"benchmark_id": "offline", "schedule": [{
        "attempt_key": "P7C-AF-001-lite-r01-a00", "case_id": "P7C-AF-001",
        "profile": "lite", "repetition": 1, "status": "completed",
    }]}
    attempts = {"P7C-AF-001-lite-r01-a00": {
        "status": "completed", "expected_answer_match": True, "abstained": False,
        "retrieval_candidates": [{"chunk_id": "c1"}],
        "evidence_references": [{"chunk_id": "c1"}], "citations": [{"chunk_id": "c1"}],
        "authorization_audit": {}, "elapsed_ms": 20.0,
    }}
    first = score_artifacts(dataset, snapshot, progress, attempts)
    second = score_artifacts(dataset, snapshot, progress, attempts)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    row = first["units"][0]
    assert row["retrieval_recall_at_10"] == 1.0
    assert row["precision_at_5"] == 0.2
    assert row["hit_rate_at_5"] == 1.0
    assert row["hit_rate_at_10"] == 1.0
    assert row["recall_at_5"] == 1.0
    assert row["evidence_coverage"] == 1.0
    assert row["mrr_at_10"] == 1.0
    assert row["ndcg_at_10"] == 1.0
    assert row["exact_answer_match"] is True
    assert row["input_tokens"] is None and row["cost_usd"] is None
    assert first["summary"]["execution_status_counts"] == {"completed": 1}


def test_offline_scorer_retains_failed_units_in_denominator():
    dataset = {"cases": [{"case_id": "P7C-US-001", "category": "unsupported",
                          "answer_match": "not_applicable", "expected_abstention": True,
                          "expected_evidence": [{"source_document": "x.txt", "locator": "line:1"}]}]}
    progress = {"schedule": [{"attempt_key": "k", "case_id": "P7C-US-001",
                               "profile": "research", "repetition": 1, "status": "timed_out"}]}
    report = score_artifacts(dataset, {"evidence_to_chunks": []}, progress, {})
    assert report["summary"]["scheduled_units"] == 1
    assert report["summary"]["execution_success_rate"] == 0.0
    assert report["summary"]["abstention_correct"]["denominator"] == 0
    assert report["units"][0]["status"] == "timed_out"


def test_warmup_units_are_accounted_but_excluded_from_scored_results():
    dataset = {"cases": [{"case_id": "P7C-AF-001", "category": "answerable_factual",
                          "answer_match": "exact", "expected_answer": "Northstar",
                          "expected_abstention": False, "expected_evidence": []}]}
    progress = {"benchmark_id": "warmup-run", "warmup": True, "schedule": [{
        "attempt_key": "P7C-AF-001-lite-r01-a00", "case_id": "P7C-AF-001",
        "profile": "lite", "repetition": 1, "status": "completed"}]}
    report = score_artifacts(dataset, {"evidence_to_chunks": []}, progress,
                             {"P7C-AF-001-lite-r01-a00": {"status": "completed"}})
    assert report["warmup"] is True
    assert report["summary"]["scheduled_units"] == 0
    assert report["summary"]["warmup_units_excluded"] == 1
    assert report["summary"]["warmup_status_counts"] == {"completed": 1}
    assert report["units"] == []


def test_denied_and_unsupported_cases_do_not_get_answerable_retrieval_scores():
    dataset = {"cases": [{"case_id": "P7C-US-001", "category": "unsupported",
                          "answer_match": "not_applicable", "expected_abstention": True,
                          "expected_evidence": [{"source_document": "x.txt", "locator": "line:1"}]}]}
    snapshot = {"evidence_to_chunks": [{"case_id": "P7C-US-001", "source_document": "x.txt",
                                        "locator": "line:1", "indexed_chunks": [{"chunk_id": "c1"}]}]}
    progress = {"schedule": [{"attempt_key": "k", "case_id": "P7C-US-001",
                               "profile": "lite", "repetition": 1, "status": "completed"}]}
    attempts = {"k": {"status": "completed", "abstained": True,
                       "retrieval_candidates": [{"chunk_id": "c1"}],
                       "evidence_references": [], "citations": [],
                       "authorization_audit": {}, "token_usage": {}}}
    row = score_artifacts(dataset, snapshot, progress, attempts)["units"][0]
    assert row["abstention_correct"] is True
    assert row["retrieval_recall_at_10"] is None
    assert row["mrr_at_10"] is None
    assert row["citation_precision"] is None


def test_posthoc_acl_audit_checks_snapshot_metadata_not_only_runner_flag():
    dataset = {"cases": [{"case_id": "P7C-AT-002", "category": "acl_tenant_isolation",
                          "answer_match": "not_applicable", "expected_abstention": True,
                          "acl_context": {"request_principal": "user:denied", "expected_access": "deny"},
                          "tenant_context": {"request_tenant_id": "tenant-a"},
                          "expected_evidence": [{"source_document": "restricted.txt", "locator": "line:1"}]}]}
    snapshot = {"evidence_to_chunks": [{"case_id": "P7C-AT-002", "source_document": "restricted.txt",
                                        "locator": "line:1", "indexed_chunks": [{"chunk_id": "c1"}]}],
                "points": [{"chunk_id": "c1", "document_id": "d1", "tenant_id": "tenant-a",
                            "acl_allowed_principals": ["user:allowed"],
                            "acl_denied_principals": ["user:denied"]}]}
    progress = {"schedule": [{"attempt_key": "k", "case_id": "P7C-AT-002",
                               "profile": "lite", "repetition": 1, "status": "completed"}]}
    attempts = {"k": {"status": "completed", "abstained": False,
                       "retrieval_candidates": [{"chunk_id": "c1", "document_id": "d1"}],
                       "evidence_references": [], "citations": [],
                       "authorization_audit": {"violation": False}, "token_usage": {}}}
    row = score_artifacts(dataset, snapshot, progress, attempts)["units"][0]
    assert row["acl_audit_pass"] is False
