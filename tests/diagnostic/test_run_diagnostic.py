import csv
import json
from pathlib import Path
import pytest

from evaluation.dataset import EvaluationDataset
from scripts.diagnostic.run_diagnostic import (
    classify_http_outcome, diagnostic_stamp, dry_run, export_human_audit, retrieval_metric_values,
    score_human_audit, score_response, sha256_file, summarize, verify_snapshot_integrity,
    wilson_interval, redact_secret, _future_run_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation/datasets/omnirag-phase8-eval-1.0.1.json"
SNAPSHOT = ROOT / "docs/audit/phase8_eval_corpus_manifest.json"
HISTORICAL = ROOT / "data/phase8/benchmarks/phase8-pilot-20261104-b5"


def test_hash_stamp_and_dataset_schema_are_diagnostic_not_pinned(tmp_path):
    raw_hash = sha256_file(DATASET)
    assert raw_hash.startswith("6a99cdd3")
    assert "DIAGNOSTIC" in diagnostic_stamp(raw_hash)
    assert raw_hash in diagnostic_stamp(raw_hash)
    assert "7b44b8f6" in diagnostic_stamp(raw_hash)
    dataset = EvaluationDataset.model_validate(json.loads(DATASET.read_text(encoding="utf-8")))
    assert len(dataset.cases) == 40
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert verify_snapshot_integrity(snapshot) == snapshot["snapshot_sha256"]


def test_secret_redaction_recurses_through_response_artifacts():
    assert redact_secret({"answer": "token=abc", "items": ["abc"]}, "abc") == {
        "answer": "token=[REDACTED]", "items": ["[REDACTED]"],
    }


def test_seven_retrieval_metrics_and_wilson_interval():
    metrics = retrieval_metric_values(["noise", "hit", "other"], {"hit", "miss"})
    assert metrics == {
        "recall_at_5": 0.5, "recall_at_10": 0.5,
        "precision_at_5": 0.2, "hit_rate_at_5": 1.0,
        "hit_rate_at_10": 1.0, "mrr_at_10": 0.5,
        "ndcg_at_10": pytest.approx(0.38685280723454163),
    }
    assert wilson_interval(0, 10) == pytest.approx([0.0, 0.2775327998628892])
    assert wilson_interval(5, 0) is None


def test_leakage_detected_from_restricted_candidate_context_citation_and_answer_marker():
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    case_data = next(c for c in raw["cases"] if c["category"] == "acl_tenant_isolation"
                     and c["acl_context"]["expected_access"] == "deny")
    from evaluation.dataset import EvaluationCase
    case = EvaluationCase.model_validate(case_data)
    snapshot = {"documents": [{"filename": "12_restricted_team.txt", "tenant_id": "other-tenant",
                                "acl_allowed_principals": ["user:other"]}],
                "points": [{"chunk_id": "restricted", "tenant_id": "other-tenant",
                            "acl_allowed_principals": ["user:other"]}]}
    response = {
        "profile_used": "core", "answer": "cobalt lantern 731", "abstained": False,
        "retrieval_inspection": [{"chunk_id": "restricted", "document_id": "doc", "fusion_rank": 1}],
        "evidence_references": [{"chunk_id": "restricted", "document_id": "doc"}],
        "citations": [{"chunk_id": "restricted", "document_id": "doc"}],
    }
    row = score_response(case, response, snapshot, 9.0)
    assert row["restricted_chunk_leakage"] == {
        "candidates": ["restricted"], "final_context": ["restricted"],
        "citations": ["restricted"], "answer_restricted_marker": True,
        "answer_restricted_marker_authorized": False,
        "answer_restricted_marker_unauthorized": True,
        "answer_restricted_marker_acl_unknown": False,
    }
    summary = summarize([row])
    assert summary["leakage"]["candidates"]["leaks"] == 1


def test_authorized_marker_in_answer_is_not_classified_as_acl_leak():
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    from evaluation.dataset import EvaluationCase
    case_data = next(c for c in raw["cases"] if c["case_id"] == "P7C-AT-001")
    case = EvaluationCase.model_validate(case_data)
    snapshot = {"documents": [{"filename": "12_restricted_team.txt", "tenant_id": "phase7a-tenant-a",
                                "acl_allowed_principals": ["user:phase7a-user-a"],
                                "acl_denied_principals": ["user:phase7a-user-b"]}], "points": []}
    response = {"profile_used": "lite", "answer": case.expected_answer, "abstained": False,
                "retrieval_inspection": [], "evidence_references": [], "citations": []}
    row = score_response(case, response, snapshot)
    assert row["restricted_chunk_leakage"]["answer_restricted_marker"] is True
    assert row["restricted_chunk_leakage"]["answer_restricted_marker_authorized"] is True
    assert row["restricted_chunk_leakage"]["answer_restricted_marker_unauthorized"] is False
    assert row["acl_outcome_pass"] is True
    assert row["restricted_chunk_leakage"]["candidates"] == []


def test_successful_slow_http_200_is_completed_with_latency_budget_violation():
    assert classify_http_outcome(200, 151_000) == ("completed", True)
    assert classify_http_outcome(504, 151_000) == ("timed_out", True)
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    from evaluation.dataset import EvaluationCase
    case = EvaluationCase.model_validate(next(c for c in raw["cases"] if c["case_id"] == "P7C-AT-001"))
    response = {"profile_used": "lite", "answer": "", "abstained": True}
    row = score_response(case, response, {"documents": [], "points": []}, 151_000)
    status, violation = classify_http_outcome(200, 151_000)
    row.update({"status": status, "http_status": 200,
                "latency_budget_exceeded": violation})
    summary = summarize([row])
    assert row["status"] == "completed"
    assert summary["profiles"]["lite"]["timeout_count"] == 0
    assert summary["profiles"]["lite"]["latency_budget_violation_count"] == 1


def test_future_provenance_pins_dataset_snapshot_collection_model_and_runtime():
    snapshot = {"snapshot_sha256": "snapshot", "collection": "omnirag_eval_example", "point_count": 12}
    settings = {"physical_collection": "omnirag_eval_example",
                "embedding_space": {"alias": "local", "model_id": "embed", "dimensions": 768},
                "models": {"generation": {"provider": "openai_compatible", "model_id": "llama-local"}},
                "limits": {"max_cost_usd": 0}}
    result = _future_run_provenance("dataset", snapshot, settings, "http://localhost:8001", None)
    assert result["dataset_sha256"] == "dataset"
    assert result["corpus_snapshot_sha256"] == "snapshot"
    assert result["qdrant_collection"] == "omnirag_eval_example"
    assert result["qdrant_point_count"] == 12
    assert result["qdrant_point_count_source"].startswith("pinned_corpus_snapshot_manifest")
    assert result["embedding"]["dimensions"] == 768
    assert result["generation_model"]["model_id"] == "llama-local"


def test_summary_separates_answer_confusion_and_profiles_are_data_driven():
    records = [
        {"profile": "research", "expected_abstention": False, "answer_match_type": "exact",
         "case_id": "c1",
         "answer_correct_raw": False, "abstained": True, "abstention_correct": False,
         "retrieval_metrics": {}, "restricted_chunk_leakage": {"candidates": [], "final_context": [],
         "citations": [], "answer_restricted_marker": False}},
        {"profile": "research", "expected_abstention": True, "answer_match_type": "evidence_based",
         "case_id": "c2",
         "answer_correct_raw": None, "abstained": True, "abstention_correct": True,
         "retrieval_metrics": {}, "restricted_chunk_leakage": {"candidates": [], "final_context": [],
         "citations": [], "answer_restricted_marker": False}},
    ]
    summary = summarize(records)
    assert list(summary["profiles"]) == ["research"]
    assert summary["profiles"]["research"]["answer_confusion"] == {
        "answerable_correct_exact": 0, "answerable_incorrect_exact": 0,
        "answerable_incorrectly_abstained": 1, "answerable_answered_unscored": 0,
        "unanswerable_correctly_abstained": 1, "unanswerable_answered": 0,
        "unanswerable_answered_insufficient_heuristic": 0,
        "unanswerable_answered_partial_or_strong_heuristic": 0,
        "unanswerable_answered_sufficiency_unknown": 0,
        "answerable_n": 1, "unanswerable_n": 1,
    }


def test_dry_run_uses_historical_artifacts_only_as_fixture_and_sends_no_requests():
    result = dry_run(DATASET, SNAPSHOT, HISTORICAL)
    assert result["mode"] == "dry_run_no_api_requests"
    assert result["api_requests"] == 0
    assert result["live_execution_not_authorized"] is True
    assert "not included as diagnostic results" in result["historical_fixture"]


def test_human_audit_export_duplicates_ten_rows_and_reports_manual_metrics(tmp_path):
    records = [{"case_id": f"c{i}", "profile": "lite", "category": "answerable_factual",
                "query": "question", "answer": "answer", "response": {"claims": []}}
               for i in range(12)]
    output = tmp_path / "audit.csv"
    count = export_human_audit(records, {"points": []}, output)
    assert count == 22
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig", newline="")))
    assert sum(row["review_copy"] == "2" for row in rows) == 10
    rows[0].update({"supported_by_evidence": "Y", "cited_chunk_supports_claim": "Y",
                    "answer_correct": "Y", "reviewer": "A"})
    pair = next(row for row in rows if row["audit_item_id"] == rows[0]["audit_item_id"]
                and row["review_copy"] == "2")
    pair.update({"supported_by_evidence": "Y", "cited_chunk_supports_claim": "N", "reviewer": "B"})
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metrics = score_human_audit(output)
    assert metrics["faithfulness_supported_claim_rate"]["rate"] == 1.0
    assert metrics["citation_accuracy_rate"]["denominator"] == 2
    assert metrics["reviewer_agreement"] == {"agreements": 0, "paired_rows": 1, "rate": 0.0}
