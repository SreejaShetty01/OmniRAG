import csv
import json
from pathlib import Path
import uuid

import pytest

from scripts.diagnostic.prepare_m2_review import (
    PINNED_DATASET_SHA256,
    prepare_exports,
    sha256_file,
    source_hashes,
    stable_id,
    write_csv,
)
from scripts.diagnostic.score_review import cohens_kappa, wilson_interval, write_scoring_reports


def test_csv_round_trip_and_stable_ids(tmp_path):
    row_id = stable_id("R", "case|profile|dataset")
    assert row_id == stable_id("R", "case|profile|dataset")
    path = tmp_path / "review.csv"
    write_csv(path, ["row_id", "text"], [{"row_id": row_id, "text": "héllo"}])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row == {"row_id": row_id, "text": "héllo"}


def test_wilson_interval_known_toy_case():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=0.0001)
    assert high == pytest.approx(0.7634, abs=0.0001)
    assert wilson_interval(0, 0) is None


def test_cohens_kappa_toy_cases():
    kappa, agreement, n = cohens_kappa(["Y", "N", "Y", "N"], ["Y", "N", "Y", "N"])
    assert (kappa, agreement, n) == (1.0, 1.0, 4)
    kappa, agreement, n = cohens_kappa(["Y", "Y", "N", "N"], ["N", "N", "Y", "Y"])
    assert (kappa, agreement, n) == (-1.0, 0.0, 4)


def test_scorer_does_not_report_metrics_with_incomplete_human_labels(tmp_path):
    stamp = f"DIAGNOSTIC — dataset sha256 {'a' * 64} != pinned {PINNED_DATASET_SHA256}"
    (tmp_path / "review_manifest.json").write_text(json.dumps({
        "diagnostic_stamp": stamp, "dataset_sha256": "a" * 64,
    }), encoding="utf-8")
    fields = ["diagnostic_stamp", "row_id", "profile", "answer_status", "answer_correct",
              "claims_supported_by_context", "citation_supports_claim", "context_status", "context_text"]
    write_csv(tmp_path / "answers_to_label.csv", fields, [{
        "diagnostic_stamp": stamp, "row_id": "R1", "profile": "lite",
        "answer_status": "FINAL_ANSWER_STORED", "answer_correct": "",
        "claims_supported_by_context": "", "citation_supports_claim": "",
        "context_status": "NOT_STORED", "context_text": "",
    }])
    deterministic_fields = ["diagnostic_stamp", "row_id", "case_id", "profile",
                            "deterministic_answer_correct"]
    write_csv(tmp_path / "deterministic_grounding_review.csv", deterministic_fields, [
        {"diagnostic_stamp": stamp, "row_id": f"G{i}", "case_id": f"C{i}",
         "profile": p, "deterministic_answer_correct": str(v).lower()}
        for p, count in (("lite", 9), ("core", 4), ("research", 6))
        for i, v in enumerate([True] * count, 1)
    ])
    simple = ["diagnostic_stamp", "row_id", "evidence_row_id", "claim_review_id",
              "evidence_sufficient", "supported_by_context", "context_status", "context_text"]
    for filename in ("case_evidence_sheet.csv", "claim_review.csv", "second_reviewer.csv"):
        write_csv(tmp_path / filename, simple, [])
    key_fields = ["diagnostic_stamp", "row_id", "case_id", "profile", "expected_answerable"]
    write_csv(tmp_path / "key_for_scoring.csv", key_fields, [{
        "diagnostic_stamp": stamp, "row_id": "R1", "case_id": "C1",
        "profile": "lite", "expected_answerable": "true",
    }])
    write_csv(tmp_path / "gold_key_facts_for_review.csv", key_fields, [{
        "diagnostic_stamp": stamp, "row_id": "R1", "case_id": "C1",
        "profile": "lite", "expected_answerable": "true",
    }])
    report = write_scoring_reports(tmp_path)
    assert report["status"].startswith("PARTIAL")
    assert report["metric_status"]["end_to_end_answerable_accuracy"] == "NOT_MEASURABLE_LABELS_INCOMPLETE"
    assert all(metric["metric"] == "deterministic_correctness_on_scored_subset"
               for metric in report["metrics"])


def test_blind_export_has_no_verifier_fields_and_hashes_unchanged():
    root = Path(__file__).resolve().parents[2]
    out_dir = prepare_exports(root=root, run_id="review-test-" + uuid.uuid4().hex[:8])
    blind_path = out_dir / "answers_to_label.csv"
    with blind_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    assert len(rows) == 85
    assert not any("verifier" in name.casefold() or "abstention_reason" in name.casefold()
                   for name in fields)
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert sum(row["answer_status"] == "DRAFT_NOT_STORED" for row in rows) == 36
    assert sum(row["answer_status"] == "FINAL_ANSWER_STORED" for row in rows) == 49
    report = json.loads((out_dir / "source_hashes_before_after.json").read_text(encoding="utf-8"))
    assert report["identical"] is True
    assert report["source_hashes_before"] == report["source_hashes_after"]
    manifest = json.loads((out_dir / "review_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_sha256"] != PINNED_DATASET_SHA256
    assert (out_dir / "deterministic_grounding_review.csv").exists()
    with (out_dir / "deterministic_grounding_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        deterministic = list(csv.DictReader(stream))
    assert len(deterministic) == 19
    assert sum(row["profile"] == "lite" for row in deterministic) == 9
    assert sum(row["profile"] == "core" for row in deterministic) == 4
    assert sum(row["profile"] == "research" for row in deterministic) == 6
    assert all(row["deterministic_answer_correct"] in {"true", "false"} for row in deterministic)
    source = [json.loads(line) for line in (root / "data/diagnostic/posthoc-remediation-20261107T072431Z-663cec3c/attempts_scored.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = {(row["case_id"], row["profile"]): str(row["answer_correct_raw"]).lower()
              for row in source if isinstance(row.get("answer_correct_raw"), bool)}
    assert all(scores[(row["case_id"], row["profile"])] == row["deterministic_answer_correct"]
               for row in deterministic)
    with (out_dir / "case_evidence_sheet.csv").open(encoding="utf-8-sig", newline="") as stream:
        evidence = list(csv.DictReader(stream))
    assert len(evidence) == 99
    assert all(row["final_context_status"].startswith("NOT_STORED") for row in evidence)
    assert (out_dir / "context_hash_comparison.csv").exists()
    with (out_dir / "claim_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        claims = list(csv.DictReader(stream))
    assert len(claims) == 159
    assert all(row["claim_review_id"] and row["row_id"] for row in claims)


def test_input_hash_helper_is_stable(tmp_path):
    path = tmp_path / "data" / "diagnostic" / "old-run" / "source.json"
    path.parent.mkdir(parents=True)
    path.write_text("unchanged", encoding="utf-8")
    dataset_path = tmp_path / "evaluation" / "datasets" / "omnirag-phase8-eval-1.0.1.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("{}", encoding="utf-8")
    before = sha256_file(path)
    mapped = source_hashes(tmp_path)
    assert mapped["data/diagnostic/old-run/source.json"] == before
    assert sha256_file(path) == before
