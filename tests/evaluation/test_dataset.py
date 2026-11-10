from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.settings import Settings
from evaluation.dataset import (
    EvaluationDataset, create_manifest, dataset_checksum, validate_dataset,
    validate_manifest, write_release,
)
from evaluation.phase7c_seed import build_dataset
from experiments.runtime import ExperimentConfig, ExperimentRunStore, ExperimentRunner


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests" / "fixtures" / "phase7_smoke"


@pytest.fixture
def dataset_dict():
    return build_dataset().model_dump(mode="json")


def test_checked_in_dataset_and_manifest_validate_against_real_corpus():
    releases = {
        "1.0.0": {"validation_status": "passed_with_provisional_cases",
                  "verified_case_count": 36, "provisional_case_count": 4,
                  "sha256": "a8be8506bf56af28cc86d05e1f925572214c43c5e12b62ec7e15a577f4364725"},
        "1.0.1": {"validation_status": "passed", "verified_case_count": 40,
                  "provisional_case_count": 0, "sha256": None},
    }
    for version, expected in releases.items():
        stem = f"omnirag-phase8-eval-{version}"
        data_path = ROOT / "evaluation" / "datasets" / f"{stem}.json"
        manifest_path = ROOT / "evaluation" / "datasets" / f"{stem}.manifest.json"
        dataset = EvaluationDataset.model_validate_json(data_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validated = validate_dataset(dataset, corpus_root=CORPUS, repository_root=ROOT)
        validate_manifest(validated, manifest, corpus_root=CORPUS, repository_root=ROOT)
        assert len(validated.cases) == 40
        assert manifest["validation_status"] == expected["validation_status"]
        assert manifest["verified_case_count"] == expected["verified_case_count"]
        assert manifest["provisional_case_count"] == expected["provisional_case_count"]
        if expected["sha256"]:
            assert manifest["dataset_sha256"] == expected["sha256"]
    latest = EvaluationDataset.model_validate_json(
        (ROOT / "evaluation" / "datasets" / "omnirag-phase8-eval-1.0.1.json").read_text(encoding="utf-8"))
    latest_acl = [case for case in latest.cases if case.category == "acl_tenant_isolation"]
    assert len(latest_acl) == 4
    assert all(case.ground_truth_status == "verified" for case in latest_acl)


def test_duplicate_ids_are_rejected(dataset_dict):
    dataset_dict["cases"].append(copy.deepcopy(dataset_dict["cases"][0]))
    with pytest.raises(ValidationError, match="duplicate case_id"):
        EvaluationDataset.model_validate(dataset_dict)


def test_missing_required_fields_are_rejected(dataset_dict):
    del dataset_dict["cases"][0]["query"]
    with pytest.raises(ValidationError, match="query"):
        EvaluationDataset.model_validate(dataset_dict)


@pytest.mark.parametrize("field,value", [
    ("category", "made_up_category"),
    ("profile_coverage", ["lite", "analyst"]),
])
def test_invalid_categories_and_profiles_are_rejected(dataset_dict, field, value):
    dataset_dict["cases"][0][field] = value
    with pytest.raises(ValidationError):
        EvaluationDataset.model_validate(dataset_dict)


def test_missing_source_document_is_rejected(dataset_dict):
    dataset_dict["cases"][0]["expected_source_documents"] = ["missing.md"]
    with pytest.raises(ValueError, match="source document does not exist"):
        validate_dataset(dataset_dict, corpus_root=CORPUS, repository_root=ROOT)


def test_evidence_quote_and_location_must_resolve(dataset_dict):
    dataset_dict["cases"][0]["expected_evidence"][0]["quote"] = "A fabricated gateway named Southstar."
    with pytest.raises(ValueError, match="evidence quote does not match"):
        validate_dataset(dataset_dict, corpus_root=CORPUS, repository_root=ROOT)


def test_unsupported_case_cannot_include_fabricated_answer(dataset_dict):
    case = next(row for row in dataset_dict["cases"] if row["category"] == "unsupported")
    case["expected_answer"] = "A made-up answer"
    with pytest.raises(ValidationError, match="unsupported cases must not contain"):
        EvaluationDataset.model_validate(dataset_dict)


def test_conflict_case_requires_multiple_supported_values(dataset_dict):
    case = next(row for row in dataset_dict["cases"] if row["category"] == "conflicting_evidence")
    case["expected_conflict"]["values"] = []
    with pytest.raises(ValueError, match="expected_conflict.values"):
        validate_dataset(dataset_dict, corpus_root=CORPUS, repository_root=ROOT)


def test_acl_and_tenant_cases_are_explicit_and_verified_after_execution(dataset_dict):
    cases = [row for row in dataset_dict["cases"] if row["category"] == "acl_tenant_isolation"]
    assert {row["acl_context"]["expected_access"] for row in cases} == {"allow", "deny"}
    assert all(row["tenant_context"]["document_tenant_id"] for row in cases)
    assert all(row["tenant_context"]["request_tenant_id"] for row in cases)
    assert all(row["ground_truth_status"] == "verified" for row in cases)
    assert all("test_phase7c_acl_dataset.py" in row["verification_note"] for row in cases)
    case = cases[0]
    case["acl_context"] = None
    with pytest.raises(ValidationError, match="ACL cases require"):
        EvaluationDataset.model_validate(dataset_dict)


def test_prompt_injection_case_requires_explicit_security_behavior(dataset_dict):
    case = next(row for row in dataset_dict["cases"] if row["category"] == "prompt_injection")
    case["security_expectation"] = None
    with pytest.raises(ValidationError, match="security_expectation"):
        EvaluationDataset.model_validate(dataset_dict)


def test_manifest_generation_and_consistency(dataset_dict):
    manifest = create_manifest(dataset_dict, corpus_root=CORPUS,
                               repository_root=ROOT, created_at="2026-11-03T00:00:00Z")
    validate_manifest(dataset_dict, manifest, corpus_root=CORPUS, repository_root=ROOT)
    assert manifest["case_count"] == 40
    assert sum(manifest["category_distribution"].values()) == 40
    assert manifest["dataset_sha256"] == dataset_checksum(dataset_dict)
    broken = {**manifest, "case_count": 35}
    with pytest.raises(ValueError, match="case_count"):
        validate_manifest(dataset_dict, broken, corpus_root=CORPUS, repository_root=ROOT)


def test_checksum_is_stable_across_json_key_order(dataset_dict):
    reordered = json.loads(json.dumps(dataset_dict, sort_keys=True))
    assert dataset_checksum(dataset_dict) == dataset_checksum(reordered)


def test_versioned_release_is_immutable_and_new_version_gets_new_files(tmp_path, dataset_dict):
    first_json, first_manifest = write_release(
        dataset_dict, output_dir=tmp_path, corpus_root=CORPUS,
        repository_root=ROOT, created_at="2026-11-03T00:00:00Z",
    )
    first_digest = dataset_checksum(dataset_dict)
    changed = copy.deepcopy(dataset_dict)
    changed["cases"][0]["tags"].append("new-tag")
    with pytest.raises(ValueError, match="Refusing to replace immutable dataset version"):
        write_release(changed, output_dir=tmp_path, corpus_root=CORPUS, repository_root=ROOT)
    changed["dataset_version"] = "1.0.2"
    second_json, second_manifest = write_release(
        changed, output_dir=tmp_path, corpus_root=CORPUS,
        repository_root=ROOT, created_at="2026-11-04T00:00:00Z",
    )
    assert first_json.exists() and first_manifest.exists()
    assert second_json.exists() and second_manifest.exists()
    assert first_json != second_json
    assert dataset_checksum(EvaluationDataset.model_validate_json(first_json.read_text())) == first_digest


def test_phase7b_experiment_manifest_can_reference_exact_dataset_version(dataset_dict):
    dataset = EvaluationDataset.model_validate(dataset_dict)
    config = ExperimentConfig(dataset_id=dataset.dataset_id, dataset_version=dataset.dataset_version)
    settings = Settings(_env_file=None)
    manifest = ExperimentRunner(ExperimentRunStore(ROOT / "data" / "unused-test-path")).make_manifest(
        profile="core", config=config, settings=settings,
    )
    assert manifest["dataset"] == {"id": "omnirag-phase8-eval", "version": "1.0.1"}
    assert manifest["experiment_config"]["dataset_id"] == dataset.dataset_id
    assert manifest["experiment_config"]["dataset_version"] == dataset.dataset_version
