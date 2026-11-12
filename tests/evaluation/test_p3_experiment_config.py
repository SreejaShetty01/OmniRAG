import json
import hashlib
from pathlib import Path

from eval.dataset import EvaluationDataset, dataset_checksum


ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "experiments/p3_claim_entailment/v1.0.0"
P3_HTTP = ROOT / "experiments/p3_claim_entailment/v1.1.0"


def test_p3_config_pins_immutable_development_inputs_and_paired_conditions():
    config = json.loads((P3 / "config.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / config["dataset"]["path"]).read_text(encoding="utf-8"))

    assert config["status"] == "prepared_not_run"
    assert dataset_checksum(EvaluationDataset.model_validate(dataset)) == config["dataset"]["dataset_sha256"]
    assert len(dataset["cases"]) == config["dataset"]["case_count"] == 40
    assert config["corpus"]["collection"] == "omnirag_eval_phase8_20261104"
    assert config["corpus"]["snapshot_sha256"] == (
        "2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847")
    assert config["shared_conditions"]["budget"] == {
        "max_cost_usd": 0, "max_total_tokens": 4_200_000, "max_provider_calls": 3_000}
    assert config["design"]["paired"] is True
    assert config["design"]["same_case_profile_order_in_both_conditions"] is True


def test_p3_changes_only_verification_role_and_uses_offline_scorer():
    config = json.loads((P3 / "config.json").read_text(encoding="utf-8"))
    control, treatment = config["conditions"]["control"], config["conditions"]["treatment"]
    verification_keys = {"verification_provider", "verification_model_id", "verification_endpoint",
                         "effective_verifier"}
    assert {key for key in control if control[key] != treatment[key]} == verification_keys
    assert config["scoring"]["implementation"] == "eval/phase8_scorer.py"
    assert config["scoring"]["scorer_version"] == "phase8-offline-2"
    assert config["scoring"]["offline_only"] is True
    assert "UNKNOWN" in config["scoring"]["claim_level_false_acceptance_rejection"]
    assert config["reproducibility"]["execution_authorized"] is False


def test_p3_manifest_pins_config_dataset_snapshot_and_nonexecution_state():
    manifest = json.loads((P3 / "manifest.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256((P3 / "config.json").read_bytes()).hexdigest()
    assert manifest["status"] == "prepared_not_run"
    assert manifest["config_sha256"] == config_hash
    assert manifest["dataset_checksum"] == (
        "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1")
    assert manifest["snapshot_sha256"] == (
        "2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847")
    assert manifest["execution_authorized"] is False


def test_p3_http_minicheck_config_has_disjoint_partitions_and_only_switch_varies():
    config = json.loads((P3_HTTP / "config.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / config["dataset"]["path"]).read_text(encoding="utf-8"))
    assert config["status"] == "prepared_not_run"
    assert dataset_checksum(EvaluationDataset.model_validate(dataset)) == config["dataset"]["dataset_sha256"]
    calibration = set(config["partitions"]["calibration_case_ids"])
    evaluation = set(config["partitions"]["evaluation_case_ids"])
    assert len(calibration) == len(evaluation) == 20
    assert not calibration & evaluation
    assert calibration | evaluation == {case["case_id"] for case in dataset["cases"]}
    control = config["conditions"]["control"]["experiment"]
    treatment = config["conditions"]["treatment"]["experiment"]
    assert control["verification"] == treatment["verification"] is True
    assert control["semantic_verification"] is False
    assert treatment["semantic_verification"] is True
    assert {key for key in control if control[key] != treatment[key]} == {"semantic_verification"}
    assert config["partitions"]["threshold"] is None
    assert config["verifier"]["model_digest"].startswith("UNKNOWN")
    assert config["execution"]["authorized"] is False
    assert config["execution"]["sealed_set_authorized"] is False


def test_p3_http_manifest_pins_config_and_unexecuted_state():
    manifest = json.loads((P3_HTTP / "manifest.json").read_text(encoding="utf-8"))
    config_hash = hashlib.sha256((P3_HTTP / "config.json").read_bytes()).hexdigest()
    assert manifest["config_sha256"] == config_hash
    assert manifest["status"] == "prepared_not_run"
    assert manifest["dataset_checksum"] == (
        "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1")
    assert manifest["execution_authorized"] is False
