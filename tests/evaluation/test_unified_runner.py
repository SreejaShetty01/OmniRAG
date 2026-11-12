import json
import pytest

from eval.runner import (
    DEFAULT_BENCHMARK, DEFAULT_DATASET, DEFAULT_SNAPSHOT,
    evaluate_saved_run, write_outputs,
)


@pytest.mark.skipif(
    not DEFAULT_SNAPSHOT.is_file() or not (DEFAULT_BENCHMARK / "progress.json").is_file(),
    reason="Private Phase 8 snapshot and historical B5 artifacts are not published",
)
def test_historical_b5_replay_is_pinned_and_labeled_incomplete(tmp_path):
    report = evaluate_saved_run(DEFAULT_DATASET, DEFAULT_SNAPSHOT, DEFAULT_BENCHMARK)
    assert report["baseline_validity"] == "invalid_incomplete_historical_pilot"
    assert report["source_run"]["scheduled_units"] == 9
    assert report["source_run"]["seed"] == 20261104
    assert report["source_run"]["schedule_sha256"]
    assert report["dataset"]["sha256"] == "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1"
    assert report["snapshot"]["sha256"] == "2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847"
    assert all(report["profiles"][profile]["n_scheduled"] == 3
               for profile in ("lite", "core", "research"))
    assert report["profiles"]["lite"]["cost_usd"]["available_requests"] == 0
    assert report["source_run"]["budget_usage_final"]["total_tokens"] == 15398
    assert report["profiles"]["lite"]["budget_accounting_deltas"] == {
        "available_attempts": 3, "provider_calls": 5, "input_tokens": 2738,
        "output_tokens": 125, "total_tokens": 2863,
    }
    assert report["profiles"]["core"]["budget_accounting_deltas"]["total_tokens"] == 4417
    assert report["profiles"]["research"]["budget_accounting_deltas"]["total_tokens"] == 8118
    assert report["profiles"]["lite"]["provider_models"] == [
        {"provider": "openai_compatible", "model": "llama3.2:3b"}
    ]

    paths = write_outputs(report, tmp_path)
    assert set(paths) == {"benchmark_results", "performance_benchmark", "retrieval_ablation", "summary"}
    serialized = paths["benchmark_results"].read_text(encoding="utf-8")
    dataset = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    assert dataset["cases"][0]["query"] not in serialized
    ablation = json.loads(paths["retrieval_ablation"].read_text(encoding="utf-8"))
    assert next(v for v in ablation["variants"] if v["variant"] == "dense_only")["metrics"] is None
