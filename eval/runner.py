"""Offline eval/reporting for immutable Phase 8 benchmark artifacts.

This runner never contacts the API, Qdrant, or an inference provider. It validates
the pinned dataset/snapshot identities, scores a saved run, and emits sanitized
JSON plus a readable Markdown summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from eval.dataset import dataset_checksum
from eval.phase8_safety import verify_snapshot_manifest
from eval.phase8_scorer import score_directory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval/datasets/omnirag-phase8-eval-1.0.1.json"
DEFAULT_SNAPSHOT = ROOT / "docs/audit/phase8_eval_corpus_manifest.json"
DEFAULT_BENCHMARK = ROOT / "data/phase8/benchmarks/phase8-pilot-20261104-b5"
DEFAULT_OUTPUT = ROOT / "eval/results/phase8-b5-offline"
PROFILES = ("lite", "core", "research")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    sample = [float(value) for value in values]
    return {
        "n": len(sample),
        "mean": statistics.mean(sample) if sample else None,
        "p50": statistics.median(sample) if sample else None,
        "p95": _percentile(sample, 0.95),
        "max": max(sample) if sample else None,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(row[key]) for row in rows
              if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None}


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [row[key] for row in rows if isinstance(row.get(key), bool)]
    return {"numerator": sum(values), "denominator": len(values),
            "rate": sum(values) / len(values) if values else None}


def _safe_attempt_files(benchmark_dir: Path) -> dict[str, dict[str, Any]]:
    attempts_dir = benchmark_dir / "attempts"
    attempts: dict[str, dict[str, Any]] = {}
    for path in sorted(attempts_dir.glob("*.json")):
        attempts[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return attempts


def _profile_summary(profile: str, rows: list[dict[str, Any]],
                     attempts: dict[str, dict[str, Any]],
                     budget_by_attempt: dict[str, dict[str, int]]) -> dict[str, Any]:
    profile_rows = [row for row in rows if row.get("profile") == profile]
    profile_attempts = [attempts[row["attempt_key"]] for row in profile_rows
                        if row.get("attempt_key") in attempts]
    statuses = Counter(str(row.get("status", "missing")) for row in profile_rows)
    warnings = Counter(code for attempt in profile_attempts
                       for code in (attempt.get("warning_codes") or attempt.get("failure_codes") or []))
    timings = [attempt.get("timings_ms") or {} for attempt in profile_attempts]
    token_rows = [attempt.get("token_usage") or {} for attempt in profile_attempts]
    token_rows = [row for row in token_rows
                  if isinstance(row.get("input_tokens"), int) and
                  isinstance(row.get("output_tokens"), int)]
    costs = [attempt["cost_usd"] for attempt in profile_attempts
             if isinstance(attempt.get("cost_usd"), (int, float))]
    model_identities = sorted({
        (str(metadata["provider"]), str(metadata["model"]))
        for attempt in profile_attempts
        if (metadata := attempt.get("generation_metadata") or {}).get("provider")
        and metadata.get("model")
    })
    profile_budget = [budget_by_attempt[row["attempt_key"]]
                      for row in profile_rows if row.get("attempt_key") in budget_by_attempt]
    retrieval_metric_names = (
        "recall_at_5", "recall_at_10", "precision_at_5", "hit_rate_at_5",
        "hit_rate_at_10", "mrr_at_10", "ndcg_at_10", "evidence_precision",
        "evidence_coverage", "citation_precision", "citation_recall",
    )
    return {
        "n_scheduled": len(profile_rows),
        "status_counts": dict(sorted(statuses.items())),
        "latency_ms": _distribution([
            attempt["elapsed_ms"] for attempt in profile_attempts
            if isinstance(attempt.get("elapsed_ms"), (int, float))]),
        "stage_latency_ms": {
            stage: _distribution([timing[stage] for timing in timings
                                  if isinstance(timing.get(stage), (int, float))])
            for stage in ("generation", "verification")
        },
        "retrieval_and_answer_metrics": {
            metric: _mean(profile_rows, metric) for metric in retrieval_metric_names
        },
        "answer_exact_match": _rate(profile_rows, "exact_answer_match"),
        "abstention_correct": _rate(profile_rows, "abstention_correct"),
        "acl_audit_pass": _rate(profile_rows, "acl_audit_pass"),
        "provider_reported_usage": {
            "available_requests": len(token_rows),
            "input_tokens_total": sum(row["input_tokens"] for row in token_rows),
            "output_tokens_total": sum(row["output_tokens"] for row in token_rows),
            "input_tokens_mean": (statistics.mean(row["input_tokens"] for row in token_rows)
                                  if token_rows else None),
            "output_tokens_mean": (statistics.mean(row["output_tokens"] for row in token_rows)
                                   if token_rows else None),
        },
        "cost_usd": {"available_requests": len(costs),
                     "total": sum(costs) if costs else None},
        "provider_models": [{"provider": provider, "model": model}
                            for provider, model in model_identities],
        "budget_accounting_deltas": {
            "available_attempts": len(profile_budget),
            "provider_calls": sum(row["calls"] for row in profile_budget),
            "input_tokens": sum(row["input_tokens"] for row in profile_budget),
            "output_tokens": sum(row["output_tokens"] for row in profile_budget),
            "total_tokens": sum(row["total_tokens"] for row in profile_budget),
        },
        "warning_counts": dict(sorted(warnings.items())),
    }


def _budget_deltas(progress: dict[str, Any],
                   attempts: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Derive per-attempt deltas from cumulative ledger snapshots in schedule order."""
    previous = {key: 0 for key in ("calls", "input_tokens", "output_tokens", "total_tokens")}
    deltas: dict[str, dict[str, int]] = {}
    for scheduled in progress.get("schedule", []):
        key = scheduled.get("attempt_key")
        snapshot = (attempts.get(key, {}).get("budget_usage") or {}) if key else {}
        values = {name: snapshot.get(name) for name in previous}
        if not all(isinstance(value, int) and value >= 0 for value in values.values()):
            continue
        current = {name: int(value) for name, value in values.items()}
        if any(current[name] < previous[name] for name in previous):
            # A reset or malformed ledger makes subsequent per-attempt deltas ambiguous.
            continue
        deltas[key] = {name: current[name] - previous[name] for name in previous}
        previous = current
    return deltas


def evaluate_saved_run(dataset_path: str | Path, snapshot_path: str | Path,
                       benchmark_dir: str | Path) -> dict[str, Any]:
    """Validate and score a saved run without making network or model calls."""
    dataset_file = Path(dataset_path)
    snapshot_file = Path(snapshot_path)
    run_dir = Path(benchmark_dir)
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))

    actual_dataset_checksum = dataset_checksum(dataset)
    declared_dataset = progress.get("dataset") or {}
    if (declared_dataset.get("id") != dataset.get("dataset_id") or
            declared_dataset.get("version") != dataset.get("dataset_version") or
            declared_dataset.get("sha256") != actual_dataset_checksum):
        raise ValueError("Saved run dataset identity/checksum does not match the supplied dataset")
    verified_snapshot = verify_snapshot_manifest(snapshot)
    if progress.get("snapshot_sha256") != verified_snapshot["snapshot_sha256"]:
        raise ValueError("Saved run snapshot checksum does not match the supplied snapshot")

    score = score_directory(dataset_file, snapshot_file, run_dir)
    attempts = _safe_attempt_files(run_dir)
    budget_by_attempt = _budget_deltas(progress, attempts)
    units = score.get("units", [])
    keyed_units = []
    for scheduled, scored in zip(progress.get("schedule", []), units):
        keyed_units.append({**scored, "attempt_key": scheduled.get("attempt_key")})
    profiles = {profile: _profile_summary(profile, keyed_units, attempts, budget_by_attempt)
                for profile in PROFILES}

    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "offline_historical_artifact_replay",
        "baseline_validity": "invalid_incomplete_historical_pilot",
        "source_run": {
            "benchmark_id": progress.get("benchmark_id"),
            "status": progress.get("status"),
            "created_at_utc": progress.get("created_at"),
            "finished_at_utc": progress.get("finished_at"),
            "scheduled_units": len(progress.get("schedule", [])),
            "schedule_sha256": progress.get("schedule_sha256"),
            "seed": progress.get("seed"),
            "concurrency": progress.get("concurrency"),
            "warmup": progress.get("warmup"),
            "request_timeout_seconds": progress.get("request_timeout_seconds"),
            "run_timeout_seconds": progress.get("run_timeout_seconds"),
            "budget_caps": progress.get("budget_caps"),
            "budget_usage_final": progress.get("budget_usage"),
            "budget_delta_accounting": "derived from monotonically increasing per-attempt cumulative snapshots in schedule order",
            "collection": progress.get("collection"),
            "cache_policy": progress.get("application_retrieval_cache_policy"),
        },
        "dataset": {
            "id": dataset.get("dataset_id"), "version": dataset.get("dataset_version"),
            "sha256": actual_dataset_checksum, "case_count": len(dataset.get("cases", [])),
            "ground_truth_status_counts": dict(sorted(Counter(
                case.get("ground_truth_status", "unknown") for case in dataset.get("cases", [])
            ).items())),
        },
        "snapshot": {
            "sha256": verified_snapshot["snapshot_sha256"],
            "corpus_id": verified_snapshot.get("corpus_id"),
            "corpus_version": verified_snapshot.get("corpus_version"),
            "collection": verified_snapshot.get("collection"),
            "document_count": verified_snapshot.get("document_count"),
            "point_count": verified_snapshot.get("point_count"),
            "evidence_mapping_count": len(verified_snapshot.get("evidence_to_chunks", [])),
        },
        "scorer_version": score.get("scorer_version"),
        "summary": score.get("summary", {}),
        "profiles": profiles,
        "units": units,
        "unavailable_metrics": {
            "ragas_faithfulness": "not measured; ragas is not installed and no paid/local judge was invoked",
            "answer_relevancy": "not measured; no labeled semantic judge output is stored",
            "context_precision_recall": "only ID-based coverage/precision from stored evidence mappings is available",
            "time_to_first_token": "not recorded in saved attempts",
            "provider_call_counts": "not available per unit in these safe artifacts",
            "cost_per_request": "cost_usd is null in all B5 attempts",
            "agentic_tool_accuracy": "no human-labeled expected tool/step traces",
            "throughput_and_concurrency": "B5 is serial and too small; no controlled concurrency data",
            "hardware_and_resource_peaks": "not recorded per request in the source run artifacts",
        },
    }


def retrieval_ablation_record(report: dict[str, Any]) -> dict[str, Any]:
    """Describe actual measured variants and explicitly leave absent variants null."""
    return {
        "source_run_id": report["source_run"]["benchmark_id"],
        "dataset_sha256": report["dataset"]["sha256"],
        "snapshot_sha256": report["snapshot"]["sha256"],
        "variants": [
            {"variant": "dense_only", "status": "not_measured", "metrics": None,
             "reason": "No saved dense-only run; isolated retrieval service was inaccessible."},
            {"variant": "sparse_only", "status": "not_measured", "metrics": None,
             "reason": "No saved sparse-only run; isolated retrieval service was inaccessible."},
            {"variant": "hybrid_fusion_fallback", "status": "historical_pilot_only",
             "metrics": {profile: report["profiles"][profile]["retrieval_and_answer_metrics"]
                         for profile in PROFILES},
             "reason": "Nine B5 units; one repetition and a single positive retrieval case per profile."},
            {"variant": "hybrid_with_reranker", "status": "not_measured", "metrics": None,
             "reason": "The baseline reranker was unconfigured; B5 records fallback warnings."},
        ],
        "profile_comparison": {
            "status": "small_unpaired_inference_not_claimed",
            "profiles": {profile: report["profiles"][profile] for profile in PROFILES},
            "limitation": "Three cases and one repetition per profile are not a controlled quality comparison.",
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Offline OmniRAG Evaluation Summary", "",
        f"- Source run: `{report['source_run']['benchmark_id']}`",
        f"- Classification: **{report['baseline_validity']}**",
        f"- Dataset: `{report['dataset']['id']}@{report['dataset']['version']}` "
        f"(canonical SHA-256 `{report['dataset']['sha256']}`)",
        f"- Snapshot SHA-256: `{report['snapshot']['sha256']}`",
        f"- Units: {report['summary'].get('scheduled_units')} historical pilot requests",
        "", "## Profile results", "",
        "| Profile | n | E2E p50 ms | E2E p95 ms | E2E max ms | Exact answer | Correct abstention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        data = report["profiles"][profile]
        latency = data["latency_ms"]
        exact = data["answer_exact_match"]
        abstention = data["abstention_correct"]
        lines.append(
            f"| {profile} | {data['n_scheduled']} | {latency['p50']} | {latency['p95']} "
            f"| {latency['max']} | {exact['numerator']}/{exact['denominator']} "
            f"| {abstention['numerator']}/{abstention['denominator']} |"
        )
    lines.extend([
        "", "## Interpretation", "",
        "This is an offline replay of a nine-request historical pilot, not a complete or "
        "statistically valid baseline. Dense-only, sparse-only, and reranker variants were "
        "not measured. Null metrics are unavailable, not zero. No model or API calls were made.",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ablation = retrieval_ablation_record(report)
    performance = {
        "source_run_id": report["source_run"]["benchmark_id"],
        "dataset_sha256": report["dataset"]["sha256"],
        "snapshot_sha256": report["snapshot"]["sha256"],
        "profiles": {profile: {
            "latency_ms": report["profiles"][profile]["latency_ms"],
            "stage_latency_ms": report["profiles"][profile]["stage_latency_ms"],
            "provider_reported_usage": report["profiles"][profile]["provider_reported_usage"],
            "cost_usd": report["profiles"][profile]["cost_usd"],
        } for profile in PROFILES},
        "limitations": report["unavailable_metrics"],
    }
    files = {
        "benchmark_results": output / "benchmark_results.json",
        "performance_benchmark": output / "performance_benchmark.json",
        "retrieval_ablation": output / "retrieval_ablation.json",
        "summary": output / "evaluation_summary.md",
    }
    for key, value in (("benchmark_results", report), ("performance_benchmark", performance),
                       ("retrieval_ablation", ablation)):
        files[key].write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    files["summary"].write_text(render_markdown(report), encoding="utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_saved_run(args.dataset, args.snapshot, args.benchmark_dir)
    paths = write_outputs(report, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"classification: {report['baseline_validity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
