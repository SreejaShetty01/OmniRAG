"""Run selected Phase 8 case/profile/repetition units through the evaluation API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omnirag.config.settings import Settings
from eval.dataset import (
    DATASET_ID, DATASET_VERSION, EvaluationDataset, dataset_checksum,
    validate_dataset, validate_manifest,
)
from eval.phase8_runner import (
    DEFAULT_RUN_TIMEOUT_SECONDS, EXPECTED_DATASET_SHA256, MAX_PROVIDER_CALLS,
    MAX_TOTAL_TOKENS, Phase8BenchmarkRunner, WARMUP_CASE_ID, schedule_units,
)
from eval.phase8_safety import verify_snapshot_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/datasets/omnirag-phase8-eval-1.0.1.json")
    parser.add_argument("--dataset-manifest", type=Path,
                        default=ROOT / "eval/datasets/omnirag-phase8-eval-1.0.1.manifest.json")
    parser.add_argument("--snapshot-manifest", type=Path,
                        default=ROOT / "docs/audit/phase8_eval_corpus_manifest.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--benchmark-id")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--profiles", default="lite,core,research")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--request-timeout", type=float, default=150.0)
    parser.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT_SECONDS)
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS)
    parser.add_argument("--max-provider-calls", type=int, default=MAX_PROVIDER_CALLS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    if args.warmup:
        if not args.benchmark_id:
            parser.error("--warmup requires --benchmark-id shared with the measured run")
        if args.case_ids and args.case_ids != [WARMUP_CASE_ID]:
            parser.error(f"--warmup only permits --case-id {WARMUP_CASE_ID}")
        if args.repetitions not in (None, 1):
            parser.error("--warmup requires exactly one repetition")
        args.case_ids = [WARMUP_CASE_ID]
        args.repetitions = 1
    else:
        args.repetitions = 5 if args.repetitions is None else args.repetitions
    args.benchmark_id = args.benchmark_id or f"phase8-{uuid4()}"
    if args.output_root is None:
        args.output_root = ROOT / ("data/phase8/warmups" if args.warmup else "data/phase8/benchmarks")

    dataset_dict = json.loads(args.dataset.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    corpus_root = ROOT / "tests/fixtures/phase7_smoke"
    dataset = validate_dataset(dataset_dict, corpus_root=corpus_root, repository_root=ROOT)
    validate_manifest(dataset, dataset_manifest, corpus_root=corpus_root, repository_root=ROOT)
    actual_checksum = dataset_checksum(dataset)
    if actual_checksum != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset checksum mismatch; refusing to schedule requests")
    if (dataset.dataset_id, dataset.dataset_version) != (DATASET_ID, DATASET_VERSION):
        raise ValueError("Dataset identity mismatch; refusing to schedule requests")

    snapshot = verify_snapshot_manifest(json.loads(args.snapshot_manifest.read_text(encoding="utf-8")))
    settings = Settings(_env_file=(ROOT / ".env", ROOT / ".env.phase8"))
    if settings.auth_shared_secret is None:
        raise ValueError("Trusted proxy shared secret is unavailable to the benchmark process")
    secret = settings.auth_shared_secret.get_secret_value()

    cases = dataset.cases
    if args.case_ids:
        selected = set(args.case_ids)
        unknown = selected - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"Unknown dataset case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case.case_id in selected]
    profiles = tuple(item.strip().casefold() for item in args.profiles.split(",") if item.strip())
    if not profiles or not set(profiles).issubset({"lite", "core", "research"}):
        raise ValueError("Profiles must be a non-empty subset of lite,core,research")
    if args.warmup and set(profiles) != {"lite", "core", "research"}:
        parser.error("--warmup requires profiles lite,core,research")
    units = schedule_units(cases, profiles=profiles, repetitions=args.repetitions, seed=args.seed)
    if not units:
        raise ValueError("No case/profile units match the requested schedule")
    runner = Phase8BenchmarkRunner(
        base_url=args.base_url, secret=secret, snapshot_manifest=snapshot,
        output_root=args.output_root, request_timeout=args.request_timeout,
        run_timeout=args.run_timeout, max_cost_usd=args.max_cost_usd,
        max_total_tokens=args.max_total_tokens, max_provider_calls=args.max_provider_calls,
    )
    preflight = runner.preflight(args.benchmark_id)
    result = runner.run(units, benchmark_id=args.benchmark_id, seed=args.seed,
                        resume=args.resume, warmup=args.warmup)
    counts = {}
    for row in result["schedule"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"benchmark_id": args.benchmark_id, "status": result["status"],
                      "scheduled": len(units), "states": counts,
                      "preflight": preflight,
                      "budget_usage": result.get("budget_usage"),
                      "progress_path": str(args.output_root / args.benchmark_id / "progress.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
