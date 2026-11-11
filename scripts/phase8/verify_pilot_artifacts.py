"""Read-only verification of Phase 8 pilot links and denied-evidence leakage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omnirag.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "docs/audit/phase8_eval_corpus_manifest.json")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "evaluation/datasets/omnirag-phase8-eval-1.0.1.json")
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--output", type=Path,
                        help="Optional safe JSON summary path; this does not alter Phase 7b artifacts")
    args = parser.parse_args()
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if len(progress.get("schedule", [])) != 9:
        raise SystemExit("Pilot schedule does not contain exactly nine units")
    cases = {item["case_id"]: item for item in dataset["cases"]}
    restricted = [item for item in snapshot["points"]
                  if item["source_basename"] == "12_restricted_team.txt"]
    restricted_ids = {str(item[key]) for item in restricted for key in ("chunk_id", "document_id")}
    marker = re.compile(r"cobalt\s+lantern\s+731", re.IGNORECASE)
    rows = []
    for scheduled in progress["schedule"]:
        run_id = scheduled.get("experiment_run_id")
        if not run_id:
            raise SystemExit(f"Missing experiment run ID for {scheduled['attempt_key']}")
        run_dir = args.data_dir / "experiments" / "runs" / run_id
        artifact_paths = [run_dir / name for name in ("manifest.json", "trace.json", "result.json")]
        if not all(path.is_file() for path in artifact_paths):
            raise SystemExit(f"Incomplete Phase 7b artifact set for {run_id}")
        manifest, trace, result = [json.loads(path.read_text(encoding="utf-8"))
                                   for path in artifact_paths]
        case = cases[scheduled["case_id"]]
        if (manifest.get("run_id") != run_id or trace.get("run_id") != run_id or
                manifest.get("parent_run_id") is not None or
                manifest.get("dataset") != {"id": dataset["dataset_id"],
                                             "version": dataset["dataset_version"]} or
                result.get("profile") != scheduled["profile"]):
            raise SystemExit(f"Phase 7b artifact linkage mismatch for {run_id}")
        leakage = False
        if (case.get("category") == "acl_tenant_isolation" and
                case.get("acl_context", {}).get("expected_access") == "deny"):
            for path in artifact_paths:
                raw = path.read_text(encoding="utf-8")
                if marker.search(raw) or any(point_id in raw for point_id in restricted_ids):
                    leakage = True
            if leakage:
                raise SystemExit(f"Denied evidence appears in Phase 7b artifacts for {run_id}")
        rows.append({"case_id": scheduled["case_id"], "profile": scheduled["profile"],
                     "run_id": run_id, "artifact_files": 3,
                     "artifact_bytes": sum(path.stat().st_size for path in artifact_paths),
                     "manifest_trace_linkage": True,
                     "dataset_linkage": True,
                     "parent_run_id": manifest.get("parent_run_id"),
                     "result_status": result.get("status"),
                     "denied_artifact_leak": leakage})
    if len({row["run_id"] for row in rows}) != 9:
        raise SystemExit("Pilot run IDs are not unique")
    report = {"status": "passed", "benchmark_id": progress.get("benchmark_id"),
              "unit_count": len(rows),
              "phase7b_artifact_bytes": sum(row["artifact_bytes"] for row in rows),
              "runs": rows}
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
