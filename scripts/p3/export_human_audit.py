"""Build a blank audit template or export saved P3 attempt artifacts for review."""

import argparse
import json
from pathlib import Path

from evaluation.p3_grounding_scorer import write_audit_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts-dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("evaluation/p3_human_audit_template.csv"))
    args = parser.parse_args()
    attempts = []
    if args.attempts_dir:
        for path in sorted(args.attempts_dir.glob("*.json")):
            attempts.append(json.loads(path.read_text(encoding="utf-8")))
    count = write_audit_csv(args.output, attempts, fill_empty_slots=True)
    print(f"wrote {count} audit slots to {args.output}")


if __name__ == "__main__":
    main()
