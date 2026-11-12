"""Score one saved Phase 8 run offline; performs no API or model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.phase8_scorer import score_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "eval/datasets/omnirag-phase8-eval-1.0.1.json")
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "docs/audit/phase8_eval_corpus_manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score_directory(args.dataset, args.snapshot, args.benchmark_dir)
    rendered = json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
