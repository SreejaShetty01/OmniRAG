"""Unified CLI for OmniRAG evaluation workflows."""
from __future__ import annotations

import argparse
import runpy
import sys

COMMANDS = {
    "benchmark": ("scripts.eval.run_benchmark", "Run the evaluation benchmark."),
    "bootstrap": ("scripts.eval.bootstrap_corpus", "Validate and bootstrap the evaluation corpus."),
    "score": ("scripts.eval.score_run", "Score a completed evaluation run."),
    "acl-audit": ("scripts.eval.validate_acl", "Validate evaluation ACL isolation."),
    "verify-artifacts": ("scripts.eval.verify_pilot_artifacts", "Verify evaluation artifacts."),
    "diagnostic": ("scripts.diagnostic.run_diagnostic", "Run the evaluation diagnostic workflow."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniRAG evaluation command runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        subparsers.add_parser(name, help=description)
    args, remainder = parser.parse_known_args()
    module = COMMANDS[args.command][0]
    sys.argv = [module, *remainder]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
