"""Validate a human annotation work queue; convert only verified approvals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.dataset import dataset_checksum
from evaluation.realworld_annotations import (HumanAnnotation, annotation_summary,
                                              convert_human_approved_annotations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("evaluation/realworld/sealed_dataset.json"))
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--versions", type=Path,
                        help="JSON mapping corpus-relative source path to exact version")
    parser.add_argument("--output", type=Path,
                        help="optional output path; only fully verified annotations convert")
    parser.add_argument("--dataset-id", default="omnirag-realworld-sealed")
    parser.add_argument("--dataset-version", default="0.1.0")
    parser.add_argument("--corpus-id", default="omnirag-realtech-m1p2")
    parser.add_argument("--corpus-version", default="v1")
    parser.add_argument("--corpus-manifest", default="evaluation/realworld/corpus_manifest.json")
    args = parser.parse_args()
    raw = json.loads(args.annotations.read_text(encoding="utf-8-sig"))
    records = [HumanAnnotation.model_validate(item) for item in raw["cases"]]
    print(json.dumps(annotation_summary(records), indent=2))
    if args.output:
        if args.corpus_root is None or args.versions is None:
            parser.error("--output requires both --corpus-root and --versions")
        versions = json.loads(args.versions.read_text(encoding="utf-8-sig"))
        dataset = convert_human_approved_annotations(
            records, dataset_id=args.dataset_id, dataset_version=args.dataset_version,
            corpus_id=args.corpus_id, corpus_version=args.corpus_version,
            corpus_manifest=args.corpus_manifest, corpus_root=args.corpus_root,
            document_versions=versions)
        payload = dataset.model_dump(mode="json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"converted": True, "case_count": len(dataset.cases),
                          "dataset_sha256": dataset_checksum(dataset),
                          "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
