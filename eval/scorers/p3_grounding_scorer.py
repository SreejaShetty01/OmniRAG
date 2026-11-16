"""Offline-only P3 scoring and human-review CSV preparation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

AUDIT_FIELDS = ["record_type", "sample_slot", "run_id", "case_id", "item_id", "text",
                "evidence_references", "prediction", "score", "human_label", "reviewer", "notes"]


def audit_rows(attempts: Iterable[dict[str, Any]], claim_limit: int = 50,
               citation_limit: int = 30) -> list[dict[str, str]]:
    """Export saved claims/citations only; all human-label columns remain blank."""
    claims, citations = [], []
    for artifact in attempts:
        run_id = str(artifact.get("run_id", ""))
        case_id = str(artifact.get("case_id", ""))
        for claim in artifact.get("claims", []):
            claims.append({"record_type": "claim", "run_id": run_id, "case_id": case_id,
                           "item_id": str(claim.get("claim_id", "")),
                           "text": str(claim.get("claim_text", "")),
                           "evidence_references": json.dumps(claim.get("supporting_evidence_ids", [])),
                           "prediction": str(claim.get("status", "")),
                           "score": "" if claim.get("verification_score") is None else
                           str(claim["verification_score"]), "human_label": "", "reviewer": "", "notes": ""})
        artifact_claims = artifact.get("claims", [])
        for citation in artifact.get("citations", []):
            evidence_id = citation.get("evidence_id")
            related = [claim.get("status") for claim in artifact_claims
                       if evidence_id and evidence_id in claim.get("supporting_evidence_ids", [])]
            prediction = ("supported" if related and all(value == "supported" for value in related) else
                          "unsupported" if any(value in {"unsupported", "contradicted"} for value in related)
                          else "uncertain")
            citations.append({"record_type": "citation", "run_id": run_id, "case_id": case_id,
                              "item_id": str(citation.get("citation_id", citation.get("evidence_id", ""))),
                              "text": str(citation.get("quote") or citation.get("source") or ""),
                              "evidence_references": json.dumps([citation.get("chunk_id")]
                                                                if citation.get("chunk_id") else []),
                              "prediction": prediction,
                              "score": "", "human_label": "", "reviewer": "", "notes": ""})
    rows = []
    for kind, collected, limit in (("claim", claims, claim_limit), ("citation", citations, citation_limit)):
        for index, row in enumerate(sorted(collected, key=lambda item: (item["run_id"], item["case_id"], item["item_id"]))[:limit], 1):
            rows.append({**row, "sample_slot": f"{kind.upper()}-{index:03d}"})
    return rows


def write_audit_csv(path: str | Path, attempts: Iterable[dict[str, Any]], *,
                    fill_empty_slots: bool = False) -> int:
    rows = audit_rows(attempts)
    if fill_empty_slots:
        present = {(row["record_type"], row["sample_slot"]) for row in rows}
        for kind, count in (("claim", 50), ("citation", 30)):
            for index in range(1, count + 1):
                slot = f"{kind.upper()}-{index:03d}"
                if (kind, slot) not in present:
                    rows.append({field: "" for field in AUDIT_FIELDS} | {
                        "record_type": kind, "sample_slot": slot,
                        "evidence_references": "[]", "human_label": "", "reviewer": "", "notes": ""})
    rows.sort(key=lambda item: (item["record_type"], item["sample_slot"]))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def score_grounding(attempts: Iterable[dict[str, Any]], cases: dict[str, dict[str, Any]],
                    human_reviews: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Score offline artifacts; unknown/missing human labels are never imputed."""
    artifacts = list(attempts)
    checked_claims = [claim for artifact in artifacts for claim in artifact.get("claims", [])
                      if claim.get("status") in {"supported", "unsupported", "contradicted"}]
    unsupported = sum(claim.get("status") == "unsupported" for claim in checked_claims)
    unsupported_claim_rate = unsupported / len(checked_claims) if checked_claims else None
    answerable = [artifact for artifact in artifacts
                  if cases.get(str(artifact.get("case_id")), {}).get("expected_abstention") is False
                  and artifact.get("status", "completed") in {"completed", "completed_with_warnings",
                                                                "completed_with_errors"}
                  and isinstance(artifact.get("abstained"), bool)]
    false_abstentions = sum(bool(artifact["abstained"]) for artifact in answerable)
    all_reviews = list(human_reviews)
    reviewed = [row for row in all_reviews if row.get("record_type") == "citation" and
                row.get("human_label") in {"supported", "unsupported"}]
    reviewed_claims = [row for row in all_reviews if row.get("record_type") == "claim" and
                       row.get("human_label") in {"supported", "unsupported"}]
    predicted_supported = [row for row in reviewed if row.get("prediction") == "supported"]
    true_supported = sum(row.get("human_label") == "supported" for row in predicted_supported)
    submitted_citations = [row for row in all_reviews if row.get("record_type") == "citation"]
    reviewed_with_evidence = sum(bool(row.get("evidence_references")) for row in reviewed)
    return {
        "unsupported_claim_rate": {"numerator": unsupported, "denominator": len(checked_claims),
                                   "rate": unsupported_claim_rate},
        "citation_entailment_precision": {"numerator": true_supported,
                                           "denominator": len(predicted_supported),
                                           "rate": true_supported / len(predicted_supported)
                                           if predicted_supported else None},
        "citation_entailment_coverage": {"numerator": reviewed_with_evidence,
                                         "denominator": len(submitted_citations),
                                         "rate": reviewed_with_evidence / len(submitted_citations)
                                         if submitted_citations else None},
        "false_abstention_rate": {"numerator": false_abstentions, "denominator": len(answerable),
                                  "rate": false_abstentions / len(answerable) if answerable else None},
        "unreviewed_claim_count": max(0, sum(1 for artifact in artifacts for claim in artifact.get("claims", [])) -
                                       len(reviewed_claims)),
        "human_reviewed_records": len(reviewed) + len(reviewed_claims),
    }


def score_directory(attempts_dir: str | Path, dataset_path: str | Path,
                    human_review_csv: str | Path) -> dict[str, Any]:
    root = Path(attempts_dir)
    attempts = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in dataset["cases"]}
    with Path(human_review_csv).open(newline="", encoding="utf-8") as stream:
        reviews = list(csv.DictReader(stream))
    return score_grounding(attempts, cases, reviews)
