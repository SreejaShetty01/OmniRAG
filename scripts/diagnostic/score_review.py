"""Score completed M2 human-review CSVs without inventing missing labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from scripts.diagnostic.prepare_m2_review import PINNED_DATASET_SHA256, write_csv


class IncompleteReview(ValueError):
    pass


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0,total]")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def cohens_kappa(first: Iterable[str], second: Iterable[str]) -> tuple[float | None, float | None, int]:
    a, b = list(first), list(second)
    if len(a) != len(b):
        raise ValueError("paired label lists must have equal length")
    n = len(a)
    if n == 0:
        return None, None, 0
    categories = sorted(set(a) | set(b))
    observed = sum(left == right for left, right in zip(a, b)) / n
    expected = sum((sum(x == category for x in a) / n) *
                   (sum(y == category for y in b) / n) for category in categories)
    kappa = None if math.isclose(1 - expected, 0.0) else (observed - expected) / (1 - expected)
    return kappa, observed, n


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_scoring_reports(review_dir: Path) -> dict:
    manifest_path = review_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stamp = manifest["diagnostic_stamp"]
    if manifest.get("dataset_sha256") != stamp.split("sha256 ", 1)[1].split(" !=", 1)[0]:
        raise ValueError("review manifest dataset hash does not match its diagnostic stamp")
    if manifest.get("dataset_sha256") == PINNED_DATASET_SHA256:
        raise ValueError("diagnostic dataset unexpectedly equals the pinned official dataset")

    answer_rows = read_csv(review_dir / "answers_to_label.csv")
    deterministic_rows = read_csv(review_dir / "deterministic_grounding_review.csv")
    evidence_rows = read_csv(review_dir / "case_evidence_sheet.csv")
    claim_rows = read_csv(review_dir / "claim_review.csv")
    second_rows = read_csv(review_dir / "second_reviewer.csv")
    key_rows = read_csv(review_dir / "key_for_scoring.csv")
    gold_rows = read_csv(review_dir / "gold_key_facts_for_review.csv")
    all_csv_rows = [*answer_rows, *deterministic_rows, *evidence_rows, *claim_rows,
                    *second_rows, *key_rows, *gold_rows]
    if any(row.get("diagnostic_stamp") != stamp for row in all_csv_rows):
        raise ValueError("CSV diagnostic stamp is missing or inconsistent")

    gaps: dict[str, list[str]] = {}
    completeness: dict[str, dict[str, int]] = {}

    def complete(name: str, rows: list[dict[str, str]], field: str,
                 allowed: set[str], *, context_required: bool = False) -> bool:
        missing = [row.get("row_id") or row.get("evidence_row_id") or row.get("claim_review_id") or "?"
                   for row in rows if row.get(field, "").strip() not in allowed]
        if context_required:
            missing.extend((row.get("row_id") or row.get("evidence_row_id") or row.get("claim_review_id") or "?")
                           for row in rows if row.get("context_status") != "AVAILABLE" or not row.get("context_text", "").strip())
        completeness.setdefault(name, {})[field] = len(rows) - len(set(missing))
        if missing:
            gaps[name] = sorted(set(missing))
            return False
        return True

    answer_correct_complete = complete("unscored_answer_correctness", answer_rows,
                                       "answer_correct", {"Y", "N", "Partial"})
    deterministic_correct_complete = len(deterministic_rows) == 19 and all(
        row.get("deterministic_answer_correct") in {"true", "false"} for row in deterministic_rows)
    human_correct_rows = [row for row in answer_rows if row.get("answer_status") == "FINAL_ANSWER_STORED"
                          and row.get("row_id") in {g.get("row_id") for g in gold_rows if g.get("expected_answerable") == "true"}]
    human_answerable_complete = len(human_correct_rows) == 44 and complete(
        "human_answerable_correctness", human_correct_rows, "answer_correct", {"Y", "N", "Partial"})
    end_to_end_available = deterministic_correct_complete and human_answerable_complete

    profile_key = {(row["row_id"]): row for row in key_rows}
    gold_key = {row["row_id"]: row for row in gold_rows}
    metrics: list[dict[str, object]] = []

    def add_proportion(name: str, profile: str, successes: int, total: int,
                       method: str, coverage: str) -> None:
        interval = wilson_interval(successes, total)
        metrics.append({"metric": name, "profile": profile, "numerator": successes,
                        "denominator": total, "value": successes / total if total else None,
                        "wilson_95": list(interval) if interval else None,
                        "method": method, "label_coverage": coverage,
                        "provenance": stamp})

    if deterministic_correct_complete:
        for profile in ("lite", "core", "research"):
            rows = [row for row in deterministic_rows if row["profile"] == profile]
            successes = sum(row["deterministic_answer_correct"] == "true" for row in rows)
            add_proportion("deterministic_correctness_on_scored_subset", profile, successes, len(rows),
                           "saved deterministic answer_match score; Wilson 95% CI",
                           f"complete for the pre-scored subset ({len(rows)} rows); not full answerable coverage")

    if end_to_end_available:
        for profile in ("lite", "core", "research"):
            det = [row for row in deterministic_rows if row["profile"] == profile]
            human = [row for row in human_correct_rows
                     if profile_key.get(row["row_id"], {}).get("profile") == profile]
            successes = sum(row["deterministic_answer_correct"] == "true" for row in det)
            successes += sum(row["answer_correct"] == "Y" for row in human)
            add_proportion("end_to_end_answerable_accuracy", profile, successes, 33,
                           "deterministic correct + human Y; abstentions and Partial are misses; Wilson 95% CI",
                           "complete: 33 answerable cases")
            answered_denominator = len(det) + len(human)
            add_proportion("accuracy_among_answered_answerable", profile, successes, answered_denominator,
                           "same numerator over shipped answerable answers; Wilson 95% CI",
                           f"complete: {answered_denominator} shipped answers")
            add_proportion("human_adjudicated_accuracy_unscored_answered", profile,
                           sum(row["answer_correct"] == "Y" for row in human), len(human),
                           "human Y / answered previously-unscored answerable rows; Wilson 95% CI",
                           f"complete: {len(human)} labeled answers")

    # No prompt context or abstained draft was saved. These are structural gaps,
    # so context/gate metrics cannot become available by filling arbitrary CSV cells.
    metric_status = {
        "answer_correctness_for_unscored_answers": "MEASURABLE_AFTER_HUMAN_LABELS",
        "end_to_end_answerable_accuracy": "MEASURABLE_AFTER_HUMAN_LABELS" if end_to_end_available else "NOT_MEASURABLE_LABELS_INCOMPLETE",
        "accuracy_among_answered": "MEASURABLE_AFTER_HUMAN_LABELS" if end_to_end_available else "NOT_MEASURABLE_LABELS_INCOMPLETE",
        "unsupported_answer_rate": "NOT_MEASURABLE_EXACT_FINAL_CONTEXT_NOT_STORED",
        "claim_level_supported_fraction": "NOT_MEASURABLE_EXACT_FINAL_CONTEXT_NOT_STORED",
        "citation_claim_support": "NOT_MEASURABLE_EXACT_FINAL_CONTEXT_NOT_STORED",
        "evidence_sufficiency": "NOT_MEASURABLE_EXACT_FINAL_CONTEXT_NOT_STORED",
        "abstention_gate_decomposition": "NOT_MEASURABLE_ABORTED_DRAFT_ANSWERS_NOT_STORED",
        "verification_gate_confusion_matrix": "NOT_MEASURABLE_ABORTED_DRAFT_ANSWERS_NOT_STORED",
        "counterfactual_gate_off": "NOT_MEASURABLE_ABORTED_DRAFT_ANSWERS_NOT_STORED",
        "unsupported_drafts_blocked": "NOT_MEASURABLE_ABORTED_DRAFT_ANSWERS_NOT_STORED",
        "verification_gate_false_positives": "NOT_MEASURABLE_ABORTED_DRAFT_ANSWERS_NOT_STORED",
        "second_reviewer_kappa": "NOT_MEASURABLE_SECOND_REVIEW_LABELS_INCOMPLETE",
        "unanswerable_answered_unsupported": "NOT_MEASURABLE_HUMAN_LABELS_INCOMPLETE",
    }
    kappa_results = {}
    second_by_id = {row.get("row_id"): row for row in second_rows}
    for label_field in ("answer_correct", "claims_supported_by_context", "citation_supports_claim"):
        paired_first, paired_second = [], []
        for row in answer_rows:
            other = second_by_id.get(row.get("row_id"))
            if other and row.get(label_field, "").strip() and other.get(label_field, "").strip():
                paired_first.append(row[label_field].strip())
                paired_second.append(other[label_field].strip())
        kappa, agreement, n = cohens_kappa(paired_first, paired_second)
        kappa_results[label_field] = {"kappa": kappa, "raw_agreement": agreement, "n": n}

    ledger_rows = []
    for metric in metrics:
        ci = metric["wilson_95"]
        ci_text = "NOT APPLICABLE" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        ledger_rows.append(f"| {metric['metric']} ({metric['profile']}) | {metric['numerator']}/{metric['denominator']} | {ci_text} | {metric['method']} | {metric['label_coverage']} | {stamp} |")
    not_permitted = [f"| {name} | {status} | No eligible complete labels and/or required context/draft artifact. |"
                     for name, status in metric_status.items() if status.startswith("NOT_MEASURABLE")]
    ledger = (f"> {stamp}\n# Claims ledger\n\n## PERMITTED claims\n\n| Metric | Numerator/denominator | Wilson 95% CI | Method | Label coverage | Provenance |\n|---|---:|---|---|---|---|\n"
              + ("\n".join(ledger_rows) if ledger_rows else "| None | — | — | — | No complete human-labeled metric. | " + stamp + " |")
              + "\n\n## NOT PERMITTED / NOT MEASURABLE\n\n| Metric | Status | Reason |\n|---|---|---|\n"
              + ("\n".join(not_permitted) if not_permitted else "| None | — | — |")
              + "\n\nDeterministic subset scores are not full answerable-case accuracy. Answer-level all/some/none labels are never used to calculate claim-level fractions.\n")
    (review_dir / "claims_ledger.md").write_text(ledger, encoding="utf-8")

    report = {"diagnostic_stamp": stamp, "status": "PARTIAL; human labels incomplete",
              "metric_status": metric_status, "metrics": metrics,
              "label_coverage": completeness, "missing_row_ids_by_metric": gaps,
              "second_reviewer_agreement": kappa_results,
              "draft_and_context_limitations": {
                  "abstained_drafts_stored": False, "final_context_stored": False,
                  "claim_verdicts_stored": True,
                  "verification_gate_false_positive_counts": "NOT MEASURABLE",
                  "claim_level_fractions_from_answer_level_labels": "PROHIBITED",
              }, "stage2_fp_cause": "NOT GENERATED: no stored blocked drafts can be adjudicated"}
    (review_dir / "score_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"> {stamp}", "# M2 review scoring report", "", "Human labels are currently blank. Only complete deterministic subset scores are permitted.", "",
             "## Permitted metrics", "", "| Metric | Profile | n correct / n | Estimate | Wilson 95% CI |", "|---|---|---:|---:|---|"]
    for item in metrics:
        ci = item["wilson_95"]
        lines.append(f"| {item['metric']} | {item['profile']} | {item['numerator']}/{item['denominator']} | {item['value']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    if not metrics:
        lines.append("| No complete metrics | — | — | — | — |")
    lines += ["", "## Not measurable", ""]
    lines += [f"- `{name}`: {status}" for name, status in metric_status.items() if status.startswith("NOT_MEASURABLE")]
    lines += ["", "`fp_cause_stage2.csv` was not generated because no abstained draft text is stored. Review rows contain no reconstructed draft/context text.", ""]
    (review_dir / "score_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_dir", type=Path)
    args = parser.parse_args()
    report = write_scoring_reports(args.review_dir)
    print(f"{report['diagnostic_stamp']}\nscore report written; incomplete labels are listed as NOT MEASURABLE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
