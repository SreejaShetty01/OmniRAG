"""Prepare human-review exports from saved OmniRAG diagnostic artifacts only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
PINNED_DATASET_SHA256 = "7b44b8f616b2df74768ff962c30d3a311000ef0bffd41015f7bb49123ca7af1"
DEFAULT_DATASET = Path("evaluation/datasets/omnirag-phase8-eval-1.0.1.json")
DEFAULT_SNAPSHOT = Path("docs/audit/phase8_eval_corpus_manifest.json")
DEFAULT_ATTEMPTS = Path("data/diagnostic/posthoc-remediation-20261107T072431Z-663cec3c/attempts_scored.jsonl")
PROFILE_ORDER = {"lite": 0, "core": 1, "research": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path, *, exclude_review_id: str | None = None) -> dict[str, str]:
    """Hash the existing diagnostic artifacts and pinned review inputs."""
    files: set[Path] = set()
    diagnostic_root = root / "data" / "diagnostic"
    if diagnostic_root.exists():
        files.update(p for p in diagnostic_root.rglob("*") if p.is_file()
                     and not (exclude_review_id and exclude_review_id in p.relative_to(diagnostic_root).parts))
    for relative in (DEFAULT_DATASET, DEFAULT_DATASET.with_suffix(".manifest.json"), DEFAULT_SNAPSHOT):
        path = root / relative
        if path.is_file():
            files.add(path)
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(files, key=lambda value: value.as_posix())}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _evidence_text(case: dict) -> str:
    items = []
    for item in case.get("expected_evidence", []):
        items.append(canonical_json({
            "source_document": item.get("source_document"),
            "locator": item.get("locator"),
            "quote": item.get("quote"),
        }))
    return "\n".join(items)


def _claim_records(attempt: dict) -> list[dict]:
    response = attempt.get("response") or {}
    claims = response.get("claims")
    return claims if isinstance(claims, list) else []


def _selected_evidence_hash(attempt: dict) -> str:
    response = attempt.get("response") or {}
    references = response.get("evidence_references")
    if not isinstance(references, list):
        return ""
    safe = [{key: item.get(key) for key in ("evidence_id", "chunk_id", "excerpt", "document_id")}
            for item in references if isinstance(item, dict)]
    return hashlib.sha256(canonical_json(safe).encode("utf-8")).hexdigest() if safe else ""


def inventory_row(attempt: dict, case: dict, profile_row_id: str, stamp: str,
                  has_gold_mapping: bool) -> dict:
    response = attempt.get("response") or {}
    return {
        "inventory_row_id": profile_row_id,
        "case_id": attempt.get("case_id", ""),
        "profile": attempt.get("profile", ""),
        "answerable": str(not bool(case.get("expected_abstention"))).lower(),
        "case_id_present": str(bool(attempt.get("case_id"))).lower(),
        "question_present": str(bool(attempt.get("query"))).lower(),
        "gold_answer_present": str(bool(case.get("expected_answer"))).lower(),
        "gold_evidence_text_present": str(bool(case.get("expected_evidence"))).lower(),
        "gold_evidence_chunk_mapping_present": str(has_gold_mapping).lower(),
        "retrieved_candidate_ids_present": str(bool(attempt.get("candidate_ids_by_stage"))).lower(),
        "final_context_text_present": "false",
        "draft_answer_present": "false",
        "corrected_answer_separately_present": "false",
        "final_answer_present": str(isinstance(response.get("answer"), str)).lower(),
        "claim_verdicts_present": str(bool(_claim_records(attempt))).lower(),
        "abstention_reason_present": str(bool(response.get("abstention_reason"))).lower(),
        "citations_present": str(isinstance(response.get("citations"), list)).lower(),
        "deterministic_correctness_field_present": str("answer_correct_raw" in attempt).lower(),
        "deterministic_correctness_populated": str(isinstance(attempt.get("answer_correct_raw"), bool)).lower(),
        "selected_evidence_reference_sha256_not_context_hash": _selected_evidence_hash(attempt),
        "diagnostic_stamp": stamp,
    }


def prepare_exports(*, root: Path = ROOT, run_id: str | None = None,
                    dataset_path: Path = DEFAULT_DATASET, snapshot_path: Path = DEFAULT_SNAPSHOT,
                    attempts_path: Path = DEFAULT_ATTEMPTS) -> Path:
    dataset_path = dataset_path if dataset_path.is_absolute() else root / dataset_path
    snapshot_path = snapshot_path if snapshot_path.is_absolute() else root / snapshot_path
    attempts_path = attempts_path if attempts_path.is_absolute() else root / attempts_path
    actual_dataset_hash = sha256_file(dataset_path)
    stamp = (f"DIAGNOSTIC — dataset sha256 {actual_dataset_hash} != pinned "
             f"{PINNED_DATASET_SHA256}")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    attempts = load_jsonl(attempts_path)
    if len(attempts) != 120:
        raise ValueError(f"Expected 120 saved attempts, found {len(attempts)}")
    case_order = {case["case_id"]: i for i, case in enumerate(dataset["cases"])}
    cases = {case["case_id"]: case for case in dataset["cases"]}
    if set(row["case_id"] for row in attempts) - set(cases):
        raise ValueError("Attempt data contains case IDs absent from the selected dataset")
    unique = {(row["case_id"], row["profile"]) for row in attempts}
    if len(unique) != len(attempts):
        raise ValueError("Attempt data contains duplicate case/profile rows")
    if len(unique) != 120:
        raise ValueError("Expected one attempt per 40 cases and three profiles")

    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("review-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not run_id.startswith("review-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in run_id):
        raise ValueError("run_id must be a safe review- prefixed identifier")
    out_dir = root / "data" / "diagnostic" / run_id
    before = source_hashes(root, exclude_review_id=run_id)
    out_dir.mkdir(parents=True, exist_ok=False)

    attempt_by_key = {(row["case_id"], row["profile"]): row for row in attempts}
    ordered = sorted(attempts, key=lambda row: (case_order[row["case_id"]], PROFILE_ORDER[row["profile"]]))
    mapping_by_case: dict[str, list] = {}
    for row in snapshot.get("evidence_to_chunks", []):
        mapping_by_case.setdefault(row.get("case_id", ""), []).append(row)
    inventory = [inventory_row(row, cases[row["case_id"]], f"I{i:03d}", stamp,
                               bool(mapping_by_case.get(row["case_id"])))
                 for i, row in enumerate(ordered, 1)]
    inventory_fields = list(inventory[0])
    write_csv(out_dir / "field_inventory.csv", inventory_fields, inventory)

    evidence_sheet = []
    evidence_key_by_case_profile = {}
    for case_id in sorted((c for c in cases if not cases[c].get("expected_abstention")), key=case_order.get):
        case = cases[case_id]
        for profile in ("lite", "core", "research"):
            attempt = attempt_by_key[(case_id, profile)]
            refs_hash = _selected_evidence_hash(attempt)
            row_id = stable_id("E", f"{case_id}|{profile}|{actual_dataset_hash}")
            evidence_key_by_case_profile[(case_id, profile)] = row_id
            mapped = []
            for mapping in mapping_by_case.get(case_id, []):
                if mapping.get("source_document") in {item.get("source_document") for item in case.get("expected_evidence", [])}:
                    mapped.extend(item.get("chunk_id") for item in mapping.get("indexed_chunks", [])
                                  if isinstance(item, dict) and item.get("chunk_id"))
            inspection = (attempt.get("response") or {}).get("retrieval_inspection") or []
            selected_ids = sorted({item.get("chunk_id") for item in inspection
                                   if isinstance(item, dict) and item.get("selected") and item.get("chunk_id")})
            evidence_sheet.append({
                "evidence_row_id": row_id,
                "case_id": case_id,
                "profile": profile,
                "question": case.get("query", ""),
                "gold_key_facts": canonical_json({"expected_answer": case.get("expected_answer"),
                                                    "acceptable_answer_variants": case.get("acceptable_answer_variants", [])}),
                "gold_evidence_chunk_ids": canonical_json(sorted(set(str(v) for v in mapped))),
                "gold_evidence_text": _evidence_text(case),
                "selected_evidence_chunk_ids": canonical_json(selected_ids),
                "retrieval_candidate_ids_by_stage": canonical_json(attempt.get("candidate_ids_by_stage", {})),
                "final_context_text": "",
                "final_context_status": "NOT_STORED; profile-specific row used conservatively",
                "selected_evidence_reference_sha256_not_context_hash": refs_hash,
                "evidence_sufficient": "",
                "reviewer_id": "",
                "notes": "Exact packed prompt context was not persisted; evidence-reference hash is not a context hash.",
                "diagnostic_stamp": stamp,
            })
    evidence_fields = list(evidence_sheet[0])
    write_csv(out_dir / "case_evidence_sheet.csv", evidence_fields, evidence_sheet)

    context_comparisons = []
    evidence_hash_equal_count = 0
    for case_id in sorted(cases, key=case_order.get):
        hashes = {profile: _selected_evidence_hash(attempt_by_key[(case_id, profile)])
                  for profile in ("lite", "core", "research")}
        comparable = [value for value in hashes.values() if value]
        evidence_hash_equal = bool(comparable) and len(comparable) == 3 and len(set(comparable)) == 1
        evidence_hash_equal_count += int(evidence_hash_equal)
        context_comparisons.append({
            "case_id": case_id,
            "lite_final_context_sha256": "",
            "core_final_context_sha256": "",
            "research_final_context_sha256": "",
            "final_context_hash_comparison": "NOT_MEASURABLE_CONTEXT_NOT_STORED",
            "lite_selected_evidence_reference_sha256_not_context_hash": hashes["lite"],
            "core_selected_evidence_reference_sha256_not_context_hash": hashes["core"],
            "research_selected_evidence_reference_sha256_not_context_hash": hashes["research"],
            "evidence_reference_hashes_equal_not_context_hashes": str(evidence_hash_equal).lower(),
            "diagnostic_stamp": stamp,
        })
    write_csv(out_dir / "context_hash_comparison.csv", list(context_comparisons[0]), context_comparisons)

    blind_candidates = []
    deterministic = []
    key_rows = []
    attempt_review_id = {}
    for attempt in ordered:
        case_id, profile = attempt["case_id"], attempt["profile"]
        case = cases[case_id]
        answerable = not bool(case.get("expected_abstention"))
        deterministic_score = attempt.get("answer_correct_raw")
        is_deterministic = answerable and not attempt.get("abstained") and isinstance(deterministic_score, bool)
        if is_deterministic:
            row_id = stable_id("G", f"{case_id}|{profile}|{actual_dataset_hash}")
            deterministic.append({
                "row_id": row_id, "case_id": case_id, "profile": profile,
                "question": attempt.get("query", ""), "answer_text": attempt.get("answer", ""),
                "context_text": "", "context_status": "NOT_STORED",
                "citations": canonical_json((attempt.get("response") or {}).get("citations", [])),
                "deterministic_answer_correct": str(deterministic_score).lower(),
                "deterministic_answer_match_type": attempt.get("answer_match_type", ""),
                "claims_supported_by_context": "", "unsupported_claim_text": "",
                "citation_supports_claim": "", "reviewer_id": "", "notes": "",
                "diagnostic_stamp": stamp,
            })
            attempt_review_id[(case_id, profile)] = row_id
        else:
            row_id = stable_id("R", f"{case_id}|{profile}|{actual_dataset_hash}")
            if answerable and attempt.get("abstained"):
                answer_status = "DRAFT_NOT_STORED"
                answer_text = ""
            elif not attempt.get("abstained"):
                answer_status = "FINAL_ANSWER_STORED"
                answer_text = attempt.get("answer", "")
            else:
                # Unanswerable cases correctly abstained, so they are not answer rows,
                # but preserve the attempt key for any stored claim record.
                row_id = stable_id("U", f"{case_id}|{profile}|{actual_dataset_hash}")
                attempt_review_id[(case_id, profile)] = row_id
                key_rows.append({
                    "row_id": row_id, "case_id": case_id, "profile": profile,
                    "shipped_vs_abstained": "abstained",
                    "expected_answerable": "false", "deterministic_answer_correct": "",
                    "verifier_claim_verdicts_and_reasons": canonical_json([
                        {key: claim.get(key) for key in ("claim_id", "claim_text", "status", "explanation",
                                                          "supporting_evidence_ids", "methods", "verification_score")}
                        for claim in _claim_records(attempt)
                    ]),
                    "abstention_reason": (attempt.get("response") or {}).get("abstention_reason") or "",
                    "diagnostic_stamp": stamp,
                })
                continue
            citations = canonical_json((attempt.get("response") or {}).get("citations", []))
            blind_candidates.append({
                "row_id": row_id, "question": attempt.get("query", ""),
                "context_text": "", "context_status": "NOT_STORED",
                "answer_text": answer_text, "answer_status": answer_status,
                "citations": citations, "answer_correct": "",
                "claims_supported_by_context": "", "unsupported_claim_text": "",
                "citation_supports_claim": "", "reviewer_id": "", "notes": "",
                "diagnostic_stamp": stamp,
            })
            attempt_review_id[(case_id, profile)] = row_id

        key_rows.append({
            "row_id": row_id, "case_id": case_id, "profile": profile,
            "shipped_vs_abstained": "abstained" if attempt.get("abstained") else "shipped",
            "expected_answerable": str(answerable).lower(),
            "deterministic_answer_correct": "" if not isinstance(deterministic_score, bool) else str(deterministic_score).lower(),
            "verifier_claim_verdicts_and_reasons": canonical_json([
                {key: claim.get(key) for key in ("claim_id", "claim_text", "status", "explanation",
                                                  "supporting_evidence_ids", "methods", "verification_score")}
                for claim in _claim_records(attempt)
            ]),
            "abstention_reason": (attempt.get("response") or {}).get("abstention_reason") or "",
            "diagnostic_stamp": stamp,
        })
    deterministic.sort(key=lambda row: (case_order[row["case_id"]], PROFILE_ORDER[row["profile"]]))
    blind_candidates.sort(key=lambda row: row["row_id"])
    blind_fields = list(blind_candidates[0]) if blind_candidates else []
    deterministic_fields = list(deterministic[0]) if deterministic else []
    key_rows.sort(key=lambda row: row["row_id"])
    write_csv(out_dir / "answers_to_label.csv", blind_fields, blind_candidates)
    write_csv(out_dir / "deterministic_grounding_review.csv", deterministic_fields, deterministic)
    write_csv(out_dir / "key_for_scoring.csv", list(key_rows[0]), key_rows)

    gold_rows = []
    for attempt in ordered:
        case_id, profile = attempt["case_id"], attempt["profile"]
        row_id = attempt_review_id.get((case_id, profile))
        if not row_id:
            continue
        case = cases[case_id]
        gold_rows.append({
            "row_id": row_id,
            "expected_answerable": str(not bool(case.get("expected_abstention"))).lower(),
            "gold_key_facts": canonical_json({"expected_answer": case.get("expected_answer"),
                                                "acceptable_answer_variants": case.get("acceptable_answer_variants", [])}),
            "diagnostic_stamp": stamp,
        })
    gold_rows.sort(key=lambda row: row["row_id"])
    write_csv(out_dir / "gold_key_facts_for_review.csv", list(gold_rows[0]), gold_rows)

    second_ids = deterministic_second_review_subset(blind_candidates, min_count=16, fraction=0.20)
    second_rows = [dict(row) for row in blind_candidates if row["row_id"] in second_ids]
    write_csv(out_dir / "second_reviewer.csv", blind_fields, second_rows)

    claim_rows = []
    for attempt in ordered:
        row_id = attempt_review_id.get((attempt["case_id"], attempt["profile"]))
        for index, claim in enumerate(_claim_records(attempt), 1):
            claim_rows.append({
                "claim_review_id": stable_id("C", f"{attempt['case_id']}|{attempt['profile']}|{index}|{actual_dataset_hash}"),
                "row_id": row_id or "",
                "claim_text": claim.get("claim_text", ""),
                "context_text": "",
                "context_status": "NOT_STORED",
                "supported_by_context": "",
                "citation_supports_claim": "",
                "reviewer_id": "",
                "notes": "One record per stored claim. Claim-verifier outputs are sealed in key_for_scoring.csv.",
                "diagnostic_stamp": stamp,
            })
    claim_rows.sort(key=lambda row: row["claim_review_id"])
    claim_fields = ["claim_review_id", "row_id", "claim_text", "context_text", "context_status",
                    "supported_by_context", "citation_supports_claim", "reviewer_id", "notes", "diagnostic_stamp"]
    write_csv(out_dir / "claim_review.csv", claim_fields, claim_rows)

    stamp_line = f"> {stamp}"
    guide = f"""{stamp_line}
# M2 labeling guide

## Review order and blinding

1. Review `case_evidence_sheet.csv`, `answers_to_label.csv`, `deterministic_grounding_review.csv`, `gold_key_facts_for_review.csv`, `claim_review.csv`, and `second_reviewer.csv`.
2. Do not open `key_for_scoring.csv` until stage 1 labels are complete and the review files are returned. It contains profile, outcome, verifier statuses/reasons, and abstention reason.
3. `second_reviewer.csv` repeats 17 answer rows using the same row IDs. The second reviewer must label them independently before comparing with first-reviewer values.

## Important artifact gap

The saved attempts do not contain the exact packed final context or the pre-verification draft answer for abstained responses. Context cells are intentionally blank and marked `NOT_STORED`; evidence-reference hashes are not context hashes. Do not infer or reconstruct the prompt context. All 36 abstained drafts are unavailable, so those answer rows cannot be adjudicated from these artifacts. Claim texts and verifier verdicts may exist, but the verdicts remain sealed in `key_for_scoring.csv`.

## Label definitions

- `evidence_sufficient`: use `Y` only if the supplied exact final context contains enough evidence for the expected answer; `N` if it does not; `Partial` if it supports only part of the key facts. Since exact contexts are absent, leave these blank unless an authorized source of the exact context is separately supplied.
- `answer_correct`: judge the answer against the aligned `gold_key_facts_for_review.csv`, not world knowledge. `Y` means all material key facts are correct, `N` means a material error or unsupported specific answer, and `Partial` means only some key facts are correct. For an expected-unanswerable case, any specific factual answer is `N`; a refusal/abstention is correct under the dataset policy.
- `claims_supported_by_context`: answer-level label only. `all` means every material claim is supported by the exact context, `some` means a mixture, and `none` means no material claims are supported. Never use this field to calculate a claim-level fraction.
- `claim_review.csv`: one row per stored claim. Mark `supported_by_context` as `Y`, `N`, or `Partial` using only exact context; mark whether an actual citation supports that claim as `Y`, `N`, or `NA`. Do not label from verifier status.
- `unsupported_claim_text`: quote only claims not supported by the supplied context. Do not create claims absent from the answer.
- `citation_supports_claim`: `Y` only when the cited evidence supports the claim, `N` when it does not, and `NA` when no citation/claim mapping exists.
- `reviewer_id` and `notes`: identify the reviewer and record concise reasons or ambiguity.

Because exact contexts are not present, context sufficiency, context grounding, citation support, verifier false positives, and claim-supported fractions are currently NOT MEASURABLE from this release. Do not fill them using memory or external sources.
"""
    (out_dir / "LABELING_GUIDE.md").write_text(guide, encoding="utf-8")

    profile_counts = {}
    for profile in PROFILE_ORDER:
        profile_attempts = [row for row in ordered if row["profile"] == profile]
        answerable = [row for row in profile_attempts if not cases[row["case_id"]].get("expected_abstention")]
        profile_counts[profile] = {
            "answerable": len(answerable),
            "deterministically_scored_answered": sum(not r.get("abstained") and isinstance(r.get("answer_correct_raw"), bool) for r in answerable),
            "answerable_abstentions": sum(bool(r.get("abstained")) for r in answerable),
            "answered_unscored": sum(not r.get("abstained") and r.get("answer_correct_raw") is None for r in answerable),
            "unanswerable_answered": sum(cases[r["case_id"]].get("expected_abstention") and not r.get("abstained") for r in profile_attempts),
        }
    counts_total = {key: sum(value[key] for value in profile_counts.values())
                    for key in next(iter(profile_counts.values()))}
    expected_partition = {"answerable": 99, "deterministically_scored_answered": 19,
                          "answerable_abstentions": 36, "answered_unscored": 44,
                          "unanswerable_answered": 5}
    if counts_total != expected_partition:
        raise ValueError(f"Partition mismatch: expected {expected_partition}, found {counts_total}")

    after = source_hashes(root, exclude_review_id=run_id)
    mismatches = [name for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)]
    hash_record = {"diagnostic_stamp": stamp, "source_hashes_before": before,
                   "source_hashes_after": after, "identical": not mismatches,
                   "changed_or_missing_sources": mismatches}
    (out_dir / "source_hashes_before_after.json").write_text(json.dumps(hash_record, indent=2) + "\n", encoding="utf-8")

    report = f"""{stamp_line}
# M2 review export inventory

- Run ID: `{run_id}`
- Dataset file SHA-256: `{actual_dataset_hash}`
- Snapshot manifest SHA-256: `{sha256_file(snapshot_path)}`
- Snapshot: `{snapshot.get('collection')}`, {snapshot.get('point_count')} points; this is not the separate M1 corpus.
- Attempts loaded: {len(ordered)} (40 each for Lite, Core, Research).
- Existing-source hash comparison: **{'PASS' if not mismatches else 'FAIL'}**, {len(before)} files hashed before and after.
- Final packed context text: **NOT STORED** in saved attempts. Final-context hashes therefore cannot be compared. Profile-specific evidence rows were emitted for all 33 answerable cases (99 rows) to avoid treating one profile as shared evidence.
- Evidence-reference hashes were equal across all three profiles for {evidence_hash_equal_count}/40 case records. These are hashes of saved response evidence-reference records, explicitly not final-context hashes; see `context_hash_comparison.csv`.
- Abstention drafts: **NOT STORED**; 36 blind rows contain a blank answer marked `DRAFT_NOT_STORED`.
- Distinct corrected answers: **NOT STORED**; only final answers and a correction-attempt count are present.
- Per-claim status and explanation: available for {sum(bool(_claim_records(row)) for row in ordered)}/120 attempts; {len(claim_rows)} stored claim rows were exported for independent review.
- Measurement categories remain separate: deterministic answer correctness is preserved only for the 19 scored shipped answers; human-adjudicated answer correctness is pending labels; unsupported-by-context rate requires exact packed context; verification-gate false positives and unsupported drafts blocked require the missing abstained drafts.
- Canonical dataset answerability partition: **19 deterministic scored + 36 answerable abstentions + 44 answered-unscored = 99 answerable attempts**; additional unanswerable answered rows: {counts_total['unanswerable_answered']}.
- Per-profile counts: `{json.dumps(profile_counts, sort_keys=True)}`.
- Export rows: case evidence {len(evidence_sheet)}, blind answer labels {len(blind_candidates)}, deterministic grounding/citation review {len(deterministic)}, second reviewer {len(second_rows)}, individual claim review {len(claim_rows)}.
- Metrics are not scored because human labels are blank. See `LABELING_GUIDE.md`.

## Proposed minimal future logging change (not implemented)

At the existing answer-generation/verification boundary, optionally persist a restricted evaluation artifact containing the exact packed context (ordered chunk IDs plus text or an integrity-linked protected reference), the pre-verification draft, any corrected answer, and per-claim verdict/reason. Keep it outside general logs, apply evaluation-only access and retention controls, and link it to the request/attempt ID. This is necessary to distinguish unsupported-by-context answers, verifier false positives, and unsupported drafts blocked without reconstructing evidence.
"""
    (out_dir / "inventory_report.md").write_text(report, encoding="utf-8")
    manifest_out = {
        "diagnostic_stamp": stamp, "run_id": run_id,
        "dataset_sha256": actual_dataset_hash,
        "snapshot_sha256": sha256_file(snapshot_path),
        "attempt_source": attempts_path.relative_to(root).as_posix(),
        "attempt_count": len(ordered), "profile_counts": profile_counts,
        "answer_partition_verified": counts_total == expected_partition,
        "source_hashes_identical_before_after": not mismatches,
        "source_hashes_before_after": "source_hashes_before_after.json",
        "context_text_persisted": False, "abstention_drafts_persisted": False,
        "corrected_answer_separately_persisted": False,
        "exports": ["field_inventory.csv", "case_evidence_sheet.csv", "context_hash_comparison.csv", "answers_to_label.csv",
                    "deterministic_grounding_review.csv", "gold_key_facts_for_review.csv",
                    "key_for_scoring.csv", "second_reviewer.csv", "claim_review.csv",
                    "LABELING_GUIDE.md", "inventory_report.md", "source_hashes_before_after.json"],
    }
    (out_dir / "review_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8")
    return out_dir


def deterministic_second_review_subset(rows: list[dict], *, min_count: int, fraction: float) -> set[str]:
    eligible = [row for row in rows if row.get("answer_text") and row.get("answer_status") == "FINAL_ANSWER_STORED"]
    count = max(min_count, round(len(rows) * fraction))
    count = min(count, len(eligible))
    if count == 0:
        return set()
    # Evenly spaced deterministic positions over the sorted eligible rows.
    eligible.sort(key=lambda row: str(row["row_id"]))
    indexes = sorted({min(len(eligible) - 1, int((i + 0.5) * len(eligible) / count)) for i in range(count)})
    # Stable fill for any index collisions from tiny sample sizes.
    for index in range(len(eligible)):
        if len(indexes) >= count:
            break
        if index not in indexes:
            indexes.append(index)
    return {str(eligible[index]["row_id"]) for index in sorted(indexes[:count])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--dataset", default=DEFAULT_DATASET.as_posix())
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT.as_posix())
    parser.add_argument("--attempts", default=DEFAULT_ATTEMPTS.as_posix())
    args = parser.parse_args()
    out_dir = prepare_exports(run_id=args.run_id, dataset_path=Path(args.dataset),
                              snapshot_path=Path(args.snapshot), attempts_path=Path(args.attempts))
    print(f"{out_dir.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
