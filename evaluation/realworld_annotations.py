"""Human annotation lifecycle and conversion into the existing evaluator schema.

This work queue is deliberately separate from EvaluationDataset. Conversion
accepts only human-reviewed, approved annotations and delegates final evidence
validation to evaluation.dataset.validate_dataset.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.dataset import (EvaluationCase, EvaluationDataset, EvidenceReference,
                                validate_dataset)

AnnotationCategory = Literal[
    "exact_identifier", "code_understanding", "multi_document_reasoning",
    "tables_and_numerical", "version_conflict", "ambiguous", "unanswerable",
]
AnnotationStatus = Literal["DRAFT", "ANNOTATED", "REVIEWED", "VERIFIED"]
_NEXT_STATUS = {"DRAFT": "ANNOTATED", "ANNOTATED": "REVIEWED", "REVIEWED": "VERIFIED"}


class AnnotationSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_document: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class HumanAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^P7C-[A-Z]{2}-[0-9]{3}$")
    query: str = ""
    category: AnnotationCategory
    difficulty: Literal["easy", "medium", "hard"] | None = None
    expected_answerability: Literal["answerable", "unanswerable"] | None = None
    expected_answer: str | None = None
    evidence_spans: list[AnnotationSpan] = Field(default_factory=list)
    author_id: str | None = None
    reviewer_id: str | None = None
    status: AnnotationStatus = "DRAFT"
    review_outcome: Literal["approved", "rejected"] | None = None
    review_notes: str | None = None
    expected_behavior: str = ""

    @model_validator(mode="after")
    def lifecycle_requirements(self):
        if self.status in {"ANNOTATED", "REVIEWED", "VERIFIED"}:
            if not self.query.strip() or not self.author_id or not self.difficulty or not self.expected_answerability:
                raise ValueError("annotated cases require query, author, difficulty, and answerability")
            if self.expected_answerability == "answerable" and (not self.expected_answer or not self.evidence_spans):
                raise ValueError("answerable annotations require a human answer and evidence spans")
            if not self.evidence_spans:
                raise ValueError("annotations require evidence spans, including a documented gap for unanswerable cases")
            if self.category == "unanswerable" and self.expected_answerability != "unanswerable":
                raise ValueError("unanswerable category requires unanswerable answerability")
        if self.status in {"REVIEWED", "VERIFIED"}:
            if not self.reviewer_id or self.review_outcome is None or self.review_notes is None:
                raise ValueError("reviewed cases require reviewer, outcome, and notes")
        if self.status == "VERIFIED" and self.review_outcome != "approved":
            raise ValueError("verified cases require an approved review outcome")
        return self


def transition_annotation(annotation: HumanAnnotation, target_status: AnnotationStatus, *,
                          actor_id: str, review_outcome: str | None = None,
                          review_notes: str | None = None) -> HumanAnnotation:
    """Apply one forward lifecycle transition with distinct author/reviewer roles."""
    if _NEXT_STATUS.get(annotation.status) != target_status:
        raise ValueError(f"invalid annotation transition: {annotation.status} -> {target_status}")
    if not actor_id.strip():
        raise ValueError("transition actor_id is required")
    row = annotation.model_dump(mode="python")
    if target_status == "ANNOTATED":
        row.update(status=target_status, author_id=actor_id)
    elif target_status == "REVIEWED":
        if actor_id == annotation.author_id:
            raise ValueError("reviewer must be distinct from the annotation author")
        if review_outcome not in {"approved", "rejected"} or review_notes is None:
            raise ValueError("review transition requires an outcome and notes")
        row.update(status=target_status, reviewer_id=actor_id,
                   review_outcome=review_outcome, review_notes=review_notes)
    else:
        if actor_id != annotation.reviewer_id or annotation.review_outcome != "approved":
            raise ValueError("only the approving reviewer can verify an annotation")
        row.update(status=target_status)
    return HumanAnnotation.model_validate(row)


_CATEGORY_MAP = {
    "exact_identifier": "answerable_factual",
    "code_understanding": "answerable_factual",
    "multi_document_reasoning": "multi_document",
    "tables_and_numerical": "answerable_factual",
    "version_conflict": "version_sensitive",
    "ambiguous": "answerable_factual",
    "unanswerable": "unsupported",
}


def _span_to_evidence(span: AnnotationSpan, corpus_root: Path,
                      document_versions: dict[str, str]) -> EvidenceReference:
    rel = Path(span.source_document)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe evidence document path: {span.source_document}")
    if document_versions.get(span.source_document) != span.document_version:
        raise ValueError(f"evidence version mismatch: {span.source_document} {span.document_version}")
    path = (corpus_root / rel).resolve()
    root = corpus_root.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"evidence document is missing or outside corpus: {span.source_document}")
    from evaluation.dataset import _read_source
    lines = _read_source(path)
    full = "\n".join(lines)
    if span.start_offset >= span.end_offset or span.end_offset > len(full):
        raise ValueError(f"invalid evidence offsets for {span.source_document}")
    quote = full[span.start_offset:span.end_offset]
    if not quote.strip():
        raise ValueError("evidence span cannot be blank")
    starts: list[int] = []
    cursor = 0
    for text in lines:
        starts.append(cursor)
        cursor += len(text) + 1
    start_unit = max((i for i, start in enumerate(starts) if start <= span.start_offset), default=0)
    end_unit = max((i for i, start in enumerate(starts) if start <= span.end_offset - 1), default=0)
    if start_unit != end_unit:
        raise ValueError("evidence span must fit one existing evaluator line/page locator")
    locator_type = "page" if path.suffix.casefold() == ".pdf" else "line"
    return EvidenceReference(source_document=span.source_document,
                             locator=f"{locator_type}:{start_unit + 1}", quote=quote)


def convert_human_approved_annotations(
    annotations: list[HumanAnnotation | dict], *, dataset_id: str, dataset_version: str,
    corpus_id: str, corpus_version: str, corpus_manifest: str,
    corpus_root: str | Path, document_versions: dict[str, str],
) -> EvaluationDataset:
    """Convert VERIFIED/approved records; validate with the unchanged evaluator."""
    parsed = [item if isinstance(item, HumanAnnotation) else HumanAnnotation.model_validate(item)
              for item in annotations]
    ids = [item.case_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate annotation case_id")
    queries = [" ".join(item.query.casefold().split()) for item in parsed if item.query.strip()]
    if len(queries) != len(set(queries)):
        raise ValueError("duplicate annotation query")
    cases = []
    for item in parsed:
        if item.status != "VERIFIED" or item.review_outcome != "approved":
            raise ValueError(f"{item.case_id}: annotation is not human-approved and VERIFIED")
        if not item.expected_behavior.strip():
            raise ValueError(f"{item.case_id}: expected_behavior is required")
        evidence = [_span_to_evidence(span, Path(corpus_root), document_versions)
                    for span in item.evidence_spans]
        if not evidence:
            raise ValueError(f"{item.case_id}: evaluator schema requires source evidence")
        is_unanswerable = item.expected_answerability == "unanswerable"
        expected_answer = None if is_unanswerable else item.expected_answer
        answer_match = "not_applicable" if is_unanswerable else "evidence_based"
        evaluator_category = "unsupported" if is_unanswerable else _CATEGORY_MAP[item.category]
        cases.append(EvaluationCase(
            case_id=item.case_id, category=evaluator_category, query=item.query,
            expected_answer=expected_answer, answer_match=answer_match,
            expected_source_documents=sorted({e.source_document for e in evidence}),
            expected_evidence=evidence, expected_behavior=item.expected_behavior,
            expected_abstention=is_unanswerable, profile_coverage=["lite", "core", "research"],
            difficulty=item.difficulty, ground_truth_status="verified",
            verification_note=f"Human author {item.author_id}; approved reviewer {item.reviewer_id}: {item.review_notes}",
        ))
    dataset = EvaluationDataset(schema_version="1.0", dataset_id=dataset_id,
        dataset_version=dataset_version, corpus_id=corpus_id, corpus_version=corpus_version,
        corpus_manifest=corpus_manifest, known_limitations=[], cases=cases)
    return validate_dataset(dataset, corpus_root=corpus_root)


def annotation_summary(annotations: list[HumanAnnotation | dict]) -> dict:
    parsed = [item if isinstance(item, HumanAnnotation) else HumanAnnotation.model_validate(item)
              for item in annotations]
    return {"case_count": len(parsed), "status_counts": dict(Counter(x.status for x in parsed)),
            "category_counts": dict(Counter(x.category for x in parsed)),
            "human_approved_count": sum(x.status == "VERIFIED" and x.review_outcome == "approved"
                                         for x in parsed),
            "awaiting_human_count": sum(x.status != "VERIFIED" for x in parsed)}
