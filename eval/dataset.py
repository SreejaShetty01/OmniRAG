"""Versioned, corpus-backed evaluation dataset schema and validation helpers.

The module intentionally does not run OmniRAG or score answers. It validates a
human-authored reference dataset against the synthetic source corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pymupdf as fitz
from docx import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator


DATASET_ID = "omnirag-phase8-eval"
DATASET_VERSION = "1.0.1"
CORPUS_ID = "phase7a-smoke-corpus"
CORPUS_VERSION = "phase7a-synthetic-v1"
CORPUS_RELATIVE_ROOT = Path("tests/fixtures/phase7_smoke")
CORPUS_MANIFEST = "docs/audit/smoke_corpus_manifest.md"
CATEGORIES = (
    "answerable_factual", "multi_document", "version_sensitive", "unsupported",
    "conflicting_evidence", "citation_provenance", "acl_tenant_isolation",
    "prompt_injection", "profile_sensitive",
)
PROFILES = ("lite", "core", "research")


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_document: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^P7C-[A-Z]{2}-[0-9]{3}$")
    category: Literal[
        "answerable_factual", "multi_document", "version_sensitive", "unsupported",
        "conflicting_evidence", "citation_provenance", "acl_tenant_isolation",
        "prompt_injection", "profile_sensitive",
    ]
    query: str = Field(min_length=1)
    expected_answer: str | None = None
    acceptable_answer_variants: list[str] = Field(default_factory=list)
    answer_match: Literal["exact", "semantic", "evidence_based", "not_applicable"]
    expected_source_documents: list[str] = Field(min_length=1)
    expected_evidence: list[EvidenceReference] = Field(min_length=1)
    expected_citations: list[str] = Field(default_factory=list)
    expected_behavior: str = Field(min_length=1)
    expected_abstention: bool | None = None
    expected_conflict: dict[str, Any] | None = None
    profile_coverage: list[Literal["lite", "core", "research"]] = Field(min_length=1)
    acl_context: dict[str, Any] | None = None
    tenant_context: dict[str, Any] | None = None
    security_expectation: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"]
    ground_truth_status: Literal["verified", "provisional"]
    verification_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_category_expectations(self):
        if self.category == "unsupported":
            if self.expected_answer is not None or self.acceptable_answer_variants:
                raise ValueError("unsupported cases must not contain an expected answer or answer variants")
            if self.expected_abstention is not True:
                raise ValueError("unsupported cases must explicitly expect abstention")
            if self.answer_match != "not_applicable":
                raise ValueError("unsupported cases must set answer_match to not_applicable")
        elif self.category == "conflicting_evidence":
            if not self.expected_conflict or len(self.expected_source_documents) < 2:
                raise ValueError("conflicting_evidence cases require conflict details and at least two sources")
            if self.expected_abstention is not False:
                raise ValueError("conflicting_evidence cases must explicitly expect a conflict-aware response")
        elif self.category == "acl_tenant_isolation":
            if not self.acl_context or not self.tenant_context:
                raise ValueError("ACL cases require explicit acl_context and tenant_context")
            if self.acl_context.get("expected_access") not in {"allow", "deny"}:
                raise ValueError("acl_context.expected_access must be allow or deny")
        elif self.category == "prompt_injection":
            if not self.security_expectation:
                raise ValueError("prompt_injection cases require an explicit security_expectation")
        if self.category in {"answerable_factual", "multi_document", "version_sensitive",
                             "citation_provenance", "profile_sensitive"} and not self.expected_answer:
            raise ValueError(f"{self.category} cases require a grounded expected_answer")
        if self.expected_answer and self.answer_match == "not_applicable":
            raise ValueError("cases with expected answers cannot use not_applicable answer_match")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")
    dataset_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    corpus_id: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    corpus_manifest: str = Field(min_length=1)
    known_limitations: list[str]
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [case.case_id for case in self.cases]
        duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate case_id values: {', '.join(duplicates)}")
        return self


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


_ANSWER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "says", "the", "to", "what", "which", "with",
}


def _read_source(path: Path) -> list[str]:
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        with fitz.open(path) as document:
            return [page.get_text("text") for page in document]
    if suffix == ".docx":
        doc = Document(path)
        return [p.text for p in doc.paragraphs] + [
            " ".join(cell.text for row in table.rows for cell in row.cells)
            for table in doc.tables
        ]
    return path.read_text(encoding="utf-8").splitlines()


def validate_dataset(dataset: EvaluationDataset | dict[str, Any], *, corpus_root: str | Path,
                     repository_root: str | Path | None = None) -> EvaluationDataset:
    """Validate structure plus each source and quoted evidence location.

    Raises ValueError with all source/evidence problems, rather than dropping
    records. Pydantic errors are preserved for schema-level failures.
    """
    parsed = dataset if isinstance(dataset, EvaluationDataset) else EvaluationDataset.model_validate(dataset)
    root = Path(corpus_root).resolve()
    errors: list[str] = []
    if not root.is_dir():
        raise ValueError(f"corpus_root does not exist or is not a directory: {root}")
    if repository_root is not None:
        manifest = (Path(repository_root) / parsed.corpus_manifest).resolve()
        if not manifest.is_file():
            errors.append(f"corpus manifest is missing: {parsed.corpus_manifest}")
        elif not manifest.is_relative_to(Path(repository_root).resolve()):
            errors.append(f"corpus manifest escapes repository root: {parsed.corpus_manifest}")
    for case in parsed.cases:
        evidence_sources = {e.source_document for e in case.expected_evidence}
        if not set(case.expected_source_documents).issubset(evidence_sources):
            errors.append(f"{case.case_id}: every expected_source_document must have expected_evidence")
        if not set(case.expected_citations).issubset(set(case.expected_source_documents)):
            errors.append(f"{case.case_id}: expected_citations must reference expected_source_documents")
        source_lines: dict[str, list[str]] = {}
        for source in set(case.expected_source_documents) | evidence_sources:
            rel = Path(source)
            if rel.is_absolute() or ".." in rel.parts:
                errors.append(f"{case.case_id}: source path must remain within corpus: {source}")
                continue
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                errors.append(f"{case.case_id}: source document does not exist in corpus: {source}")
                continue
            try:
                source_lines[source] = _read_source(path)
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{case.case_id}: cannot read source {source}: {type(exc).__name__}")
        for evidence in case.expected_evidence:
            lines = source_lines.get(evidence.source_document)
            if lines is None:
                continue
            locator = re.fullmatch(r"line:([1-9][0-9]*)", evidence.locator)
            page_locator = re.fullmatch(r"page:([1-9][0-9]*)", evidence.locator)
            if locator:
                index = int(locator.group(1)) - 1
                if index >= len(lines):
                    errors.append(f"{case.case_id}: {evidence.source_document} has no {evidence.locator}")
                    continue
                haystack = lines[index]
            elif page_locator:
                index = int(page_locator.group(1)) - 1
                if index >= len(lines):
                    errors.append(f"{case.case_id}: {evidence.source_document} has no {evidence.locator}")
                    continue
                haystack = lines[index]
            else:
                errors.append(f"{case.case_id}: locator must use line:N or page:N: {evidence.locator}")
                continue
            if _normalise(evidence.quote) not in _normalise(haystack):
                errors.append(f"{case.case_id}: evidence quote does not match {evidence.source_document} {evidence.locator}")
        if case.expected_answer and case.answer_match == "exact":
            evidence_text = " ".join(e.quote for e in case.expected_evidence)
            if _normalise(case.expected_answer) not in _normalise(evidence_text):
                errors.append(f"{case.case_id}: exact expected_answer is not present in expected_evidence")
        elif case.expected_answer and case.category in {
            "answerable_factual", "multi_document", "version_sensitive", "citation_provenance", "profile_sensitive",
        }:
            evidence_text = _normalise(" ".join(e.quote for e in case.expected_evidence))
            source_text = _normalise(" ".join(case.expected_source_documents))
            terms = [term.casefold() for term in re.findall(r"[A-Za-z0-9]+", case.expected_answer)
                     if term.casefold() not in _ANSWER_STOPWORDS]
            missing_terms = [term for term in terms
                             if term not in evidence_text and term not in source_text]
            if missing_terms:
                errors.append(f"{case.case_id}: expected_answer contains terms not supported by evidence or source references: "
                              + ", ".join(sorted(set(missing_terms))))
        if case.category == "conflicting_evidence":
            conflicts = case.expected_conflict or {}
            entries = conflicts.get("values", [])
            if len(entries) < 2:
                errors.append(f"{case.case_id}: expected_conflict.values must document at least two values")
            for entry in entries:
                src = entry.get("source_document")
                value = str(entry.get("value", ""))
                if src not in evidence_sources or not value:
                    errors.append(f"{case.case_id}: each conflicting value needs a cited evidence source and value")
                elif not any(e.source_document == src and _normalise(value) in _normalise(e.quote)
                             for e in case.expected_evidence):
                    errors.append(f"{case.case_id}: conflicting value {value!r} is absent from source evidence {src}")
        if case.category == "acl_tenant_isolation":
            if case.ground_truth_status == "verified" and not case.verification_note:
                errors.append(f"{case.case_id}: verified ACL case requires verification_note")
        if case.category == "prompt_injection" and not case.expected_behavior.strip():
            errors.append(f"{case.case_id}: prompt injection requires expected security behavior")
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return parsed


def dataset_checksum(dataset: EvaluationDataset | dict[str, Any]) -> str:
    parsed = dataset if isinstance(dataset, EvaluationDataset) else EvaluationDataset.model_validate(dataset)
    canonical = json.dumps(parsed.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def create_manifest(dataset: EvaluationDataset | dict[str, Any], *, corpus_root: str | Path,
                    created_at: str | None = None,
                    repository_root: str | Path | None = None) -> dict[str, Any]:
    parsed = validate_dataset(dataset, corpus_root=corpus_root, repository_root=repository_root)
    distributions = Counter(case.category for case in parsed.cases)
    sources = sorted({source for case in parsed.cases for source in case.expected_source_documents})
    provisional = sum(case.ground_truth_status == "provisional" for case in parsed.cases)
    return {
        "manifest_schema_version": "1.0",
        "dataset_id": parsed.dataset_id,
        "dataset_version": parsed.dataset_version,
        "created_at": created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "corpus_id": parsed.corpus_id,
        "corpus_version": parsed.corpus_version,
        "corpus_manifest": parsed.corpus_manifest,
        "case_count": len(parsed.cases),
        "verified_case_count": len(parsed.cases) - provisional,
        "provisional_case_count": provisional,
        "category_distribution": dict(sorted(distributions.items())),
        "source_documents": sources,
        "schema_version": parsed.schema_version,
        "validation_status": "passed_with_provisional_cases" if provisional else "passed",
        "dataset_sha256": dataset_checksum(parsed),
        "known_limitations": parsed.known_limitations,
    }


def validate_manifest(dataset: EvaluationDataset | dict[str, Any], manifest: dict[str, Any], *,
                      corpus_root: str | Path, repository_root: str | Path | None = None) -> None:
    expected = create_manifest(dataset, corpus_root=corpus_root, created_at=manifest.get("created_at"),
                               repository_root=repository_root)
    differences = [key for key, value in expected.items() if manifest.get(key) != value]
    if differences:
        raise ValueError("Dataset manifest is inconsistent for: " + ", ".join(differences))


def write_release(dataset: EvaluationDataset | dict[str, Any], *, output_dir: str | Path,
                  corpus_root: str | Path, repository_root: str | Path | None = None,
                  created_at: str | None = None) -> tuple[Path, Path]:
    parsed = validate_dataset(dataset, corpus_root=corpus_root, repository_root=repository_root)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{parsed.dataset_id}-{parsed.dataset_version}"
    data_path, manifest_path = directory / f"{stem}.json", directory / f"{stem}.manifest.json"
    data = json.dumps(parsed.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    if data_path.exists():
        existing = EvaluationDataset.model_validate_json(data_path.read_text(encoding="utf-8"))
        if dataset_checksum(existing) != dataset_checksum(parsed):
            raise ValueError(f"Refusing to replace immutable dataset version {parsed.dataset_version}")
    else:
        data_path.write_text(data, encoding="utf-8", newline="\n")
    new_manifest = create_manifest(parsed, corpus_root=corpus_root, created_at=created_at,
                                   repository_root=repository_root)
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(parsed, existing_manifest, corpus_root=corpus_root,
                          repository_root=repository_root)
    else:
        manifest_path.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8", newline="\n")
    return data_path, manifest_path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(f"eval/datasets/{DATASET_ID}-{DATASET_VERSION}.json"))
    parser.add_argument("--manifest", type=Path, default=Path(
        f"eval/datasets/{DATASET_ID}-{DATASET_VERSION}.manifest.json"))
    parser.add_argument("--corpus", type=Path, default=CORPUS_RELATIVE_ROOT)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    dataset = EvaluationDataset.model_validate(_load_json(args.dataset))
    validate_dataset(dataset, corpus_root=args.corpus, repository_root=args.repository_root)
    manifest = _load_json(args.manifest)
    validate_manifest(dataset, manifest, corpus_root=args.corpus, repository_root=args.repository_root)
    print(json.dumps({"validation_status": manifest["validation_status"],
                      "case_count": manifest["case_count"],
                      "verified_case_count": manifest["verified_case_count"],
                      "provisional_case_count": manifest["provisional_case_count"],
                      "category_distribution": manifest["category_distribution"],
                      "dataset_sha256": manifest["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
