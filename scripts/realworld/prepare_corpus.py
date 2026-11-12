"""Validate the acquired real-world corpus offline; no APIs or models are called."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/realworld/corpus_manifest.json"
CORPUS_ROOT = ROOT / "data/realworld_corpus"
ALLOWED_EXTENSIONS = {".md", ".rst", ".py", ".proto", ".rs"}
MAX_SOURCE_BYTES = 50 * 1024 * 1024
FULL_GIT_ID = re.compile(r"^[0-9a-f]{40,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_set_sha256(sources: list[dict]) -> str:
    content_set = [{"document_id": row.get("document_id"), "local_path": row.get("local_path"),
                    "repository_commit": row.get("repository_commit"), "sha256": row.get("sha256"),
                    "file_size_bytes": row.get("file_size_bytes")}
                   for row in sorted(sources, key=lambda item: str(item.get("document_id", "")))]
    canonical = json.dumps(content_set, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest(manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("corpus_id") != "omnirag-realtech-m1p2":
        errors.append("unexpected corpus_id")
    target = manifest.get("target_collection", "")
    if not re.fullmatch(r"omnirag_eval_[a-z0-9][a-z0-9_-]{0,80}", target):
        errors.append("target collection name is invalid")
    if target in {"omnirag", "omnirag_eval_phase8_20261104"}:
        errors.append("target collection collides with a protected collection")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("manifest sources must be a non-empty list")
        return errors
    ids: set[str] = set()
    paths: set[str] = set()
    for row in sources:
        source_id = row.get("document_id")
        if not source_id or source_id in ids:
            errors.append(f"missing or duplicate document_id: {source_id!r}")
        ids.add(source_id)
        local_path = row.get("local_path")
        rel = Path(local_path or "")
        if not local_path or rel.is_absolute() or ".." in rel.parts or not str(local_path).startswith("sources/"):
            errors.append(f"{source_id}: unsafe or missing local_path")
        elif local_path in paths:
            errors.append(f"{source_id}: duplicate local_path {local_path}")
        paths.add(local_path)
        for field in ("source_repository", "source_url", "original_path", "repository_commit",
                      "license", "license_evidence", "retrieved_at", "sha256", "file_size_bytes",
                      "file_format", "category", "publisher"):
            if row.get(field) is None or row.get(field) == "":
                errors.append(f"{source_id}: missing {field}")
        if row.get("status") != "CLEARED" or row.get("cleared_for_reuse") is not True:
            errors.append(f"{source_id}: source is not cleared for reuse")
        if row.get("intended_use_clearance") != "CLEARED":
            errors.append(f"{source_id}: intended-use licence clearance is invalid")
        if not FULL_GIT_ID.fullmatch(row.get("repository_commit") or ""):
            errors.append(f"{source_id}: repository commit must be a full Git object ID")
        if not SHA256.fullmatch(row.get("sha256") or ""):
            errors.append(f"{source_id}: invalid SHA-256")
        if not isinstance(row.get("file_size_bytes"), int) or row["file_size_bytes"] <= 0:
            errors.append(f"{source_id}: invalid file_size_bytes")
        ext = Path(str(local_path or "")).suffix.casefold()
        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"{source_id}: unexpected file type {ext!r}")
        if not isinstance(row.get("license_evidence"), list) or not row["license_evidence"]:
            errors.append(f"{source_id}: licence evidence is required")
        if urlsplit(str(row.get("source_url", ""))).hostname != "github.com":
            errors.append(f"{source_id}: source URL is not on github.com")
        if urlsplit(str(row.get("retrieval_url", ""))).hostname != "raw.githubusercontent.com":
            errors.append(f"{source_id}: retrieval URL is not raw.githubusercontent.com")
        if row.get("source_url_verification") != "VERIFIED_OFFICIAL_PINNED_REVISION":
            errors.append(f"{source_id}: official pinned source URL is not verified")
        if row.get("file_format") != Path(str(local_path or "")).suffix.lstrip(".").lower():
            errors.append(f"{source_id}: file_format does not match local path extension")

    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        errors.append("repository revision records are missing")
    else:
        for repo in repositories:
            name = repo.get("repository_id")
            if not FULL_GIT_ID.fullmatch(repo.get("commit") or ""):
                errors.append(f"{name}: repository commit must be a full Git object ID")
            if not repo.get("license_file_path") or not repo.get("license"):
                errors.append(f"{name}: repository license provenance is incomplete")
            repo_docs = [row for row in sources if row.get("document_id", "").startswith(f"{name}:")]
            if not repo_docs:
                errors.append(f"{name}: no selected document inventory")
            if any(row.get("repository_commit") != repo.get("commit") for row in repo_docs):
                errors.append(f"{name}: document commit differs from repository revision")
    if manifest.get("source_count") != len(sources):
        errors.append("source_count does not match sources array")
    expected_set_hash = content_set_sha256(sources)
    if manifest.get("content_set_sha256") != expected_set_hash:
        errors.append("content_set_sha256 does not match the sorted source inventory")
    return errors


def validate_corpus(manifest: dict, corpus_root: str | Path = CORPUS_ROOT,
                    evaluation_manifest: str | Path = MANIFEST) -> dict:
    root = Path(corpus_root).resolve()
    errors = validate_manifest(manifest)
    expected: set[str] = set()
    hashes: dict[str, str] = {}
    verified_hashes: set[str] = set()
    duplicate_hash_count = 0
    total_bytes = 0
    for row in manifest.get("sources", []):
        rel = Path(row.get("local_path", ""))
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            errors.append(f"{row.get('document_id')}: local path escapes corpus root")
            continue
        expected.add(rel.as_posix())
        if not path.is_file():
            errors.append(f"{row.get('document_id')}: source file missing: {rel.as_posix()}")
            continue
        size = path.stat().st_size
        actual_hash = sha256_file(path)
        total_bytes += size
        if size != row.get("file_size_bytes"):
            errors.append(f"{row.get('document_id')}: file size mismatch")
        if actual_hash != row.get("sha256"):
            errors.append(f"{row.get('document_id')}: SHA-256 mismatch")
        else:
            verified_hashes.add(row.get("document_id", "unknown"))
        if actual_hash in hashes:
            duplicate_hash_count += 1
            errors.append(f"duplicate content hash: {row.get('document_id')} duplicates {hashes[actual_hash]}")
        hashes[actual_hash] = row.get("document_id", "unknown")

    source_root = root / "sources"
    actual_files = {path.relative_to(root).as_posix() for path in source_root.rglob("*") if path.is_file()} if source_root.is_dir() else set()
    for unexpected in sorted(actual_files - expected):
        errors.append(f"unmanifested source file: {unexpected}")
    for absent in sorted(expected - actual_files):
        errors.append(f"manifest file absent from sources tree: {absent}")

    license_entries = manifest.get("license_files", [])
    verified_license_count = 0
    for entry in license_entries:
        path = (root / entry.get("path", "")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"license evidence file missing: {entry.get('path')}")
            continue
        if path.stat().st_size != entry.get("file_size_bytes"):
            errors.append(f"license evidence size mismatch: {entry.get('path')}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"license evidence SHA-256 mismatch: {entry.get('path')}")
        else:
            verified_license_count += 1

    sidecar_values = {}
    for sidecar in ("metadata/source_licences.json", "metadata/repository_revisions.json",
                    "metadata/acquisition_report.json"):
        path = root / sidecar
        if not path.is_file():
            errors.append(f"required acquisition sidecar missing: {sidecar}")
        else:
            try:
                sidecar_values[sidecar] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                errors.append(f"invalid acquisition sidecar JSON: {sidecar}")

    revision_records = manifest.get("repositories", [])
    license_sidecar = sidecar_values.get("metadata/source_licences.json", {})
    revisions_sidecar = sidecar_values.get("metadata/repository_revisions.json", {})
    report_sidecar = sidecar_values.get("metadata/acquisition_report.json", {})
    if license_sidecar.get("repositories") != revision_records:
        errors.append("source_licences repository records differ from manifest")
    if revisions_sidecar.get("repositories") != revision_records:
        errors.append("repository_revisions records differ from manifest")
    if license_sidecar.get("license_files") != license_entries:
        errors.append("source_licences license file list differs from manifest")
    if report_sidecar.get("document_count") != len(manifest.get("sources", [])):
        errors.append("acquisition_report document count differs from manifest")
    if report_sidecar.get("content_set_sha256") != manifest.get("content_set_sha256"):
        errors.append("acquisition_report content-set checksum differs from manifest")
    expected_license_paths = {str(entry.get("path")) for entry in license_entries}
    license_dir = root / "metadata/licenses"
    actual_license_paths = {p.relative_to(root).as_posix() for p in license_dir.rglob("*") if p.is_file()} if license_dir.is_dir() else set()
    for missing in sorted(expected_license_paths - actual_license_paths):
        errors.append(f"license evidence missing from metadata tree: {missing}")
    for unexpected in sorted(actual_license_paths - expected_license_paths):
        errors.append(f"unmanifested license file: {unexpected}")

    eval_path = Path(evaluation_manifest)
    if eval_path.is_file():
        try:
            eval_manifest = json.loads(eval_path.read_text(encoding="utf-8-sig"))
            if eval_manifest != manifest:
                errors.append("eval/realworld corpus manifest differs from data corpus manifest")
            recorded_manifest_hash = report_sidecar.get("manifest_sha256")
            if recorded_manifest_hash and sha256_file(eval_path) != recorded_manifest_hash:
                errors.append("acquisition_report manifest_sha256 does not match manifest bytes")
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append("eval/realworld corpus manifest is invalid JSON")

    disk_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0
    return {"valid": not errors, "document_count": len(manifest.get("sources", [])),
            "repository_count": len(manifest.get("repositories", [])),
            "total_file_bytes": total_bytes, "total_file_mib": round(total_bytes / (1024 ** 2), 3),
            "corpus_disk_bytes": disk_bytes, "corpus_disk_mib": round(disk_bytes / (1024 ** 2), 3),
            "category_distribution": dict(sorted(Counter(row.get("category") for row in manifest.get("sources", [])).items())),
            "sha256_verified_count": len(verified_hashes), "license_files_verified_count": verified_license_count,
            "duplicate_content_hashes": duplicate_hash_count,
            "unexpected_file_count": len(actual_files - expected), "error_count": len(errors),
            "errors": errors}


def leakage_candidates(dataset: dict, dev_dataset: dict, corpus_root: Path) -> list[dict]:
    """Report duplicate/near-identical query text and corpus wording for review."""
    from difflib import SequenceMatcher

    old = [(case.get("case_id", ""), " ".join(case.get("query", "").casefold().split()))
           for case in dev_dataset.get("cases", [])]
    files = [p for p in corpus_root.rglob("*") if p.is_file() and p.suffix.casefold() in ALLOWED_EXTENSIONS]
    corpus_text = []
    for path in files:
        try:
            corpus_text.append((str(path.relative_to(corpus_root)),
                                " ".join(path.read_text(encoding="utf-8").casefold().split())))
        except (UnicodeError, OSError):
            continue
    findings = []
    for case in dataset.get("cases", []):
        query = " ".join(case.get("query", "").casefold().split())
        if not query:
            continue
        for case_id, prior in old:
            score = SequenceMatcher(None, query, prior).ratio()
            if score >= 0.8:
                findings.append({"case_id": case.get("case_id"), "kind": "development_query_overlap",
                                 "reference": case_id, "similarity": round(score, 3), "review_required": True})
        for name, content in corpus_text:
            if query and query in content:
                findings.append({"case_id": case.get("case_id"), "kind": "corpus_wording_overlap",
                                 "reference": name, "review_required": True})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--report", type=Path, help="optional output path for integrity report JSON")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": f"manifest load failed: {type(exc).__name__}"}, indent=2))
        return 2
    report = validate_corpus(manifest, args.corpus_root, args.manifest)
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
