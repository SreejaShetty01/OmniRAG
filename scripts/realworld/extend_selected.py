"""Add a second fixed batch of reviewed files from already pinned repositories."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.realworld.acquire_selected import OUT, EVAL_MANIFEST, REPOSITORIES, _download, scan_file_notices
from scripts.realworld.prepare_corpus import content_set_sha256

EXTRA = [
    ("fastapi", "docs/en/docs/tutorial/background-tasks.md", "architecture"),
    ("fastapi", "docs/en/docs/tutorial/bigger-applications.md", "architecture"),
    ("fastapi", "docs/en/docs/tutorial/body-fields.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/body-multiple-params.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/body-nested-models.md", "data_validation"),
    ("fastapi", "docs/en/docs/tutorial/body-updates.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/cors.md", "security"),
    ("fastapi", "docs/en/docs/tutorial/handling-errors.md", "troubleshooting"),
    ("fastapi", "docs/en/docs/tutorial/header-params.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/metadata.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/middleware.md", "architecture"),
    ("fastapi", "docs/en/docs/tutorial/request-files.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/request-forms-and-files.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/testing.md", "testing"),
    ("fastapi", "docs/en/docs/tutorial/security/oauth2-jwt.md", "security"),
    ("fastapi", "docs/en/docs/advanced/additional-responses.md", "api_reference"),
    ("fastapi", "docs/en/docs/advanced/response-headers.md", "api_reference"),
    ("fastapi", "docs/en/docs/advanced/response-cookies.md", "api_reference"),
    ("fastapi", "docs/en/docs/advanced/testing-websockets.md", "testing"),
    ("fastapi", "docs/en/docs/deployment/https.md", "configuration"),
    ("pydantic", "docs/api/aliases.md", "data_validation"),
    ("pydantic", "docs/api/annotated_handlers.md", "data_validation"),
    ("pydantic", "docs/api/dataclasses.md", "data_validation"),
    ("pydantic", "docs/api/errors.md", "troubleshooting"),
    ("pydantic", "docs/api/functional_serializers.md", "data_validation"),
    ("pydantic", "docs/api/functional_validators.md", "data_validation"),
    ("pydantic", "docs/api/networks.md", "api_reference"),
    ("pydantic", "docs/api/standard_library_types.md", "api_reference"),
    ("pydantic", "docs/api/validate_call.md", "api_reference"),
    ("pydantic", "docs/api/version.md", "versioning"),
    ("pydantic", "docs/api/pydantic_core.md", "architecture"),
    ("pydantic", "docs/concepts/alias.md", "data_validation"),
    ("pydantic", "docs/concepts/dataclasses.md", "data_validation"),
    ("pydantic", "docs/concepts/forward_annotations.md", "data_validation"),
    ("pydantic", "docs/concepts/performance.md", "architecture"),
    ("pydantic", "docs/concepts/pydantic_settings.md", "configuration"),
    ("pydantic", "docs/concepts/unions.md", "data_validation"),
    ("pydantic", "docs/concepts/validation_decorator.md", "data_validation"),
    ("pydantic", "docs/internals/resolving_annotations.md", "architecture"),
    ("pydantic", "docs/errors/validation_errors.md", "troubleshooting"),
    ("qdrant", "docs/roadmap/README.md", "versioning"),
    ("qdrant", "docs/roadmap/roadmap-2022.md", "versioning"),
    ("qdrant", "docs/roadmap/roadmap-2023.md", "versioning"),
    ("qdrant", "lib/api/src/grpc/proto/collections.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/proto/points_service.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/proto/qdrant.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/proto/snapshots_service.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/proto/health_check.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/models.rs", "source_code"),
    ("qdrant", "lib/api/src/rest/schema.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/facet.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/distance_matrix.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/query.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/sharding_keys.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/state_management.rs", "source_code"),
    ("qdrant", "lib/collection/src/config.rs", "source_code"),
    ("requests", "docs/community/recommended.rst", "architecture"),
    ("requests", "docs/community/vulnerabilities.rst", "security"),
    ("requests", "docs/community/updates.rst", "versioning"),
    ("requests", "docs/community/out-there.rst", "overview"),
    ("requests", "docs/community/support.rst", "troubleshooting"),
    ("requests", "docs/community/release-process.rst", "versioning"),
    ("requests", "src/requests/cookies.py", "source_code"),
    ("requests", "src/requests/compat.py", "source_code"),
    ("requests", "src/requests/hooks.py", "source_code"),
    ("requests", "src/requests/status_codes.py", "source_code"),
    ("requests", "src/requests/_internal_utils.py", "source_code"),
    ("requests", "src/requests/_types.py", "source_code"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def extend() -> dict:
    data_manifest = OUT / "manifest.json"
    if not data_manifest.is_file():
        raise FileNotFoundError("initial acquired corpus manifest is missing")
    manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    existing_ids = {row["document_id"] for row in manifest["sources"]}
    existing_hashes = {row["sha256"] for row in manifest["sources"]}
    staged = []
    for repo_name, original_path, category in EXTRA:
        repo = REPOSITORIES[repo_name]
        document_id = f"{repo_name}:{original_path}"
        target = OUT / "sources" / repo_name / original_path
        if document_id in existing_ids or target.exists():
            raise ValueError(f"refusing duplicate or pre-existing path: {document_id}")
        body, retrieval_url = _download(repo, original_path)
        digest = hashlib.sha256(body).hexdigest()
        if digest in existing_hashes or any(digest == item["sha256"] for item in staged):
            raise ValueError(f"duplicate content hash for {document_id}")
        notice_hits = scan_file_notices(body)
        staged.append({"repo_name": repo_name, "original_path": original_path,
                       "category": category, "document_id": document_id,
                       "body": body, "retrieval_url": retrieval_url, "sha256": digest,
                       "notice_hits": notice_hits})

    new_rows = []
    for item in staged:
        repo_name, original_path, body = item["repo_name"], item["original_path"], item["body"]
        repo = REPOSITORIES[repo_name]
        target = OUT / "sources" / repo_name / original_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        now = _now()
        license_entry = next(x for x in manifest["license_files"] if x["repository"] == repo_name and x["path"].endswith("/LICENSE"))
        new_rows.append({
            "document_id": item["document_id"], "source_id": item["document_id"],
            "source_repository": f"https://github.com/{repo['slug']}",
            "source_url": f"https://github.com/{repo['slug']}/blob/{repo['commit']}/{original_path}",
            "retrieval_url": item["retrieval_url"], "original_path": original_path,
            "local_path": f"sources/{repo_name}/{original_path}",
            "repository_commit": repo["commit"], "repository_release": repo["release"],
            "version": repo["release"], "publisher": repo["publisher"],
            "license": repo["license"], "license_version": repo["license"], "licence": repo["license"],
            "license_evidence": [repo["license_url"]], "license_evidence_url": repo["license_url"],
            "license_evidence_sha256": license_entry["sha256"],
            "cleared_for_reuse": not bool(item["notice_hits"]),
            "intended_use_clearance": "PENDING_FILE_NOTICE_REVIEW" if item["notice_hits"] else "PENDING_MANUAL_REVIEW",
            "intended_use_scope": manifest["sources"][0]["intended_use_scope"],
            "attribution_requirements": manifest["sources"][0]["attribution_requirements"],
            "processing_restrictions": "Selected first-party upstream file; byte-preserved; root license and notice retained.",
            "retrieved_at": now, "retrieval_date": now[:10], "sha256": item["sha256"],
            "file_size_bytes": len(body), "file_format": target.suffix.lstrip(".").lower(),
            "expected_format": target.suffix.lstrip(".").lower(), "category": item["category"],
            "status": "BLOCKED", "source_type": "git_repository_file",
            "source_url_verification": "VERIFIED_OFFICIAL_PINNED_REVISION",
            "file_level_notice_scan": item["notice_hits"] or "no matched file-level license/copyright markers; manual review still required",
        })
    manifest["sources"].extend(new_rows)
    manifest["source_count"] = len(manifest["sources"])
    manifest["content_set_sha256"] = content_set_sha256(manifest["sources"])
    manifest["created_at"] = _now()
    manifest["status"] = "acquired_pending_integrity_validation"
    for repo in manifest["repositories"]:
        repo["source_inventory"] = [row["original_path"] for row in manifest["sources"]
                                     if row["document_id"].startswith(f"{repo['repository_id']}:")]
    for name, value in (("manifest.json", manifest),):
        (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    EVAL_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    licenses = json.loads((OUT / "metadata/source_licences.json").read_text(encoding="utf-8"))
    licenses["repositories"] = manifest["repositories"]
    (OUT / "metadata/source_licences.json").write_text(json.dumps(licenses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    revisions = json.loads((OUT / "metadata/repository_revisions.json").read_text(encoding="utf-8"))
    revisions["repositories"] = manifest["repositories"]
    (OUT / "metadata/repository_revisions.json").write_text(json.dumps(revisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = json.loads((OUT / "metadata/acquisition_report.json").read_text(encoding="utf-8"))
    report.update({"document_count": len(manifest["sources"]),
                   "python_source_file_count": sum(row["file_format"] == "py" for row in manifest["sources"]),
                   "category_distribution": dict(sorted(Counter(row["category"] for row in manifest["sources"]).items())),
                   "total_source_bytes": sum(row["file_size_bytes"] for row in manifest["sources"]),
                   "total_source_mib": round(sum(row["file_size_bytes"] for row in manifest["sources"]) / 1024**2, 3),
                   "status": "acquired_pending_integrity_validation"})
    (OUT / "metadata/acquisition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"added_count": len(new_rows), "total_document_count": len(manifest["sources"]),
            "retrieved_bytes": sum(len(item["body"]) for item in staged),
            "status": "acquired_pending_integrity_validation"}


if __name__ == "__main__":
    print(json.dumps(extend(), indent=2))
