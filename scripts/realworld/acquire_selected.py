"""Acquire a fixed, reviewable set of files at immutable upstream commits."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/realworld_corpus"
EVAL_MANIFEST = ROOT / "eval/realworld/corpus_manifest.json"
LICENSE_CACHE = ROOT / ".m1p2_acquire"
MAX_FILE_BYTES = 5 * 1024 * 1024
FILE_NOTICE_PATTERN = re.compile(
    r"SPDX-License-Identifier|Copyright|All rights reserved|Creative Commons|CC-BY|licensed under",
    re.IGNORECASE,
)
sys.path.insert(0, str(ROOT))
from scripts.realworld.prepare_corpus import content_set_sha256

REPOSITORIES = {
    "fastapi": {"slug": "fastapi/fastapi", "commit": "bb8c2a64981d3e806575a445115f29eddf014c77",
                "release": "0.115.6", "license": "MIT", "license_sha256": "4EC89FFC81485B97FEC584B2D4A961032EEFFE834453894FD9C1274906CC744E",
                "license_url": "https://github.com/fastapi/fastapi/blob/bb8c2a64981d3e806575a445115f29eddf014c77/LICENSE",
                "publisher": "FastAPI project"},
    "pydantic": {"slug": "pydantic/pydantic", "commit": "5bd3a6507b749fcd4833173fba88b3690ff77170",
                 "release": "v2.10.4", "license": "MIT", "license_sha256": "A9E186F3CA16B5EEF84318E7A701721351A00CB7B8AE3A4394B67B49E3529EF3",
                 "license_url": "https://github.com/pydantic/pydantic/blob/5bd3a6507b749fcd4833173fba88b3690ff77170/LICENSE",
                 "publisher": "Pydantic project"},
    "qdrant": {"slug": "qdrant/qdrant", "commit": "9c20e9e27960228019b4606f137ca82b42fc3e66",
               "release": "v1.12.1", "license": "Apache-2.0", "license_sha256": "C71D239DF91726FC519C6EB72D318EC65820627232B2F796219E87DCF35D0AB4",
               "license_url": "https://github.com/qdrant/qdrant/blob/9c20e9e27960228019b4606f137ca82b42fc3e66/LICENSE",
               "publisher": "Qdrant project"},
    "requests": {"slug": "psf/requests", "commit": "6e83187b8feb273ed4c6cdab5efd8d54901dfab3",
                 "release": "v2.34.2", "license": "Apache-2.0", "license_sha256": "09E8A9BCEC8067104652C168685AB0931E7868F9C8284B66F5AE6EDAE5F1130B",
                 "license_url": "https://github.com/psf/requests/blob/6e83187b8feb273ed4c6cdab5efd8d54901dfab3/LICENSE",
                 "publisher": "Python Software Foundation / psf"},
}

SELECTIONS = [
    ("fastapi", "docs/en/docs/tutorial/first-steps.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/path-params.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/query-params.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/body.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/response-model.md", "api_reference"),
    ("fastapi", "docs/en/docs/tutorial/dependencies/index.md", "architecture"),
    ("fastapi", "docs/en/docs/tutorial/response-status-code.md", "api_reference"),
    ("fastapi", "docs/en/docs/async.md", "architecture"),
    ("fastapi", "docs/en/docs/advanced/advanced-dependencies.md", "architecture"),
    ("fastapi", "docs/en/docs/advanced/middleware.md", "architecture"),
    ("fastapi", "docs/en/docs/advanced/settings.md", "configuration"),
    ("fastapi", "docs/en/docs/deployment/docker.md", "configuration"),
    ("fastapi", "docs/en/docs/deployment/versions.md", "versioning"),
    ("fastapi", "docs/en/docs/reference/apirouter.md", "api_reference"),
    ("pydantic", "docs/concepts/models.md", "data_validation"),
    ("pydantic", "docs/concepts/fields.md", "data_validation"),
    ("pydantic", "docs/concepts/validators.md", "data_validation"),
    ("pydantic", "docs/concepts/strict_mode.md", "data_validation"),
    ("pydantic", "docs/concepts/config.md", "configuration"),
    ("pydantic", "docs/concepts/serialization.md", "data_validation"),
    ("pydantic", "docs/concepts/json.md", "data_validation"),
    ("pydantic", "docs/concepts/json_schema.md", "api_reference"),
    ("pydantic", "docs/concepts/conversion_table.md", "tables_numerical"),
    ("pydantic", "docs/concepts/type_adapter.md", "data_validation"),
    ("pydantic", "docs/api/base_model.md", "api_reference"),
    ("pydantic", "docs/api/types.md", "api_reference"),
    ("pydantic", "docs/internals/architecture.md", "architecture"),
    ("qdrant", "README.md", "overview"),
    ("qdrant", "docs/QUICK_START.md", "api_reference"),
    ("qdrant", "docs/DEVELOPMENT.md", "configuration"),
    ("qdrant", "docs/grpc/docs.md", "api_reference"),
    ("qdrant", "lib/api/src/grpc/proto/points.proto", "protocol_schema"),
    ("qdrant", "lib/api/src/grpc/proto/collections_service.proto", "protocol_schema"),
    ("qdrant", "lib/collection/src/collection/search.rs", "source_code"),
    ("qdrant", "lib/collection/src/collection/point_ops.rs", "source_code"),
    ("requests", "docs/user/quickstart.rst", "api_reference"),
    ("requests", "docs/user/authentication.rst", "security"),
    ("requests", "docs/user/advanced.rst", "architecture"),
    ("requests", "docs/user/install.rst", "configuration"),
    ("requests", "docs/api.rst", "api_reference"),
    ("requests", "docs/community/faq.rst", "troubleshooting"),
    ("requests", "src/requests/sessions.py", "source_code"),
    ("requests", "src/requests/api.py", "source_code"),
    ("requests", "src/requests/adapters.py", "source_code"),
    ("requests", "src/requests/models.py", "source_code"),
    ("requests", "src/requests/auth.py", "source_code"),
    ("requests", "src/requests/exceptions.py", "source_code"),
    ("requests", "src/requests/utils.py", "source_code"),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _download(repo: dict, original_path: str) -> tuple[bytes, str]:
    path = quote(original_path, safe="/._-()")
    url = f"https://raw.githubusercontent.com/{repo['slug']}/{repo['commit']}/{path}"
    request = Request(url, headers={"User-Agent": "OmniRAG-realworld-corpus/1.0"})
    with urlopen(request, timeout=30) as response:
        if not response.geturl().startswith("https://raw.githubusercontent.com/"):
            raise ValueError("redirect left approved raw.githubusercontent.com host")
        body = response.read(MAX_FILE_BYTES + 1)
    if not body or len(body) > MAX_FILE_BYTES:
        raise ValueError(f"empty or over-size file: {original_path}")
    return body, url


def scan_file_notices(body: bytes) -> list[str]:
    """Return line-level license/copyright candidates for human review."""
    text = body.decode("utf-8", errors="replace")
    return [f"line {number}: {line.strip()[:180]}"
            for number, line in enumerate(text.splitlines(), 1)
            if FILE_NOTICE_PATTERN.search(line)]


def _license_copy(name: str, expected_sha: str, destination: Path) -> dict:
    source = LICENSE_CACHE / name / "LICENSE"
    if not source.is_file():
        raise FileNotFoundError(f"verified license cache is missing: {source}")
    body = source.read_bytes()
    actual = hashlib.sha256(body).hexdigest().upper()
    if actual != expected_sha:
        raise ValueError(f"pinned {name} LICENSE hash mismatch")
    dest = destination / name / "LICENSE"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest().upper() != expected_sha:
            raise ValueError(f"existing {name} LICENSE does not match pinned digest")
    else:
        dest.write_bytes(body)
    return {"repository": name, "path": f"metadata/licenses/{name}/LICENSE",
            "source_url": REPOSITORIES[name]["license_url"], "license": REPOSITORIES[name]["license"],
            "sha256": actual.lower(), "file_size_bytes": len(body),
            "retrieved_at": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds").replace("+00:00", "Z"),
            "retrieval_note": "Downloaded from official raw URL at pinned commit during this task; hash rechecked from preserved bytes."}


def acquire() -> dict:
    if (OUT / "manifest.json").exists():
        raise FileExistsError(f"refusing to replace completed corpus manifest: {OUT / 'manifest.json'}")
    OUT.mkdir(parents=True, exist_ok=True)
    for repo in REPOSITORIES:
        (OUT / "sources" / repo).mkdir(parents=True, exist_ok=True)

    licenses = []
    for repo, metadata in REPOSITORIES.items():
        licenses.append(_license_copy(repo, metadata["license_sha256"], OUT / "metadata/licenses"))

    # Preserve the upstream Requests NOTICE file as required for any later redistribution.
    notice_path = OUT / "metadata/licenses/requests/NOTICE"
    if notice_path.exists():
        notice = notice_path.read_bytes()
        notice_url = f"https://raw.githubusercontent.com/psf/requests/{REPOSITORIES['requests']['commit']}/NOTICE"
    else:
        notice, notice_url = _download(REPOSITORIES["requests"], "NOTICE")
        notice_path.parent.mkdir(parents=True, exist_ok=True)
        notice_path.write_bytes(notice)
    licenses.append({"repository": "requests", "path": "metadata/licenses/requests/NOTICE",
                     "source_url": notice_url, "license": "Apache-2.0 related notice file",
                     "sha256": hashlib.sha256(notice).hexdigest(), "file_size_bytes": len(notice),
                     "retrieved_at": _utc_now(), "retrieval_note": "Upstream NOTICE preserved verbatim."})

    documents = []
    seen_hashes: dict[str, str] = {}
    for repo_name, original_path, category in SELECTIONS:
        repo = REPOSITORIES[repo_name]
        target = OUT / "sources" / repo_name / original_path
        if target.exists():
            body = target.read_bytes()
            retrieval_url = f"https://raw.githubusercontent.com/{repo['slug']}/{repo['commit']}/{quote(original_path, safe='/._-()')}"
            if not body or len(body) > MAX_FILE_BYTES:
                raise ValueError(f"existing source file is empty or over-size: {target}")
        else:
            body, retrieval_url = _download(repo, original_path)
        digest = hashlib.sha256(body).hexdigest()
        if digest in seen_hashes:
            raise ValueError(f"duplicate source content: {original_path} duplicates {seen_hashes[digest]}")
        seen_hashes[digest] = f"{repo_name}:{original_path}"
        notice_hits = scan_file_notices(body)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(body)
        fmt = target.suffix.lstrip(".").lower()
        source_url = f"https://github.com/{repo['slug']}/blob/{repo['commit']}/{original_path}"
        license_evidence = repo["license_url"]
        documents.append({
            "document_id": f"{repo_name}:{original_path}",
            "source_id": f"{repo_name}:{original_path}",
            "source_repository": f"https://github.com/{repo['slug']}",
            "source_url": source_url,
            "retrieval_url": retrieval_url,
            "original_path": original_path,
            "local_path": f"sources/{repo_name}/{original_path}",
            "repository_commit": repo["commit"],
            "repository_release": repo["release"],
            "version": repo["release"],
            "publisher": repo["publisher"],
            "license": repo["license"],
            "license_version": repo["license"],
            "licence": repo["license"],
            "license_evidence": [license_evidence],
            "license_evidence_url": license_evidence,
            "cleared_for_reuse": False,
            "intended_use_clearance": "PENDING_MANUAL_FILE_NOTICE_REVIEW",
            "intended_use_scope": "local storage, text extraction, chunking, indexing, and internal evaluation; redistribution is outside this task",
            "attribution_requirements": "Preserve repository LICENSE and NOTICE/source notices. If corpus files are redistributed, include required license and attribution notices and mark modifications as applicable.",
            "processing_restrictions": "Selected first-party upstream files only; source copied byte-for-byte; re-check any third-party notice before expanding selection.",
            "retrieved_at": _utc_now(),
            "retrieval_date": datetime.now(timezone.utc).date().isoformat(),
            "sha256": digest,
            "file_size_bytes": len(body),
            "file_format": fmt,
            "expected_format": fmt,
            "category": category,
            "status": "BLOCKED",
            "source_type": "git_repository_file",
            "source_url_verification": "VERIFIED_OFFICIAL_PINNED_REVISION",
            "file_level_notice_scan": notice_hits or ["No notice-pattern matches; manual license review still required"],
        })

    revisions = []
    for name, repo in REPOSITORIES.items():
        revisions.append({"repository_id": name, "repository_url": f"https://github.com/{repo['slug']}",
                          "commit": repo["commit"], "release_tag": repo["release"],
                          "license_file_path": f"metadata/licenses/{name}/LICENSE", "license": repo["license"],
                          "source_inventory": [x[1] for x in SELECTIONS if x[0] == name],
                          "acquisition_status": "acquired_selected_files",
                          "license_sha256": repo["license_sha256"].lower()})
    source_licenses = {"schema_version": "1.0", "repositories": revisions, "license_files": licenses,
                       "reuse_scope": "local corpus storage, text extraction, chunking, indexing, and evaluation; no redistribution",
                       "conditions": ["Retain source license texts and notices.",
                                      "Do not distribute the corpus without separate license/attribution review.",
                                      "Review selected file notices; exclude uncertain third-party material."]}
    revision_doc = {"schema_version": "1.0", "repositories": revisions}
    retrieved_at = _utc_now()
    manifest = {"manifest_schema_version": "1.0", "corpus_id": "omnirag-realtech-m1p2",
                "corpus_version": "v1", "created_at": retrieved_at,
                "target_collection": "omnirag_eval_omnirag_realtech_nomic768_v1",
                "embedding_space": "nomic-embed-text-768", "status": "acquired_pending_integrity_validation",
                "source_count": len(documents), "sources": documents, "repositories": revisions,
                "license_files": licenses, "known_limitations": [
                    "Local-only use is documented; redistribution requires a separate license review.",
                    "No full repository history or unselected files were acquired."]}
    manifest["content_set_sha256"] = content_set_sha256(documents)

    manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    (OUT / "manifest.json").write_text(manifest_json, encoding="utf-8")
    EVAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EVAL_MANIFEST.write_text(manifest_json, encoding="utf-8")
    (OUT / "metadata/source_licences.json").write_text(json.dumps(source_licenses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "metadata/repository_revisions.json").write_text(json.dumps(revision_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total_bytes = sum(item["file_size_bytes"] for item in documents)
    report = {"status": "acquired_pending_integrity_validation", "retrieved_at": retrieved_at,
              "document_count": len(documents), "python_source_file_count": sum(
                  x["file_format"] == "py" for x in documents), "repository_count": len(revisions),
              "category_distribution": dict(sorted(Counter(x["category"] for x in documents).items())),
              "total_source_bytes": total_bytes, "total_source_mib": round(total_bytes / (1024 ** 2), 3),
              "all_selected_hashes_computed_from_retrieved_bytes": True,
              "external_operations": "official raw GitHub URLs only; no inference or application API calls"}
    (OUT / "metadata/acquisition_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2))
