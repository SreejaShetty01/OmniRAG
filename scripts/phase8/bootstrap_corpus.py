"""Build and audit the isolated Phase 8 corpus using the application ingestion path."""

from __future__ import annotations

import hashlib
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from evaluation.dataset import (
    CORPUS_ID,
    CORPUS_VERSION,
    DATASET_ID,
    DATASET_VERSION,
    dataset_checksum,
    validate_dataset,
)
from evaluation.phase8_safety import require_evaluation_collection
from indexing.qdrant_indexer import PAYLOAD_INDEX_SCHEMAS
from infrastructure.provider_budget import activate_run_budget
from ingestion.pipeline import IngestionPipeline, IngestionStatus
from models.chunk import AccessControlMetadata


DATASET_PATH = ROOT / "evaluation" / "datasets" / f"{DATASET_ID}-{DATASET_VERSION}.json"
DATASET_MANIFEST_PATH = ROOT / "evaluation" / "datasets" / f"{DATASET_ID}-{DATASET_VERSION}.manifest.json"
STANDARD_USER = "phase8-evaluation-user"
STANDARD_TENANT = "phase8-evaluation-tenant"
ACL_USER_A = "user:phase7a-user-a"
ACL_USER_B = "user:phase7a-user-b"
ACL_TENANT = "phase7a-tenant-a"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _html_visible_body(value: str) -> str:
    from html.parser import HTMLParser

    class BodyText(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.in_body = False
            self.skip_depth = 0
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "body":
                self.in_body = True
            if tag in {"script", "style", "noscript"}:
                self.skip_depth += 1

        def handle_endtag(self, tag):
            if tag == "body":
                self.in_body = False
            if tag in {"script", "style", "noscript"} and self.skip_depth:
                self.skip_depth -= 1

        def handle_data(self, data):
            if self.in_body and not self.skip_depth:
                self.parts.append(data)

    parser = BodyText()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


def _source_version(filename: str) -> str:
    match = re.search(r"_v([0-9]+)\.", filename)
    return f"v{match.group(1)}" if match else CORPUS_VERSION


def _point_summary(point) -> dict[str, Any]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    allowed_provenance = (
        "embedder_provider", "embedder_model", "embedding_dim", "sparse_method",
        "sparse_tokenizer_version", "chunker_version", "enrichment_mode",
        "ingestion_schema_version",
    )
    return {
        "point_id": str(point.id),
        "document_id": payload.get("document_id"),
        "chunk_id": payload.get("chunk_id"),
        "parent_id": payload.get("parent_id"),
        "source": payload.get("source"),
        "source_basename": Path(str(payload.get("source", ""))).name,
        "source_hash": payload.get("source_hash"),
        "content_hash": payload.get("content_hash"),
        "version": payload.get("version"),
        "is_active": payload.get("is_active"),
        "tenant_id": metadata.get("tenant_id"),
        "acl_allowed_principals": sorted(payload.get("acl_allowed_principals") or []),
        "acl_denied_principals": sorted(payload.get("acl_denied_principals") or []),
        "page_number": payload.get("page_number"),
        "source_span": {"start_line": metadata.get("start_line"),
                         "end_line": metadata.get("end_line"),
                         "chunk_index": metadata.get("chunk_index")},
        "provenance": {key: payload.get(key) for key in allowed_provenance},
    }


def _scroll_all(client, collection: str) -> list[Any]:
    points = []
    offset = None
    while True:
        page, offset = client.scroll(collection_name=collection, limit=256, offset=offset,
                                     with_payload=True, with_vectors=False)
        points.extend(page)
        if offset is None:
            return points


def _evidence_mappings(dataset, corpus_root: Path, point_summaries: list[dict[str, Any]],
                       points: list[Any]) -> list[dict[str, Any]]:
    from evaluation.dataset import _read_source

    mapping = []
    for case in dataset.cases:
        for evidence in case.expected_evidence:
            source_path = corpus_root / evidence.source_document
            source_rows = _read_source(source_path)
            if evidence.locator.startswith("line:"):
                index = int(evidence.locator.split(":", 1)[1]) - 1
            else:
                index = int(evidence.locator.split(":", 1)[1]) - 1
            if index < 0 or index >= len(source_rows):
                raise ValueError(f"{case.case_id}: locator no longer resolves: {evidence.locator}")
            source_excerpt = source_rows[index]
            if _normalise(evidence.quote) not in _normalise(source_excerpt):
                raise ValueError(f"{case.case_id}: quote does not resolve at {evidence.locator}")
            indexed_excerpt = (_html_visible_body(evidence.quote)
                               if Path(evidence.source_document).suffix.casefold() in {".html", ".htm"}
                               else evidence.quote)
            if Path(evidence.source_document).suffix.casefold() in {".md", ".markdown"}:
                # The Markdown parser removes heading markers before indexing.
                indexed_excerpt = re.sub(r"^\s*#{1,6}\s*", "", indexed_excerpt)
            is_restricted = evidence.source_document == "12_restricted_team.txt"
            tenant = ACL_TENANT if is_restricted else STANDARD_TENANT
            candidates = [item for item in point_summaries
                          if item["source_basename"] == evidence.source_document
                          and item["tenant_id"] == tenant and item["is_active"] is True]
            matched = []
            # Chunk text is inspected transiently only. The retained mapping is the
            # source locator plus stable source/content hashes and current chunk IDs.
            for summary, point in zip(point_summaries, points, strict=True):
                if summary not in candidates:
                    continue
                content = str((point.payload or {}).get("content", ""))
                if _normalise(indexed_excerpt) and _normalise(indexed_excerpt) in _normalise(content):
                    matched.append({"point_id": summary["point_id"], "document_id": summary["document_id"],
                                    "chunk_id": summary["chunk_id"], "content_hash": summary["content_hash"],
                                    "source_span": summary["source_span"]})
            if not matched:
                raise ValueError(f"{case.case_id}: no indexed chunk maps to {evidence.source_document} {evidence.locator}")
            mapping.append({
                "case_id": case.case_id,
                "source_document": evidence.source_document,
                "source_hash": next(item["source_hash"] for item in point_summaries
                                    if item["source_basename"] == evidence.source_document
                                    and item["tenant_id"] == tenant),
                "locator": evidence.locator,
                "quote_sha256": _sha256(evidence.quote.encode("utf-8")),
                "match_method": ("validated_source_locator_then_visible_html_body_match" if
                                 Path(evidence.source_document).suffix.casefold() in {".html", ".htm"} else
                                 "validated_source_locator_then_exact_normalized_quote_in_current_chunk"),
                "indexed_chunks": matched,
            })
    return mapping


def run(*, audit_existing: bool = False) -> dict[str, Any]:
    settings = get_settings()
    collection = require_evaluation_collection(settings.qdrant_collection)
    if settings.validation_max_cost_usd is None or settings.validation_max_cost_usd != 0:
        raise ValueError("Phase 8 bootstrap requires an explicit zero USD validation cap")
    if settings.validation_max_total_tokens is None or settings.validation_max_calls is None:
        raise ValueError("Phase 8 bootstrap requires total-token and call caps")
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    dataset_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    corpus_root = ROOT / "tests" / "fixtures" / "phase7_smoke"
    dataset = validate_dataset(raw, corpus_root=corpus_root, repository_root=ROOT)
    checksum = dataset_checksum(dataset)
    if checksum != "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1":
        raise ValueError("Checked-in dataset checksum differs from the approved 1.0.1 release")
    if dataset_manifest.get("dataset_sha256") != checksum:
        raise ValueError("Dataset sidecar manifest checksum does not match dataset")

    from indexing.embeddings import provider_from_config
    from infrastructure.qdrant_client import get_qdrant_client

    client = get_qdrant_client()
    if client.collection_exists(collection):
        if audit_existing:
            return audit_existing_snapshot(settings=settings, dataset=dataset,
                                           dataset_manifest=dataset_manifest, corpus_root=corpus_root,
                                           client=client, collection=collection)
        raise ValueError(f"Evaluation collection already exists; refusing to overwrite: {collection}")
    if audit_existing:
        raise ValueError("Cannot audit an absent evaluation collection")

    snapshot_root = settings.data_dir / "phase8" / "corpus" / f"{DATASET_ID}-{DATASET_VERSION}"
    if snapshot_root.exists() and any(snapshot_root.rglob("*")):
        # Reuse byte-identical staged source files after an interrupted run, never replace them.
        pass
    snapshot_root.mkdir(parents=True, exist_ok=True)
    indexer = __import__("indexing.qdrant_indexer", fromlist=["QdrantIndexer"]).QdrantIndexer(
        client=client, collection=collection)
    pipeline = IngestionPipeline(indexer=indexer, embedding_provider=provider_from_config())
    documents = []
    source_paths: dict[tuple[str, str], Path] = {}
    expected_sources = dataset_manifest.get("source_documents")
    if not expected_sources:
        raise ValueError("Dataset manifest has no source document inventory")

    for filename in expected_sources:
        source = corpus_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Dataset-referenced source is missing: {filename}")
        raw_bytes = source.read_bytes()
        source_hash = _sha256(raw_bytes)
        restricted = filename == "12_restricted_team.txt"
        tenant = ACL_TENANT if restricted else STANDARD_TENANT
        tenants = [tenant]
        for tenant_id in tenants:
            destination_dir = snapshot_root / tenant_id
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / filename
            if destination.exists():
                if _sha256(destination.read_bytes()) != source_hash:
                    raise ValueError(f"Refusing to overwrite non-matching staged source: {filename}")
            else:
                shutil.copyfile(source, destination)
            source_paths[(filename, tenant_id)] = destination
            access = (AccessControlMetadata(allowed_principals=[ACL_USER_A],
                                            denied_principals=[ACL_USER_B], sensitivity="restricted")
                      if restricted else
                      AccessControlMetadata(allowed_principals=[f"user:{STANDARD_USER}"],
                                            denied_principals=[]))
            documents.append({"filename": filename, "tenant_id": tenant_id,
                              "source_hash": source_hash, "version": _source_version(filename),
                              "access_control": access})

    budget_run_id = f"phase8-bootstrap-{checksum[:16]}"
    ingestion_results = []
    with activate_run_budget(budget_run_id, settings) as budget:
        if budget is None:
            raise ValueError("Provider budget was not activated for evaluation bootstrap")
        for document in documents:
            source = source_paths[(document["filename"], document["tenant_id"])]
            result = pipeline.ingest(source, version=document["version"],
                                     access_control=document["access_control"],
                                     tenant_id=document["tenant_id"])
            if result.status != IngestionStatus.INDEXED:
                raise RuntimeError(f"Ingestion did not index {document['filename']}: {result.status.value}/"
                                   f"{result.error_type or 'no error type'}")
            ingestion_results.append({"filename": document["filename"],
                                      "tenant_id": document["tenant_id"],
                                      "document_id": result.document_id,
                                      "source_hash": result.source_hash,
                                      "version": document["version"],
                                      "chunk_count": result.chunk_count,
                                      "acl_allowed_principals": document["access_control"].allowed_principals,
                                      "acl_denied_principals": document["access_control"].denied_principals})
        budget_snapshot = budget.snapshot()

    if not client.collection_exists(collection):
        raise RuntimeError("Application ingestion did not create the evaluation collection")
    info = client.get_collection(collection)
    point_count = client.count(collection, exact=True).count
    raw_points = _scroll_all(client, collection)
    if len(raw_points) != point_count:
        raise RuntimeError(f"Qdrant scroll count {len(raw_points)} differs from exact count {point_count}")
    point_summaries = [_point_summary(point) for point in raw_points]
    payload_indexes = sorted((getattr(info, "payload_schema", None) or {}).keys())
    required_indexes = sorted(PAYLOAD_INDEX_SCHEMAS)
    missing_indexes = sorted(set(required_indexes) - set(payload_indexes))
    if missing_indexes:
        raise RuntimeError(f"Evaluation collection is missing payload indexes: {missing_indexes}")

    params = info.config.params
    dense_name = settings.dense_vector_name
    sparse_name = settings.sparse_vector_name
    dense_config = params.vectors.get(dense_name) if isinstance(params.vectors, dict) else None
    sparse_config = (params.sparse_vectors or {}).get(sparse_name)
    if dense_config is None or dense_config.size != settings.embedding_dimensions:
        raise RuntimeError("Evaluation dense vector schema does not match configured dimensions")
    if sparse_config is None:
        raise RuntimeError("Evaluation sparse vector schema is missing")

    expected_docs = {(item["filename"], item["tenant_id"]) for item in documents}
    indexed_docs = {(item["source_basename"], item["tenant_id"]) for item in point_summaries
                    if item["is_active"] is True}
    missing_docs = expected_docs - indexed_docs
    if missing_docs:
        raise RuntimeError(f"Sources without active indexed points: {sorted(missing_docs)}")
    expected_tenants = {item["tenant_id"] for item in documents}
    observed_tenants = {item["tenant_id"] for item in point_summaries}
    if expected_tenants != observed_tenants:
        raise RuntimeError(f"Tenant metadata mismatch: expected {expected_tenants}, got {observed_tenants}")
    for document in documents:
        matching = [item for item in point_summaries if item["source_basename"] == document["filename"]
                    and item["tenant_id"] == document["tenant_id"]]
        if any(item["source_hash"] != document["source_hash"] or
               item["version"] != document["version"] or
               item["acl_allowed_principals"] != sorted(document["access_control"].allowed_principals) or
               item["acl_denied_principals"] != sorted(document["access_control"].denied_principals)
               for item in matching):
            raise RuntimeError(f"Point metadata failed source/ACL/version audit: {document['filename']}")
        if any(item["provenance"].get("embedding_dim") != settings.embedding_dimensions or
               not item["provenance"].get("embedder_provider") or
               not item["provenance"].get("embedder_model") for item in matching):
            raise RuntimeError(f"Point embedding provenance failed audit: {document['filename']}")

    evidence_mapping = _evidence_mappings(dataset, corpus_root, point_summaries, raw_points)
    core = {
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "dataset_sha256": checksum, "corpus_id": CORPUS_ID, "corpus_version": CORPUS_VERSION,
        "collection": collection,
        "vector_schema": {
            "dense_name": dense_name, "dense_size": dense_config.size,
            "dense_distance": getattr(dense_config.distance, "value", str(dense_config.distance)),
            "sparse_name": sparse_name,
            "sparse_modifier": getattr(sparse_config.modifier, "value", str(sparse_config.modifier)),
        },
        "required_payload_indexes": required_indexes,
        "documents": sorted(ingestion_results, key=lambda x: (x["filename"], x["tenant_id"])),
        "points": sorted(point_summaries, key=lambda x: x["point_id"]),
        "evidence_to_chunks": evidence_mapping,
        "known_exclusion": "21_scanned_style.pdf is not referenced by any 1.0.1 case and is excluded; OCR is not enabled.",
    }
    snapshot_hash = _canonical_hash(core)
    manifest = {
        **core,
        "snapshot_sha256": snapshot_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(ingestion_results),
        "point_count": point_count,
        "payload_indexes_observed": payload_indexes,
        "validation_budget": budget_snapshot,
        "dataset_case_count": len(dataset.cases),
        "evidence_mapping_count": len(evidence_mapping),
        "audit_status": "passed",
        "hash_method": "SHA-256 over canonical JSON core with sorted keys and compact separators; created_at and budget snapshot excluded.",
    }
    output_dir = settings.data_dir / "phase8" / "snapshots" / snapshot_hash
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                              encoding="utf-8")
    return {"audit_status": "passed", "collection": collection, "snapshot_sha256": snapshot_hash,
            "manifest_path": str(output_dir / "manifest.json"), "document_count": len(ingestion_results),
            "point_count": point_count, "evidence_mapping_count": len(evidence_mapping),
            "payload_indexes": len(payload_indexes), "budget": budget_snapshot}


def audit_existing_snapshot(*, settings, dataset, dataset_manifest, corpus_root: Path,
                            client, collection: str) -> dict[str, Any]:
    """Audit existing points only; does not ingest, embed, upsert, or delete."""
    snapshot_root = settings.data_dir / "phase8" / "corpus" / f"{DATASET_ID}-{DATASET_VERSION}"
    if not snapshot_root.is_dir():
        raise FileNotFoundError("The staged Phase 8 corpus snapshot is missing")
    raw_points = _scroll_all(client, collection)
    point_count = client.count(collection, exact=True).count
    if len(raw_points) != point_count:
        raise RuntimeError("Qdrant scroll count does not match exact point count")
    point_summaries = [_point_summary(point) for point in raw_points]
    info = client.get_collection(collection)
    payload_indexes = sorted((getattr(info, "payload_schema", None) or {}).keys())
    required_indexes = sorted(PAYLOAD_INDEX_SCHEMAS)
    missing_indexes = sorted(set(required_indexes) - set(payload_indexes))
    if missing_indexes:
        raise RuntimeError(f"Evaluation collection is missing payload indexes: {missing_indexes}")
    params = info.config.params
    dense_name, sparse_name = settings.dense_vector_name, settings.sparse_vector_name
    dense_config = params.vectors.get(dense_name) if isinstance(params.vectors, dict) else None
    sparse_config = (params.sparse_vectors or {}).get(sparse_name)
    if dense_config is None or dense_config.size != settings.embedding_dimensions or sparse_config is None:
        raise RuntimeError("Evaluation dense/sparse vector schema failed audit")

    documents = []
    for filename in dataset_manifest["source_documents"]:
        source = corpus_root / filename
        staged_tenant = ACL_TENANT if filename == "12_restricted_team.txt" else STANDARD_TENANT
        staged = snapshot_root / staged_tenant / filename
        if not source.is_file() or not staged.is_file():
            raise FileNotFoundError(f"Missing source or staged file for {filename}")
        source_hash = _sha256(source.read_bytes())
        if _sha256(staged.read_bytes()) != source_hash:
            raise ValueError(f"Staged source hash mismatch for {filename}")
        restricted = filename == "12_restricted_team.txt"
        access = (AccessControlMetadata(allowed_principals=[ACL_USER_A],
                                        denied_principals=[ACL_USER_B], sensitivity="restricted")
                  if restricted else
                  AccessControlMetadata(allowed_principals=[f"user:{STANDARD_USER}"],
                                        denied_principals=[]))
        tenant = ACL_TENANT if restricted else STANDARD_TENANT
        matching = [item for item in point_summaries if item["source_basename"] == filename
                    and item["tenant_id"] == tenant]
        if not matching:
            raise RuntimeError(f"No indexed point for expected source {filename}")
        if any(item["source_hash"] != source_hash or item["version"] != _source_version(filename) or
               item["acl_allowed_principals"] != sorted(access.allowed_principals) or
               item["acl_denied_principals"] != sorted(access.denied_principals) or
               not item["is_active"] for item in matching):
            raise RuntimeError(f"Point source/tenant/ACL/version audit failed for {filename}")
        if any(item["provenance"].get("embedding_dim") != settings.embedding_dimensions or
               not item["provenance"].get("embedder_provider") or
               not item["provenance"].get("embedder_model") for item in matching):
            raise RuntimeError(f"Point embedding provenance failed audit for {filename}")
        documents.append({"filename": filename, "tenant_id": tenant,
                          "document_id": matching[0]["document_id"], "source_hash": source_hash,
                          "version": _source_version(filename), "chunk_count": len(matching),
                          "acl_allowed_principals": sorted(access.allowed_principals),
                          "acl_denied_principals": sorted(access.denied_principals)})

    expected_tenants = {item["tenant_id"] for item in documents}
    observed_tenants = {item["tenant_id"] for item in point_summaries}
    if expected_tenants != observed_tenants:
        raise RuntimeError(f"Unexpected tenant IDs in evaluation snapshot: {observed_tenants}")
    expected_docs = {(item["filename"], item["tenant_id"]) for item in documents}
    actual_docs = {(item["source_basename"], item["tenant_id"]) for item in point_summaries}
    if expected_docs != actual_docs:
        raise RuntimeError("Indexed document inventory differs from dataset source inventory")

    evidence_mapping = _evidence_mappings(dataset, corpus_root, point_summaries, raw_points)
    core = {
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "dataset_sha256": dataset_checksum(dataset), "corpus_id": CORPUS_ID,
        "corpus_version": CORPUS_VERSION, "collection": collection,
        "vector_schema": {
            "dense_name": dense_name, "dense_size": dense_config.size,
            "dense_distance": getattr(dense_config.distance, "value", str(dense_config.distance)),
            "sparse_name": sparse_name,
            "sparse_modifier": getattr(sparse_config.modifier, "value", str(sparse_config.modifier)),
        },
        "required_payload_indexes": required_indexes,
        "documents": sorted(documents, key=lambda x: (x["filename"], x["tenant_id"])),
        "points": sorted(point_summaries, key=lambda x: x["point_id"]),
        "evidence_to_chunks": evidence_mapping,
        "known_exclusion": "21_scanned_style.pdf is not referenced by any 1.0.1 case and is excluded; OCR is not enabled.",
    }
    snapshot_hash = _canonical_hash(core)
    output_dir = settings.data_dir / "phase8" / "snapshots" / snapshot_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("snapshot_sha256") != snapshot_hash:
            raise ValueError("Existing snapshot directory contains a different manifest")
        return {"audit_status": previous.get("audit_status"), "collection": collection,
                "snapshot_sha256": snapshot_hash, "manifest_path": str(manifest_path),
                "document_count": previous.get("document_count"),
                "point_count": previous.get("point_count"),
                "evidence_mapping_count": previous.get("evidence_mapping_count"),
                "payload_indexes": len(previous.get("payload_indexes_observed", [])),
                "budget_usage": previous.get("validation_budget", {}).get("usage", "UNKNOWN")}
    manifest = {
        **core, "snapshot_sha256": snapshot_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents), "point_count": point_count,
        "payload_indexes_observed": payload_indexes,
        "validation_budget": {
            "status": "active during ingestion; usage snapshot unavailable because the initial audit stopped before manifest finalization",
            "max_cost_usd": settings.validation_max_cost_usd,
            "max_total_tokens": settings.validation_max_total_tokens,
            "max_calls": settings.validation_max_calls,
            "usage": "UNKNOWN",
        },
        "dataset_case_count": len(dataset.cases), "evidence_mapping_count": len(evidence_mapping),
        "audit_status": "passed",
        "hash_method": "SHA-256 over canonical JSON core with sorted keys and compact separators; created_at and budget snapshot excluded.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"audit_status": "passed", "collection": collection, "snapshot_sha256": snapshot_hash,
            "manifest_path": str(manifest_path), "document_count": len(documents),
            "point_count": point_count, "evidence_mapping_count": len(evidence_mapping),
            "payload_indexes": len(payload_indexes), "budget_usage": "UNKNOWN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-existing", action="store_true",
                        help="Audit an already-ingested collection without provider calls or Qdrant writes")
    args = parser.parse_args()
    print(json.dumps(run(audit_existing=args.audit_existing), sort_keys=True))
