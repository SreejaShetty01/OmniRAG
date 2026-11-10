"""Read-only audit of an ingested M1 corpus; snapshot creation is opt-in."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from uuid import uuid4
from pathlib import Path
import sys
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from indexing.qdrant_indexer import PAYLOAD_INDEX_SCHEMAS
from scripts.realworld.prepare_corpus import CORPUS_ROOT, MANIFEST, validate_corpus


def _value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def audit_collection(client, manifest: dict, corpus_root: Path, *, progress: dict,
                     user_id: str, tenant_id: str, evaluation_manifest: Path = MANIFEST) -> dict:
    integrity = validate_corpus(manifest, corpus_root, evaluation_manifest)
    collection = manifest.get("target_collection", "")
    errors = list(integrity["errors"])
    if not collection.startswith("omnirag_eval_") or collection in {"omnirag", "omnirag_eval_phase8_20261104"}:
        errors.append("target collection is not an isolated M1 evaluation collection")
    if progress.get("collection") != collection or progress.get("status") != "completed":
        errors.append("ingestion progress does not prove a completed run for this collection")
    if progress.get("content_set_sha256") != manifest.get("content_set_sha256"):
        errors.append("ingestion progress content-set checksum does not match manifest")

    info = client.get_collection(collection_name=collection)
    params = _value(_value(_value(info, "config"), "params"), "vectors")
    dense_name, sparse_name = "dense", "sparse"
    dense = _value(params, dense_name)
    dense_size = _value(dense, "size")
    dense_distance = _value(_value(dense, "distance"), "value", _value(dense, "distance"))
    sparse_config = _value(_value(_value(info, "config"), "params"), "sparse_vectors")
    sparse = _value(sparse_config, sparse_name)
    sparse_modifier = _value(_value(sparse, "modifier"), "value", _value(sparse, "modifier"))
    if dense is None or dense_size != 768 or str(dense_distance).casefold() != "cosine":
        errors.append("dense collection vector must be dense/768/Cosine")
    if sparse is None or str(sparse_modifier).casefold() != "idf":
        errors.append("sparse collection vector must be sparse/IDF")

    payload_schema = _value(info, "payload_schema", {}) or {}
    payload_indexes = sorted(payload_schema)
    missing_indexes = sorted(set(PAYLOAD_INDEX_SCHEMAS) - set(payload_indexes))
    if missing_indexes:
        errors.append("required payload indexes are missing")

    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(collection_name=collection, offset=offset,
            limit=512, with_payload=True, with_vectors=False)
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    active = []
    inactive_count = 0
    for point in points:
        payload = _value(point, "payload", {}) or {}
        if not payload.get("is_active", False):
            inactive_count += 1
            continue
        active.append(point)
    if inactive_count:
        errors.append("inactive or staged points are present in the evaluation collection")

    manifest_rows = {row["sha256"]: row for row in manifest.get("sources", [])}
    progress_rows = {row.get("source_sha256"): row for row in progress.get("schedule", [])}
    by_hash: dict[str, list[dict]] = {}
    seen_point_ids: set[str] = set()
    duplicate_point_ids = 0
    acl_checks = 0
    tenant_checks = 0
    version_checks = 0
    provenance_checks = 0
    provenance_failures = 0
    for point in active:
        point_id = str(_value(point, "id", ""))
        if point_id in seen_point_ids:
            duplicate_point_ids += 1
        seen_point_ids.add(point_id)
        payload = _value(point, "payload", {}) or {}
        source_hash = payload.get("source_hash")
        by_hash.setdefault(str(source_hash), []).append(payload)
        source = manifest_rows.get(source_hash)
        if source is None:
            errors.append("active Qdrant point has an unmanifested source hash")
            continue
        run_row = progress_rows.get(source_hash)
        expected_document_id = run_row.get("api_document_id") if run_row else None
        if expected_document_id and payload.get("document_id") != expected_document_id:
            errors.append("Qdrant document ID differs from the API ingestion record")
        if payload.get("version") != source.get("repository_commit"):
            errors.append("Qdrant version differs from the pinned repository commit")
        version_checks += 1
        metadata = payload.get("metadata") or {}
        allowed = payload.get("acl_allowed_principals") or []
        denied = payload.get("acl_denied_principals") or []
        if allowed != [f"user:{user_id}"] or denied:
            errors.append("Qdrant ACL principal payload is incorrect")
        acl_checks += 1
        if metadata.get("tenant_id") != tenant_id:
            errors.append("Qdrant tenant payload is incorrect")
        tenant_checks += 1
        if (payload.get("embedder_provider") != "openai_compatible" or
                payload.get("embedder_model") != "nomic-embed-text" or
                payload.get("embedding_dim") != 768):
            provenance_failures += 1
            errors.append("Qdrant embedding provenance differs from configured embedding space")
        provenance_checks += 1
    if duplicate_point_ids:
        errors.append("duplicate Qdrant point IDs were returned")

    missing_hashes = sorted(set(manifest_rows) - set(by_hash))
    extra_hashes = sorted(set(by_hash) - set(manifest_rows))
    if missing_hashes:
        errors.append("one or more manifest source hashes have no active Qdrant points")
    expected_chunk_count = 0
    expected_chunk_count_known = True
    for source_hash in manifest_rows:
        row = progress_rows.get(source_hash)
        if row is None or row.get("status") not in {"indexed", "unchanged"}:
            errors.append("a manifest source lacks a successful per-file ingestion status")
            expected_chunk_count_known = False
        elif not isinstance(row.get("chunk_count"), int) or row["chunk_count"] <= 0:
            expected_chunk_count_known = False
        else:
            expected_chunk_count += row["chunk_count"]
    actual_chunk_count = len(active)
    if expected_chunk_count_known and expected_chunk_count != actual_chunk_count:
        errors.append("Qdrant active point count differs from per-file ingestion chunk counts")

    return {
        "valid": not errors, "collection": collection,
        "source_file_count": len(manifest_rows), "source_hashes_accounted": len(set(manifest_rows) - set(missing_hashes)),
        "missing_source_hashes": missing_hashes, "unmanifested_source_hashes": extra_hashes,
        "actual_active_point_count": actual_chunk_count,
        "expected_chunk_count": expected_chunk_count if expected_chunk_count_known else None,
        "expected_chunk_count_status": "verified" if expected_chunk_count_known else "unknown_from_ingestion_responses",
        "inactive_point_count": inactive_count, "duplicate_point_ids": duplicate_point_ids,
        "acl_points_checked": acl_checks, "tenant_points_checked": tenant_checks,
        "version_points_checked": version_checks, "provenance_points_checked": provenance_checks,
        "provenance_failures": provenance_failures,
        "dense_vector": {"name": dense_name, "size": dense_size, "distance": str(dense_distance)},
        "sparse_vector": {"name": sparse_name, "modifier": str(sparse_modifier)},
        "payload_indexes": payload_indexes, "missing_payload_indexes": missing_indexes,
        "collection_disk_data_size": _value(info, "disk_data_size"),
        "collection_ram_data_size": _value(info, "ram_data_size"),
        "integrity": integrity, "errors": errors,
    }


def collection_sample_fingerprint(client, collection: str, *, sample_limit: int = 64) -> dict:
    """Read-only stable point-count/sample hash matching the Phase 8 baseline rule."""
    if not client.collection_exists(collection):
        raise RuntimeError(f"Protected collection is missing: {collection}")
    points = []
    offset = None
    while True:
        batch, next_offset = client.scroll(collection_name=collection, offset=offset,
            limit=512, with_payload=True, with_vectors=False)
        points.extend(batch)
        if next_offset is None:
            break
        offset = next_offset
    ordered = sorted(points, key=lambda point: str(_value(point, "id", "")))
    selected = ordered[:sample_limit]
    fields = ("document_id", "chunk_id", "source_hash", "content_hash", "generation", "is_active")
    rows = [{"id": str(_value(point, "id", "")), **{
        key: (_value(point, "payload", {}) or {}).get(key) for key in fields}}
        for point in selected]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"collection": collection, "point_count": len(points),
            "sampled_point_count": len(rows), "sample_rule": f"first {sample_limit} by point ID",
            "sha256": hashlib.sha256(canonical).hexdigest()}


def record_protected_baselines(client, output_path: Path) -> dict:
    """Record dev/Phase8 collection counts and stable sampled hashes without writing Qdrant."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite protected baseline record: {output_path}")
    record = {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "collections": [collection_sample_fingerprint(client, name)
                              for name in ("omnirag", "omnirag_eval_phase8_20261104")]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    temp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, output_path)
    return record


def compare_protected_baselines(client, baseline_path: Path) -> dict:
    """Read-only compare dev/Phase8 point counts and samples against a pre-ingest record."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old = {row["collection"]: row for row in baseline.get("collections", [])}
    current = [collection_sample_fingerprint(client, name)
               for name in ("omnirag", "omnirag_eval_phase8_20261104")]
    diffs = []
    for row in current:
        prior = old.get(row["collection"])
        if prior is None or any(prior.get(key) != row.get(key)
                                for key in ("point_count", "sha256", "sampled_point_count")):
            diffs.append(row["collection"])
    return {"preserved": not diffs, "changed_collections": diffs,
            "baseline_recorded_at": baseline.get("recorded_at"), "current": current}


def create_snapshot_and_hash(client, *, collection: str, qdrant_url: str,
                             output_path: Path, api_key: str | None = None,
                             timeout: float = 600.0) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing snapshot: {output_path}")
    snapshot = client.create_snapshot(collection_name=collection, wait=True)
    name = _value(snapshot, "name")
    if not name:
        raise RuntimeError("Qdrant did not return a snapshot name")
    parts = urlsplit(qdrant_url)
    endpoint = (f"{parts.scheme}://{parts.netloc}/collections/{quote(collection, safe='')}/"
                f"snapshots/{quote(str(name), safe='')}")
    headers = {"Accept": "application/octet-stream"}
    if api_key:
        headers["api-key"] = api_key
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(Request(endpoint, headers=headers), timeout=timeout) as response:
            content = response.read()
        temp_path.write_bytes(content)
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {"snapshot_name": str(name), "snapshot_path": str(output_path),
            "snapshot_bytes": output_path.stat().st_size, "snapshot_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--evaluation-manifest", type=Path,
                        default=ROOT / "evaluation/realworld/corpus_manifest.json")
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--principal-user-id", default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--snapshot-path", type=Path, default=None)
    parser.add_argument("--record-protected-baselines", type=Path, default=None)
    parser.add_argument("--compare-protected-baselines", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    from qdrant_client import QdrantClient
    client = QdrantClient(url=args.qdrant_url)
    try:
        if args.record_protected_baselines:
            record = record_protected_baselines(client, args.record_protected_baselines)
            print(json.dumps({"status": "recorded", **record}, indent=2))
            return 0
        preservation = None
        if args.compare_protected_baselines:
            preservation = compare_protected_baselines(client, args.compare_protected_baselines)
        if args.progress is None:
            raise SystemExit("--progress is required unless recording protected baselines")
        if not args.principal_user_id or not args.tenant_id:
            raise SystemExit("--principal-user-id and --tenant-id are required for corpus audit")
        progress = json.loads(args.progress.read_text(encoding="utf-8"))
        report = audit_collection(client, manifest, args.corpus_root, progress=progress,
            user_id=args.principal_user_id, tenant_id=args.tenant_id,
            evaluation_manifest=args.evaluation_manifest)
        if preservation is not None:
            report["protected_collection_preservation"] = preservation
            if not preservation["preserved"]:
                report["valid"] = False
                report["errors"].append("development or Phase 8 evaluation collection changed")
        if report["valid"] and args.snapshot_path:
            snapshot = create_snapshot_and_hash(client, collection=manifest["target_collection"],
                qdrant_url=args.qdrant_url, output_path=args.snapshot_path)
            report["snapshot"] = snapshot
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
