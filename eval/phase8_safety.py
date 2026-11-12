"""Safety checks shared by Phase 8 evaluation bootstrap and execution tools."""

import hashlib
import json
import re


EVALUATION_COLLECTION_PREFIX = "omnirag_eval_"


def require_evaluation_collection(collection: str) -> str:
    """Accept only a dedicated evaluation collection name; reject dev/invalid targets."""
    if not isinstance(collection, str) or not re.fullmatch(
        r"omnirag_eval_[a-z0-9][a-z0-9_-]{0,80}", collection
    ):
        raise ValueError("Phase 8 tools require an omnirag_eval_ collection name")
    if collection == "omnirag":
        raise ValueError("Development collection is forbidden for Phase 8 tools")
    return collection


def verify_snapshot_manifest(manifest: dict) -> dict:
    """Verify the B2 canonical hash and audited release identity before use."""
    if not isinstance(manifest, dict) or manifest.get("audit_status") != "passed":
        raise ValueError("Phase 8 requires a passed evaluation snapshot audit")
    if (manifest.get("dataset_id"), manifest.get("dataset_version"), manifest.get("dataset_sha256")) != (
        "omnirag-phase8-eval", "1.0.1",
        "7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1",
    ):
        raise ValueError("Snapshot does not match the approved immutable evaluation dataset")
    require_evaluation_collection(manifest.get("collection"))
    core_keys = (
        "dataset_id", "dataset_version", "dataset_sha256", "corpus_id", "corpus_version",
        "collection", "vector_schema", "required_payload_indexes", "documents", "points",
        "evidence_to_chunks", "known_exclusion",
    )
    if any(key not in manifest for key in core_keys):
        raise ValueError("Snapshot manifest is missing a hashed core field")
    core = {key: manifest[key] for key in core_keys}
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != manifest.get("snapshot_sha256"):
        raise ValueError("Snapshot canonical checksum does not match its manifest")
    if manifest.get("point_count") != len(manifest.get("points", [])):
        raise ValueError("Snapshot point count does not match point manifest")
    if manifest.get("document_count") != len(manifest.get("documents", [])):
        raise ValueError("Snapshot document count does not match document manifest")
    return manifest
