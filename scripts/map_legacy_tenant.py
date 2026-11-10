"""Explicit, dry-run-first migration for records without tenant attribution."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TABLES = {
    "collection": ("collections", "collection_id"),
    "conversation": ("conversations", "conversation_id"),
    "document": ("documents", "document_id"),
}


def _audit(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(), **event}) + "\n")


def run_mapping(mapping: dict, *, sqlite_path: Path, qdrant_client, collection: str,
                apply: bool, audit_path: Path) -> list[dict]:
    sqlite_rows = mapping.get("sqlite", [])
    qdrant_rows = mapping.get("qdrant", [])
    actions: list[dict] = []
    if sqlite_rows:
        if not sqlite_path.is_file():
            raise FileNotFoundError(f"SQLite database does not exist: {sqlite_path}")
        with sqlite3.connect(sqlite_path) as db:
            for item in sqlite_rows:
                kind, record_id = item.get("kind"), item.get("id")
                tenant_id, owner_id = item.get("tenant_id"), item.get("owner_id")
                if kind not in TABLES or not all((record_id, tenant_id, owner_id)):
                    raise ValueError("Each SQLite mapping requires kind, id, tenant_id and owner_id")
                table, id_column = TABLES[kind]
                row = db.execute(f"SELECT user_id,tenant_id FROM {table} WHERE {id_column}=?",
                                 (record_id,)).fetchone()
                if row is None:
                    raise ValueError(f"Mapped SQLite record not found: {kind}/{record_id}")
                current_owner, current_tenant = row
                if current_owner != owner_id:
                    raise ValueError(f"Owner mismatch for {kind}/{record_id}; refusing tenant mapping")
                if current_tenant and current_tenant != tenant_id:
                    raise ValueError(f"Record {kind}/{record_id} already belongs to another tenant")
                action = "already_mapped" if current_tenant == tenant_id else "map_sqlite"
                actions.append({"kind": kind, "id": record_id, "tenant_id": tenant_id,
                                "owner_id": owner_id, "action": action})

        if apply:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = sqlite_path.with_name(f"{sqlite_path.name}.{stamp}.bak")
            shutil.copy2(sqlite_path, backup)
            with sqlite3.connect(sqlite_path) as db:
                for item in sqlite_rows:
                    table, id_column = TABLES[item["kind"]]
                    db.execute(f"UPDATE {table} SET tenant_id=? WHERE {id_column}=? AND user_id=? "
                               "AND (tenant_id IS NULL OR tenant_id='')",
                               (item["tenant_id"], item["id"], item["owner_id"]))
            _audit(audit_path, {"action": "sqlite_backup", "backup": str(backup)})

    if qdrant_rows and qdrant_client is None:
        raise ValueError("Qdrant mappings require a configured Qdrant client")
    if qdrant_rows:
        from qdrant_client import models
        for item in qdrant_rows:
            document_id, tenant_id, owner_id = item.get("document_id"), item.get("tenant_id"), item.get("owner_id")
            if not all((document_id, tenant_id, owner_id)):
                raise ValueError("Each Qdrant mapping requires document_id, tenant_id and owner_id")
            records, _ = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id))]),
                limit=1000, with_payload=True, with_vectors=False,
            )
            if not records:
                raise ValueError(f"No Qdrant points found for document {document_id}")
            needs_mapping = False
            for record in records:
                payload = record.payload or {}
                metadata = dict(payload.get("metadata") or {})
                current_tenant = metadata.get("tenant_id")
                if current_tenant and current_tenant != tenant_id:
                    raise ValueError(f"Qdrant document {document_id} already belongs to another tenant")
                action = "already_mapped" if current_tenant == tenant_id else "map_qdrant"
                actions.append({"kind": "qdrant_point", "id": str(record.id),
                                "document_id": document_id, "tenant_id": tenant_id,
                                "owner_id": owner_id, "action": action})
                needs_mapping = needs_mapping or action == "map_qdrant"
            if apply and needs_mapping:
                missing_tenant = models.Filter(must=[
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="metadata.tenant_id")),
                ])
                qdrant_client.set_payload(
                    collection_name=collection, payload={"tenant_id": tenant_id}, key="metadata",
                    points=models.FilterSelector(filter=missing_tenant), wait=True,
                )
                mapped_filter = models.Filter(must=[
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                    models.FieldCondition(key="metadata.tenant_id", match=models.MatchValue(value=tenant_id)),
                ])
                allowed = [f"user:{owner_id}"]
                qdrant_client.set_payload(collection_name=collection,
                                          payload={"allowed_principals": allowed}, key="access_control",
                                          points=models.FilterSelector(filter=mapped_filter), wait=True)
                qdrant_client.set_payload(collection_name=collection,
                                          payload={"acl_allowed_principals": allowed},
                                          points=models.FilterSelector(filter=mapped_filter), wait=True)
                for action in actions:
                    if (action["kind"] == "qdrant_point" and action["document_id"] == document_id
                            and action["action"] == "map_qdrant"):
                        _audit(audit_path, {"mode": "apply", **action})

    for action in actions:
        if apply and action["kind"] == "qdrant_point" and action["action"] == "map_qdrant":
            continue
        _audit(audit_path, {"mode": "apply" if apply else "dry_run", **action})
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path, help="Explicit JSON record-to-tenant mapping")
    parser.add_argument("--sqlite", type=Path, help="SQLite file; defaults to OMNIRAG_DATA_DIR/omnirag.sqlite3")
    parser.add_argument("--collection", help="Qdrant collection; defaults to OMNIRAG_QDRANT_COLLECTION")
    parser.add_argument("--audit-log", type=Path, default=Path("docs/audit/legacy_tenant_mapping.jsonl"))
    parser.add_argument("--apply", action="store_true", help="Apply mappings (default is dry-run)")
    args = parser.parse_args()

    from config.settings import get_settings
    settings = get_settings()
    sqlite_path = args.sqlite or (settings.data_dir / "omnirag.sqlite3")
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    qdrant_client = None
    if mapping.get("qdrant"):
        from infrastructure.qdrant_client import get_qdrant_client
        qdrant_client = get_qdrant_client()
    actions = run_mapping(mapping, sqlite_path=sqlite_path, qdrant_client=qdrant_client,
                          collection=args.collection or settings.qdrant_collection,
                          apply=args.apply, audit_path=args.audit_log)
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "actions": len(actions),
                      "already_mapped": sum(row["action"] == "already_mapped" for row in actions)}, indent=2))


if __name__ == "__main__":
    main()
