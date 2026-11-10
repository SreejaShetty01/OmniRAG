import json
import sqlite3

from scripts.map_legacy_tenant import run_mapping


def test_tenant_mapping_dry_run_apply_and_idempotent_rerun(tmp_path):
    database = tmp_path / "product.sqlite3"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE collections(collection_id TEXT, user_id TEXT, tenant_id TEXT DEFAULT '')")
        db.execute("INSERT INTO collections VALUES('c1','owner-a','')")
    mapping = {"sqlite": [{"kind": "collection", "id": "c1", "owner_id": "owner-a",
                           "tenant_id": "tenant-a"}]}
    audit = tmp_path / "audit.jsonl"

    dry = run_mapping(mapping, sqlite_path=database, qdrant_client=None,
                      collection="shared", apply=False, audit_path=audit)
    assert dry[0]["action"] == "map_sqlite"
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT tenant_id FROM collections WHERE collection_id='c1'").fetchone()[0] == ""

    applied = run_mapping(mapping, sqlite_path=database, qdrant_client=None,
                           collection="shared", apply=True, audit_path=audit)
    assert applied[0]["action"] == "map_sqlite"
    assert list(tmp_path.glob("product.sqlite3.*.bak"))
    rerun = run_mapping(mapping, sqlite_path=database, qdrant_client=None,
                        collection="shared", apply=True, audit_path=audit)
    assert rerun[0]["action"] == "already_mapped"
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT tenant_id FROM collections WHERE collection_id='c1'").fetchone()[0] == "tenant-a"
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("mode") == "dry_run" for event in events)
    assert any(event.get("action") == "sqlite_backup" for event in events)
