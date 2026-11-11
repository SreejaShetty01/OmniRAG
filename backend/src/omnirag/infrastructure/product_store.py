"""Small durable SQLite store for product preferences and UI-owned metadata."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from omnirag.config.settings import get_settings

_UNSET = object()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProductStore:
    def __init__(self, database_path: Path | str | None = None):
        settings = get_settings()
        self.database_path = Path(database_path or settings.data_dir / "omnirag.sqlite3")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_lock = RLock()
        self.initialize()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    def initialize(self) -> None:
        with self._schema_lock, self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT PRIMARY KEY, preferred_name TEXT, onboarding_skipped INTEGER NOT NULL DEFAULT 0,
                    default_profile TEXT NOT NULL DEFAULT 'core', escalation_preference TEXT NOT NULL DEFAULT 'when_needed',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
                    selected_profile TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                    response_json TEXT, profile TEXT, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, tenant_id TEXT NOT NULL DEFAULT '',
                    UNIQUE(user_id, name)
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
                    source_path TEXT NOT NULL, filename TEXT NOT NULL, version TEXT, collection_id TEXT,
                    status TEXT NOT NULL, source_type TEXT, document_type TEXT, source_hash TEXT,
                    chunk_count INTEGER NOT NULL DEFAULT 0, size_bytes INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(user_id, updated_at DESC);
            """)
            # Add tenant ownership to databases created by earlier builds.
            for table in ("conversations", "collections"):
                columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
                if "tenant_id" not in columns:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''")

    def preferences(self, user_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM preferences WHERE user_id=?", (user_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO preferences(user_id,updated_at) VALUES(?,?)", (user_id, _now()))
                return {"user_id": user_id, "preferred_name": None, "onboarding_skipped": False,
                        "default_profile": "core", "escalation_preference": "when_needed"}
            result = dict(row)
            result["onboarding_skipped"] = bool(result["onboarding_skipped"])
            return result

    def update_preferences(self, user_id: str, *, preferred_name: Any = _UNSET,
                           onboarding_skipped: Any = _UNSET, default_profile: Any = _UNSET,
                           escalation_preference: Any = _UNSET) -> dict[str, Any]:
        current = self.preferences(user_id)
        values = {
            "preferred_name": current["preferred_name"] if preferred_name is _UNSET else preferred_name,
            "onboarding_skipped": current["onboarding_skipped"] if onboarding_skipped is _UNSET else onboarding_skipped,
            "default_profile": current["default_profile"] if default_profile is _UNSET else default_profile,
            "escalation_preference": current["escalation_preference"] if escalation_preference is _UNSET else escalation_preference,
        }
        with self._connect() as db:
            db.execute("""INSERT INTO preferences(user_id,preferred_name,onboarding_skipped,default_profile,
                escalation_preference,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                preferred_name=excluded.preferred_name,onboarding_skipped=excluded.onboarding_skipped,
                default_profile=excluded.default_profile,escalation_preference=excluded.escalation_preference,
                updated_at=excluded.updated_at""",
                (user_id, values["preferred_name"], int(values["onboarding_skipped"]),
                 values["default_profile"], values["escalation_preference"], _now()))
        return self.preferences(user_id)

    def create_conversation(self, user_id: str, title: str, profile: str,
                            conversation_id: str | None = None, tenant_id: str = "") -> str:
        conversation_id, now = conversation_id or str(uuid4()), _now()
        with self._connect() as db:
            db.execute("INSERT INTO conversations(conversation_id,user_id,title,selected_profile,created_at,updated_at,tenant_id) "
                       "VALUES(?,?,?,?,?,?,?)",
                       (conversation_id, user_id, title, profile, now, now, tenant_id))
        return conversation_id

    def conversation(self, user_id: str, conversation_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as db:
            sql = "SELECT * FROM conversations WHERE user_id=? AND conversation_id=?"
            params: tuple[Any, ...] = (user_id, conversation_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            row = db.execute(sql, params).fetchone()
            return dict(row) if row else None

    def list_conversations(self, user_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            sql = "SELECT conversation_id,title,selected_profile,created_at,updated_at FROM conversations WHERE user_id=?"
            params: tuple[Any, ...] = (user_id,)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            rows = db.execute(sql + " ORDER BY updated_at DESC", params).fetchall()
            return [dict(row) for row in rows]

    def save_chat_turn(self, user_id: str, conversation_id: str, user_text: str,
                       answer: dict[str, Any], selected_profile: str,
                       execution_profile: str, tenant_id: str | None = None) -> tuple[str, str]:
        user_message_id, assistant_message_id, now = str(uuid4()), str(uuid4()), _now()
        with self._connect() as db:
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
                       (user_message_id, conversation_id, user_id, "user", user_text, None, selected_profile, now))
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)",
                       (assistant_message_id, conversation_id, user_id, "assistant", answer.get("answer", ""),
                        json.dumps(answer, ensure_ascii=False), execution_profile, now))
            sql = "UPDATE conversations SET selected_profile=?,updated_at=? WHERE conversation_id=? AND user_id=?"
            params: tuple[Any, ...] = (selected_profile, now, conversation_id, user_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            db.execute(sql, params)
        return user_message_id, assistant_message_id

    def messages(self, user_id: str, conversation_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            sql = ("SELECT message_id,role,content,response_json,profile,created_at FROM messages "
                   "WHERE user_id=? AND conversation_id=?")
            params: tuple[Any, ...] = (user_id, conversation_id)
            if tenant_id is not None:
                sql += " AND EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id=messages.conversation_id " \
                       "AND c.user_id=messages.user_id AND c.tenant_id=?)"
                params += (tenant_id,)
            rows = db.execute(sql + " ORDER BY rowid", params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["response"] = json.loads(item.pop("response_json")) if item["response_json"] else None
                result.append(item)
            return result

    def delete_conversation(self, user_id: str, conversation_id: str, tenant_id: str | None = None) -> bool:
        with self._connect() as db:
            sql = "DELETE FROM conversations WHERE user_id=? AND conversation_id=?"
            params: tuple[Any, ...] = (user_id, conversation_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            cursor = db.execute(sql, params)
            return cursor.rowcount > 0

    def create_collection(self, user_id: str, name: str, description: str, tenant_id: str = "") -> str:
        collection_id, now = str(uuid4()), _now()
        with self._connect() as db:
            db.execute("INSERT INTO collections(collection_id,user_id,name,description,created_at,tenant_id) "
                       "VALUES(?,?,?,?,?,?)", (collection_id, user_id, name, description, now, tenant_id))
        return collection_id

    def collection(self, user_id: str, collection_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as db:
            sql = "SELECT * FROM collections WHERE user_id=? AND collection_id=?"
            params: tuple[Any, ...] = (user_id, collection_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            row = db.execute(sql, params).fetchone()
            return dict(row) if row else None

    def list_collections(self, user_id: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""SELECT c.collection_id,c.name,c.description,c.created_at,
                COUNT(d.document_id) AS document_count FROM collections c LEFT JOIN documents d
                ON d.collection_id=c.collection_id AND d.user_id=c.user_id WHERE c.user_id=?
                AND (? IS NULL OR c.tenant_id=?) GROUP BY c.collection_id ORDER BY c.name""",
                              (user_id, tenant_id, tenant_id)).fetchall()
            return [dict(row) for row in rows]

    def delete_collection(self, user_id: str, collection_id: str, tenant_id: str | None = None) -> bool:
        with self._connect() as db:
            sql = "DELETE FROM collections WHERE user_id=? AND collection_id=?"
            params: tuple[Any, ...] = (user_id, collection_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            cursor = db.execute(sql, params)
            if cursor.rowcount:
                db.execute("UPDATE documents SET collection_id=NULL WHERE user_id=? AND collection_id=? AND tenant_id=?",
                           (user_id, collection_id, tenant_id or ""))
            return cursor.rowcount > 0

    def upsert_document(self, *, document_id: str, user_id: str, tenant_id: str,
                        source_path: str, filename: str, version: str | None,
                        collection_id: str | None, status: str, source_type: str | None,
                        document_type: str | None, source_hash: str | None, chunk_count: int,
                        size_bytes: int, error_message: str | None = None) -> None:
        now = _now()
        with self._connect() as db:
            db.execute("""INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET user_id=excluded.user_id,tenant_id=excluded.tenant_id,
                source_path=excluded.source_path,filename=excluded.filename,version=excluded.version,
                collection_id=excluded.collection_id,status=excluded.status,source_type=excluded.source_type,
                document_type=excluded.document_type,source_hash=excluded.source_hash,chunk_count=excluded.chunk_count,
                size_bytes=excluded.size_bytes,error_message=excluded.error_message,updated_at=excluded.updated_at""",
                (document_id,user_id,tenant_id,source_path,filename,version,collection_id,status,source_type,
                 document_type,source_hash,chunk_count,size_bytes,error_message,now,now))

    def document(self, user_id: str, document_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        with self._connect() as db:
            sql = "SELECT * FROM documents WHERE user_id=? AND document_id=?"
            params: tuple[Any, ...] = (user_id, document_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            row = db.execute(sql, params).fetchone()
            return dict(row) if row else None

    def list_documents(self, user_id: str, *, collection_id: str | None = None,
                       source_type: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
        clauses, params = ["user_id=?"], [user_id]
        if tenant_id is not None:
            clauses.append("tenant_id=?")
            params.append(tenant_id)
        if collection_id is not None:
            clauses.append("collection_id=?")
            params.append(collection_id)
        if source_type is not None:
            clauses.append("source_type=?")
            params.append(source_type)
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents WHERE " + " AND ".join(clauses) +
                              " ORDER BY updated_at DESC", params).fetchall()
            return [dict(row) for row in rows]

    def update_document_collection(self, user_id: str, document_id: str,
                                   collection_id: str | None, tenant_id: str | None = None) -> bool:
        with self._connect() as db:
            sql = "UPDATE documents SET collection_id=?,updated_at=? WHERE user_id=? AND document_id=?"
            params: tuple[Any, ...] = (collection_id, _now(), user_id, document_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            cursor = db.execute(sql, params)
            return cursor.rowcount > 0

    def delete_document_record(self, user_id: str, document_id: str, tenant_id: str | None = None) -> bool:
        with self._connect() as db:
            sql = "DELETE FROM documents WHERE user_id=? AND document_id=?"
            params: tuple[Any, ...] = (user_id, document_id)
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params += (tenant_id,)
            cursor = db.execute(sql, params)
            return cursor.rowcount > 0
