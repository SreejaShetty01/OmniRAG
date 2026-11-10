"""Authenticated upload and document lifecycle endpoints over the shared pipeline."""

import asyncio
import hashlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam, Request, status
from models.chunk import AccessControlMetadata

from api.dependencies import (
    RequestIdentity, get_ingestion_pipeline, get_product_store, get_qdrant_indexer, get_request_identity,
)
from api.routes.collections import _set_document_collection
from api.schemas import DocumentResponse
from config.settings import get_settings
from ingestion.loaders.code_loader import CODE_LANGUAGES, GENERIC_CODE_EXTENSIONS
from ingestion.loaders.document_loader import DOCUMENT_EXTENSIONS
from ingestion.pipeline import IngestionPipeline, IngestionResult, IngestionStatus
from infrastructure.product_store import ProductStore
from ingestion.structure.normalize import stable_id

router = APIRouter(prefix="/documents", tags=["documents"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]
Store = Annotated[ProductStore, Depends(get_product_store)]
Pipeline = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
Indexer = Annotated[object, Depends(get_qdrant_indexer)]

ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | set(CODE_LANGUAGES) | GENERIC_CODE_EXTENSIONS


def _safe_filename(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", Path(name.replace("\\", "/")).name)
    safe = re.sub(r"[^\w.() -]", "_", normalized, flags=re.UNICODE).strip(" .")
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=422, detail="A valid filename is required")
    if Path(safe).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document or code file type")
    return safe[:180]


def _doc_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".markdown": "markdown",
            ".txt": "text", ".rst": "text", ".html": "html", ".htm": "html"}.get(
                suffix, "code" if suffix in CODE_LANGUAGES or suffix in GENERIC_CODE_EXTENSIONS else "other")


def _source_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return "code" if suffix in CODE_LANGUAGES or suffix in GENERIC_CODE_EXTENSIONS else "document"


def _upload_exceeds_limit(current_size: int, part_size: int, max_bytes: int) -> bool:
    return current_size + part_size > max_bytes


def _ingestion_response_status(result: IngestionResult, *, already_exists: bool) -> int:
    if result.status == IngestionStatus.FAILED:
        if result.error_type == "SourceLoadError":
            return status.HTTP_422_UNPROCESSABLE_CONTENT
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_200_OK if already_exists else status.HTTP_201_CREATED


@router.get("", response_model=list[DocumentResponse])
def list_documents(identity: Identity, store: Store,
                   collection_id: str | None = None,
                   source_type: str | None = QueryParam(default=None, pattern="^(document|code)$")):
    if collection_id and not store.collection(identity.user_id, collection_id, identity.tenant_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    rows = store.list_documents(identity.user_id, collection_id=collection_id,
                                source_type=source_type, tenant_id=identity.tenant_id)
    return [_public_document(row) for row in rows]


@router.put("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, identity: Identity, store: Store, pipeline: Pipeline,
                          indexer: Indexer,
                          filename: str = QueryParam(min_length=1, max_length=255),
                          version: str | None = QueryParam(default=None, max_length=100),
                          collection_id: str | None = None):
    filename = _safe_filename(filename)
    if collection_id and not store.collection(identity.user_id, collection_id, identity.tenant_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    max_bytes = get_settings().max_upload_bytes
    body = bytearray()
    async for part in request.stream():
        if _upload_exceeds_limit(len(body), len(part), max_bytes):
            raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit")
        body.extend(part)
    if not body:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    user_dir = hashlib.sha256(f"{identity.tenant_id}\0{identity.user_id}".encode()).hexdigest()[:24]
    root = (get_settings().data_dir / "uploads" / user_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / filename).resolve()
    if root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid upload filename")
    temp = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        await asyncio.to_thread(temp.write_bytes, bytes(body))
        await asyncio.to_thread(os.replace, temp, target)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)

    document_id = stable_id("doc", str(target))
    existing = store.document(identity.user_id, document_id, identity.tenant_id)
    source_hash = hashlib.sha256(body).hexdigest()
    effective_version = version or (
        existing["version"] if existing and existing["source_hash"] == source_hash else
        datetime.now(timezone.utc).strftime("upload-%Y%m%dT%H%M%SZ"))
    store.upsert_document(
        document_id=document_id, user_id=identity.user_id, tenant_id=identity.tenant_id,
        source_path=str(target), filename=filename, version=effective_version,
        collection_id=collection_id, status="processing", source_type=_source_type(filename),
        document_type=_doc_type(filename), source_hash=source_hash,
        chunk_count=existing["chunk_count"] if existing else 0, size_bytes=len(body),
    )
    acl = AccessControlMetadata(allowed_principals=[f"user:{identity.user_id}"])
    result = await asyncio.to_thread(pipeline.ingest, target, version=effective_version,
                                     access_control=acl, tenant_id=identity.tenant_id,
                                     collection_id=collection_id)
    document_id = result.document_id or document_id
    if result.status == IngestionStatus.UNCHANGED and collection_id:
        _set_document_collection(indexer, document_id, identity, collection_id)
    store.upsert_document(
        document_id=document_id, user_id=identity.user_id, tenant_id=identity.tenant_id,
        source_path=str(target), filename=filename, version=effective_version,
        collection_id=collection_id, status=result.status.value, source_type=_source_type(filename),
        document_type=_doc_type(filename), source_hash=result.source_hash,
        chunk_count=result.chunk_count, size_bytes=len(body), error_message=result.error_message,
    )
    response_status = _ingestion_response_status(result, already_exists=existing is not None)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=response_status,
                        content=DocumentResponse.model_validate(
                            _public_document(store.document(identity.user_id, document_id, identity.tenant_id))).model_dump())


@router.post("/{document_id}/reingest", response_model=DocumentResponse)
async def reingest_document(document_id: str, identity: Identity, store: Store, pipeline: Pipeline):
    record = store.document(identity.user_id, document_id, identity.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    acl = AccessControlMetadata(allowed_principals=[f"user:{identity.user_id}"])
    result = await asyncio.to_thread(pipeline.ingest, record["source_path"], version=record["version"],
                                     access_control=acl, tenant_id=identity.tenant_id,
                                     collection_id=record["collection_id"])
    store.upsert_document(
        document_id=document_id, user_id=identity.user_id, tenant_id=identity.tenant_id,
        source_path=record["source_path"], filename=record["filename"], version=record["version"],
        collection_id=record["collection_id"], status=result.status.value,
        source_type=_source_type(record["filename"]), document_type=_doc_type(record["filename"]),
        source_hash=result.source_hash or record["source_hash"], chunk_count=result.chunk_count or record["chunk_count"],
        size_bytes=record["size_bytes"], error_message=result.error_message,
    )
    saved = store.document(identity.user_id, document_id, identity.tenant_id)
    if result.status == IngestionStatus.FAILED:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=_ingestion_response_status(result, already_exists=True),
            content=DocumentResponse.model_validate(_public_document(saved)).model_dump(),
        )
    return _public_document(saved)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, identity: Identity, store: Store, pipeline: Pipeline):
    record = store.document(identity.user_id, document_id, identity.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    pipeline.delete_document(document_id)
    uploads_root = (get_settings().data_dir / "uploads").resolve()
    source_path = Path(record["source_path"]).resolve()
    if uploads_root in source_path.parents:
        source_path.unlink(missing_ok=True)
    store.delete_document_record(identity.user_id, document_id, identity.tenant_id)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, identity: Identity, store: Store):
    record = store.document(identity.user_id, document_id, identity.tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _public_document(record)


def _public_document(row):
    result = {key: row[key] for key in (
        "document_id", "filename", "status", "version", "collection_id", "source_type",
        "document_type", "source_hash", "chunk_count", "size_bytes", "error_message",
        "created_at", "updated_at",
    )}
    # Safe ingestion errors can include their input path; keep the useful failure text
    # while avoiding disclosure of the per-user storage location.
    error = result.get("error_message")
    source_path = row.get("source_path")
    if error and source_path:
        result["error_message"] = str(error).replace(str(source_path), row.get("filename") or "uploaded file")
    return result
