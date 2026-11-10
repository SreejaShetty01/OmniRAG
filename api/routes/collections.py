import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import models

from api.dependencies import RequestIdentity, get_product_store, get_qdrant_indexer, get_request_identity
from api.schemas import CollectionCreate, CollectionResponse, DocumentCollectionUpdate
from indexing.qdrant_indexer import QdrantIndexer
from infrastructure.product_store import ProductStore

router = APIRouter(prefix="/collections", tags=["collections"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]
Store = Annotated[ProductStore, Depends(get_product_store)]
Indexer = Annotated[QdrantIndexer, Depends(get_qdrant_indexer)]


def _owned_points_filter(document_id: str, identity: RequestIdentity):
    principal = f"user:{identity.user_id}"
    return models.Filter(must=[
        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
        models.FieldCondition(key="metadata.tenant_id", match=models.MatchValue(value=identity.tenant_id)),
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        models.FieldCondition(key="acl_allowed_principals", match=models.MatchAny(any=[principal])),
    ], must_not=[models.FieldCondition(key="acl_denied_principals", match=models.MatchAny(any=[principal]))])


def _set_document_collection(indexer: QdrantIndexer, document_id: str,
                             identity: RequestIdentity, collection_id: str | None) -> None:
    selector = _owned_points_filter(document_id, identity)
    if not indexer.client.collection_exists(indexer.collection):
        return
    if collection_id is None:
        indexer.client.delete_payload(collection_name=indexer.collection,
                                      keys=["metadata.collection_id"], points=selector, wait=True)
    else:
        indexer.client.set_payload(collection_name=indexer.collection,
                                   payload={"collection_id": collection_id}, key="metadata",
                                   points=selector, wait=True)


@router.get("", response_model=list[CollectionResponse])
def list_collections(identity: Identity, store: Store):
    return store.list_collections(identity.user_id, identity.tenant_id)


@router.post("", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
def create_collection(payload: CollectionCreate, identity: Identity, store: Store):
    try:
        collection_id = store.create_collection(identity.user_id, payload.name, payload.description,
                                                identity.tenant_id)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="A collection with this name already exists") from None
    return next(item for item in store.list_collections(identity.user_id, identity.tenant_id)
                if item["collection_id"] == collection_id)


@router.put("/documents/{document_id}")
def assign_document(document_id: str, payload: DocumentCollectionUpdate, identity: Identity,
                    store: Store, indexer: Indexer):
    document = store.document(identity.user_id, document_id, identity.tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if payload.collection_id and not store.collection(identity.user_id, payload.collection_id, identity.tenant_id):
        raise HTTPException(status_code=404, detail="Collection not found")
    _set_document_collection(indexer, document_id, identity, payload.collection_id)
    store.update_document_collection(identity.user_id, document_id, payload.collection_id, identity.tenant_id)
    return {"document_id": document_id, "collection_id": payload.collection_id}


@router.delete("/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: str, identity: Identity, store: Store, indexer: Indexer):
    collection = store.collection(identity.user_id, collection_id, identity.tenant_id)
    if collection is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    docs = store.list_documents(identity.user_id, collection_id=collection_id, tenant_id=identity.tenant_id)
    for document in docs:
        _set_document_collection(indexer, document["document_id"], identity, None)
    store.delete_collection(identity.user_id, collection_id, identity.tenant_id)
