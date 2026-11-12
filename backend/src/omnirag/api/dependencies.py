"""Shared application services and explicit request identity boundary."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel

from omnirag.config.settings import get_settings
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.indexing.qdrant_indexer import QdrantIndexer
from omnirag.ingestion.pipeline import IngestionPipeline
from omnirag.infrastructure.product_store import ProductStore
from omnirag.models.query import Query
from omnirag.profiles.execution import ProfileExecutionService
from omnirag.retrieval.contracts import RetrievalScope
from omnirag.retrieval.hybrid_retriever import HybridRetriever


class RequestIdentity(BaseModel):
    user_id: str
    tenant_id: str


@lru_cache(maxsize=1)
def get_product_store() -> ProductStore:
    return ProductStore()


@lru_cache(maxsize=1)
def get_qdrant_indexer() -> QdrantIndexer:
    return QdrantIndexer()


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(indexer=get_qdrant_indexer())


@lru_cache(maxsize=1)
def get_profile_execution_service() -> ProfileExecutionService:
    indexer = get_qdrant_indexer()
    retriever = HybridRetriever(client=indexer.client, collection=indexer.collection)
    answer_pipeline = AnswerPipeline()
    return ProfileExecutionService(retriever, answer_pipeline)


def get_request_identity(request: Request) -> RequestIdentity:
    """Use a fixed dev identity locally; production requires trusted auth middleware."""
    settings = get_settings()
    identity = getattr(request.state, "authenticated_identity", None)
    if identity is not None:
        if isinstance(identity, RequestIdentity):
            return identity
        if isinstance(identity, dict) and identity.get("user_id") and identity.get("tenant_id"):
            return RequestIdentity.model_validate(identity)
        raise HTTPException(status_code=401, detail="Invalid authenticated identity")
    if settings.environment.casefold() == "development":
        return RequestIdentity(user_id=settings.development_user_id,
                               tenant_id=settings.development_tenant_id)
    raise HTTPException(status_code=503, detail="A trusted authentication provider is not configured")


def get_retrieval_scope(identity: Annotated[RequestIdentity, Depends(get_request_identity)]) -> RetrievalScope:
    return RetrievalScope(principal_ids=(f"user:{identity.user_id}",), tenant_id=identity.tenant_id)
