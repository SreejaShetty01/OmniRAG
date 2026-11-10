"""Lazy construction of the single shared Qdrant client."""

from functools import lru_cache

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("Install the optional qdrant extra to enable Qdrant indexing") from exc
    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant_url, api_key=api_key)
