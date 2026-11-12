"""Provider-neutral dense embedding interface and configurable HTTP adapter."""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from omnirag.config.model_registry import ModelRegistry
from omnirag.config.settings import get_settings
from omnirag.infrastructure.provider_budget import (BudgetExceeded, after_provider_call,
                                            before_provider_call, estimate_tokens)

class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HttpEmbeddingProvider:
    """Adapter for endpoints accepting an OpenAI-compatible `{model,input}` JSON body."""

    def __init__(self, endpoint: str, model_id: str, api_key: str | None = None,
                 timeout_seconds: float = 30, retries: int = 3):
        if not endpoint or not model_id:
            raise ValueError("Embedding endpoint and model ID must be configured")
        self.endpoint, self.model_id, self.api_key = endpoint, model_id, api_key
        self.provider_name = "openai_compatible"
        self.timeout_seconds, self.retries = timeout_seconds, retries

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_id, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            input_estimate = sum(estimate_tokens(text) for text in texts)
            ticket = before_provider_call(self.provider_name, self.model_id, input_estimate, 0)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read())
                rows = payload.get("data")
                if not isinstance(rows, list):
                    raise EmbeddingError("Embedding response has no data array")
                rows.sort(key=lambda row: row.get("index", 0))
                vectors = [row.get("embedding") for row in rows]
                if len(vectors) != len(texts):
                    raise EmbeddingError("Embedding response count does not match request")
                usage = payload.get("usage") or {}
                actual_input = usage.get("prompt_tokens") or usage.get("total_tokens") or input_estimate
                after_provider_call(ticket, input_tokens=actual_input, output_tokens=0,
                                    cost_usd=payload.get("cost_usd"))
                return vectors
            except BudgetExceeded:
                raise
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, EmbeddingError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429:
                    break
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2 ** attempt), 2.0))
        logger_error = type(last_error).__name__ if last_error else "unknown error"
        raise EmbeddingError(f"Embedding request failed after retries ({logger_error})") from None


def provider_from_config(registry: ModelRegistry | None = None) -> EmbeddingProvider:
    settings = get_settings()
    selection = (registry or ModelRegistry()).embedding
    provider_name = selection.provider or settings.embedding_provider
    if provider_name not in {None, "openai_compatible"}:
        raise EmbeddingError(f"No embedding adapter is registered for provider {provider_name!r}")
    endpoint = selection.endpoint or settings.embedding_endpoint
    model_id = selection.model_id or settings.embedding_model_id
    if not endpoint or not model_id:
        raise EmbeddingError("Configure embedding_endpoint and embedding_model_id before ingestion")
    key = settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None
    return HttpEmbeddingProvider(endpoint, model_id, key)


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider, *, batch_size: int | None = None,
                 expected_dimensions: int | None = None):
        self.provider = provider
        settings = get_settings()
        self.batch_size = batch_size if batch_size is not None else settings.embedding_batch_size
        self.expected_dimensions = expected_dimensions if expected_dimensions is not None else settings.embedding_dimensions
        if self.batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.cache_capacity = 2048
        self.dimensions: int | None = self.expected_dimensions

    def embed(self, texts: Sequence[str], *, cache_scope: str | None = None) -> list[list[float]]:
        normalized = [text.strip() for text in texts]
        if any(not text for text in normalized):
            raise ValueError("Embedding inputs must not be empty")
        results: list[list[float] | None] = [None] * len(normalized)
        missing: dict[str, str] = {}
        positions: dict[str, list[int]] = {}
        for index, text in enumerate(normalized):
            identity = {
                "text": text,
                "scope": cache_scope,
                "provider": type(self.provider).__name__,
                "model": getattr(self.provider, "model_id", None),
                "dimensions": self.expected_dimensions,
            }
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
            positions.setdefault(key, []).append(index)
            if cache_scope is None or key not in self._cache:
                missing[key] = text
            else:
                results[index] = self._cache[key]
                self._cache.move_to_end(key)
        keys = list(missing)
        for start in range(0, len(keys), self.batch_size):
            batch_keys = keys[start:start + self.batch_size]
            vectors = self.provider.embed([missing[key] for key in batch_keys])
            if len(vectors) != len(batch_keys):
                raise EmbeddingError("Provider returned an unexpected vector count")
            for key, vector in zip(batch_keys, vectors, strict=True):
                if not vector or any(not isinstance(value, (int, float)) for value in vector):
                    raise EmbeddingError("Provider returned an invalid vector")
                vector = [float(value) for value in vector]
                if self.dimensions is None:
                    self.dimensions = len(vector)
                if len(vector) != self.dimensions:
                    raise EmbeddingError(f"Expected {self.dimensions} dimensions, received {len(vector)}")
                if cache_scope is not None:
                    self._cache[key] = vector
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_capacity:
                        self._cache.popitem(last=False)
                for index in positions[key]:
                    results[index] = vector
        return [vector for vector in results if vector is not None]
