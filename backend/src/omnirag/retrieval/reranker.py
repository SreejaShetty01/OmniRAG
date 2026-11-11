"""Optional provider-neutral reranker contract and configurable HTTP adapter."""

import json
import math
from dataclasses import dataclass
from typing import Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from omnirag.infrastructure.provider_budget import (BudgetExceeded, after_provider_call,
                                            before_provider_call, estimate_tokens)

from omnirag.config.settings import get_settings
from omnirag.retrieval.contracts import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class RerankScore:
    candidate_id: str
    score: float


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[RerankScore]: ...


class RerankerError(RuntimeError):
    pass


class HttpReranker:
    """Adapter contract: POST `{model, query, documents}` and read `{results:[{id,score}]}'."""

    def __init__(self, endpoint: str, model_id: str, api_key: str | None = None,
                 timeout_seconds: float = 10.0, batch_size: int = 64):
        if not endpoint or not model_id:
            raise ValueError("Reranker endpoint and model ID must be configured")
        if timeout_seconds <= 0 or batch_size <= 0:
            raise ValueError("Reranker timeout and batch size must be positive")
        self.endpoint, self.model_id, self.api_key = endpoint, model_id, api_key
        self.timeout_seconds, self.batch_size = timeout_seconds, batch_size

    def rerank(self, query: str, candidates: Sequence[RetrievalCandidate]) -> list[RerankScore]:
        ranked: list[RerankScore] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start:start + self.batch_size]
            payload = {"model": self.model_id, "query": query,
                       "documents": [{"id": candidate.identity, "text": candidate.chunk.content}
                                     for candidate in batch]}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
            input_estimate = estimate_tokens(query + "\n" + "\n".join(c.chunk.content for c in batch))
            ticket = before_provider_call("http", self.model_id, input_estimate, 0)
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read())
                rows = body.get("results")
                if not isinstance(rows, list):
                    raise ValueError("Reranker response has no results array")
                parsed = [RerankScore(str(row["id"]), float(row["score"])) for row in rows]
                if any(not math.isfinite(item.score) for item in parsed):
                    raise ValueError("Reranker scores must be finite")
                usage = body.get("usage") or {}
                actual_input = usage.get("prompt_tokens") or usage.get("total_tokens") or input_estimate
                after_provider_call(ticket, input_tokens=actual_input, output_tokens=0,
                                    cost_usd=body.get("cost_usd"))
                ranked.extend(parsed)
            except BudgetExceeded:
                raise
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
                raise RerankerError(f"Reranker request failed ({type(exc).__name__})") from None
        valid_ids = {candidate.identity for candidate in candidates}
        returned_ids = [result.candidate_id for result in ranked]
        if any(candidate_id not in valid_ids for candidate_id in returned_ids):
            raise RerankerError("Reranker returned an unknown candidate ID")
        if len(returned_ids) != len(set(returned_ids)):
            raise RerankerError("Reranker returned duplicate candidate IDs")
        return ranked


def reranker_from_settings() -> Reranker:
    settings = get_settings()
    provider = settings.reranker_provider
    if provider not in {None, "http"}:
        raise ValueError(f"No reranker adapter is registered for provider {provider!r}")
    if not settings.reranker_endpoint or not settings.reranker_model_id:
        raise ValueError("Configure reranker_endpoint and reranker_model_id to enable reranking")
    key = settings.reranker_api_key.get_secret_value() if settings.reranker_api_key else None
    return HttpReranker(settings.reranker_endpoint, settings.reranker_model_id, key,
                        settings.reranker_timeout_seconds)
