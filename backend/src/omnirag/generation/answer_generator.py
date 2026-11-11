"""Structured, evidence-grounded answer generation without retrieval/provider coupling."""

import json
import time
from dataclasses import dataclass

from omnirag.config.settings import get_settings
from omnirag.llm.llm_client import ChatMessage, ChatProvider, ChatRequest, provider_from_settings
from omnirag.retrieval.contracts import RetrievalResult
from omnirag.generation.prompt_templates import build_generation_messages
from omnirag.verification.claim_extractor import extract_claims
from omnirag.experimentation.runtime import record_stage


class GenerationOutputError(ValueError):
    pass


@dataclass(slots=True)
class GenerationDraft:
    answer: str
    claims: list
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    cost_usd: float | None
    warnings: list[str]


class AnswerGenerator:
    def __init__(self, provider: ChatProvider | None = None, model_id: str | None = None):
        self.provider = provider
        self.model_id = model_id

    def generate(self, query: str, retrieval: RetrievalResult,
                 *, correction_note: str | None = None, response_style: str = "balanced",
                 max_output_tokens: int | None = None,
                 conversation_context: list[dict[str, str]] | None = None) -> GenerationDraft:
        provider, model = self._provider()
        context_started = time.perf_counter()
        messages, warnings = build_generation_messages(query, retrieval.evidence, correction_note,
                                                       response_style, conversation_context)
        record_stage("context_construction", (time.perf_counter() - context_started) * 1000,
                     metadata={"evidence_count": len(retrieval.evidence)})
        settings = get_settings()
        request = ChatRequest(
            model=model,
            messages=[ChatMessage(item["role"], item["content"]) for item in messages],
            temperature=settings.generation_temperature,
            max_output_tokens=max_output_tokens or settings.generation_max_output_tokens,
            response_format={"type": "json_object"},
        )
        started = time.perf_counter()
        response = provider.complete(request)
        if response.raw_metadata.get("fallback_used"):
            warnings.append("configured_generation_fallback_used")
        elapsed = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(response.content)
            answer = payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("answer must be a non-empty string")
            claims = extract_claims(payload)
        except (ValueError, TypeError, AttributeError) as exc:
            raise GenerationOutputError(f"Invalid structured generation output ({type(exc).__name__})") from None
        return GenerationDraft(answer.strip(), claims, response.provider, response.model,
                               response.input_tokens, response.output_tokens,
                               response.latency_ms or elapsed, response.cost_usd, warnings)

    def _provider(self) -> tuple[ChatProvider, str]:
        if self.provider is not None and self.model_id:
            return self.provider, self.model_id
        if self.provider is not None or self.model_id:
            raise ValueError("Provider and model_id must be supplied together")
        return provider_from_settings()
