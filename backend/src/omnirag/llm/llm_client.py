"""Provider-neutral chat completion interfaces and an opt-in OpenAI-compatible adapter."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from omnirag.infrastructure.provider_budget import (BudgetExceeded, after_provider_call,
                                            before_provider_call, estimate_tokens)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatRequest:
    model: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.0
    max_output_tokens: int | None = None
    response_format: dict | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResponse:
    content: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None
    raw_metadata: dict = field(default_factory=dict)


class ChatProvider(Protocol):
    def complete(self, request: ChatRequest) -> ChatResponse: ...

    def stream(self, request: ChatRequest) -> Iterator[str]: ...


class ProviderError(RuntimeError):
    """Normalized provider failure; never includes request credentials or prompt text."""


def _resolve_chat_completions_endpoint(endpoint: str) -> str:
    """Resolve an OpenAI-compatible base URL to its chat-completions route."""
    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    route = "/chat/completions"
    while path.endswith(route + route):
        path = path[:-len(route)]
    if not path.endswith(route):
        path += route
    return urlunsplit(parsed._replace(path=path))


class OpenAICompatibleChatProvider:
    """Adapter for endpoints implementing the common chat-completions wire format."""

    def __init__(self, endpoint: str, api_key: str | None = None, timeout_seconds: float = 30.0,
                 retries: int = 1, provider_name: str = "openai_compatible"):
        if not endpoint or timeout_seconds <= 0 or retries < 0:
            raise ValueError("Endpoint, positive timeout, and non-negative retries are required")
        self.endpoint = _resolve_chat_completions_endpoint(endpoint)
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.provider_name = provider_name

    def complete(self, request: ChatRequest) -> ChatResponse:
        payload = {"model": request.model,
                   "messages": [{"role": item.role, "content": item.content} for item in request.messages],
                   "temperature": request.temperature}
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            input_estimate = estimate_tokens("\n".join(item.content for item in request.messages))
            ticket = before_provider_call(self.provider_name, request.model, input_estimate,
                                          request.max_output_tokens or 0)
            try:
                with urlopen(Request(self.endpoint, data=body, headers=headers, method="POST"),
                             timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read())
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Completion content must be text")
                usage = data.get("usage") or {}
                actual_input = usage.get("prompt_tokens") or input_estimate
                actual_output = usage.get("completion_tokens") or estimate_tokens(content)
                reported_cost = data.get("cost_usd") if isinstance(data.get("cost_usd"), (int, float)) else None
                after_provider_call(ticket, input_tokens=actual_input, output_tokens=actual_output,
                                    cost_usd=reported_cost)
                return ChatResponse(
                    content=content, provider=self.provider_name,
                    model=str(data.get("model", request.model)),
                    input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    cost_usd=data.get("cost_usd") if isinstance(data.get("cost_usd"), (int, float)) else None,
                    raw_metadata={"finish_reason": data["choices"][0].get("finish_reason")},
                )
            except BudgetExceeded:
                raise
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError,
                    TypeError, AttributeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(0.2 * (2 ** attempt), 1.0))
        raise ProviderError(f"Chat provider failed ({type(last_error).__name__})") from None

    def stream(self, request: ChatRequest) -> Iterator[str]:
        """Yield text deltas from a common server-sent-events chat stream."""
        payload = {"model": request.model,
                   "messages": [{"role": item.role, "content": item.content} for item in request.messages],
                   "temperature": request.temperature, "stream": True}
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        input_estimate = estimate_tokens("\n".join(item.content for item in request.messages))
        output_limit = request.max_output_tokens
        if output_limit is None:
            from omnirag.config.settings import get_settings
            output_limit = get_settings().generation_max_output_tokens
        ticket = before_provider_call(self.provider_name, request.model, input_estimate, output_limit)
        output_text = []
        try:
            with urlopen(Request(self.endpoint, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST"), timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        output = "".join(output_text)
                        after_provider_call(ticket, input_tokens=input_estimate,
                                            output_tokens=estimate_tokens(output), cost_usd=None)
                        return
                    event = json.loads(data)
                    delta = event["choices"][0].get("delta", {}).get("content")
                    if isinstance(delta, str):
                        output_text.append(delta)
                        yield delta
        except BudgetExceeded:
            raise
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Chat stream failed ({type(exc).__name__})") from None


class FallbackChatProvider:
    """Explicitly configured single fallback; never hides which provider handled a call."""

    def __init__(self, primary: ChatProvider, fallback: ChatProvider, fallback_model: str):
        self.primary, self.fallback, self.fallback_model = primary, fallback, fallback_model

    def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            return self.primary.complete(request)
        except BudgetExceeded:
            raise
        except Exception as primary_error:
            try:
                response = self.fallback.complete(ChatRequest(
                    model=self.fallback_model, messages=request.messages,
                    temperature=request.temperature, max_output_tokens=request.max_output_tokens,
                    response_format=request.response_format, metadata=request.metadata,
                ))
                response.raw_metadata["fallback_used"] = True
                response.raw_metadata["primary_failure"] = type(primary_error).__name__
                return response
            except Exception as fallback_error:
                if isinstance(fallback_error, BudgetExceeded):
                    raise
                raise ProviderError(
                    f"Primary and configured fallback failed ({type(primary_error).__name__}; "
                    f"{type(fallback_error).__name__})") from None

    def stream(self, request: ChatRequest) -> Iterator[str]:
        yield from self.primary.stream(request)


def provider_from_settings(*, verification: bool = False) -> tuple[ChatProvider, str]:
    from omnirag.config.settings import get_settings
    settings = get_settings()
    prefix = "verification" if verification else "generation"
    provider = getattr(settings, f"{prefix}_provider")
    endpoint = getattr(settings, f"{prefix}_endpoint")
    model = getattr(settings, f"{prefix}_model_id")
    key_setting = getattr(settings, f"{prefix}_api_key")
    timeout = getattr(settings, f"{prefix}_timeout_seconds")
    if not endpoint or not model:
        raise ValueError(f"Configure {prefix}_endpoint and {prefix}_model_id to enable this provider")
    if provider not in (None, "openai_compatible"):
        raise ValueError(f"No chat adapter is registered for provider {provider!r}")
    key = key_setting.get_secret_value() if key_setting else None
    retries = settings.generation_max_retries if not verification else 0
    primary = OpenAICompatibleChatProvider(endpoint, key, timeout, retries, provider or "openai_compatible")
    fallback_suffix = "verification_fallback" if verification else "generation_fallback"
    fallback_provider_name = getattr(settings, f"{fallback_suffix}_provider")
    fallback_model = getattr(settings, f"{fallback_suffix}_model_id")
    fallback_endpoint = getattr(settings, f"{fallback_suffix}_endpoint")
    fallback_key_setting = getattr(settings, f"{fallback_suffix}_api_key")
    fallback_configured = any((fallback_provider_name, fallback_model, fallback_endpoint, fallback_key_setting))
    if fallback_configured:
        if not fallback_model or not fallback_endpoint:
            raise ValueError(f"Configured {fallback_suffix} requires both model ID and endpoint")
        if fallback_provider_name not in (None, "openai_compatible"):
            raise ValueError(f"No chat adapter is registered for fallback provider {fallback_provider_name!r}")
        fallback_key = fallback_key_setting.get_secret_value() if fallback_key_setting else None
        fallback = OpenAICompatibleChatProvider(fallback_endpoint, fallback_key, timeout, 0,
                                                fallback_provider_name or "openai_compatible_fallback")
        primary = FallbackChatProvider(primary, fallback, fallback_model)
    return primary, model
