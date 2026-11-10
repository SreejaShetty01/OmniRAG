import json

import llm.llm_client as llm_client
from llm.llm_client import (
    ChatMessage,
    ChatRequest,
    OpenAICompatibleChatProvider,
    _resolve_chat_completions_endpoint,
)


def test_chat_endpoint_resolution_appends_route_and_preserves_complete_urls():
    assert _resolve_chat_completions_endpoint("http://ollama:11434/v1") == (
        "http://ollama:11434/v1/chat/completions"
    )
    assert _resolve_chat_completions_endpoint("http://ollama:11434/v1/") == (
        "http://ollama:11434/v1/chat/completions"
    )
    assert _resolve_chat_completions_endpoint(
        "http://ollama:11434/v1/chat/completions"
    ) == "http://ollama:11434/v1/chat/completions"
    assert _resolve_chat_completions_endpoint(
        "http://ollama:11434/v1/chat/completions/chat/completions"
    ) == "http://ollama:11434/v1/chat/completions"


def test_chat_adapter_posts_to_resolved_url_and_parses_success(monkeypatch):
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "model": "llama3.2:3b",
                "choices": [{"message": {"content": '{"answer":"Evidence-backed answer","claims":[]}'},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            }).encode()

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(llm_client, "urlopen", fake_urlopen)
    provider = OpenAICompatibleChatProvider(
        "http://ollama:11434/v1", timeout_seconds=7, retries=0,
    )

    result = provider.complete(ChatRequest(
        "llama3.2:3b",
        [ChatMessage("system", "Return JSON."), ChatMessage("user", "Question")],
        temperature=0,
        max_output_tokens=400,
        response_format={"type": "json_object"},
    ))

    assert seen["url"] == "http://ollama:11434/v1/chat/completions"
    assert seen["payload"]["model"] == "llama3.2:3b"
    assert seen["payload"]["messages"][1] == {"role": "user", "content": "Question"}
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["timeout"] == 7
    assert result.content == '{"answer":"Evidence-backed answer","claims":[]}'
    assert result.model == "llama3.2:3b"
    assert result.raw_metadata["finish_reason"] == "stop"
    assert result.input_tokens == 12
    assert result.output_tokens == 8
