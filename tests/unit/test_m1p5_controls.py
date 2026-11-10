import logging

from pydantic import SecretStr

import config.settings as settings_module
from api import main
from experiments.runtime import ExperimentRunStore
from infrastructure.rate_limit import PrincipalRateLimiter
from infrastructure.runtime_metrics import RuntimeMetrics
from utils.logger import JsonLogFormatter


def test_rate_limit_is_per_principal_and_reports_retry_after():
    now = [10.0]
    limiter = PrincipalRateLimiter(1, clock=lambda: now[0])
    assert limiter.check(("tenant:a", "user"))[0] is True
    allowed, retry = limiter.check(("tenant:a", "user"))
    assert allowed is False
    assert retry > 0
    assert limiter.check(("tenant", "a:user"))[0] is True
    now[0] = 60.0
    assert limiter.check(("tenant:a", "user"))[0] is True


def test_evaluation_limit_allows_six_hundred_serial_units():
    limiter = PrincipalRateLimiter(1200, clock=lambda: 1.0)
    assert all(limiter.check(("eval-tenant", "dataset-principal"))[0] for _ in range(600))


def test_api_rate_limit_returns_429_with_retry_after():
    from infrastructure.rate_limit import too_many_requests_response
    response = too_many_requests_response("req-1", 15)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "15"
    assert "rate_limit_exceeded" in response.body.decode()


def test_runtime_metrics_tracks_status_latency_provider_and_budget_usage():
    metrics = RuntimeMetrics()
    metrics.record(200, 22, {"run_id": "r1", "calls": 1, "input_tokens": 4,
                             "output_tokens": 2, "total_tokens": 6,
                             "estimated_cost_usd": 0.0})
    metrics.record(200, 30, {"run_id": "r1", "calls": 1, "input_tokens": 4,
                             "output_tokens": 2, "total_tokens": 6,
                             "estimated_cost_usd": 0.0})
    metrics.record(503, 1500, None)
    result = metrics.snapshot()
    assert result["request_count_by_status"] == {"200": 2, "503": 1}
    assert result["error_count_by_status"] == {"503": 1}
    assert result["provider_calls_observed"] == 1
    assert result["latest_budget_usage"]["total_tokens"] == 6
    assert result["queue_depth"] is None


def test_structured_log_formatter_redacts_configured_secret():
    record = logging.LogRecord("test", logging.ERROR, "x.py", 1,
                               "provider failed: CANARY_SECRET", (), None)
    rendered = JsonLogFormatter(("CANARY_SECRET",)).format(record)
    assert "CANARY_SECRET" not in rendered
    assert "[REDACTED]" in rendered


def test_experiment_manifest_trace_and_result_redact_configured_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(settings_module, "get_settings", lambda: settings_module.Settings(
        auth_shared_secret=SecretStr("CANARY_SECRET")))
    store = ExperimentRunStore(tmp_path)
    for filename in ("manifest.json", "trace.json", "result.json", "evaluation.json"):
        store.write_json("secret-check", filename, {"message": "prefix CANARY_SECRET suffix"})
        artifact_text = (tmp_path / "secret-check" / filename).read_text(encoding="utf-8")
        assert "CANARY_SECRET" not in artifact_text
        assert "[REDACTED]" in artifact_text


def test_readiness_checks_ollama_only_when_configured(monkeypatch):
    class Qdrant:
        def get_collections(self):
            return []

    monkeypatch.setattr(main, "get_qdrant_client", Qdrant)
    monkeypatch.setattr(main.settings, "ollama_health_url", "http://ollama.invalid/api/tags")

    class HealthyResponse:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: HealthyResponse())
    assert main.readiness()["status"] == "ready"


def test_readiness_reports_ollama_unavailable_without_exposing_endpoint(monkeypatch):
    class Qdrant:
        def get_collections(self):
            return []

    monkeypatch.setattr(main, "get_qdrant_client", Qdrant)
    monkeypatch.setattr(main.settings, "ollama_health_url", "http://ollama.invalid/api/tags")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()))
    response = main.readiness()
    assert response.status_code == 503
    assert response.body == b'{"status":"not_ready","dependency":"ollama"}'


def test_readiness_reports_qdrant_unavailable_without_exposing_connection_details(monkeypatch):
    monkeypatch.setattr(main, "get_qdrant_client",
                        lambda: (_ for _ in ()).throw(OSError("private-qdrant-host:6333")))
    response = main.readiness()
    assert response.status_code == 503
    assert response.body == b'{"status":"not_ready","dependency":"qdrant"}'
