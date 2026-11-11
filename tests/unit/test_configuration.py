from omnirag.config.profiles import ProfileName, get_profile_config
import pytest
from pydantic import ValidationError

from omnirag.config.settings import Settings


def test_environment_settings_load_and_validate(monkeypatch):
    monkeypatch.setenv("OMNIRAG_API_PORT", "8123")
    monkeypatch.setenv("OMNIRAG_DEFAULT_PROFILE", "research")
    settings = Settings(_env_file=None)
    assert settings.api_port == 8123
    assert settings.default_profile == "research"


def test_production_requires_strong_secret_and_explicit_identity_allowlist():
    with pytest.raises(ValidationError, match="AUTH_SHARED_SECRET"):
        Settings(_env_file=None, environment="production",
                 accepted_identities_json='[{"user_id":"u","tenant_id":"t"}]')
    with pytest.raises(ValidationError, match="non-empty accepted identity allowlist"):
        Settings(_env_file=None, environment="production", auth_shared_secret="x" * 32)

    settings = Settings(
        _env_file=None, environment="production", auth_shared_secret="x" * 32,
        accepted_identities_json='[{"user_id":"u","tenant_id":"t"}]',
    )
    assert settings.accepted_identities == (("u", "t"),)


def test_profile_policies_scale_and_research_features_are_enabled():
    lite = get_profile_config(ProfileName.LITE)
    core = get_profile_config(ProfileName.CORE)
    research = get_profile_config(ProfileName.RESEARCH)
    assert lite.retrieval_depth.value == "shallow"
    assert core.retrieval_depth.value == "standard"
    assert research.retrieval_depth.value == "deep"
    assert lite.context_budget_tokens < core.context_budget_tokens < research.context_budget_tokens
    assert not lite.defaults_benchmarked and not core.defaults_benchmarked and not research.defaults_benchmarked
    assert core.reranking_enabled and core.sufficiency_checks_enabled
    assert research.multi_query_enabled and research.evidence_ledger_enabled
    assert research.orchestration == "langgraph"
