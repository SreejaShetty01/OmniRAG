import pytest
from pydantic import ValidationError

from omnirag.api.dependencies import RequestIdentity
from omnirag.api.routes.product import application_settings
from omnirag.api.trusted_proxy import identity_is_allowed
from omnirag.config.settings import Settings


def test_isolated_instance_refuses_non_evaluation_collection():
    with pytest.raises(ValidationError, match="must start with 'omnirag_eval_'"):
        Settings(_env_file=None, require_eval_collection_prefix=True,
                 qdrant_collection="omnirag")


def test_identity_allowlist_is_pair_scoped_and_empty_list_preserves_existing_mode():
    allowed = (("reader", "tenant-a"),)
    assert identity_is_allowed({"user_id": "reader", "tenant_id": "tenant-a"}, allowed)
    assert not identity_is_allowed({"user_id": "reader", "tenant_id": "tenant-b"}, allowed)
    assert not identity_is_allowed({"user_id": "other", "tenant_id": "tenant-a"}, allowed)
    assert identity_is_allowed({"user_id": "any", "tenant_id": "tenant"}, ())


def test_settings_reports_configured_collection_embedding_budget_and_identities(monkeypatch):
    settings = Settings(
        _env_file=None,
        environment="production",
        auth_shared_secret="x" * 32,
        qdrant_collection="omnirag_eval_rw_omnirag_realtech_nomic768",
        require_eval_collection_prefix=True,
        embedding_provider="openai_compatible",
        embedding_model_id="nomic-embed-text",
        embedding_space_alias="nomic-embed-text-768",
        embedding_model_digest=None,
        embedding_dimensions=768,
        validation_max_cost_usd=0,
        validation_max_total_tokens=4_200_000,
        validation_max_calls=3_000,
        accepted_identities_json='[{"user_id":"corpus-reader","tenant_id":"realworld"}]',
    )
    monkeypatch.setattr("omnirag.api.routes.product.get_settings", lambda: settings)

    result = application_settings(RequestIdentity(user_id="corpus-reader", tenant_id="realworld"))

    assert result["physical_collection"] == settings.qdrant_collection
    assert result["embedding_space"] == {
        "alias": "nomic-embed-text-768", "provider": "openai_compatible",
        "model_id": "nomic-embed-text", "model_digest": None, "dimensions": 768,
    }
    assert result["limits"]["max_cost_usd"] == 0
    assert result["limits"]["max_total_tokens"] == 4_200_000
    assert result["limits"]["max_calls"] == 3_000
    assert result["accepted_identities"] == [{"user_id": "corpus-reader", "tenant_id": "realworld"}]
    assert "secret" not in str(result).casefold()
