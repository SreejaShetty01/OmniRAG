"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False,
        env_prefix="OMNIRAG_",
    )

    app_name: str = "OmniRAG"
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    default_profile: str = "core"
    data_dir: Path = Path("data")
    development_user_id: str = "local-development-user"
    development_tenant_id: str = "local-development-tenant"
    auth_shared_secret: SecretStr | None = None
    accepted_identities_json: str = "[]"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    rate_limit_requests_per_minute: int = Field(default=120, ge=1)
    qdrant_url: str = "http://localhost:6333"
    ollama_health_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    cache_ttl_seconds: int = Field(default=300, ge=0)
    max_upload_bytes: int = Field(default=20_000_000, gt=0)
    chunk_max_tokens: int = Field(default=500, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    chunk_contextualization_enabled: bool = True
    embedding_provider: str | None = None
    embedding_model_id: str | None = None
    embedding_endpoint: str | None = None
    embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_space_alias: str | None = None
    embedding_model_digest: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_provenance_override: bool = False
    chunker_version: str = "structure-v1"
    ingestion_schema_version: str = "1"
    validation_max_cost_usd: float | None = Field(default=None, ge=0)
    validation_max_total_tokens: int | None = Field(default=None, gt=0)
    validation_max_calls: int | None = Field(default=None, gt=0)
    provider_price_table_json: str = "{}"
    qdrant_collection: str = "omnirag"
    require_eval_collection_prefix: bool = False
    sparse_vector_name: str = "sparse"
    dense_vector_name: str = "dense"
    reranker_provider: str | None = None
    reranker_model_id: str | None = None
    reranker_endpoint: str | None = None
    reranker_api_key: SecretStr | None = None
    reranker_timeout_seconds: float = Field(default=10.0, gt=0)
    generation_provider: str | None = None
    generation_model_id: str | None = None
    generation_endpoint: str | None = None
    generation_api_key: SecretStr | None = None
    generation_fallback_provider: str | None = None
    generation_fallback_model_id: str | None = None
    generation_fallback_endpoint: str | None = None
    generation_fallback_api_key: SecretStr | None = None
    generation_timeout_seconds: float = Field(default=30.0, gt=0)
    generation_max_retries: int = Field(default=1, ge=0, le=5)
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    verification_provider: str | None = None
    verification_model_id: str | None = None
    verification_endpoint: str | None = None
    verification_api_key: SecretStr | None = None
    verification_fallback_provider: str | None = None
    verification_fallback_model_id: str | None = None
    verification_fallback_endpoint: str | None = None
    verification_fallback_api_key: SecretStr | None = None
    verification_timeout_seconds: float = Field(default=15.0, gt=0)
    verification_model_digest: str | None = None
    verification_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    generation_max_output_tokens: int = Field(default=1200, gt=0)
    generation_max_corrections: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_isolated_evaluation_settings(self):
        import json
        if self.require_eval_collection_prefix and not self.qdrant_collection.startswith("omnirag_eval_"):
            raise ValueError("isolated evaluation collection must start with 'omnirag_eval_'")
        try:
            identities = json.loads(self.accepted_identities_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("accepted_identities_json must be a JSON array") from exc
        if not isinstance(identities, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("user_id"), str)
            or not item["user_id"] or not isinstance(item.get("tenant_id"), str)
            or not item["tenant_id"] for item in identities
        ):
            raise ValueError("accepted_identities_json must contain user_id/tenant_id objects")
        if self.environment.casefold() == "production":
            secret = self.auth_shared_secret.get_secret_value() if self.auth_shared_secret else ""
            if len(secret.encode("utf-8")) < 32:
                raise ValueError("production requires OMNIRAG_AUTH_SHARED_SECRET with at least 32 characters")
            if not identities:
                raise ValueError("production requires a non-empty accepted identity allowlist")
        if self.verification_provider == "minicheck_http" and not all((
            self.verification_endpoint, self.verification_model_id, self.verification_model_digest,
            self.verification_threshold is not None,
        )):
            raise ValueError("MiniCheck requires endpoint, model ID, pinned digest, and calibrated threshold")
        return self

    @property
    def accepted_identities(self) -> tuple[tuple[str, str], ...]:
        import json
        return tuple((item["user_id"], item["tenant_id"])
                     for item in json.loads(self.accepted_identities_json))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings instance."""
    return Settings()
