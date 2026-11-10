"""Provider-neutral model selection configuration."""

from pydantic import BaseModel, ConfigDict


class ModelSelection(BaseModel):
    """Logical model slots; concrete identifiers are supplied by deployment config."""

    model_config = ConfigDict(extra="forbid")

    provider: str | None = None
    model_id: str | None = None
    endpoint: str | None = None


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: ModelSelection = ModelSelection()
    embedding: ModelSelection = ModelSelection()
    reranking: ModelSelection = ModelSelection()
    verification: ModelSelection = ModelSelection()
    research: ModelSelection = ModelSelection()
    generation_fallback: ModelSelection = ModelSelection()
    verification_fallback: ModelSelection = ModelSelection()


def registry_from_settings(settings) -> ModelRegistry:
    """Expose configured provider slots through the shared provider-neutral registry."""
    return ModelRegistry(
        generation=ModelSelection(provider=settings.generation_provider,
                                  model_id=settings.generation_model_id,
                                  endpoint=settings.generation_endpoint),
        embedding=ModelSelection(provider=settings.embedding_provider,
                                 model_id=settings.embedding_model_id,
                                 endpoint=settings.embedding_endpoint),
        reranking=ModelSelection(provider=settings.reranker_provider,
                                 model_id=settings.reranker_model_id,
                                 endpoint=settings.reranker_endpoint),
        verification=ModelSelection(provider=settings.verification_provider,
                                    model_id=settings.verification_model_id,
                                    endpoint=settings.verification_endpoint),
        generation_fallback=ModelSelection(provider=settings.generation_fallback_provider,
                                           model_id=settings.generation_fallback_model_id,
                                           endpoint=settings.generation_fallback_endpoint),
        verification_fallback=ModelSelection(provider=settings.verification_fallback_provider,
                                             model_id=settings.verification_fallback_model_id,
                                             endpoint=settings.verification_fallback_endpoint),
    )
