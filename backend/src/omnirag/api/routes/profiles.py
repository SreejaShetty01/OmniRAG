from fastapi import APIRouter

from omnirag.config.model_registry import registry_from_settings
from omnirag.config.profiles import PROFILE_CONFIGS
from omnirag.config.settings import get_settings

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("")
def profile_catalog():
    registry = registry_from_settings(get_settings())
    return {"profiles": [{
        "name": config.name.value,
        "description": {
            "lite": "Concise responses with a smaller retrieval and evidence budget.",
            "core": "Deeper retrieval and available evidence checking.",
            "research": "Bounded multi-step investigation and cross-document synthesis.",
        }[config.name.value],
        "policy": {
            "dense_candidates": config.dense_candidates,
            "sparse_candidates": config.sparse_candidates,
            "evidence_token_budget": config.context_budget_tokens,
            "reranking_enabled": config.reranking_enabled,
            "verification_depth": config.verification_depth.value,
            "defaults_benchmarked": config.defaults_benchmarked,
        },
    } for config in PROFILE_CONFIGS.values()],
        "configured_models": {
            key: bool(selection.model_id)
            for key, selection in (("generation", registry.generation),
                                   ("verification", registry.verification),
                                   ("reranking", registry.reranking))
        }}
