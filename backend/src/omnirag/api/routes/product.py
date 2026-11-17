"""Safe settings and evaluation availability endpoints."""

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from omnirag.api.dependencies import RequestIdentity, get_request_identity
from omnirag.api.schemas import EvaluationStatus
from omnirag.config.model_registry import registry_from_settings
from omnirag.config.profiles import PROFILE_CONFIGS
from omnirag.config.settings import get_settings

router = APIRouter(tags=["product"])
Identity = Annotated[RequestIdentity, Depends(get_request_identity)]


@router.get("/settings")
def application_settings(identity: Identity):
    settings = get_settings()
    registry = registry_from_settings(settings)
    return {
        "user_id": identity.user_id,
        "tenant_id": identity.tenant_id,
        "default_profile": settings.default_profile,
        "physical_collection": settings.qdrant_collection,
        "embedding_space": {
            "alias": settings.embedding_space_alias,
            "provider": settings.embedding_provider or "openai_compatible",
            "model_id": settings.embedding_model_id,
            "model_digest": settings.embedding_model_digest or None,
            "dimensions": settings.embedding_dimensions,
        },
        "accepted_identities": [{"user_id": user, "tenant_id": tenant}
                                for user, tenant in settings.accepted_identities],
        "profiles": [name.value for name in PROFILE_CONFIGS],
        "models": {
            key: {"provider": selection.provider, "model_id": selection.model_id,
                  "configured": bool(selection.model_id and selection.endpoint)}
            for key, selection in (("generation", registry.generation),
                                   ("verification", registry.verification),
                                   ("reranking", registry.reranking),
                                   ("embedding", registry.embedding))
        },
        "limits": {"max_upload_bytes": settings.max_upload_bytes,
                   "generation_max_output_tokens": settings.generation_max_output_tokens,
                   "max_cost_usd": settings.validation_max_cost_usd,
                   "max_total_tokens": settings.validation_max_total_tokens,
                   "max_calls": settings.validation_max_calls,
                   "requests_per_minute_per_principal": settings.rate_limit_requests_per_minute},
        "health_dependencies": {"qdrant": True,
                                "ollama": bool(settings.ollama_health_url)},
    }


def _gold_dataset_available() -> bool:
    path = Path(__file__).resolve().parents[5] / "eval" / "eval_dataset.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, list) and bool(data)


@router.get("/evaluations/status", response_model=EvaluationStatus)
def evaluation_status(identity: Identity):
    dataset_available = _gold_dataset_available()
    return EvaluationStatus(
        # Dataset presence alone does not mean a runner can produce persisted results.
        available=False, dataset_available=dataset_available,
        suites=["profile-comparison", "retrieval", "grounded-generation"] if dataset_available else [],
        message=("Labeled evaluation data exists, but no persisted evaluation runner is configured." if dataset_available else
                 "No representative labeled question/evidence dataset is configured; quality scores are unavailable."),
    )


@router.post("/evaluations/runs", status_code=status.HTTP_202_ACCEPTED)
def start_evaluation(identity: Identity):
    if not _gold_dataset_available():
        raise HTTPException(status_code=409,
                            detail="Evaluation cannot run because no labeled gold dataset is configured")
    raise HTTPException(status_code=501, detail="No persisted evaluation runner is configured yet")
