"""FastAPI product API, shared service composition and production-safe static hosting."""

import logging
import os
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from omnirag.api.routes.chat import router as chat_router
from omnirag.api.routes.collections import router as collections_router
from omnirag.api.routes.conversations import router as conversations_router
from omnirag.api.routes.documents import router as documents_router
from omnirag.api.routes.preferences import router as preferences_router
from omnirag.api.routes.product import router as product_router
from omnirag.api.routes.profiles import router as profiles_router
from omnirag.api.dependencies import get_qdrant_indexer
from omnirag.api.trusted_proxy import identity_is_allowed, verify_proxy_identity
from omnirag.infrastructure.provider_budget import (BudgetExceeded, activate_run_budget,
                                            current_budget_snapshot)
from omnirag.infrastructure.rate_limit import PrincipalRateLimiter, too_many_requests_response
from omnirag.infrastructure.runtime_metrics import runtime_metrics
from omnirag.config.profiles import PROFILE_CONFIGS
from omnirag.config.settings import get_settings
from omnirag.infrastructure.qdrant_client import get_qdrant_client
from omnirag.utils.logger import configure_logging

settings = get_settings()
_log_secrets = tuple(secret.get_secret_value() for secret in (
    settings.auth_shared_secret, settings.qdrant_api_key, settings.embedding_api_key,
    settings.generation_api_key, settings.generation_fallback_api_key,
    settings.verification_api_key, settings.verification_fallback_api_key,
    settings.reranker_api_key,
) if secret is not None)
configure_logging(settings.log_level, _log_secrets)
logger = logging.getLogger(__name__)

def _interactive_docs_enabled(environment: str) -> bool:
    return environment.casefold() != "production"


docs_enabled = _interactive_docs_enabled(settings.environment)
app = FastAPI(title=settings.app_name, version="0.6.0",
              description="Shared OmniRAG knowledge workspace API",
              docs_url="/docs" if docs_enabled else None,
              redoc_url="/redoc" if docs_enabled else None,
              openapi_url="/openapi.json" if docs_enabled else None)
rate_limiter = PrincipalRateLimiter(settings.rate_limit_requests_per_minute)


@app.on_event("startup")
def ensure_existing_qdrant_payload_indexes():
    """Add missing filter indexes to an existing collection without touching vectors."""
    indexer = get_qdrant_indexer()
    if indexer.client.collection_exists(indexer.collection):
        indexer.ensure_payload_indexes()


app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID", "X-OmniRAG-Run-ID",
                   "X-OmniRAG-User-ID", "X-OmniRAG-Tenant-ID", "X-OmniRAG-Auth-Timestamp",
                   "X-OmniRAG-Auth-Signature"],
    expose_headers=["X-Request-ID", "X-OmniRAG-Budget-Usage"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    started = time.perf_counter()
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    shared_secret = settings.auth_shared_secret.get_secret_value() if settings.auth_shared_secret else None
    identity = verify_proxy_identity(
        request.headers.get("X-OmniRAG-User-ID"), request.headers.get("X-OmniRAG-Tenant-ID"),
        request.headers.get("X-OmniRAG-Auth-Timestamp"),
        request.headers.get("X-OmniRAG-Auth-Signature"), shared_secret,
    )
    if identity is not None:
        allowed = settings.accepted_identities
        if not identity_is_allowed(identity, allowed):
            return JSONResponse(status_code=403, content={"detail": "Identity is not enabled for this instance"})
        request.state.authenticated_identity = identity
    elif settings.environment.casefold() == "development":
        identity = {"user_id": settings.development_user_id,
                    "tenant_id": settings.development_tenant_id}
    if request.url.path not in {"/health", "/health/ready"} and identity is not None:
        allowed, retry_after = rate_limiter.check(
            (identity["tenant_id"], identity["user_id"])
        )
        if not allowed:
            response = too_many_requests_response(request_id, retry_after)
            runtime_metrics.record(429, (time.perf_counter() - started) * 1000, None)
            response.headers["X-Request-ID"] = request_id
            return response
    run_id = request.headers.get("X-OmniRAG-Run-ID") or request_id
    if len(run_id) > 128 or "\n" in run_id:
        run_id = request_id
    with activate_run_budget(run_id, settings):
        response = await call_next(request)
        usage = current_budget_snapshot()
    if usage is not None:
        import json
        response.headers["X-OmniRAG-Budget-Usage"] = json.dumps(usage, separators=(",", ":"))
    runtime_metrics.record(response.status_code, (time.perf_counter() - started) * 1000, usage)
    response.headers["X-Request-ID"] = request_id
    logger.info("HTTP request completed", extra={"request_id": request_id,
                "method": request.method, "path": request.url.path,
                "status_code": response.status_code})
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "validation_error",
        "message": "The request did not match the API contract.", "request_id": getattr(request.state, "request_id", None),
        "fields": [{"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
                   for error in exc.errors()]}})


@app.exception_handler(BudgetExceeded)
async def provider_budget_exceeded(request: Request, exc: BudgetExceeded):
    return JSONResponse(status_code=429, content={"error": {
        "code": "provider_budget_exceeded", "message": exc.reason,
        "request_id": getattr(request.state, "request_id", None),
        "run_id": request.headers.get("X-OmniRAG-Run-ID"),
        "budget": current_budget_snapshot(),
    }})


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Unhandled API failure", extra={"request_id": request_id,
                    "error_type": type(exc).__name__})
    return JSONResponse(status_code=500, content={"error": {"code": "internal_error",
        "message": "The request could not be completed.", "request_id": request_id}})


v1 = "/api/v1"
for router in (preferences_router, chat_router, conversations_router, documents_router,
               collections_router, product_router):
    app.include_router(router, prefix=v1)
app.include_router(profiles_router, prefix=v1)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.get("/health/ready", tags=["system"])
def readiness():
    try:
        get_qdrant_client().get_collections()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready", "dependency": "qdrant"})
    if settings.ollama_health_url:
        import urllib.request
        try:
            with urllib.request.urlopen(settings.ollama_health_url, timeout=2) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError("Ollama health endpoint returned a non-success status")
        except Exception:
            return JSONResponse(status_code=503, content={"status": "not_ready", "dependency": "ollama"})
    return {"status": "ready", "service": settings.app_name}


@app.get("/metrics", tags=["system"])
def metrics(request: Request):
    """Authenticated, process-local operational counters; not a distributed metrics backend."""
    if not (getattr(request.state, "authenticated_identity", None) or
            settings.environment.casefold() == "development"):
        return JSONResponse(status_code=503, content={"detail": "A trusted authentication provider is not configured"})
    return runtime_metrics.snapshot()


@app.get("/profiles", tags=["profiles"])
def profiles() -> dict[str, list[str]]:
    return {"profiles": [profile.value for profile in PROFILE_CONFIGS]}


frontend_dist = Path(os.environ.get(
    "OMNIRAG_FRONTEND_DIST",
    str(Path(__file__).resolve().parents[4] / "frontend" / "dist"),
))
if frontend_dist.is_dir():
    assets = frontend_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str):
        if frontend_path.startswith(("api/", "health")):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        candidate = (frontend_dist / frontend_path).resolve()
        if frontend_dist.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist / "index.html")
