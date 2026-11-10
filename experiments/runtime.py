"""Reproducible experiment configuration, traces, manifests, and replay checks."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import inspect
import math
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable
from uuid import uuid4
from utils.atomic_file import atomic_replace

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAFE_STAGE_NAMES = {
    "request_initialization", "profile_execution", "query_processing", "embedding",
    "dense_retrieval", "sparse_retrieval", "fusion", "reranking", "evidence_selection",
    "query_expansion", "research_planning", "context_construction", "generation",
    "citation_validation", "claim_verification", "research_escalation", "final_response",
    "retrieval",
}
_SAFE_TRACE_METADATA = {"profile", "candidate_count", "citation_count", "evidence_count",
                        "subquery_count", "provider_call_count", "status", "abstained"}


class ExperimentConfig(BaseModel):
    """Optional overrides; None inherits the active profile's existing behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dense_retrieval: bool | None = None
    sparse_retrieval: bool | None = None
    hybrid_fusion: bool | None = None
    reranking: bool | None = None
    query_expansion: bool | None = None
    evidence_selection: bool | None = None
    verification: bool | None = None
    semantic_verification: bool | None = None
    research_escalation: bool | None = None
    dataset_id: str | None = Field(default=None, min_length=1, max_length=128)
    dataset_version: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="before")
    @classmethod
    def validate_dataset_labels(cls, values):
        if isinstance(values, dict):
            for key in ("dataset_id", "dataset_version"):
                value = values.get(key)
                if value is not None and (not isinstance(value, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value
                )):
                    raise ValueError(f"{key} must be a short identifier, not a path or free text")
        return values

    @model_validator(mode="after")
    def validate_switches(self):
        if self.dense_retrieval is False and self.sparse_retrieval is False:
            raise ValueError("At least one of dense_retrieval or sparse_retrieval must remain enabled")
        if self.hybrid_fusion is True and (
            self.dense_retrieval is False or self.sparse_retrieval is False
        ):
            raise ValueError("hybrid_fusion requires both dense and sparse retrieval")
        return self

    def apply_to_policy(self, policy, profile_config):
        updates = {
            "dense_retrieval_enabled": self.dense_retrieval if self.dense_retrieval is not None else True,
            "sparse_retrieval_enabled": self.sparse_retrieval if self.sparse_retrieval is not None else True,
            "hybrid_fusion_enabled": (self.hybrid_fusion if self.hybrid_fusion is not None else
                                      self.dense_retrieval is not False and self.sparse_retrieval is not False),
            "reranking_enabled": self.reranking if self.reranking is not None else policy.reranking_enabled,
            "evidence_selection_enabled": self.evidence_selection if self.evidence_selection is not None else True,
        }
        if self.dense_retrieval is None:
            updates["dense_retrieval_enabled"] = True
        if self.sparse_retrieval is None:
            updates["sparse_retrieval_enabled"] = True
        if self.hybrid_fusion is None:
            updates["hybrid_fusion_enabled"] = True
        return policy.model_copy(update=updates)

    def resolved_switches(self, profile_config) -> dict[str, bool]:
        dense = self.dense_retrieval if self.dense_retrieval is not None else True
        sparse = self.sparse_retrieval if self.sparse_retrieval is not None else True
        return {
            "dense_retrieval": dense,
            "sparse_retrieval": sparse,
            "hybrid_fusion": self.hybrid_fusion if self.hybrid_fusion is not None else dense and sparse,
            "reranking": self.reranking if self.reranking is not None else profile_config.reranking_enabled,
            "query_expansion": self.query_expansion if self.query_expansion is not None else profile_config.multi_query_enabled,
            "evidence_selection": self.evidence_selection if self.evidence_selection is not None else True,
            "verification": self.verification if self.verification is not None else profile_config.claim_verification_enabled,
            "semantic_verification": self.semantic_verification if self.semantic_verification is not None else True,
            "research_escalation": self.research_escalation if self.research_escalation is not None else True,
        }


class StageTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    duration_ms: float | None = Field(default=None, ge=0)
    status: str
    metadata: dict[str, int | float | bool | str] = Field(default_factory=dict)
    error_type: str | None = None


class StageTraceCollector:
    def __init__(self):
        self.stages: list[StageTrace] = []

    def record(self, name: str, duration_ms: float | None, status: str = "success",
               metadata: dict[str, Any] | None = None, error_type: str | None = None) -> None:
        if name not in _SAFE_STAGE_NAMES:
            raise ValueError("Unsupported experiment stage name")
        if status not in {"success", "failure", "not_applicable", "not_run"}:
            raise ValueError("Invalid stage status")
        safe = {}
        for key, value in (metadata or {}).items():
            if key not in _SAFE_TRACE_METADATA:
                continue
            if isinstance(value, (str, int, float, bool)):
                if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value):
                    continue
                safe[key] = value
        if error_type is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", error_type):
            error_type = None
        self.stages.append(StageTrace(name=name, duration_ms=duration_ms, status=status,
                                      metadata=safe, error_type=error_type))


_trace_context: ContextVar[StageTraceCollector | None] = ContextVar("omnirag_experiment_trace", default=None)


def current_trace() -> StageTraceCollector | None:
    return _trace_context.get()


def record_stage(name: str, duration_ms: float | None, status: str = "success",
                 metadata: dict[str, Any] | None = None, error_type: str | None = None) -> None:
    collector = current_trace()
    if collector is not None:
        collector.record(name, duration_ms, status, metadata, error_type)


class ExperimentRunStore:
    """Atomic JSON artifacts stored under the configured application data directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id or ""):
            raise ValueError("Invalid run ID")
        return self.root / run_id

    def write_json(self, run_id: str, filename: str, value: dict[str, Any]) -> None:
        if filename not in {"manifest.json", "trace.json", "result.json", "evaluation.json"}:
            raise ValueError("Unsupported experiment artifact")
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / filename
        temp = target.with_suffix(target.suffix + ".tmp")
        value = _redact_configured_secrets(value)
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        atomic_replace(temp, target)

    def read_json(self, run_id: str, filename: str) -> dict[str, Any]:
        if filename not in {"manifest.json", "trace.json", "result.json", "evaluation.json"}:
            raise ValueError("Unsupported experiment artifact")
        return json.loads((self.run_dir(run_id) / filename).read_text(encoding="utf-8"))


def _redact_configured_secrets(value: Any) -> Any:
    """Redact configured credentials from all persisted Phase 7b artifact strings."""
    from config.settings import get_settings
    settings = get_settings()
    secret_fields = (
        "auth_shared_secret", "qdrant_api_key", "embedding_api_key", "generation_api_key",
        "generation_fallback_api_key", "verification_api_key", "verification_fallback_api_key",
        "reranker_api_key",
    )
    secrets = [getattr(settings, field).get_secret_value()
               for field in secret_fields if getattr(settings, field, None) is not None]

    def visit(item):
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            for secret in secrets:
                if secret:
                    item = item.replace(secret, "[REDACTED]")
        return item

    return visit(value)


def _git_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=2, check=True).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_worktree_state() -> str:
    try:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                timeout=3, check=True).stdout
        return "dirty" if status.strip() else "clean"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _available_settings(settings) -> dict[str, Any]:
    return {
        "generation_provider": settings.generation_provider,
        "generation_model": settings.generation_model_id,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model_id,
        "reranker_provider": settings.reranker_provider,
        "reranker_model": settings.reranker_model_id,
        "verification_provider": settings.verification_provider,
        "verification_model": settings.verification_model_id,
        "verification_model_digest": settings.verification_model_digest,
        "verification_threshold": settings.verification_threshold,
        "generation_timeout_seconds": settings.generation_timeout_seconds,
        "generation_max_retries": settings.generation_max_retries,
        "generation_temperature": settings.generation_temperature,
        "reranker_timeout_seconds": settings.reranker_timeout_seconds,
        "verification_timeout_seconds": settings.verification_timeout_seconds,
        "embedding_dimensions": settings.embedding_dimensions,
        "validation_max_cost_usd": settings.validation_max_cost_usd,
        "validation_max_total_tokens": settings.validation_max_total_tokens,
        "validation_max_calls": settings.validation_max_calls,
    }


def _safe_manifest_config(value: Any) -> Any:
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
                continue
            if any(term in key.casefold() for term in ("secret", "credential", "content",
                                                        "prompt", "query", "path", "endpoint", "key")):
                continue
            safe[key] = _safe_manifest_config(item)
        return safe
    if isinstance(value, list):
        return [_safe_manifest_config(item) for item in value
                if isinstance(item, (dict, list, str, int, float, bool))]
    if isinstance(value, str):
        return value if re.fullmatch(r"[A-Za-z0-9_.:/+-]{1,128}", value) else None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return None


class ReplayValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    unavailable_dependencies: list[str] = Field(default_factory=list)
    environment_differences: list[str] = Field(default_factory=list)


class ReplayUnavailable(RuntimeError):
    def __init__(self, report: ReplayValidation):
        self.report = report
        super().__init__("Experiment replay dependencies are unavailable")


class ExperimentRunner:
    REQUIRED_STAGES = (
        "request_initialization", "query_processing", "embedding", "dense_retrieval",
        "sparse_retrieval", "fusion", "reranking", "evidence_selection", "query_expansion",
        "research_planning", "context_construction",
        "generation", "citation_validation", "claim_verification", "research_escalation",
        "final_response",
    )

    def __init__(self, store: ExperimentRunStore):
        self.store = store

    def make_manifest(self, *, profile: str, config: ExperimentConfig, settings,
                      retrieval_config: dict[str, Any] | None = None,
                      resolved_switches: dict[str, bool] | None = None,
                      parent_run_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or str(uuid4())
        return {
            "schema_version": 1,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "revision": _git_revision(),
            "worktree_state": _git_worktree_state(),
            "application_version": "0.6.0",
            "profile": profile,
            "experiment_config": config.model_dump(mode="json"),
            "resolved_switches": resolved_switches or {},
            "models": {key: _available_settings(settings)[key] for key in (
                "generation_provider", "generation_model", "embedding_provider", "embedding_model",
                "reranker_provider", "reranker_model", "verification_provider", "verification_model",
                "verification_model_digest", "verification_threshold",
            )},
            "retrieval_config": _safe_manifest_config(retrieval_config or {}),
            "dataset": {"id": config.dataset_id, "version": config.dataset_version},
            "execution_parameters": {
                "generation_timeout_seconds": settings.generation_timeout_seconds,
                "generation_max_retries": settings.generation_max_retries,
                "generation_temperature": settings.generation_temperature,
                "reranker_timeout_seconds": settings.reranker_timeout_seconds,
                "verification_timeout_seconds": settings.verification_timeout_seconds,
                "verification_threshold": settings.verification_threshold,
                "embedding_dimensions": settings.embedding_dimensions,
                "validation_max_cost_usd": settings.validation_max_cost_usd,
                "validation_max_total_tokens": settings.validation_max_total_tokens,
                "validation_max_calls": settings.validation_max_calls,
            },
            "environment": {"python": platform.python_version(), "platform": sys.platform},
            "artifacts": {"trace": "trace.json", "result": "result.json",
                          "evaluation": None, "manifest": "manifest.json"},
        }

    async def run(self, *, profile: str, config: ExperimentConfig, settings,
                  execute: Callable[[], Any], retrieval_config: dict[str, Any] | None = None,
                  resolved_switches: dict[str, bool] | None = None,
                  parent_run_id: str | None = None, run_id: str | None = None) -> Any:
        manifest = self.make_manifest(profile=profile, config=config, settings=settings,
                                      retrieval_config=retrieval_config, resolved_switches=resolved_switches,
                                      parent_run_id=parent_run_id,
                                      run_id=run_id)
        run_id = manifest["run_id"]
        directory = self.store.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=False)
        self.store.write_json(run_id, "manifest.json", manifest)
        collector = StageTraceCollector()
        token = _trace_context.set(collector)
        started = time.perf_counter()
        collector.record("request_initialization", 0.0, metadata={"profile": profile})
        try:
            outcome = execute()
            response = await outcome if inspect.isawaitable(outcome) else outcome
        except Exception as exc:
            collector.record("final_response", (time.perf_counter() - started) * 1000,
                             status="failure", error_type=type(exc).__name__)
            manifest["status"] = "failed"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.store.write_json(run_id, "manifest.json", manifest)
            self._write_trace(run_id, collector)
            self.store.write_json(run_id, "result.json", {"status": "failed", "error_type": type(exc).__name__})
            raise
        finally:
            _trace_context.reset(token)
        collector.record("final_response", (time.perf_counter() - started) * 1000,
                         metadata={"abstained": response.abstained,
                                   "citation_count": len(response.citations),
                                   "evidence_count": len(response.evidence_references)})
        failure_prefixes = ("generation_failed", "research_finalize_failed", "dense_retrieval_failed",
                            "sparse_retrieval_failed", "reranker_failed", "correction_failed")
        failure_codes = [warning.split(":", 1)[0] for warning in response.warnings
                         if warning.startswith(failure_prefixes)]
        run_status = "completed_with_errors" if failure_codes else "completed"
        manifest["status"] = run_status
        manifest["models"]["generation_resolved"] = response.generation_metadata.get("model")
        manifest["models"]["generation_provider_resolved"] = response.generation_metadata.get("provider")
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.store.write_json(run_id, "manifest.json", manifest)
        self._write_trace(run_id, collector)
        self.store.write_json(run_id, "result.json", {
            "status": run_status, "response_id": response.response_id,
            "profile": response.profile_used.value, "abstained": response.abstained,
            "abstention_reason": response.abstention_reason,
            "generation_metadata": response.generation_metadata,
            "timings_ms": response.timings_ms,
            "token_usage": response.token_usage, "cost_usd": response.cost_usd,
            "citation_count": len(response.citations),
            "evidence_count": len(response.evidence_references),
            "warning_codes": [code for item in response.warnings
                              if re.fullmatch(r"[A-Za-z0-9_.-]+", code := item.split(":", 1)[0])],
            "failure_codes": failure_codes,
        })
        response.execution_metadata["experiment_run_id"] = run_id
        return response

    def _write_trace(self, run_id: str, collector: StageTraceCollector) -> None:
        present = {item.name for item in collector.stages}
        for name in self.REQUIRED_STAGES:
            if name not in present:
                collector.record(name, None, status="not_run")
        self.store.write_json(run_id, "trace.json", {
            "run_id": run_id, "stages": [stage.model_dump(mode="json") for stage in collector.stages]
        })

    def validate_replay(self, parent_run_id: str, *, settings,
                        available_datasets: set[tuple[str, str | None]] | None = None) -> ReplayValidation:
        parent = self.store.read_json(parent_run_id, "manifest.json")
        missing: list[str] = []
        differences: list[str] = []
        configured = _available_settings(settings)
        from config.profiles import PROFILE_CONFIGS, ProfileName
        saved_config = ExperimentConfig.model_validate(parent["experiment_config"])
        switches = parent.get("resolved_switches") or saved_config.resolved_switches(
            PROFILE_CONFIGS[ProfileName(parent["profile"])])
        required_keys = ["generation_provider", "generation_model"]
        if switches.get("dense_retrieval", True):
            required_keys.extend(("embedding_provider", "embedding_model"))
        if switches.get("reranking", False):
            required_keys.extend(("reranker_provider", "reranker_model"))
        parent_models = parent.get("models", {})
        for key in required_keys:
            expected = parent_models.get(key)
            if expected is None:
                missing.append(f"model:{key}:not-recorded")
            elif configured.get(key) != expected:
                missing.append(f"model:{key}:{expected}")
        recorded_models = parent_models
        for resolved_key, configured_key in (("generation_resolved", "generation_model"),
                                               ("generation_provider_resolved", "generation_provider")):
            resolved = recorded_models.get(resolved_key)
            if resolved is not None and configured.get(configured_key) != resolved:
                missing.append(f"model:{resolved_key}:{resolved}")
        dataset = parent.get("dataset", {})
        if dataset.get("id"):
            known = available_datasets or set()
            if (dataset["id"], dataset.get("version")) not in known:
                missing.append(f"dataset:{dataset['id']}@{dataset.get('version') or 'unspecified'}")
        env = parent.get("environment", {})
        if env.get("python") != platform.python_version():
            differences.append(f"python:{env.get('python')}->{platform.python_version()}")
        if env.get("platform") != sys.platform:
            differences.append(f"platform:{env.get('platform')}->{sys.platform}")
        current_revision = _git_revision()
        if parent.get("revision") and current_revision and parent["revision"] != current_revision:
            differences.append(f"revision:{parent['revision']}->{current_revision}")
        current_worktree = _git_worktree_state()
        if parent.get("worktree_state") and parent["worktree_state"] != current_worktree:
            differences.append(f"worktree_state:{parent['worktree_state']}->{current_worktree}")
        execution = parent.get("execution_parameters", {})
        current_parameters = _available_settings(settings)
        for key in ("generation_timeout_seconds", "generation_max_retries", "generation_temperature",
                    "reranker_timeout_seconds", "verification_timeout_seconds", "verification_threshold",
                    "verification_model_digest", "embedding_dimensions",
                    "validation_max_cost_usd", "validation_max_total_tokens", "validation_max_calls"):
            if execution.get(key) != current_parameters.get(key):
                differences.append(f"{key}:{execution.get(key)}->{current_parameters.get(key)}")
        return ReplayValidation(available=not missing, unavailable_dependencies=missing,
                                environment_differences=differences)

    async def replay(self, parent_run_id: str, *, settings,
                     execute: Callable[[ExperimentConfig, dict[str, Any]], Any],
                     available_datasets: set[tuple[str, str | None]] | None = None,
                     run_id: str | None = None) -> Any:
        validation = self.validate_replay(parent_run_id, settings=settings,
                                          available_datasets=available_datasets)
        if not validation.available:
            raise ReplayUnavailable(validation)
        parent = self.store.read_json(parent_run_id, "manifest.json")
        config = ExperimentConfig.model_validate(parent["experiment_config"])
        return await self.run(profile=parent["profile"], config=config, settings=settings,
                              execute=lambda: execute(config, parent.get("retrieval_config", {})),
                              retrieval_config=parent.get("retrieval_config", {}),
                              resolved_switches=parent.get("resolved_switches", {}),
                              parent_run_id=parent_run_id, run_id=run_id)

    def record_evaluation(self, run_id: str, metrics: dict[str, float | int | bool]) -> None:
        manifest = self.store.read_json(run_id, "manifest.json")
        clean: dict[str, float | int | bool] = {}
        for key, value in metrics.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", key) or any(
                term in key.casefold() for term in ("secret", "credential", "content", "prompt", "query", "key")
            ):
                raise ValueError("Evaluation metric name is not an allowed identifier")
            if not isinstance(value, (float, int, bool)):
                raise ValueError("Evaluation metrics must be numeric or boolean")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Evaluation metrics must be finite")
            clean[key] = value
        self.store.write_json(run_id, "evaluation.json", {"run_id": run_id, "metrics": clean})
        manifest["artifacts"]["evaluation"] = "evaluation.json"
        self.store.write_json(run_id, "manifest.json", manifest)
