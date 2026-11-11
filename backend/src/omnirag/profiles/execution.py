"""Single profile orchestrator reusing shared retrieval and answer components."""

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omnirag.agent.graph import ResearchGraph, ResearchLimits
from omnirag.config.profiles import PROFILE_CONFIGS, ProfileConfig, ProfileName
from omnirag.config.model_registry import registry_from_settings
from omnirag.config.settings import get_settings
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.models.query import EscalationPreference, Query
from omnirag.models.response import OmniRAGResponse
from omnirag.retrieval.contracts import RetrievalPolicy, RetrievalScope
from omnirag.retrieval.hybrid_retriever import HybridRetriever
from omnirag.infrastructure.provider_budget import BudgetExceeded
from experiments.runtime import ExperimentConfig, record_stage

logger = logging.getLogger(__name__)


class EscalationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    max_additional_tokens: int = Field(default=0, ge=0)
    max_time_seconds: float = Field(default=0, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0)


class ConversationProfileStore:
    """Thread-safe in-process preference store; replaceable by a persistent backend."""

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._profiles: dict[str, ProfileName] = {}

    def get(self, conversation_id: str) -> ProfileName | None:
        with self._lock:
            return self._profiles.get(conversation_id)

    def set(self, conversation_id: str, profile: ProfileName) -> None:
        with self._lock:
            self._profiles[conversation_id] = profile


class ProfileExecutionService:
    def __init__(self, retriever: HybridRetriever, answer_pipeline: AnswerPipeline,
                 *, research_graph: ResearchGraph | None = None,
                 conversations: ConversationProfileStore | None = None,
                 escalation: EscalationConfig | None = None):
        self.retriever = retriever
        self.answer_pipeline = answer_pipeline
        self.research_graph = research_graph or ResearchGraph(retriever, answer_pipeline)
        self.conversations = conversations or ConversationProfileStore()
        self.escalation = escalation or EscalationConfig()

    async def execute(self, query: Query, scope: RetrievalScope,
                      experiment: ExperimentConfig | None = None) -> OmniRAGResponse:
        profile = self._resolve_profile(query)
        selected_profile = profile
        query = query.model_copy(update={"selected_profile": profile})
        config = PROFILE_CONFIGS[profile]
        switches = experiment.resolved_switches(config) if experiment is not None else None
        if config.name != ProfileName.RESEARCH:
            record_stage("research_planning", 0.0, status="not_applicable")
            record_stage("query_expansion", 0.0, status="not_applicable")
        profile_started = asyncio.get_running_loop().time()
        response = await self._run_profile(query, scope, config, experiment=experiment)
        record_stage("profile_execution", (asyncio.get_running_loop().time() - profile_started) * 1000,
                     metadata={"profile": profile.value})
        response.execution_metadata.setdefault("selected_profile", selected_profile.value)
        response.execution_metadata.setdefault("actual_execution_path", profile.value)

        reason = self._escalation_reason(response)
        target = self._next_profile(profile)
        preference = query.escalation_preference
        if reason and target and preference != EscalationPreference.NEVER and (
            experiment is None or switches["research_escalation"]
        ):
            response.execution_metadata["escalation_reason"] = reason
            response.execution_metadata["escalation_target"] = target.value
            if preference in {EscalationPreference.ASK, EscalationPreference.WHEN_NEEDED}:
                response.execution_metadata["escalation_status"] = "suggested_user_consent_required"
                response.warnings.append("escalation_suggested_requires_user_selection")
                record_stage("research_escalation", 0.0, metadata={"status": "suggested"})
                self._log(response)
                return response
            if preference in {EscalationPreference.AUTO_WITH_NOTICE, EscalationPreference.ALWAYS}:
                allowed, blocked_reason = self._within_escalation_budget(target)
                if allowed:
                    try:
                        escalated_query = query.model_copy(update={"selected_profile": target})
                        upgraded = await asyncio.wait_for(
                            self._run_profile(escalated_query, scope, PROFILE_CONFIGS[target],
                                              experiment=experiment),
                            timeout=self.escalation.max_time_seconds,
                        )
                        upgraded.execution_metadata.update({
                            "selected_profile": selected_profile.value,
                            "actual_execution_path": f"{profile.value}->{target.value}",
                            "escalation_status": "auto_escalated_with_notice",
                            "escalation_reason": reason,
                            "pre_escalation_response_id": response.response_id,
                            "pre_escalation_resource_usage": response.token_usage,
                        })
                        for key, value in response.token_usage.items():
                            upgraded.token_usage[key] = upgraded.token_usage.get(key, 0) + value
                        if response.cost_usd is not None and upgraded.cost_usd is not None:
                            upgraded.cost_usd += response.cost_usd
                        for key, value in response.timings_ms.items():
                            upgraded.timings_ms[key] = upgraded.timings_ms.get(key, 0.0) + value
                        upgraded.warnings.append(f"automatically_escalated:{profile.value}_to_{target.value}")
                        record_stage("research_escalation", 0.0, metadata={"status": "executed"})
                        self._log(upgraded)
                        return upgraded
                    except BudgetExceeded:
                        raise
                    except Exception as exc:
                        response.execution_metadata["escalation_status"] = "failed"
                        response.warnings.append(f"escalation_failed:{type(exc).__name__}")
                else:
                    response.execution_metadata["escalation_status"] = "blocked_by_budget"
                    response.execution_metadata["escalation_budget_reason"] = blocked_reason
                    response.warnings.append("escalation_blocked_by_configured_budget")
        elif preference == EscalationPreference.NEVER:
            response.execution_metadata["escalation_status"] = "disabled_by_user"
        elif reason and target and experiment is not None and not switches["research_escalation"]:
            response.execution_metadata["escalation_status"] = "disabled_by_experiment"
        else:
            response.execution_metadata.setdefault("escalation_status", "not_needed")
        record_stage("research_escalation", 0.0,
                     status="success" if reason and target else "not_applicable",
                     metadata={"status": str(response.execution_metadata.get("escalation_status", "not_needed"))})
        self._log(response)
        return response

    def _resolve_profile(self, query: Query) -> ProfileName:
        if query.conversation_id:
            explicit = "selected_profile" in query.model_fields_set
            if explicit:
                self.conversations.set(query.conversation_id, query.selected_profile)
                return query.selected_profile
            return self.conversations.get(query.conversation_id) or query.selected_profile
        return query.selected_profile

    async def _run_profile(self, query: Query, scope: RetrievalScope,
                           config: ProfileConfig,
                           experiment: ExperimentConfig | None = None) -> OmniRAGResponse:
        retrieval_policy = self._retrieval_policy(config)
        if experiment is not None:
            retrieval_policy = experiment.apply_to_policy(retrieval_policy, config)
        settings = get_settings()
        registry = registry_from_settings(settings)
        if config.name == ProfileName.RESEARCH:
            limits = ResearchLimits(
                max_rounds=config.research_max_rounds,
                max_subqueries=config.research_max_subqueries,
                max_evidence=config.research_max_evidence,
                max_token_budget=config.research_max_tokens,
                max_output_tokens=config.generation_max_output_tokens,
                max_time_seconds=config.research_max_time_seconds,
                finalization_reserve_seconds=min(settings.generation_timeout_seconds,
                                                 config.research_max_time_seconds * 0.75),
                max_correction_attempts=1,
            )
            response = await self.research_graph.run(
                query, scope, retrieval_policy=retrieval_policy, limits=limits,
                query_expansion=(experiment.resolved_switches(config)["query_expansion"]
                                 if experiment is not None else config.multi_query_enabled),
                verification_enabled=(experiment.resolved_switches(config)["verification"]
                                      if experiment is not None else config.claim_verification_enabled),
                semantic_verification=(experiment.resolved_switches(config)["semantic_verification"]
                                       if experiment is not None else True))
        else:
            retrieval = await self.retriever.retrieve(query, scope=scope, policy=retrieval_policy)
            response = await asyncio.to_thread(
                self.answer_pipeline.answer, query, retrieval,
                verification_depth=("citations_only" if experiment is not None and
                                    not experiment.resolved_switches(config)["verification"]
                                    else config.verification_depth.value),
                generation_style=config.generation_style,
                generation_max_output_tokens=config.generation_max_output_tokens,
                allow_correction=(config.claim_verification_enabled and
                                  (experiment is None or experiment.resolved_switches(config)["verification"])),
                abstain_on_weak=config.abstain_on_weak_evidence,
                semantic_verification=(experiment.resolved_switches(config)["semantic_verification"]
                                       if experiment is not None else True),
            )
        response.profile_used = config.name
        response.execution_metadata.update({
            "selected_profile": config.name.value,
            "profile_policy": self._policy_summary(config, retrieval_policy),
            "verification_depth": config.verification_depth.value,
            "profile_defaults_benchmarked": config.defaults_benchmarked,
            "model_registry_slots": {
                key: {"provider": getattr(registry, key).provider,
                      "model_id": getattr(registry, key).model_id}
                for key in ("generation", "verification", "generation_fallback",
                            "verification_fallback", "research", "reranking", "embedding")
            },
        })
        if experiment is not None:
            response.execution_metadata["experiment_config"] = experiment.model_dump(mode="json")
            response.execution_metadata["experiment_switches"] = experiment.resolved_switches(config)
        return response

    @staticmethod
    def _retrieval_policy(config: ProfileConfig) -> RetrievalPolicy:
        return RetrievalPolicy(
            dense_limit=config.dense_candidates, sparse_limit=config.sparse_candidates,
            max_evidence=config.research_max_evidence if config.name == ProfileName.RESEARCH else 8,
            evidence_token_budget=(
                max(1, config.research_max_tokens - config.generation_max_output_tokens - 100)
                if config.name == ProfileName.RESEARCH else (config.context_budget_tokens or 3000)),
            expansion_token_budget=config.expansion_token_budget,
            expansion_max_count=config.expansion_max_count,
            expand_parents=config.expand_parents, expand_children=config.expand_children,
            expand_neighbours=config.expand_neighbours,
            expand_cross_references=config.expand_cross_references,
            reranking_enabled=config.reranking_enabled,
        )

    @staticmethod
    def _policy_summary(config: ProfileConfig, retrieval: RetrievalPolicy) -> dict[str, object]:
        summary = {"retrieval_candidates": {"dense": retrieval.dense_limit, "sparse": retrieval.sparse_limit},
                   "evidence_token_budget": retrieval.evidence_token_budget,
                   "expansion_max_count": retrieval.expansion_max_count,
                   "reranking_enabled": retrieval.reranking_enabled,
                   "generation_max_output_tokens": config.generation_max_output_tokens,
                   "generation_style": config.generation_style,
                   "verification_depth": config.verification_depth.value,
                   "defaults_benchmarked": config.defaults_benchmarked}
        if config.name == ProfileName.RESEARCH:
            summary["research_limits"] = {"max_rounds": config.research_max_rounds,
                                           "max_subqueries": config.research_max_subqueries,
                                           "max_evidence": config.research_max_evidence,
                                           "max_total_tokens": config.research_max_tokens,
                                           "max_time_seconds": config.research_max_time_seconds,
                                           "max_correction_attempts": 1}
        return summary

    @staticmethod
    def _next_profile(profile: ProfileName) -> ProfileName | None:
        return {ProfileName.LITE: ProfileName.CORE, ProfileName.CORE: ProfileName.RESEARCH,
                ProfileName.RESEARCH: None}[profile]

    @staticmethod
    def _escalation_reason(response: OmniRAGResponse) -> str | None:
        state = response.evidence_sufficiency.get("state")
        if response.abstained:
            return response.abstention_reason or "answer_abstained"
        if state in {"insufficient_evidence", "weak_or_unrelated_evidence"}:
            return f"evidence_sufficiency:{state}"
        if response.verification_status.value in {"unsupported", "contradicted"}:
            return f"answer_reliability:{response.verification_status.value}"
        return None

    def _within_escalation_budget(self, target: ProfileName) -> tuple[bool, str]:
        if not self.escalation.enabled:
            return False, "automatic_escalation_not_enabled"
        if self.escalation.max_additional_tokens <= 0 or self.escalation.max_time_seconds <= 0:
            return False, "additional_token_and_time_budgets_must_be_configured"
        config = PROFILE_CONFIGS[target]
        required_tokens = (config.research_max_tokens if target == ProfileName.RESEARCH else
                           (config.context_budget_tokens or 0) + config.generation_max_output_tokens)
        if required_tokens > self.escalation.max_additional_tokens:
            return False, "target_profile_exceeds_additional_token_budget"
        if target == ProfileName.RESEARCH and config.research_max_time_seconds > self.escalation.max_time_seconds:
            return False, "research_profile_exceeds_escalation_time_budget"
        if self.escalation.max_cost_usd is not None:
            return False, "cost_ceiling_cannot_be_enforced_without_provider_price_estimate"
        return True, "within_explicit_token_and_time_budgets"

    @staticmethod
    def _log(response: OmniRAGResponse) -> None:
        logger.info("Profile execution completed", extra={
            "response_id": response.response_id,
            "selected_profile": response.execution_metadata.get("selected_profile"),
            "actual_execution_path": response.execution_metadata.get("actual_execution_path"),
            "escalation_status": response.execution_metadata.get("escalation_status"),
            "verification_status": response.verification_status.value,
            "abstained": response.abstained,
            "token_usage": response.token_usage,
            "cost_usd": response.cost_usd,
            "warnings_count": len(response.warnings),
        })
