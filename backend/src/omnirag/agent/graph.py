"""Bounded LangGraph Research orchestration over shared retrieval and answer services."""

import asyncio
import hashlib
import json
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnirag.agent.evidence_ledger import LedgerEvidenceReference, ResearchLedgerEntry
from omnirag.agent.planner import DeterministicResearchPlanner, ResearchPlanner
from omnirag.agent.state import ResearchState
from omnirag.config.profiles import PROFILE_CONFIGS, ProfileName
from omnirag.generation.answer_pipeline import AnswerPipeline
from omnirag.ingestion.chunking.structure_chunker import estimate_tokens
from omnirag.models.query import Query
from omnirag.retrieval.contracts import (
    EvidenceAssessment, EvidenceAssessmentState, RetrievalFilters, RetrievalPolicy,
    RetrievalResult, RetrievalScope,
)
from omnirag.retrieval.hybrid_retriever import HybridRetriever
from omnirag.retrieval.query_analyzer import QueryAnalyzer
from omnirag.verification.contradiction_detector import detect_contradictions
from omnirag.infrastructure.provider_budget import BudgetExceeded
from omnirag.experiments.runtime import record_stage


class ResearchLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rounds: int = Field(default=3, ge=1, le=8)
    max_subqueries: int = Field(default=8, ge=1, le=32)
    max_evidence: int = Field(default=24, ge=1, le=200)
    max_token_budget: int = Field(default=6000, gt=0)
    max_output_tokens: int = Field(default=1800, gt=0)
    max_prompt_overhead_tokens: int = Field(default=100, ge=0)
    max_time_seconds: float = Field(default=60, gt=0, le=600)
    finalization_reserve_seconds: float = Field(default=30.0, gt=0)
    max_correction_attempts: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def token_budget_can_fit_generation(self):
        if self.max_token_budget <= self.max_output_tokens + self.max_prompt_overhead_tokens:
            raise ValueError("Research token budget must cover output and prompt overhead")
        return self

class ResearchGraph:
    def __init__(self, retriever: HybridRetriever, answer_pipeline: AnswerPipeline,
                 *, planner: ResearchPlanner | None = None):
        self.retriever = retriever
        self.answer_pipeline = answer_pipeline
        self.planner = planner or DeterministicResearchPlanner()
        graph = StateGraph(ResearchState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("plan", self._plan)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("assess_and_ledger", self._assess_and_ledger)
        graph.add_node("plan_followup", self._plan_followup)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "assess_and_ledger")
        graph.add_edge("assess_and_ledger", "plan_followup")
        graph.add_conditional_edges("plan_followup", self._route, {
            "retrieve": "retrieve", "finalize": "finalize",
        })
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    async def run(self, query: Query, scope: RetrievalScope, *,
                  retrieval_policy: RetrievalPolicy, limits: "ResearchLimits",
                  query_expansion: bool = True, verification_enabled: bool = True,
                  semantic_verification: bool = True):
        initial: ResearchState = {
            "query": query, "scope": scope, "retrieval_policy": retrieval_policy,
            "limits": limits, "started_at": time.perf_counter(), "round_number": 0,
            "query_analysis": None, "research_plan": [],
            "pending_queries": [], "executed_queries": [], "round_results": [],
            "all_results": [], "ledger": [], "unresolved_questions": [],
            "new_evidence_count": 0, "research_termination_reason": "",
            "contradictions": [], "warnings": [],
            "query_expansion_enabled": query_expansion,
            "verification_enabled": verification_enabled,
            "semantic_verification_enabled": semantic_verification,
        }
        result = await self.graph.ainvoke(initial, config={"recursion_limit": limits.max_rounds * 4 + 12})
        return result["response"]

    def _analyze(self, state: ResearchState) -> dict[str, Any]:
        query: Query = state["query"]
        analysis = QueryAnalyzer().analyze(query)
        return {"query_analysis": analysis, "research_termination_reason": ""}

    def _plan(self, state: ResearchState) -> dict[str, Any]:
        plan_started = time.perf_counter()
        query: Query = state["query"]
        limits: ResearchLimits = state["limits"]
        # Keep explicit textual metadata directives intact across decomposition.
        if (not state.get("query_expansion_enabled", True) or
                state["query_analysis"].extracted_filters.model_dump(exclude_defaults=True)):
            plan = [query.query_text]
        else:
            plan = self.planner.decompose(query.query_text, limits.max_subqueries)
        record_stage("research_planning", (time.perf_counter() - plan_started) * 1000,
                     metadata={"subquery_count": len(plan)})
        record_stage("query_expansion", (time.perf_counter() - plan_started) * 1000,
                     status="success" if state.get("query_expansion_enabled", True) else "not_applicable",
                     metadata={"subquery_count": len(plan)})
        return {"research_plan": plan, "pending_queries": plan, "executed_queries": []}

    async def _retrieve(self, state: ResearchState) -> dict[str, Any]:
        limits: ResearchLimits = state["limits"]
        # Preserve time for final answer generation. Retrieval used to be allowed
        # to consume the entire Research deadline, leaving finalize only 10 ms.
        # Keep the overall limit unchanged and reserve at most 75% of it, bounded
        # by the configured generation-call timeout.
        remaining = self._retrieval_time_remaining(state)
        if remaining <= 0:
            return {"round_results": [], "warnings": [*state.get("warnings", []),
                                                          "research_finalization_time_reserved"],
                    "research_termination_reason": "finalization_time_reserved"}
        queries = state.get("pending_queries", [])[:max(0, limits.max_subqueries -
                                                       len(state.get("executed_queries", [])))]
        if not queries:
            return {"round_results": [], "research_termination_reason": "subquery_budget_exhausted"}
        original: Query = state["query"]
        tasks = [asyncio.create_task(self.retriever.retrieve(
            original.model_copy(update={"query_text": subquery, "selected_profile": ProfileName.RESEARCH}),
            scope=state["scope"], policy=state["retrieval_policy"])) for subquery in queries]
        done, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        results, warnings = [], list(state.get("warnings", []))
        budget_exceeded = False
        for subquery, task in zip(queries, tasks):
            if task not in done:
                warnings.append("research_subquery_timed_out")
                continue
            try:
                results.append({"sub_question": subquery, "result": task.result()})
            except BudgetExceeded:
                budget_exceeded = True
                warnings.append("provider_budget_exceeded")
            except Exception as exc:
                warnings.append(f"research_retrieval_failed:{type(exc).__name__}")
        if budget_exceeded:
            return {"round_results": [], "research_termination_reason": "budget_exceeded",
                    "warnings": warnings}
        termination_reason = "finalization_time_reserved" if pending else ""
        if pending:
            warnings.append("research_finalization_time_reserved")
        return {
            "round_results": results,
            "all_results": [*state.get("all_results", []), *results],
            "executed_queries": [*state.get("executed_queries", []), *queries],
            "round_number": state.get("round_number", 0) + 1,
            "warnings": warnings,
            "research_termination_reason": termination_reason,
        }

    @staticmethod
    def _retrieval_time_remaining(state: ResearchState) -> float:
        limits: ResearchLimits = state["limits"]
        finalization_reserve = min(limits.finalization_reserve_seconds,
                                   limits.max_time_seconds * 0.75)
        retrieval_deadline = limits.max_time_seconds - finalization_reserve
        elapsed = time.perf_counter() - state["started_at"]
        return max(0.0, retrieval_deadline - elapsed)

    def _assess_and_ledger(self, state: ResearchState) -> dict[str, Any]:
        ledger = list(state.get("ledger", []))
        round_number = state.get("round_number", 0)
        round_results = state.get("round_results", [])
        prior_ids = {item.chunk.chunk_id for record in state.get("all_results", [])[:-len(round_results or [None])]
                     for item in record["result"].evidence} if round_results else set()
        new_ids = {item.chunk.chunk_id for record in round_results for item in record["result"].evidence} - prior_ids
        for index, record in enumerate(round_results, 1):
            result: RetrievalResult = record["result"]
            contradictions = detect_contradictions(result.evidence)
            evidence_refs = [LedgerEvidenceReference(
                document_id=item.chunk.document_id, chunk_id=item.chunk.chunk_id,
                source=item.chunk.source, version=item.chunk.version,
                section_path=item.chunk.section_path, excerpt=item.chunk.content[:400],
            ) for item in result.evidence]
            unresolved = ([record["sub_question"]] if result.assessment.state in {
                EvidenceAssessmentState.INSUFFICIENT, EvidenceAssessmentState.WEAK,
                EvidenceAssessmentState.PARTIAL,
            } else [])
            ledger.append(ResearchLedgerEntry(
                step_id=f"R{round_number}S{index}", round_number=round_number,
                sub_question=record["sub_question"], evidence_references=evidence_refs,
                supporting_findings=[item.chunk.content[:300] for item in result.evidence],
                contradictory_findings=contradictions, unresolved_questions=unresolved,
                provenance=[f"{item.chunk.source}#{item.chunk.chunk_id}" for item in result.evidence],
                evidence_sufficiency_state=result.assessment.state.value,
            ))
        unresolved = [record["sub_question"] for record in round_results
                      if record["result"].assessment.state in {
                          EvidenceAssessmentState.INSUFFICIENT, EvidenceAssessmentState.WEAK,
                          EvidenceAssessmentState.PARTIAL,
                      }]
        merged, contradictions = self._merge(state.get("all_results", []), state["limits"])
        if merged is not None:
            merged = merged.model_copy(update={
                "original_query": state["query"].query_text,
                "normalized_query": state["query_analysis"].normalized_query,
                "query_analysis": state["query_analysis"],
                "applied_filters": state["query_analysis"].extracted_filters,
            }, deep=True)
        limits: ResearchLimits = state["limits"]
        tokens = merged.selected_token_estimate if merged else 0
        reason = state.get("research_termination_reason") or ""
        if not reason:
            if merged is None:
                reason = "retrieval_failed"
            elif merged.assessment.state == EvidenceAssessmentState.STRONG:
                reason = "evidence_sufficient"
            elif not new_ids and round_number > 1:
                reason = "diminishing_evidence"
            elif tokens + limits.max_output_tokens + limits.max_prompt_overhead_tokens >= limits.max_token_budget:
                reason = "token_budget_exhausted"
            elif len(merged.evidence) >= limits.max_evidence:
                reason = "evidence_budget_exhausted"
            elif time.perf_counter() - state["started_at"] >= limits.max_time_seconds:
                reason = "time_budget_exhausted"
        return {
            "ledger": ledger, "merged_retrieval": merged, "unresolved_questions": unresolved,
            "new_evidence_count": len(new_ids),
            "contradictions": contradictions, "research_termination_reason": reason,
        }

    def _plan_followup(self, state: ResearchState) -> dict[str, Any]:
        followup_started = time.perf_counter()
        if state.get("research_termination_reason"):
            return {}
        limits: ResearchLimits = state["limits"]
        allowed = max(0, limits.max_subqueries - len(set(state.get("executed_queries", []))))
        followups = self.planner.followups(
            state.get("unresolved_questions", []),
            [record["result"] for record in state.get("all_results", [])],
            {item.casefold() for item in state.get("executed_queries", [])}, allowed,
        ) if allowed and state.get("query_expansion_enabled", True) else []
        record_stage("query_expansion", (time.perf_counter() - followup_started) * 1000,
                     status="success" if state.get("query_expansion_enabled", True) else "not_applicable",
                     metadata={"subquery_count": len(followups)})
        return {"pending_queries": followups,
                "research_termination_reason": "" if followups else "no_actionable_followup"}

    def _route(self, state: ResearchState) -> str:
        limits: ResearchLimits = state["limits"]
        if state.get("research_termination_reason"):
            return "finalize"
        if state.get("round_number", 0) >= limits.max_rounds:
            return "finalize"
        if len(state.get("executed_queries", [])) >= limits.max_subqueries:
            return "finalize"
        if not state.get("pending_queries"):
            return "finalize"
        return "retrieve"

    async def _finalize(self, state: ResearchState) -> dict[str, Any]:
        merged = state.get("merged_retrieval")
        if merged is None:
            merged, _ = self._merge(state.get("all_results", []), state["limits"])
        if merged is None:
            query: Query = state["query"]
            analysis = QueryAnalyzer().analyze(query)
            scope_json = json.dumps(state["scope"].model_dump(mode="json"), sort_keys=True)
            merged = RetrievalResult(
                original_query=query.query_text, normalized_query=analysis.normalized_query,
                query_analysis=analysis, applied_filters=RetrievalFilters(),
                authorization_scope_hash=hashlib.sha256(scope_json.encode()).hexdigest(),
                evidence=[], assessment=EvidenceAssessment(state=EvidenceAssessmentState.INSUFFICIENT,
                    rationale=["All bounded research retrieval attempts failed."]),
            )
        else:
            merged = merged.model_copy(update={
                "original_query": state["query"].query_text,
                "normalized_query": state["query_analysis"].normalized_query,
                "query_analysis": state["query_analysis"],
                "applied_filters": state["query_analysis"].extracted_filters,
            }, deep=True)
        reason = state.get("research_termination_reason")
        if not reason:
            limits: ResearchLimits = state["limits"]
            reason = ("round_budget_exhausted" if state.get("round_number", 0) >= limits.max_rounds
                      else "subquery_budget_exhausted")
        elapsed = time.perf_counter() - state["started_at"]
        failure = None
        if reason != "budget_exceeded":
            try:
                timeout = max(0.01, state["limits"].max_time_seconds - elapsed)
                response = await asyncio.wait_for(asyncio.to_thread(
                    self.answer_pipeline.answer, state["query"], merged,
                    generation_style="structured_research",
                    generation_max_output_tokens=PROFILE_CONFIGS[ProfileName.RESEARCH].generation_max_output_tokens,
                    allow_correction=(state["limits"].max_correction_attempts > 0 and
                                      state.get("verification_enabled", True)),
                    verification_depth=("full_claim_checks" if state.get("verification_enabled", True)
                                        else "citations_only"),
                    semantic_verification=state.get("semantic_verification_enabled", True),
                ), timeout=timeout)
            except BudgetExceeded:
                reason = "budget_exceeded"
            except Exception as exc:
                failure = type(exc).__name__

        from omnirag.models.response import OmniRAGResponse, VerificationStatus
        if reason == "budget_exceeded":
            response = self._budget_exceeded_response(state["query"].query_text)
        elif failure:
            response = OmniRAGResponse(answer="Research stopped before a grounded answer could be generated.",
                                       profile_used=ProfileName.RESEARCH, original_query=state["query"].query_text,
                                       abstained=True, abstention_reason="research_time_or_generation_failure",
                                       verification_status=VerificationStatus.FAILED,
                                       overall_reliability=VerificationStatus.FAILED,
                                       warnings=[f"research_finalize_failed:{failure}"])
        response.execution_metadata.update({
            "selected_profile": "research", "actual_execution_path": "research_langgraph",
            "research_steps": len(state.get("ledger", [])),
            "subquery_count": len(state.get("executed_queries", [])),
            "followup_retrieval_count": max(0, state.get("round_number", 0) - 1),
            "evidence_gathered": len(merged.evidence),
            "unresolved_questions": state.get("unresolved_questions", []),
            "research_termination_reason": reason,
            "research_elapsed_seconds": round(time.perf_counter() - state["started_at"], 3),
            "evidence_ledger": [entry.model_dump(mode="json") for entry in state.get("ledger", [])],
            "research_contradictions": state.get("contradictions", []),
        })
        evidence_chunks = {f"E{index}": item.chunk.chunk_id
                           for index, item in enumerate(merged.evidence, start=1)}
        verified_entries = []
        for entry in state.get("ledger", []):
            entry_chunks = {ref.chunk_id for ref in entry.evidence_references}
            statuses = [claim.status.value for claim in response.claims
                        if any(evidence_chunks.get(ref) in entry_chunks
                               for ref in claim.supporting_evidence_ids)]
            verification_status = ",".join(sorted(set(statuses))) if statuses else "not_run"
            verified_entries.append(entry.model_copy(update={"verification_status": verification_status}))
        response.execution_metadata["evidence_ledger"] = [
            entry.model_dump(mode="json") for entry in verified_entries]
        response.timings_ms["research_total"] = round((time.perf_counter() - state["started_at"]) * 1000, 3)
        response.execution_metadata["research_resource_usage"] = {
            "retrieval_evidence_tokens_estimated": merged.selected_token_estimate,
            "generation_input_tokens": response.token_usage.get("input_tokens", 0),
            "generation_output_tokens": response.token_usage.get("output_tokens", 0),
            "cost_usd": response.cost_usd,
        }
        response.warnings.extend(state.get("warnings", []))
        response.warnings.append(f"research_terminated:{reason}")
        return {"response": response}

    @staticmethod
    def _merge(records: list[dict[str, Any]], limits: "ResearchLimits"):
        if not records:
            return None, []
        base: RetrievalResult = records[0]["result"]
        evidence, seen, token_count = [], set(), 0
        evidence_budget = limits.max_token_budget - limits.max_output_tokens - limits.max_prompt_overhead_tokens
        for record in records:
            for item in record["result"].evidence:
                if item.chunk.chunk_id in seen:
                    continue
                size = item.estimated_tokens or estimate_tokens(item.chunk.content)
                if len(evidence) >= limits.max_evidence or token_count + size > evidence_budget:
                    continue
                evidence.append(item)
                seen.add(item.chunk.chunk_id)
                token_count += size
        contradictions = detect_contradictions(evidence)
        states = [record["result"].assessment.state for record in records]
        if contradictions:
            assessment_state = EvidenceAssessmentState.CONFLICTING
        elif states and all(state == EvidenceAssessmentState.STRONG for state in states):
            assessment_state = EvidenceAssessmentState.STRONG
        elif evidence and any(state in {EvidenceAssessmentState.STRONG,
                                        EvidenceAssessmentState.PARTIAL} for state in states):
            assessment_state = EvidenceAssessmentState.PARTIAL
        elif evidence:
            assessment_state = EvidenceAssessmentState.WEAK
        else:
            assessment_state = EvidenceAssessmentState.INSUFFICIENT
        assessment = EvidenceAssessment(state=assessment_state, calibrated=False,
                                        signals={"research_rounds": len(records),
                                                 "distinct_evidence_count": len(evidence)},
                                        rationale=["Aggregated from bounded research retrieval steps; uncalibrated."])
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        warnings = []
        inspection = []
        seen_inspection = set()
        for record in records:
            result = record["result"]
            for key, value in result.timings_ms.items():
                timings[key] = timings.get(key, 0.0) + value
            for key, value in result.candidate_counts.items():
                counts[key] = counts.get(key, 0) + value
            warnings.extend(result.warnings)
            for candidate in result.retrieval_inspection:
                key = candidate.get("chunk_id")
                if key not in seen_inspection:
                    seen_inspection.add(key)
                    inspection.append(candidate)
        merged = base.model_copy(update={
            "evidence": evidence, "assessment": assessment, "selected_token_estimate": token_count,
            "timings_ms": timings, "candidate_counts": counts,
            "retrieval_inspection": inspection, "warnings": list(dict.fromkeys(warnings)),
        }, deep=True)
        return merged, contradictions

    @staticmethod
    def _budget_exceeded_response(query_text: str):
        from omnirag.models.response import OmniRAGResponse, VerificationStatus
        return OmniRAGResponse(
            answer="Research stopped because the configured provider budget was reached.",
            profile_used=ProfileName.RESEARCH, original_query=query_text,
            abstained=True, abstention_reason="budget_exceeded",
            verification_status=VerificationStatus.NOT_RUN,
            overall_reliability=VerificationStatus.NOT_RUN,
            warnings=["provider_budget_exceeded"],
        )
