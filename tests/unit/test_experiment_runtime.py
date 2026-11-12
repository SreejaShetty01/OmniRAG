import asyncio
import json
import sys

import pytest

from omnirag.config.profiles import PROFILE_CONFIGS, ProfileName
from omnirag.config.settings import Settings
from omnirag.experiments.runtime import (
    ExperimentConfig, ExperimentRunStore, ExperimentRunner, ReplayUnavailable,
    StageTraceCollector,
)
from omnirag.models.response import OmniRAGResponse
from omnirag.profiles.execution import ProfileExecutionService
from omnirag.models.chunk import AccessControlMetadata, Chunk
from omnirag.models.document import DocumentType
from omnirag.retrieval.contracts import RetrievalCandidate, RetrievalPolicy, RetrievalScope
from omnirag.retrieval.hybrid_retriever import HybridRetriever


def _settings():
    return Settings(_env_file=None,
                    generation_provider="local", generation_model_id="chat-v1",
                    embedding_provider="local", embedding_model_id="embed-v1",
                    reranker_provider="local", reranker_model_id="rank-v1",
                    verification_provider="local", verification_model_id="verify-v1",
                    embedding_dimensions=768)


def _response():
    return OmniRAGResponse(answer="safe result", profile_used=ProfileName.CORE)


def _run(coro):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def test_default_experiment_config_inherits_profile_and_keeps_default_policy():
    profile = PROFILE_CONFIGS[ProfileName.LITE]
    policy = ProfileExecutionService._retrieval_policy(profile)
    experiment = ExperimentConfig()
    changed = experiment.apply_to_policy(policy, profile)
    assert changed.dense_retrieval_enabled and changed.sparse_retrieval_enabled
    assert changed.hybrid_fusion_enabled and changed.evidence_selection_enabled
    assert changed.reranking_enabled == policy.reranking_enabled
    assert experiment.resolved_switches(profile)["verification"] == profile.claim_verification_enabled


@pytest.mark.parametrize(("field", "expected"), [
    ("dense_retrieval", "dense_retrieval_enabled"),
    ("sparse_retrieval", "sparse_retrieval_enabled"),
    ("hybrid_fusion", "hybrid_fusion_enabled"),
    ("reranking", "reranking_enabled"),
    ("evidence_selection", "evidence_selection_enabled"),
])
def test_retrieval_ablations_apply_independently(field, expected):
    profile = PROFILE_CONFIGS[ProfileName.CORE]
    policy = ProfileExecutionService._retrieval_policy(profile)
    config = ExperimentConfig(**{field: False})
    changed = config.apply_to_policy(policy, profile)
    assert getattr(changed, expected) is False
    for other in {"dense_retrieval_enabled", "sparse_retrieval_enabled", "hybrid_fusion_enabled",
                  "reranking_enabled", "evidence_selection_enabled"} - {expected}:
        assert getattr(changed, other) == getattr(policy, other)


def test_research_ablation_switches_resolve_independently():
    profile = PROFILE_CONFIGS[ProfileName.RESEARCH]
    switches = ExperimentConfig(query_expansion=False, verification=False,
                                research_escalation=False).resolved_switches(profile)
    assert switches["query_expansion"] is False
    assert switches["verification"] is False
    assert switches["research_escalation"] is False
    assert switches["reranking"] is True


def test_semantic_verification_ablation_is_independent_of_claim_check_switch():
    profile = PROFILE_CONFIGS[ProfileName.CORE]
    control = ExperimentConfig(verification=True, semantic_verification=False).resolved_switches(profile)
    treatment = ExperimentConfig(verification=True, semantic_verification=True).resolved_switches(profile)
    assert control["verification"] == treatment["verification"] is True
    assert control["semantic_verification"] is False
    assert treatment["semantic_verification"] is True


def test_invalid_switch_combinations_are_rejected():
    with pytest.raises(ValueError, match="At least one"):
        ExperimentConfig(dense_retrieval=False, sparse_retrieval=False)
    with pytest.raises(ValueError, match="requires both"):
        ExperimentConfig(dense_retrieval=False, hybrid_fusion=True)
    with pytest.raises(ValueError):
        ExperimentConfig(extra_field=True)


def test_stage_trace_records_success_failure_and_filters_sensitive_metadata():
    trace = StageTraceCollector()
    trace.record("retrieval", 12.5, metadata={"candidate_count": 4, "query_text": "private",
                                              "api_key": "secret", "content_length": 50})
    trace.record("generation", 3.2, status="failure", error_type="TimeoutError")
    assert trace.stages[0].metadata == {"candidate_count": 4}
    assert trace.stages[0].status == "success"
    assert trace.stages[1].error_type == "TimeoutError"


def test_manifest_trace_result_and_unique_run_ids(tmp_path):
    runner = ExperimentRunner(ExperimentRunStore(tmp_path))
    config = ExperimentConfig(dataset_id="synthetic", dataset_version="v1")
    initial_statuses = []

    async def execute():
        from omnirag.experiments.runtime import record_stage
        statuses = [json.loads(path.read_text(encoding="utf-8"))["status"]
                    for path in tmp_path.glob("*/manifest.json")]
        initial_statuses.append(statuses.count("running"))
        record_stage("generation", 9, metadata={"provider_call_count": 1})
        return _response()

    first = _run(runner.run(profile="core", config=config, settings=_settings(), execute=execute))
    second = _run(runner.run(profile="core", config=config, settings=_settings(), execute=execute))
    assert initial_statuses == [1, 1]
    assert first.execution_metadata["experiment_run_id"] != second.execution_metadata["experiment_run_id"]
    run_id = first.execution_metadata["experiment_run_id"]
    manifest = runner.store.read_json(run_id, "manifest.json")
    trace = runner.store.read_json(run_id, "trace.json")
    result = runner.store.read_json(run_id, "result.json")
    assert manifest["status"] == "completed" and manifest["experiment_config"]["dataset_id"] == "synthetic"
    assert trace["run_id"] == result["response_id"] or trace["run_id"] == run_id
    assert {item["name"] for item in trace["stages"]} >= set(runner.REQUIRED_STAGES)
    assert any(item["name"] == "fusion" and item["status"] == "not_run" for item in trace["stages"])
    raw = "".join(path.read_text(encoding="utf-8") for path in runner.store.run_dir(run_id).glob("*.json"))
    assert "safe result" not in raw and "private" not in raw and "api_key" not in raw
    runner.record_evaluation(run_id, {"accuracy": 0.75})
    assert runner.store.read_json(run_id, "manifest.json")["artifacts"]["evaluation"] == "evaluation.json"


def test_failed_runs_are_identifiable_without_persisting_exception_text(tmp_path):
    runner = ExperimentRunner(ExperimentRunStore(tmp_path))

    async def fail():
        raise RuntimeError("sensitive request body")

    with pytest.raises(RuntimeError):
        _run(runner.run(profile="lite", config=ExperimentConfig(), settings=_settings(), execute=fail))
    run_id = next(tmp_path.iterdir()).name
    assert runner.store.read_json(run_id, "manifest.json")["status"] == "failed"
    assert runner.store.read_json(run_id, "result.json") == {"status": "failed", "error_type": "RuntimeError"}
    trace = runner.store.read_json(run_id, "trace.json")
    assert any(stage["name"] == "final_response" and stage["status"] == "failure"
               for stage in trace["stages"])
    assert "sensitive request body" not in (runner.store.run_dir(run_id) / "result.json").read_text()


def test_returned_provider_failure_is_recorded_as_completed_with_errors(tmp_path):
    runner = ExperimentRunner(ExperimentRunStore(tmp_path))

    async def execute():
        return OmniRAGResponse(answer="safe abstention", profile_used=ProfileName.CORE,
                               abstained=True, warnings=["generation_failed:ProviderError"])

    response = _run(runner.run(profile="core", config=ExperimentConfig(), settings=_settings(),
                               execute=execute))
    run_id = response.execution_metadata["experiment_run_id"]
    assert runner.store.read_json(run_id, "manifest.json")["status"] == "completed_with_errors"
    assert runner.store.read_json(run_id, "result.json")["failure_codes"] == ["generation_failed"]


def test_replay_creates_child_and_reports_missing_dependencies(tmp_path):
    runner = ExperimentRunner(ExperimentRunStore(tmp_path))
    settings = _settings()
    config = ExperimentConfig(dataset_id="synthetic", dataset_version="v1")

    async def initial():
        return _response()

    parent = _run(runner.run(profile="core", config=config, settings=settings, execute=initial))
    parent_id = parent.execution_metadata["experiment_run_id"]
    seen = []

    async def replay_execute(replayed_config, replayed_retrieval_config):
        seen.append((replayed_config, replayed_retrieval_config))
        return _response()

    missing_data = runner.validate_replay(parent_id, settings=settings)
    assert "dataset:synthetic@v1" in missing_data.unavailable_dependencies
    replayed = _run(runner.replay(parent_id, settings=settings, execute=replay_execute,
                                 available_datasets={("synthetic", "v1")}))
    child_id = replayed.execution_metadata["experiment_run_id"]
    assert child_id != parent_id and seen == [(config, {})]
    assert runner.store.read_json(child_id, "manifest.json")["parent_run_id"] == parent_id

    changed = _settings()
    changed.generation_model_id = "different-model"
    report = runner.validate_replay(parent_id, settings=changed)
    assert not report.available and "model:generation_model:chat-v1" in report.unavailable_dependencies
    with pytest.raises(ReplayUnavailable):
        _run(runner.replay(parent_id, settings=changed, execute=replay_execute))


def test_chat_schema_accepts_optional_experiment_without_changing_default():
    from omnirag.api.schemas import ChatRequest

    plain = ChatRequest(query_text="question")
    experimental = ChatRequest(query_text="question", experiment={"reranking": False})
    assert plain.experiment is None
    assert experimental.experiment.reranking is False


def test_chat_route_persists_opt_in_experiment_artifacts(tmp_path, monkeypatch):
    from omnirag.api import routes
    from omnirag.api.dependencies import RequestIdentity
    from omnirag.api.routes.chat import chat
    from omnirag.api.schemas import ChatRequest
    from omnirag.config.settings import Settings
    from omnirag.infrastructure.product_store import ProductStore
    from omnirag.retrieval.contracts import RetrievalPolicy, RetrievalScope

    settings = Settings(_env_file=None, data_dir=tmp_path)
    monkeypatch.setattr(routes.chat, "get_settings", lambda: settings)

    class Executor:
        def _retrieval_policy(self, profile):
            return RetrievalPolicy()

        def _policy_summary(self, profile, policy):
            return {"verification_depth": profile.verification_depth.value}

        async def execute(self, query, scope, experiment=None):
            response = OmniRAGResponse(answer="safe response", profile_used=ProfileName.CORE)
            if experiment:
                response.execution_metadata["experiment_config"] = experiment.model_dump(mode="json")
            return response

    result = _run(chat(
            ChatRequest(query_text="question", selected_profile=ProfileName.CORE,
                        experiment={"dense_retrieval": False}),
        RequestIdentity(user_id="user-a", tenant_id="tenant-a"),
        ProductStore(tmp_path / "product.sqlite3"), Executor(),
        RetrievalScope(principal_ids=("user:user-a",), tenant_id="tenant-a"),
    ))
    run_id = result.response.execution_metadata["experiment_run_id"]
    runner = ExperimentRunner(ExperimentRunStore(tmp_path / "experiments" / "runs"))
    assert runner.store.read_json(run_id, "manifest.json")["profile"] == "core"
    assert runner.store.read_json(run_id, "manifest.json")["resolved_switches"]["dense_retrieval"] is False


def test_dense_and_sparse_retrieval_ablation_switches_change_called_channels():
    class Embeddings:
        calls = 0

        def embed(self, texts, **kwargs):
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    class Search:
        def __init__(self, candidate):
            self.candidate = candidate
            self.calls = 0

        def search(self, *args):
            self.calls += 1
            return [self.candidate]

    class Expansion:
        def expand(self, *args):
            return [], 0

    def candidate(chunk_id):
        chunk = Chunk(chunk_id=chunk_id, document_id="doc", parent_id="doc",
                      content=f"evidence text {chunk_id}",
                      document_type=DocumentType.TEXT, content_hash=chunk_id, source="guide.md",
                      access_control=AccessControlMetadata())
        return RetrievalCandidate(chunk=chunk, dense_rank=1 if chunk_id == "dense" else None,
                                  sparse_rank=1 if chunk_id == "sparse" else None)

    async def exercise(policy):
        embeddings = Embeddings()
        dense = Search(candidate("dense"))
        sparse = Search(candidate("sparse"))
        retriever = HybridRetriever(
            client=None, embedding_service=embeddings, dense_retriever=dense,
            sparse_retriever=sparse, expander=Expansion(), collection="unit",
        )
        result = await retriever.retrieve(
            "question", scope=RetrievalScope(principal_ids=("user:a",), tenant_id="tenant-a"), policy=policy,
        )
        return result, embeddings, dense, sparse

    base = RetrievalPolicy(expand_parents=False, expand_children=False, expand_neighbours=False,
                           expand_cross_references=False, max_evidence=5)
    result, embedding_calls, dense, sparse = _run(exercise(base))
    assert result.candidate_counts["dense"] == 1 and result.candidate_counts["sparse"] == 1
    assert embedding_calls.calls == dense.calls == sparse.calls == 1

    dense_only = base.model_copy(update={"sparse_retrieval_enabled": False,
                                        "hybrid_fusion_enabled": False})
    result, embedding_calls, dense, sparse = _run(exercise(dense_only))
    assert result.candidate_counts["dense"] == 1 and result.candidate_counts["sparse"] == 0
    assert embedding_calls.calls == dense.calls == 1 and sparse.calls == 0

    sparse_only = base.model_copy(update={"dense_retrieval_enabled": False,
                                          "hybrid_fusion_enabled": False})
    result, embedding_calls, dense, sparse = _run(exercise(sparse_only))
    assert result.candidate_counts["dense"] == 0 and result.candidate_counts["sparse"] == 1
    assert embedding_calls.calls == dense.calls == 0 and sparse.calls == 1

    no_fusion = base.model_copy(update={"hybrid_fusion_enabled": False})
    result, _, _, _ = _run(exercise(no_fusion))
    assert [item.chunk.chunk_id for item in result.evidence] == ["dense", "sparse"]
    assert all(item.rrf_score is None for item in result.evidence)
