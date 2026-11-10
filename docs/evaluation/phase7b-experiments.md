# Phase 7b — experiment infrastructure and reproducibility

**Status:** **COMPLETE** — experiment infrastructure acceptance checks passed on 2026-11-03.
**Scope:** Instrumentation and reproducibility only. The experiment layer calls the existing OmniRAG pipeline; it does not change prompts, model selection, retrieval defaults, or profile budgets.

## Architecture and data flow

An API client opts into a run by including `experiment` in `POST /api/v1/chat`. The validated `ExperimentConfig` is passed through `ProfileExecutionService` to the existing retrieval and Research graph boundaries. Without this field, the API calls the same execution method with no override.

`ExperimentRunner` creates a UUID run directory before execution, writes a `running` manifest, installs a context-local stage collector, then records the completed or failed response. The stage collector is a `ContextVar`, so concurrent requests have separate traces and `asyncio.to_thread` work retains its request context. JSON artifacts are atomically replaced and contain no query, prompt, answer, evidence excerpts, credentials, or raw exception text.

## Experiment configuration schema

`ExperimentConfig` is a strict Pydantic model (`extra="forbid"`). Boolean fields accept `true`, `false`, or `null`. `null` means inherit the active profile or existing user policy; it is the default for every switch. With no experiment object, no experiment run is created and execution follows the normal service path.

| Field | Behavior when set to `false` | Default (`null`) |
|---|---|---|
| `dense_retrieval` | Skip query embedding and dense retrieval | Enabled, matching current shared retrieval behavior |
| `sparse_retrieval` | Skip sparse retrieval | Enabled, matching current shared retrieval behavior |
| `hybrid_fusion` | Concatenate dense results followed by sparse results; omit RRF and its structural boost | Enabled when both channels are enabled |
| `reranking` | Skip reranker construction and scoring | Inherit the profile's reranking policy |
| `query_expansion` | Research uses the original query only and does not ask the planner for follow-ups | Inherit Research `multi_query_enabled`; not applicable outside Research |
| `evidence_selection` | Turn off diversity preference while preserving the selector's evidence-count and token safety limits | Enabled |
| `verification` | Skip claim-level checking and correction; deterministic citation validation remains enabled | Inherit the profile's claim-verification policy |
| `research_escalation` | Suppress profile-to-profile escalation while preserving the response | Enabled, subject to the existing preference and budget checks |

`dense_retrieval=false` and `sparse_retrieval=false` together are rejected. Explicitly enabling hybrid fusion while disabling either channel is rejected. The same constraints are checked by `RetrievalPolicy` at the execution boundary. No absent component is synthesized to support an ablation.

`dataset_id` and `dataset_version` are optional labels, not paths and not claims that a dataset is locally present. Replay validation requires the caller to supply the available dataset/version pairs.

## Stage trace format

Each `trace.json` has a run ID and ordered `stages`. A stage contains `name`, `duration_ms` (null when it did not execute), `status` (`success`, `failure`, `not_applicable`, or `not_run`), allowlisted scalar `metadata`, and an exception class name when applicable. Instrumented stages include request initialization/profile execution, query analysis, embedding, dense and sparse retrieval, fusion, reranking, Research planning/query expansion, evidence selection, context construction, generation, citation validation, claim verification, escalation, and final response. A missing required stage is explicitly marked `not_run`; disabled or inapplicable stages are marked `not_applicable` where the pipeline can determine that.

Only scalar metadata whose key is not content/query/prompt/path/credential/token-like is admitted. Traces do not copy retrieval inspection, document text, queries, or prompts. Per-stage retrieval timings are durations; dense and sparse service calls can overlap.

## Manifest and artifact storage

For configured `data_dir`, runs are written to `data_dir/experiments/runs/<run_id>/`:

- `manifest.json`: schema version, UUID, parent ID, timestamps, execution status, local git revision/worktree state if available, API/application version, profile, requested and resolved switches, configured and response-resolved generation model identities, retrieval/execution parameters, optional dataset identity, Python/platform metadata, and artifact links.
- `trace.json`: stage duration/status records.
- `result.json`: response ID, profile, abstention/verification summary, provider/model, timings, token/cost accounting, citation/evidence counts, warning codes, and error classifications. It does not include answer text or raw warnings.
- `evaluation.json`: optional numeric/boolean evaluation metrics, linked from the manifest by `ExperimentRunner.record_evaluation`.

The run begins in `running`; normal return produces `completed` or `completed_with_errors`; a raised exception produces `failed`. Process termination can leave a `running` manifest, which is intentionally identifiable. The run directory is excluded from source control.

## Replay procedure

Replay is a Python service operation, not a new product endpoint. Load the parent ID and call `ExperimentRunner.replay(parent_id, settings=..., execute=..., available_datasets=...)`. The callback receives both the reconstructed `ExperimentConfig` and the saved retrieval/profile configuration so it can recreate the original service policy. The runner validates configured provider/model IDs, the resolved generation provider/model when recorded, dataset/version availability, and differences in revision/worktree state, Python/platform, and relevant timeout/retry/dimension settings. Missing models or datasets are returned in `ReplayValidation`; `ReplayUnavailable` prevents execution if a required dependency is missing. When validation passes, the saved configuration is executed with a new UUID and `parent_run_id`. The parent artifacts are never overwritten. The caller supplies the query/input to the execution callback; it is not recovered from artifacts because raw user content is not stored.

The replay report exposes environment differences. Matching configuration does not imply bit-for-bit output: remote/local model servers may change model weights behind aliases, runtimes may be nondeterministic, and this layer does not pin provider snapshots or random seeds.

## Known limitations

- The run API persists artifacts only for chat requests that explicitly include an experiment object; normal production traffic does not create experiment manifests.
- Replay is available through the Python `ExperimentRunner`; there is no new dashboard or replay endpoint.
- The manifest records configured provider/model IDs before generation. Resolved generation provider/model is added to the manifest after a response is available; a crashed/failed call may leave those fields null.
- The caller supplies dataset labels and the available dataset inventory; this layer does not discover corpus snapshots or hash corpus contents.
- Retrieval/citation safety limits remain active under ablations. `evidence_selection=false` disables diversity ordering, not token/count limits or citation validation.
- Stage timings are observations from existing component boundaries. Concurrent dense/sparse durations may overlap and therefore must not be summed as wall-clock latency.
- A process crash may leave a `running` manifest; no background stale-run reconciler is added in this phase.
- Reproducibility is configuration-level, not bit-for-bit for external providers or mutable model aliases.

## Phase 7b acceptance checklist

- [x] Default config inherits profile behavior; no experiment request follows the unchanged path.
- [x] Dense, sparse, hybrid fusion, reranking, query expansion, evidence-selection strategy, verification, and escalation switches are represented without adding components.
- [x] Invalid retrieval combinations are rejected.
- [x] Traces represent success, failure, not-applicable, and not-run states without sensitive content.
- [x] Run manifests record available safe metadata; UUIDs are unique; running and failed states remain visible.
- [x] Trace/result/evaluation artifacts share a run ID and manifest links.
- [x] Replay reconstructs the original config, reports unavailable models/datasets and environment differences, and uses a new child run ID.
- [x] Tests cover the experiment layer and existing Phase 7a behavior remains covered by the complete backend suite.
- [x] Complete backend suite passed; the one live-Qdrant-only gate remained skipped because its required URL was not configured.

## Verification record

Commands and results:

- `.venv\Scripts\python.exe -m pytest -q tests/unit/test_experiment_runtime.py` — **16 passed**.
- `.venv\Scripts\python.exe -m pytest -q` — **113 passed, 1 skipped, 2 warnings in 11.51s**. The skip was `tests/integration/test_live_qdrant_gates.py:14` because `OMNIRAG_7A_QDRANT_URL` was not configured. Both warnings are the existing FastAPI `on_event` deprecation notices in `api/main.py` and FastAPI's router wrapper.

The experiment suite verifies the API opt-in path, independent channel calls, invalid switch handling, trace status and metadata filtering, run/result/evaluation linkage, unique run IDs, failed-run visibility, and replay configuration/dependency checks. The full backend suite covers the existing Phase 7a tests as well.

## Final live acceptance verification — 2026-11-03

Dependency checks returned HTTP 200 from API readiness (`{"status":"ready","service":"OmniRAG"}`), Qdrant (`healthz check passed`), and Ollama (`Ollama is running`). The API container reports `openai_compatible` / `llama3.2:3b` generation and `openai_compatible` / `nomic-embed-text` embedding at 768 dimensions.

The running API container predates the Phase 7b API schema: one `POST /api/v1/chat` with the `experiment` field was rejected with HTTP 422 (`body.experiment: Extra inputs are not permitted`). It did not enter the pipeline or create a manifest. The container was not rebuilt or restarted. To verify the checked-out Phase 7b implementation without changing the running service, one live Core execution and its one replay were run directly through the workspace `ProfileExecutionService`/`ExperimentRunner` against the same running Qdrant and Ollama services. Process-local endpoint overrides pointed the host process to `localhost`; `.env` was not changed.

| Check | Original run | Replay |
|---|---|---|
| Run ID / parent | `0b61164c-7eb9-4d13-9d79-c326bb5bde4f` / no parent | `93676bb4-1a85-4002-ae05-67b53cec2b29` / `0b61164c-7eb9-4d13-9d79-c326bb5bde4f` |
| Status | `completed` | `completed` |
| Experiment | Core, `reranking=false`, dataset `phase7a-synthetic@2026-11-03` | Reconstructed config and retrieval policy both matched the parent |
| Answer | `Northstar`, not abstained | `Northstar`, not abstained |
| Provider/model | `openai_compatible` / `llama3.2:3b` | `openai_compatible` / `llama3.2:3b` |
| Citation check | One citation chunk ID matched the one evidence reference chunk ID (`chk_465f3f16001aa1b14dbc5b37`) | Same citation/evidence chunk ID match |
| Usage | 1,273 input / 46 output tokens; cost unpriced (`null`) | 1,273 input / 46 output tokens; cost unpriced (`null`) |
| Trace | 17 stages; generation and citation validation successful; reranking not applicable | 17 stages; generation and citation validation successful; reranking not applicable |

Both runs have a manifest, trace, and result summary. The child has a distinct run ID and its manifest links to the original. Original artifacts: [manifest](../../data/experiments/runs/0b61164c-7eb9-4d13-9d79-c326bb5bde4f/manifest.json), [trace](../../data/experiments/runs/0b61164c-7eb9-4d13-9d79-c326bb5bde4f/trace.json), [result](../../data/experiments/runs/0b61164c-7eb9-4d13-9d79-c326bb5bde4f/result.json). Replay artifacts: [manifest](../../data/experiments/runs/93676bb4-1a85-4002-ae05-67b53cec2b29/manifest.json), [trace](../../data/experiments/runs/93676bb4-1a85-4002-ae05-67b53cec2b29/trace.json), [result](../../data/experiments/runs/93676bb4-1a85-4002-ae05-67b53cec2b29/result.json).

**Acceptance note:** The live pipeline experiment and replay passed. The existing API container’s experiment endpoint remains unverified because its image rejected the new field; updating that container was outside this acceptance-only task. No application code, provider behavior, or container state was changed.

This execution verifies run/replay mechanics, not answer quality. Its Core answer was `Northstar`, which differs from the user-confirmed Lite UI answer `OMNI-AX7-442`; it also returned `conflicting_retrieved_evidence`, `semantic_verification_unconfigured`, and `suspicious_instruction_text_in_retrieved_evidence` warnings. No quality conclusion or optimization was performed.
