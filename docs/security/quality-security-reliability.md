# OmniRAG Quality, Security and Reliability Closure

**Date:** 2026-11-05
**Scope:** Deterministic local tests and static review only. No live inference, benchmark, failure injection, container action, or external service call was made.

## Executive finding

Quality, security, and failure-handling controls have useful deterministic regression coverage, but they do not establish production security or live-provider reliability. One application defect was confirmed and fixed: the synchronous document upload/reingestion routes returned success HTTP status codes even when the ingestion pipeline returned `FAILED`. They now return HTTP 422 for `SourceLoadError` and HTTP 503 for other failed ingestion results, while preserving the document’s persisted failure status and response body. Successful create/update status behavior remains 201/200.

Overall release evidence remains **PARTIAL**. The live inference and failure-injection checks explicitly excluded by this task remain unverified.

## Quality validation matrix

| Area | Evidence reviewed | Validation performed | Result |
|---|---|---|---|
| Grounded answer generation | `generation/answer_pipeline.py`, `generation/answer_generator.py`, `generation/prompt_templates.py` | Fake-provider supported-claim and citation tests in `tests/unit/test_generation_verification.py` | PARTIAL — deterministic unit behavior passed; no live model request |
| Unsupported-question abstention | `retrieval/sufficiency.py`, `generation/answer_pipeline.py`, `verification/abstention.py` | Insufficient-evidence and missing named-entity tests | PASS — for covered unit cases only |
| Citation correctness | `generation/citation_builder.py`, `verification/citation_validator.py` | Valid, absent, and invalid evidence IDs; one bounded correction; typed enum status; Research `_finalize` test | PASS — covered deterministic cases |
| Evidence-to-claim consistency | `verification/claim_verifier.py`, `verification/nli_verifier.py` | Numeric/identifier mismatch and unconfigured verifier tests | PARTIAL — deterministic numeric and verifier-state cases pass; semantic verifier is unconfigured and lexical checks do not establish general entailment |
| Contradictory evidence | `verification/contradiction_detector.py` | Versioned numeric contradiction test | PARTIAL — covered numeric/version example only |
| Missing or insufficient evidence | `retrieval/sufficiency.py`, `generation/answer_pipeline.py` | Insufficient retrieval returns abstention without provider call | PASS — for deterministic fixture |
| Prompt-injection resistance | `generation/prompt_templates.py` | Retrieved instruction is marked untrusted, delimited, and accompanied by system guidance | PARTIAL — prompt construction test only; no adversarial live model evaluation |
| Verification status and uncertainty | `models/response.py`, `generation/answer_pipeline.py`, `verification/claim_verifier.py` | Supported, unsupported, and unconfigured/failed verifier states | PASS — for covered deterministic cases; calibrated semantic confidence is unavailable |

## Security validation matrix

| Area | Evidence reviewed | Validation performed | Result |
|---|---|---|---|
| Principal/tenant ACL isolation | `api/trusted_proxy.py`, `api/dependencies.py`, `retrieval/query_filters.py`, `retrieval/context_expander.py` | In-memory Qdrant ACL/tenant retrieval and expansion checks in `tests/integration/test_hybrid_retrieval.py` | PARTIAL — deterministic filters passed; no live multi-tenant request in this task |
| Signed identity and timestamp handling | `api/trusted_proxy.py`, `api/main.py` | Fresh signed identity accepted; tampered tenant, stale timestamp, and absent signature rejected in `tests/unit/test_trusted_proxy.py` | PASS — deterministic HMAC/time-window checks; not a deployment/proxy configuration audit |
| Unauthorized document retrieval prevention | `retrieval/query_filters.py`, `retrieval/context_expander.py`, `tests/integration/test_phase7c_acl_dataset.py` | Existing local filter/expansion coverage was run; Phase 7c API ACL suite was not run successfully in this task because of the local test-client stall described below | PARTIAL |
| Secret redaction in logs/artifacts | `utils/logger.py`, `experiments/runtime.py` | Canary secret redaction checks for structured logs and manifest/trace/result/evaluation artifacts | PASS — configured-secret test cases; not a review of every third-party or container log sink |
| API input validation and errors | `api/routes/documents.py`, `api/routes/chat.py`, `api/schemas.py`, `api/main.py` | Filename, extension, upload-size, and query/profile contract unit tests; new failed-ingestion status mapping test | PARTIAL — tested local validation paths; no live HTTP server test |
| Rate limiting | `infrastructure/rate_limit.py`, `api/main.py` | Per-principal window, Retry-After/429, and request accounting tests in `tests/unit/test_m1p5_controls.py` | PASS — process-local limiter behavior; not a distributed/global limit |

## Reliability validation matrix

| Failure or control | Evidence reviewed | Validation performed | Result |
|---|---|---|---|
| Ollama unavailable | `api/main.py` readiness handler | Mocked health request failure returns sanitized 503 in `tests/unit/test_m1p5_controls.py` | PASS — mocked dependency failure only |
| Qdrant unavailable | `api/main.py` readiness handler | Added regression: mocked Qdrant connection exception returns sanitized `{"status":"not_ready","dependency":"qdrant"}` with HTTP 503 | PASS — mocked dependency failure only |
| Generation provider failure/timeout | `llm/llm_client.py`, `generation/answer_pipeline.py` | Fake-provider failure returns failed/abstained answer; adapter and bounded retry tests included in focused run | PARTIAL — no live provider outage or timeout injection |
| Embedding failure | `indexing/embeddings.py`, `ingestion/pipeline.py` | Mocked invalid dimensions/provider configuration and failed-ingestion cases | PASS — deterministic provider failures; no live embedding call |
| Reranker unavailable | `retrieval/hybrid_retriever.py`, `retrieval/reranker.py`, `evaluation/phase8_runner.py` | Static review confirms absent reranker construction warns and continues with fused ranking; Phase 8 runner tests encode `reranker_unavailable` as nonfatal and actual `reranker_failed_fallback_to_fusion` as fatal. Those runner tests were not rerun here. | PARTIAL — direct retrieval fallback execution was not validated; provider failures were not injected |
| Research workflow termination | `agent/graph.py`, `profiles/execution.py` | Research budget, reserved-finalization, enum-type `_finalize`, and termination tests | PASS — deterministic tests only; historical live Research request is documented in `docs/developer/end-to-end-verification.md`, not repeated here |
| Budget exhaustion | `infrastructure/provider_budget.py`, `profiles/execution.py`, `agent/graph.py` | Token/call/cost guard and Research budget termination tests | PASS — deterministic budget tests; no live provider budget trip |
| Retry/fallback behavior | `llm/llm_client.py`, `indexing/embeddings.py`, `profiles/execution.py` | Fake provider fallback metadata, bounded chat retry, embedding retry/error handling, and explicit escalation budget tests | PARTIAL — local bounded behavior passed; no live upstream outage or retry-after timing test |
| Failed ingestion HTTP status | `api/routes/documents.py` | New status mapping regression verifies source-load failure maps to 422, other failed upload/reingestion maps to 503, and successful create/update remain 201/200 | PASS — helper-level deterministic regression; direct async route test was not retained because this Windows environment stalled in the local async test path |

## Tests executed

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m pytest --basetemp .pytest-tmp-prompt3-verified tests/unit/test_m1p5_controls.py tests/unit/test_trusted_proxy.py tests/unit/test_api.py tests/unit/test_chunking_and_embeddings.py tests/unit/test_llm_client.py tests/unit/test_generation_verification.py tests/unit/test_profiles_and_research.py tests/unit/test_research_budget.py tests/unit/test_research_finalization.py tests/unit/test_provider_budget.py tests/unit/test_retrieval_components.py tests/integration/test_ingestion_pipeline.py tests/integration/test_hybrid_retrieval.py -q` | **79 passed, 2 warnings in 7.67 s** after the final HTTP status mapping. Warnings are existing FastAPI `on_event` deprecations. |
| Same focused selection using default sandbox pytest temp directory | **76 passed, 1 failed, 2 warnings**: Windows `PermissionError` on `os.replace` in the secret-redaction test’s sandbox-managed temp directory. |
| Focused rerun with workspace-local `--basetemp` | **79 passed, 2 warnings**; confirms the previous failure was caused by the temp-directory write/replace restriction, not the redaction assertion. |
| Async direct-route and reranker retrieval-path experiments | **Stalled and stopped before a result** in this Windows environment; not counted as passes. No service or provider was contacted. |

The 79-test selection includes tests for generation verification, prompt untrusted-evidence delimiting, contradictory versioned values, HMAC timestamp/signature checks, ACL/tenant Qdrant filtering, rate limiting, secret redaction, Ollama/Qdrant readiness failures, embedding errors, Research termination/finalization, and budgets. Existing full backend result from Prompt 1 remains **254 passed, 1 skipped, 2 warnings**; it is inherited evidence and was not rerun as a full suite here.

## Minimal fix

**Files changed:**

- `api/routes/documents.py` — failed synchronous upload and reingestion results now use HTTP 422 for source-load failures and HTTP 503 for other failures instead of 201/200. The persisted `failed` document record and sanitized `DocumentResponse` body are retained. Successful first upload stays 201; successful update/unchanged remains 200.
- `tests/unit/test_api.py` — verifies failure status mapping and unchanged success status semantics.
- `tests/unit/test_m1p5_controls.py` — verifies Qdrant readiness failure is a sanitized 503.
- `docs/security/quality-security-reliability.md` — this report.
- `docs/developer/end-to-end-verification.md` — Prompt 3 addendum with the fix and results.

The upload route awaited ingestion synchronously, persisted the returned `FAILED` status, then selected HTTP status solely from whether the document row already existed. That was misleading success reporting for both first ingestion and reingestion. The correction changes only response status selection; it does not bypass authentication, alter ingestion, discard its failure record, or alter ACL behavior.

## Known limitations and unverified behavior

- No live model inference, live API request, external service outage, or failure injection was performed in this task.
- Phase 7a has historical post-fix Lite/Core/Research live requests, but no current live all-profile validation; see Prompt 2 report. Research's saved request completed with a verification-driven abstention.
- Live Research success rate, provider timeout behavior, Ollama recovery, Qdrant recovery, and upstream retry behavior remain unknown.
- The phase7c API/ACL TestClient selection previously stalled in Windows local asyncio socket setup. A direct async route test and direct retrieval reranker test also stalled before results in this environment; these are test-environment limitations, not passing evidence or confirmed application failures.
- ACL filter tests use local/in-memory Qdrant, not a live Qdrant service. Signed identity unit tests do not prove reverse-proxy header scrubbing or secret rotation in deployment.
- Reranker configuration is absent in the baseline; direct live reranking is not verified. Fusion fallback is statically present and runner warning classification has existing regression cases.
- Prompt-injection resistance is a deterministic prompt-format test, not a model red-team result. Evidence sufficiency and numeric checks are not calibrated general entailment measures.
- Rate limiting is process-local and requires one API worker per instance as documented in its implementation; it is not a distributed quota.
- Qdrant startup/readiness and provider checks are bounded at the readiness adapter level, but readiness does not verify a full successful retrieval/generation transaction.
- No full backend rerun or live Qdrant gate was performed for Prompt 3. The inherited full suite and the focused run are stated separately.

## Release blockers

1. **Current live end-to-end validation remains pending.** The latest runtime image/source identity and the full upload-to-response flow have not been established in this task.
2. **Production security posture remains unverified live.** Proxy configuration, secret provisioning/rotation, TLS/network boundaries, and real cross-tenant traffic need a deployment-specific review.
3. **Reliability under real dependency failures is untested.** Live Ollama/Qdrant outage/recovery and provider timeout handling need separately approved controlled tests.
4. **Windows async test environment is unresolved.** The affected API/ACL async tests should be run in a supported CI/Linux environment or with an approved local test configuration, without weakening test assertions.
5. **Quality validity is incomplete.** Human-reviewed answer/evidence labels and a representative live quality run are needed before making correctness or security-rate claims.

## Exact next actions for Prompt 4

1. Run the stalled API/ACL integration selection in the supported CI/Linux runner and record the unmodified test results.
2. Obtain explicit approval for a small isolated live validation of one Lite, one Core, and one Research request, using a synthetic indexed corpus; record the active image/source identity and inspect citations, abstentions, provider metadata, and Research termination.
3. Separately request explicit approval before any live fault injection. Use reversible isolated dependencies and one scenario at a time, beginning with a non-destructive health/readiness check.
4. Review the deployment reverse proxy, TLS, secret management, and multi-worker rate-limit assumptions against actual deployment configuration before any release recommendation.

**Prompt 3 status:** **PARTIAL** — focused deterministic quality/security/reliability coverage passed and one confirmed HTTP status defect was corrected. Live provider reliability, deployment security, and live end-to-end quality remain unverified.
