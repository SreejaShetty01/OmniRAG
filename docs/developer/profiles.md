# Execution profiles

`ProfileExecutionService.execute(query, scope)` selects one of the shared profile policies and invokes the same `HybridRetriever` and `AnswerPipeline`. Policy defaults are starting values only; none is benchmarked. Provider/model selection comes from the global provider-neutral model registry mapping and is independent of profile identity.

| Policy | Lite | Core | Research |
|---|---:|---:|---:|
| Dense/sparse candidate depth | 12 / 12 | 40 / 40 | 50 / 50 per sub-query |
| Evidence context tokens | 700 | 3,000 | 6,000 total |
| Reranking | Disabled | Enabled when configured | Enabled when configured |
| Expansion | Parent, max 1 | Parent/child/neighbour, max 8 | All four kinds, max 12 per retrieval |
| Verification | Citation checks | Full available claim checks | Full available claim checks |
| Research rounds / sub-queries | 1 / 1 | 1 / 1 | Up to 3 / 8 |

Lite uses concise generation and does not claim deep verification. Core is the balanced default with claim checks and the Phase 4 one-attempt correction limit. Research runs a LangGraph state graph with separate analysis, planning, parallel retrieval, evidence assessment/ledger, follow-up planning, and final generation stages. Deterministic decomposition handles explicit compound questions; follow-up hops use Phase 2 `links[]`. The graph stops on sufficient evidence, diminishing new evidence, token/evidence/time ceilings, round/sub-query ceilings, or no actionable linked follow-up. It preserves unresolved questions and ledger provenance.

Research limits are validated by `ResearchLimits`; defaults are unbenchmarked and configurable at construction. The Research graph uses shared retrieval and answer services. No profile creates a separate index or pipeline. The graph is async; this Windows workspace has previously stalled while initializing asyncio socket pairs, so its full async runtime path needs verification in a deployment/test environment where the event loop can run.

Research now reserves a bounded portion of its existing `max_time_seconds` for finalization. The reserve is `min(generation_timeout_seconds, 75% of the Research deadline)`; retrieval stops at the earlier deadline and finalization uses only time left in the unchanged total budget. This addresses retrieval consuming all time before final answer generation, but does not guarantee completion when generation plus a correction exceeds the remaining budget. See `agent/graph.py` and `tests/unit/test_profiles_and_research.py`.

Profile choice is a request field. `ConversationProfileStore` provides process-local persistence with request-level switching; production persistence should replace it with a durable store before multi-worker deployment. Escalation preferences include `ask`, `auto_with_notice`, and `never` (plus legacy values). `ask` records a suggested target and performs no escalation. Automatic escalation is disabled by default, requires explicit token/time budgets, and records a notice. If a cost ceiling is configured, automatic escalation is blocked because this deployment has no provider price estimate to enforce it safely.

Profile comparison uses the same labeled cases through `evaluation.profile_benchmark`. No human-labeled profile benchmark dataset or results are included; the defaults do not imply a speed, cost, or quality winner.
