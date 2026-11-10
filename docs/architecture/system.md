# OmniRAG architecture

## Shared foundation

Lite, Core, and Research are execution policies over shared ingestion, indexing, and one Qdrant knowledge store. A profile does not select a separate database. Shared Pydantic contracts in `models/` keep application and persistence concerns separated.

The React/TypeScript client lives in `frontend/` and calls only the typed `/api/v1`
boundary. SQLite in `data/omnirag.sqlite3` stores user preferences and UI metadata;
Qdrant remains the source for chunks and vectors. Collection IDs are payload filters
on the shared Qdrant collection. SQLite and uploaded files are persisted in the
`omnirag_data` Compose volume.

## Module responsibilities

- `config/`: environment settings, model selection slots, and immutable profile defaults.
- `api/`: FastAPI startup and transport boundary.
- `ingestion/`: shared Phase 2 loading, parsing, normalized structure, cross-reference extraction, chunking, and orchestration.
- `indexing/`: provider-neutral dense embedding, sparse encoding, shared Qdrant schema, and document lifecycle operations.
- `retrieval/`: shared async dense/sparse retrieval, RRF fusion, scoped filtering, structural expansion, reranking, evidence selection, and sufficiency signals.
- `profiles/`: named Lite/Core/Research policies and the central execution/escalation service.
- `generation/` and `verification/`: shared evidence-grounded generation, citation resolution, claim checks, entailment adapter, contradiction signals, and bounded abstention/correction.
- `agent/`: bounded LangGraph Research orchestration, deterministic planning, and traceable evidence ledger. Lite and Core do not enter the graph.
- `infrastructure/`: shared service concerns such as Qdrant connectivity, caching, security, and observability.
- `evaluation/` and `experiments/`: evaluation datasets/metrics/benchmarks and isolated experiments.
- `models/`: provider-neutral validated document, chunk, query, and response contracts.

## Profile boundaries

Lite is configured for shallow retrieval and constrained context, with deterministic citation validation in the shared answer pipeline. Core enables sufficiency and full available claim checks; reranking runs only when configured. Research adds multi-query and iterative retrieval, planning, evidence ledger, cross-document verification, and bounded LangGraph orchestration. Candidate and context-token policies are configured in `config/profiles.py` (12/700 Lite, 40/3,000 Core, 50/6,000 Research per the profile policy); latency budgets are unset, and these defaults are explicitly unbenchmarked. Provider/model choices come from runtime settings and are not fixed by profile.

All profiles share infrastructure. No profile-specific vector database or separate ingestion pipeline is part of this design.

## Experiment instrumentation (Phase 7b)

An optional `experiment` object on the chat request supplies validated, per-request ablation overrides. Unspecified switches inherit current profile behavior. `experiments/runtime.py` owns the context-local trace collector, JSON run store, manifest lifecycle, and replay dependency checks. Explicit runs persist under `<data_dir>/experiments/runs/<run_id>/`; normal chat traffic does not create experiment artifacts. See [the Phase 7b experiment guide](../evaluation/phase7b-experiments.md) for the schema, trace safety rules, replay API, and limitations.

## Phase 2 ingestion and indexing

`IngestionPipeline` dispatches local documents, code, and web URLs through source-specific loaders and parsers. Parsers produce the normalized `ContentBlock` hierarchy; cross-reference extraction enriches block links; `StructureChunker` packs blocks at section and syntax boundaries; deterministic context is stored separately from source text; embeddings and sparse term vectors are generated independently; `QdrantIndexer` stages and publishes points in one named-vector collection.

PDF and DOCX parsing prefer optional Docling when installed and otherwise use PyMuPDF and python-docx. Markdown and HTML use local parsers. JavaScript and TypeScript use Tree-sitter; Python uses its standard AST/tokenizer. No ingested code is executed. Oversized blocks split at sentences or lines, with word-boundary fallback and a documented token-count estimate rather than a tokenizer-specific guarantee.

The embedding adapter accepts a configured OpenAI-compatible endpoint; `EmbeddingProvider` is the extension point for other providers. Dense embedding input can include a deterministic prefix while stored chunk content remains unchanged. Sparse terms use stable hashed indices and sublinear term frequency; Qdrant applies its collection-level IDF modifier. This avoids maintaining a separate BM25 corpus service, with hash collisions and no BM25 field-length normalization as known limitations.

Each indexed generation is first written inactive under a content-hash generation ID. It is activated only after staging succeeds, then stale point IDs are removed. Pre-publication failures trigger best-effort staged-point cleanup. Qdrant does not provide a transaction across these lifecycle operations, so deployment monitoring should treat reported cleanup failures as requiring reconciliation.

## Retrieval boundary

`HybridRetriever` coordinates one shared pipeline and returns typed evidence with provenance. The Phase 2 `dense` and `sparse` named vectors and payload are reused. RRF remains the baseline; DBSF is exposed as an isolated Qdrant experiment. Every initial and expansion query receives the same active-generation and ACL filters. Query cache entries require a caller-provided corpus generation and include authorization scope in their key.

## Generation and verification boundary

`AnswerPipeline` consumes a `RetrievalResult` directly. Provider adapters sit behind the chat interface, while citation checks and deterministic numeric/identifier checks remain provider-independent. Semantic verification is optional and stays explicitly uncertain if unavailable or if the provider fails. The response extends the shared response model using defaulted fields. `ProfileExecutionService` selects the shared retrieval/answer policy; Research additionally enters the bounded LangGraph agent workflow described above, while Lite and Core do not.
