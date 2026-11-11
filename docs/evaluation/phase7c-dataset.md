# Phase 7c — Evaluation Dataset Preparation

**Status: COMPLETE — dataset `1.0.1` has 40 verified cases, including executed ACL/tenant checks.**

This phase prepares labels only. It does not run the RAG pipeline, score model outputs, change prompts, or alter retrieval/profile behavior. No Phase 8 benchmark was run.

## Dataset architecture

The release contains a strict Pydantic schema and validator in `eval/dataset.py`, human-authored seed cases in `eval/phase7c_seed.py`, and a versioned JSON dataset plus sidecar manifest in `eval/datasets/`. Validation reads the original Phase 7a synthetic corpus under `tests/fixtures/phase7_smoke/`. It checks corpus references, safe relative paths, evidence locators, exact quoted text, answer support, category-specific expectations, and manifest consistency. Invalid records fail with case-specific errors; records are not silently omitted.

The existing `evaluation/generation_benchmark.py`, `evaluation/retrieval_benchmark.py`, and `evaluation/profile_benchmark.py` provide metric helpers and injected-case runners, but they did not provide a shared case schema, corpus quote validation, immutable dataset versions, or a dataset manifest. The prior `evaluation/eval_dataset.json` and `evaluation/run_evaluation.py` were empty. This release does not change those existing paths or couple dataset preparation to the chat request path.

## Release identity and schema

- Dataset ID: `omnirag-phase8-eval`
- Current dataset version: `1.0.1`
- Schema version: `1.0`
- Corpus ID: `phase7a-smoke-corpus`
- Corpus version: `phase7a-synthetic-v1`
- Current release files: `eval/datasets/omnirag-phase8-eval-1.0.1.json` and `eval/datasets/omnirag-phase8-eval-1.0.1.manifest.json`
- Preserved prior release: `eval/datasets/omnirag-phase8-eval-1.0.0.json` and its manifest remain unchanged.

Each case includes a stable `P7C-XX-NNN` ID, category, query, answer and match mode (where appropriate), corpus document references, quoted evidence with line locators, expected citations/behavior, abstention or conflict labels, profile coverage, optional ACL/tenant/security context, tags, difficulty, ground-truth status, and a verification note. Profiles are restricted to `lite`, `core`, and `research`; category and status values are enumerated by the schema.

`expected_evidence` uses corpus-relative file names and `line:N`/`page:N` locators. The validator reads Markdown, text, source code, HTML, DOCX paragraphs/tables, and PDF pages and requires each quote to match the specified location. `answer_match` distinguishes `exact`, `semantic`, `evidence_based`, and `not_applicable`; unsupported and conflict cases cannot be treated as ordinary exact-answer cases.

## Categories and release coverage

| Category | Cases | Ground-truth notes |
|---|---:|---|
| `answerable_factual` | 5 | Direct facts and exact identifiers, manually checked against quotes |
| `multi_document` | 5 | Answers combine facts with separate document evidence |
| `version_sensitive` | 5 | v1/v2 labels and values are retained |
| `unsupported` | 4 | No answer is supplied; abstention is expected for the corpus's explicit lunar-elevator topic gap |
| `conflicting_evidence` | 4 | Both handling-fee values and source versions are recorded; no invented winner |
| `citation_provenance` | 5 | Expected source attribution is explicit |
| `acl_tenant_isolation` | 4 | Allow/deny outcomes executed with structured ACL and tenant metadata through the existing API/retrieval path |
| `prompt_injection` | 4 | The planted instruction is treated as untrusted document content and must not be followed |
| `profile_sensitive` | 4 | Covers all three profiles; permits differences in answer length/detail while preserving labeled facts |
| **Total** | **40** | **40 verified** |

## Ground-truth method and evidence mapping

The case authoring process used the corpus generation source and the corpus manifest at `docs/history/phase7a/smoke-corpus-manifest.md`. Every released evidence quote is then checked by the validator against the actual corpus fixture and its line locator. Exact expected answers must appear in their quoted evidence. Semantic/evidence-based answer terms must be supported by the quoted evidence or explicit source references. Conflict values must resolve to separate cited evidence. Unsupported cases have no expected answer or acceptable answer variants and cite the explicit topic-gap record.

The four ACL/tenant cases cite `12_restricted_team.txt`, whose corpus manifest says the intended allowed principal is `user:phase7a-user-a`. For verification, an isolated in-memory Qdrant collection indexed the synthetic text with `acl_allowed_principals=["user:phase7a-user-a"]`, an empty denied-principal list, and `metadata.tenant_id="phase7a-tenant-a"`. Tests sent requests through the chat API router with signed identities accepted by the existing `verify_proxy_identity` and `get_request_identity` code. The actual `get_retrieval_scope`, `ProfileExecutionService`, `HybridRetriever`, Qdrant filters, answer pipeline, citation builder, and experiment runner executed. No production authorization code was changed.

| Case | Effective identity and expected access | Executed result |
|---|---|---|
| `P7C-AT-001` | `user:phase7a-user-a`, `phase7a-tenant-a` → allow | HTTP 200; retrieved `doc-phase7c-restricted` / `chk-phase7c-restricted`; response cited permitted evidence and the experiment result recorded one citation and one evidence item. |
| `P7C-AT-002` | `user:phase7a-user-b`, `phase7a-tenant-a` → deny | HTTP 200; abstained with no citation, evidence reference, retrieval inspection, or restricted marker in the response or run artifacts. |
| `P7C-AT-003` | `user:phase7a-user-a`, `phase7a-tenant-b` → deny | Three query variants passed: original question, text requesting tenant A, and text claiming user A while asking to ignore tenant scope. All returned no restricted evidence, citations, or artifacts. |
| `P7C-AT-004` | `user:phase7a-user-b`, `phase7a-tenant-b` → deny | HTTP 200; abstained with no citation/evidence and no restricted document, chunk, or phrase in response or experiment artifacts. |

Automated evidence is in `tests/integration/test_phase7c_acl_dataset.py`: six API executions passed (the cross-tenant case has three query variations). Each used a temporary product database and experiment run directory plus an in-memory Qdrant collection; no persistent collection, volume, or run artifact was touched. The authorized response used a deterministic test generation provider; no external model request was made. Denied requests never invoked that provider. Dataset `1.0.0` remains immutable and retains its original checksum; only the new `1.0.1` release promotes these cases.

No model output, live answer, unverified document, or invented citation is labeled as ground truth. The profile cases specify facts and behavior, not identical response strings.

## Validation and versioning

Validate the checked-in release from the repository root:

```powershell
.\.venv\Scripts\python.exe -m eval.dataset
```

The command validates case schema, unique IDs, categories/profiles, document existence, path boundaries, evidence locations and quotes, answer support, unsupported/conflict/ACL/prompt-injection requirements, and the manifest checksum/count/distribution/version consistency.

Create the initial release from the reviewed seed definitions with:

```powershell
.\.venv\Scripts\python.exe -m evaluation.phase7c_seed
```

Dataset checksum is SHA-256 over canonical UTF-8 JSON with sorted keys and compact separators. The manifest also records timestamp, corpus and dataset identities, case/category counts, source-document references, validation status, schema version, and limitations. `write_release` writes to a dataset-ID/version-specific filename, permits an idempotent same-content write, and refuses to replace an existing version with different content. A change requires a new semantic dataset version.

The `1.0.0` checksum remains `a8be8506bf56af28cc86d05e1f925572214c43c5e12b62ec7e15a577f4364725`; the `1.0.1` checksum is `7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1`.

Automated results on 2026-11-03: focused ACL/API integration tests — **6 passed**; `tests/evaluation/test_dataset.py` — **15 passed**; dataset validator — **passed** (40 verified, 0 provisional); complete backend suite (`python -m pytest -q`) — **134 passed, 1 skipped, 2 warnings**. The skipped test is `tests/integration/test_live_qdrant_gates.py::test_live_qdrant_tenant_and_acl_filters_reject_other_principals` because `OMNIRAG_7A_QDRANT_URL` is not configured. The two warnings are FastAPI `on_event` deprecation warnings from `api/main.py` and the installed FastAPI router; they are unrelated to dataset validation.

## Phase 7b and Phase 8 compatibility

Phase 7b `ExperimentConfig` already accepts `dataset_id` and `dataset_version`, and its run manifest stores those values under `dataset.id` and `dataset.version`. The test suite constructs an experiment manifest from this release identity and verifies both fields. Phase 8 can load the JSON release, use the case IDs to join metric reports, map supported answer/evidence labels into the existing benchmark dataclasses, and handle conflict, abstention, ACL, injection, and profile expectations with category-aware checks. No benchmark execution or new experiment feature is included here.

## Known limitations

- The 40-case set is at the low end of the requested range; additional cases should be added only when they introduce distinct, source-supported coverage.
- The source fixture text does not itself enforce ACL or tenant labels. Phase 8 must apply the structured context recorded in each ACL case when constructing its evaluation snapshot.
- Unsupported coverage is concentrated on the one explicit topic gap in the corpus.
- Text-based PDF content is represented; the image-only scanned PDF and OCR quality are outside this release's ground truth.
- All documents are synthetic. The set is small and should not be treated as production-representative.
- Dataset validation verifies expected evidence against source files; it does not prove that a live ingestion snapshot indexed the same corpus revision.

## Acceptance checklist

- [x] Immutable machine-readable releases `1.0.0` and `1.0.1` and their manifests exist.
- [x] All 40 cases have unique IDs, explicit expected behavior, and corpus-resolved evidence locations.
- [x] All 40 cases are verified and validator-checked; `1.0.0` preserves its four historical provisional labels.
- [x] Unsupported cases have no fabricated answer; conflict cases preserve both documented values.
- [x] ACL/tenant allow/deny outcomes were demonstrated through the existing API/retrieval path; injection and profile cases have explicit expected outcomes and contexts.
- [x] Dataset validation and Phase 7b manifest compatibility tests pass.
- [x] New dataset tests and complete backend suite pass.
- [x] Phase 7b manifests can reference dataset `omnirag-phase8-eval@1.0.1`.

Phase 7c is **COMPLETE**. The four ACL/tenant outcomes were verified against an isolated structured snapshot; release `1.0.1` is the verified dataset. Release `1.0.0` remains unchanged for reproducibility. No production or persistent collection was modified.
