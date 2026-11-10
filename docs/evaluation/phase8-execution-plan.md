# Phase 8 B6 — full baseline execution plan

**Status:** Plan prepared for explicit approval. The 600-unit baseline has **not** been run.
**Evidence source:** B5 pilot `phase8-pilot-20261104-b5`, completed 2026-11-03 19:06–19:10 UTC; Phase 7b manifests/traces/results and the two deterministic offline scores.
**Dataset:** `omnirag-phase8-eval@1.0.1`, SHA-256 `7b44b8f616b2df74768ff9620c30d3a311000ef0bffd41015f7bb49123ca7af1`.
**Corpus:** evaluation collection `omnirag_eval_phase8_20261104`, snapshot SHA-256 `2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847`.

## Pilot measurements

| Measurement | Observed |
|---|---:|
| Measured units | 9 (three cases × three profiles) |
| End-to-end runner time | 262.99 s |
| Mean request time | 29.212 s |
| Maximum request time | 67.672 s |
| Cumulative provider calls | 17 |
| Cumulative input / output / total tokens | 14,858 / 540 / 15,398 |
| Highest per-unit call delta | 3 |
| Highest per-unit token delta | 5,433 |
| API estimated cost | $0.00 under the local-only zero-rate table; all response-level cost fields were null |
| Client timeouts | 0; each request had a 150-second client timeout |
| Runner status | 9 responses, each with warnings and no actual provider/retrieval failure code after warning classification |

## Proposed full execution

1. Run all **40 cases × 3 profiles × 5 repetitions = 600 measured units**. Do not reduce the repetitions. Use Lite, Core and Research; concurrency 1; seed `20261104`; deterministic shuffle with rotated order for each repetition. Record each attempt and retain all 600 units, including failed, timed-out, budget-stopped, or not-run statuses.
2. Run three **unscored warm-ups** first, using the answerable dataset case `P7C-AF-001` once under each of Lite, Core and Research. This is the authoritative warm-up policy across this plan, the protocol and runner. Store them under `data/phase8/warmups/` with `warmup=true`, and use the same explicit benchmark/budget run ID as the measured run. The runner requires exactly these three units, and the scorer accounts for their statuses while excluding them from scored units. Do not claim that these requests make all system caches cold; the application retrieval cache is disabled by profile policy, while OS/model/provider caches can warm.
3. Keep one isolated API process and the evaluation collection throughout warm-up and measurement. The process-local provider-budget ledger resets on API restart, so do not restart the API between the warm-up and measured invocations. Use the same `X-OmniRAG-Run-ID` for both, and keep the two artifact output roots separate.
4. Use the existing signed trusted-proxy identity, `research_escalation=false`, and `escalation_preference=never` on every request. Use the isolated API at `http://localhost:8001` and collection `omnirag_eval_phase8_20261104` only.
5. Validate the immutable dataset and snapshot hashes, API/Qdrant/Ollama readiness, collection vector schema and payload indexes, identity secret availability, and budget caps immediately before the run. Reconfirm the development collection baseline (`omnirag`, 22 points, checksum `e5c948c2c6ae20bcec126d58d627c5b893fd1db4f1bc7c4bbdadbbd545bafc53`) after the run.
6. Preserve every Phase 7b run directory and the top-level progress file. Score only after collection, run the offline scorer twice, compare the resulting JSON byte-for-byte, and run the post-hoc ACL audit over retrieved/cited IDs. Do not overwrite prior run IDs or artifacts.

## Caps, timeouts and safety headroom

At the last recorded runtime inspection, the isolated API had `$0.00`, 500,000-token and 500-call caps. The pilot used 15,398 tokens and 17 calls. Those runtime caps are too low for a 600-unit run. B7 remediation updated the checked-in Compose override to the proposed limits, but the Compose file does not update a running API container; live values are UNKNOWN until Docker is restored and effective settings are checked after an explicitly approved API-eval recreate.

Proposed limits for approval:

| Limit | Proposed value | Basis |
|---|---:|---|
| USD tripwire | `$0.00` | Keep; only explicitly listed local Ollama provider/model IDs receive zero rates. Any other/unpriced model must be rejected before inference. This is an estimate of provider cost, not a statement that local electricity is free. |
| Total-token cap | `4,200,000` | High-water pilot allowance: 5,433 tokens/unit × 603 total requests (600 measured + 3 warm-ups) = 3,276,099; add 25% headroom and round up. The pilot mean projects to about 1.032M tokens over 603 requests. `docker-compose.phase8.yml` and runner defaults now match this proposed cap; verify effective API runtime settings before use. |
| Provider-call cap | `3,000` | High-water pilot allowance: 3 calls/unit × 603 = 1,809; add about 66% headroom and round up. The pilot mean projects to about 1,139 calls. The call meter includes the observed embedding/generation/provider subcalls and retries; the sample is small, so retain the headroom. Compose and runner defaults now match; verify effective API runtime settings before use. |
| Per-request client timeout | `150 s` | Pilot maximum was 67.672 s. The API manifest showed generation timeout 90 s with one retry; Research allows 120 s end-to-end, so keep the client timeout finite and above the profile deadline. |
| Overall client run timeout | `108,000 s` (30 h) | A 150-second ceiling across 603 serial requests is 90,450 s (25 h 7 m 30 s); 30 h leaves approximately 19% run-level margin. Runner default is now 108,000 s and the absolute Unix deadline is persisted in progress, so resumed invocations cannot reset the total window. |

After explicit approval, recreate only `api-eval` from the already updated Phase 8 Compose override if required; preserve `.env`, Qdrant and Ollama. Pass identical limits to the runner. Repeat readiness, identity, snapshot, effective budget and resource preflight checks before any warm-up. Do not start until approval covers the larger caps and every gate passes.

## Wall-clock estimate

- Pilot mean projection: `29.212 s × 603 = 17,615 s`, about **4 h 54 m** including three mean-sized warm-ups.
- High-water projection: `67.672 s × 603 = 40,806 s`, about **11 h 20 m**.
- Timeout ceiling: `150 s × 603 = 90,450 s`, about **25 h 8 m**. Use the 30-hour overall deadline and stop on exhausted budget, missing/reset budget ledger, nonzero estimated cost, timeout, provider/API failure, unknown evidence mapping, authorization violation, denied evidence leak, wrong collection, or readiness loss.

The point estimate is not a guarantee. The pilot has only three cases and all three Research attempts emitted `research_terminated`; their detailed termination reason is not in the privacy-safe artifacts. Count those as warnings in the baseline and keep their status/metrics in the denominator. The runner now stops after any HTTP/API failure, captured provider/retrieval failure code, client timeout, missing or reset budget ledger, or ACL/evidence mapping violation; it records the failed attempt and never automatically retries it. Completed attempts are preserved during resume, and an interrupted in-flight attempt is recorded as indeterminate rather than retried.

## Storage estimate

- Host benchmark artifacts used 80,402 bytes for nine pilot attempts, progress, and two score files. Linear projection to 600 units is about 5.36 MB; with 50% headroom, about 8.04 MB.
- Phase 7b manifest/trace/result artifacts used 64,471 bytes for nine runs. Linear projection to 600 is about 4.30 MB; with 50% headroom, about 6.45 MB.
- Combined measured-file projection with the stated headroom is about **14.5 MB**, excluding SQLite growth. The isolated volume currently contains 58 files totaling 445,939 bytes; `omnirag.sqlite3` is 225,280 bytes after B2 and B5 activity. No pre-pilot SQLite size was captured, so per-run SQLite growth is **UNKNOWN**. Keep at least 1 GiB free on the Docker data disk as operational reserve and verify before starting. The Windows C: volume reported approximately 275.9 GiB free.

## Machine and model constraints

- Host reported 12 logical processors, about 15.67 GiB RAM total and about 3.93 GiB physically free at the post-pilot sample. Exact CPU SKU/core count was unavailable.
- Docker containers reported a 7.59 GiB memory limit. A post-pilot point-in-time sample showed API-eval at 119.8 MiB and Qdrant at 235.1 MiB; these are not peak inference measurements.
- `ollama list` reported `llama3.2:3b` at 2.0 GB and `nomic-embed-text:latest` at 274 MB on disk. `ollama ps` was empty after the pilot, so resident/peak model memory was not captured. `nvidia-smi` was unavailable; GPU presence and VRAM are **UNKNOWN**.
- Other unrelated Docker workloads were running and some had nonzero CPU use. Before a valid baseline, record or quiesce unrelated workload through the operator’s normal process; do not silently stop their containers. During warm-ups and the run, monitor host RAM, Docker memory, Ollama process residency, CPU, and disk. Stop if memory pressure causes model unload, swapping, or request failures. A short serial pilot does not establish full-run peak resource requirements.

## Baseline interpretation and stop gates

The pilot passed infrastructure gates: nine unique runs, all three artifact files per run, deterministic offline scores, budget below caps, and zero ACL violations/leaks. It also found a quality result to preserve: unsupported-case abstention was correct in Core and Research but failed in Lite (8/9 overall). All three answerable units matched the exact answer. This does not change profile or retrieval behavior and is not treated as an infrastructure stop gate; the baseline measures the current behavior.

Warnings are not mislabeled as hard failures. The runner/scorer now preserve warning codes separately from actual provider/retrieval failure codes. Unavailable response token/cost values remain null; budget totals are reported separately.

**Approval checkpoint:** approve the proposed 4.2M-token / 3,000-call / $0 caps, the 30-hour run window, and the machine-monitoring/stop gates before any warm-up or 600-unit execution. No warm-up or full baseline has been run as part of B6.

## B7 remediation status (2026-11-04)

- Harness defaults and the isolated Compose override now specify 4,200,000 total tokens, 3,000 provider calls and a `$0.00` estimated-cost tripwire. This aligns configuration with the proposal; it does not itself authorize changing/recreating the running API service. Confirm effective runtime settings during preflight.
- Warm-up policy is consistently `P7C-AF-001` once per profile. `--warmup` requires the explicit shared benchmark ID, forces that case and one repetition, writes to the warm-ups output root by default, and warm-up scoring returns zero scored units while preserving warm-up status counts.
- The runner persists an absolute deadline in progress, validates it on resume with unchanged request/run timeouts and budget caps, and does not restart the deadline. Interrupted in-flight requests become indeterminate and receive a safe attempt record; completed artifacts are not repeated or overwritten.
- HTTP failures, provider failure codes, timeouts, invalid/missing/reset usage ledgers and ACL/evidence-mapping violations halt scheduling. The failed attempt is persisted, pending units are marked not-run with the stop reason, and no failed unit is automatically retried.
- Focused validation: `tests/evaluation/test_phase8_runner.py` and `tests/evaluation/test_phase8_scorer.py` — **18 passed** (offline `uv` environment; no network).
- Current preflight could not pass: local API and Qdrant health GETs failed with `HttpRequestException`; Docker CLI could not connect to the Docker Desktop Linux engine pipe. See `docs/evaluation/phase8-baseline-readiness.md`. No chat request or inference was made.
