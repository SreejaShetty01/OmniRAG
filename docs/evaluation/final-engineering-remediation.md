# Final engineering remediation and offline evaluation

## Scope and safety

This remediation changes the diagnostic scorer and deterministic identifier check only. The Phase 8 dataset, snapshot manifest, official runner/scorer, application access-control policy, configuration, secrets, containers, and historical run records were left unchanged. No API requests or inference were made. The offline rescore below reads saved responses and writes new artifacts; it does not rewrite the original runs.

The current worktree was already substantially modified and contained many untracked additions before this task. Changes were restricted to the two diagnostic/verification source files, their focused tests, this report, and a new post-hoc artifact directory.

## Confirmed fixes

### Authorized marker classification

The old diagnostic scorer treated any appearance of the restricted-source marker in an answer as a leak, regardless of the request principal. That made the authorized allow-case P7C-AT-001 fail the scorer even though the retrieved chunk, final evidence, and citation were permitted by its structured ACL and matching tenant.

The scorer now looks up the expected evidence source in the snapshot's document ACL metadata and evaluates the case principal and tenant. An answer marker from an authorized source is recorded as an authorized occurrence; a marker from a denied source or an unmapped source is classified as unauthorized/unknown. Candidate, final-context, and citation chunk checks remain principal- and tenant-aware through the snapshot point ACL. The allow and deny regression tests cover both outcomes.

Offline before/after for the three saved P7C-AT-001 attempts:

| Profile | Original scorer ACL result | ACL-aware rescore | Explanation |
|---|---|---|---|
| Lite | FAIL | PASS | Marker was present for the designated allowed user in the matching tenant; candidate/context/citation checks found no disallowed chunk. |
| Core | PASS | PASS | No marker in the answer. |
| Research | PASS | PASS | No marker in the answer. |

This corrects the evaluation false positive. It is not a new live security test and does not change the API's ACL behavior.

### Identifier punctuation and abstention

The deterministic identifier guard previously treated sentence-ending punctuation captured by the identifier regex as part of the identifier. It now removes only trailing sentence punctuation before comparison; internal punctuation remains significant. Focused tests cover terminal punctuation, internal punctuation, and a negative case where `A.B` must not be treated as `AB`.

Saved failed claims show that missing/invalid claim evidence references and identifiers absent from cited evidence remain reasons for abstention. The pipeline's final gate abstains on unsupported or contradicted claims after at most one bounded correction. I did not auto-attach answer-level citations to claims: the saved data does not prove which citation supports each claim, and doing so could make unsupported claims pass verification. No abstention threshold or ACL safeguard was weakened.

The punctuation correction removes a confirmed false rejection mechanism for future executions. Historical answers were not regenerated, so this report does not claim those old abstentions changed. Four saved attempts had punctuation-sensitive identifier comparisons; in some attempts, separate missing-reference failures would still require abstention. The remaining answerable-case abstentions need claim-by-claim human adjudication and better grounded evidence references.

### HTTP 200 latency classification

The diagnostic runner now labels HTTP 200–299 responses as completed even if elapsed time exceeds the 150-second latency budget. It records `latency_budget_exceeded` separately. HTTP 408/504 remain timed out, other non-2xx responses remain failures, and transport exceptions retain their failure path. A test verifies both slow HTTP 200 and HTTP 504 classification.

## Post-hoc rescore

Canonical output: `data/diagnostic/posthoc-remediation-20261107T072431Z-663cec3c/`

- `report.json`: metrics, denominators, confidence intervals, run-source information, and provenance limitations.
- `report.md`: human-readable profile summary.
- `attempts_scored.jsonl`: one rescored record per saved attempt.
- `human_audit.csv`: claim-level review export. It has 40 primary claim rows from up to 30 answered responses and 10 duplicate rows for a second reviewer. No reviewer fields are completed; faithfulness and citation accuracy remain pending human review.

The input dataset SHA-256 is `6a99cdd3389e4579a5d4e0aa55d083764595361680bd4b09af7168df8defc561`; this is diagnostic dataset v1.0.1, not the Phase 8 pinned checksum. The snapshot checksum is `2b325b263d49775788e66d3594c3d1109c319205c563a7c9253ad2b11dbf3847`, representing 18 points. The source API was configured to this Phase 8 evaluation collection, not the separate 1,467-point M1 corpus. Historical run manifests do not contain a Git commit or Docker image identity, so neither has been retroactively attributed.

### Answerability confusion and exact-answer scoring

Each profile has 33 answerable and 7 expected-abstention cases. “Exact answer accuracy” is calculated only for answerable cases with an exact gold match that the system answered; evidence-based answers without deterministic exact labels are not guessed. False abstentions and correct unanswerable abstentions are reported separately.

| Profile | Answerable exact correct | Answerable exact wrong | False abstained / 33 | Answered, exact scorer unavailable | Correct abstentions / 7 | Unanswerable answered |
|---|---:|---:|---:|---:|---:|---:|
| Lite | 5 | 4 | 1/33 | 23 | 4/7 | 3 |
| Core | 2 | 2 | 19/33 | 10 | 6/7 | 1 |
| Research | 5 | 1 | 16/33 | 11 | 6/7 | 1 |

Exact-answer accuracy among answered, exact-match cases is Lite 5/9 (Wilson 95% CI 26.7–81.1%), Core 2/4 (15.0–85.0%), and Research 5/6 (43.6–97.0%). The small denominators and wide intervals do not support broad quality claims. Overall abstention-policy agreement is Lite 36/40, Core 20/40, and Research 23/40. The Research 23/40 figure includes both answerable non-abstentions and correct abstentions for expected-unanswerable cases; it is not answer accuracy.

The earlier mixed “answer correctness” figures were Lite 6/13, Core 5/13, and Research 8/13. Those denominators included unsupported-case abstentions as if they were answer-correct; the revised report keeps that behavior under abstention correctness and reports answer accuracy only for exact-match answers actually produced.

The saved responses' sufficiency field is heuristic and uncalibrated. Among unanswerable cases that were answered, Lite has three with a stored partial/strong sufficiency state, Core has one partial/strong, and Research has one with no usable sufficiency state. These are not ground-truth judgments that the answer was supported.

### Retrieval, citations, latency, and token observations

The retrieval values below are means across 33 answerable cases with mapped evidence. They are from the saved hybrid retrieval inspections, not independent dense-only, sparse-only, hybrid, and reranker trials.

| Profile | Recall@5 | MRR@10 | Evidence precision (n) | Evidence coverage (n) | Citation precision (n) | Citation recall (n) |
|---|---:|---:|---:|---:|---:|---:|
| Lite | 0.955 | 0.970 | 0.887 (33) | 0.909 (33) | 0.887 (33) | 0.909 (33) |
| Core | 0.955 | 0.970 | 0.820 (25) | 0.697 (33) | 0.820 (25) | 0.697 (33) |
| Research | 0.924 | 0.970 | 0.922 (27) | 0.788 (33) | 0.922 (27) | 0.788 (33) |

| Profile | Latency n | p50 | p95 | max | Actual transport/HTTP timeouts | 150s latency-budget violations |
|---|---:|---:|---:|---:|---:|---:|
| Lite | 40 | 28.797 s | 41.140 s | 167.938 s | 0 | 1 |
| Core | 40 | 35.868 s | 51.516 s | 54.266 s | 0 | 0 |
| Research | 40 | 69.758 s | 97.766 s | 100.015 s | 0 | 0 |

All 120 saved responses had HTTP 200. The Lite 167.938-second attempt was a successful HTTP response that violated the latency budget; it was not a transport timeout. Provider-reported token usage was present for 37/40 responses in each profile. Totals from the available responses were Lite 49,315 input / 2,629 output tokens, Core 79,835 / 4,949, Research 146,354 / 7,336. Provider usage was absent for 3 attempts per profile.

The original saved summary counted the slow Lite HTTP 200 as one timeout; the corrected classification is zero HTTP/transport timeouts and one latency-budget violation.

Budget header observations are cumulative per run ID, not a reliable profile split: the Lite/Core source run reached 176 calls, 137,730 total tokens, and $0; the separate Research run reached 101 calls, 154,199 total tokens, and $0. Do not add the Core cumulative count to the Lite cumulative count.

The claim-status totals in `report.json` are machine-produced validation statuses, not human-verified faithfulness labels. The human audit CSV is pending review. No claim-level faithfulness or citation-accuracy score is reported as complete.

### ACL/security results

Across 24 security-case attempts (8 per profile): candidate leakage was 0/24, final-context leakage 0/24, citation leakage 0/24, and unauthorized-or-unknown answer-marker disclosure 0/24. One answer-marker occurrence was authorized for its case principal and tenant; it is not a leak. The 95% Wilson upper bound for zero events is about 13.8%; rule-of-three bound is 12.5%. These are observations on the small saved diagnostic set, not proof of general security.

The per-profile split is in `report.json` under `security_by_profile_and_expected_access`, including ACL allow/deny cases and prompt-injection cases. Prompt-injection rows do not carry an ACL expected-access label and are not counted as ACL allow/deny outcomes.

## Provenance and corpus limitations

New runs now record the dataset hash, snapshot hash, configured Qdrant collection, snapshot point count, embedding alias/provider/model/digest/dimensions, generation model, runtime limits, Git commit and tracked-diff hash, dirty-worktree state, platform/CPU/RAM when available, and optional Docker image ID/repository digests when `--api-container-name` is supplied and Docker inspection succeeds. A live Qdrant count is explicitly not claimed; the stored count is the count in the verified snapshot manifest. Missing optional provenance is stored as unavailable rather than fabricated.

The saved runs queried the 18-point Phase 8 snapshot. No live experiment against the 1,467-point M1 corpus was run because its separate isolated API/Qdrant stack was not active. No dense-only, sparse-only, or reranker comparison was run. No oracle-context experiment, correction ablation, or live post-fix Research request was run. The API and Qdrant were stopped during this offline work; there were no service changes.

## Validation

Focused command:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/diagnostic/test_run_diagnostic.py tests/unit/test_generation_verification.py
```

Result: **28 passed**. Tests cover ACL-aware authorized/denied marker handling, answer confusion counts, slow HTTP 200 versus HTTP timeout, future provenance fields, and identifier punctuation behavior.

Broader targeted command:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/diagnostic tests/unit/test_generation_verification.py tests/integration/test_phase7c_acl_dataset.py
```

This command did not complete. It emitted 28 passing progress dots and then stalled while entering the integration selection; it was interrupted. No integration pass is claimed. The deterministic in-memory ACL integration test therefore remains unverified in this run.

`git diff --check` and `git diff --cached --check` reported no whitespace errors; Git emitted only existing line-ending conversion notices for modified files.

## Changed files and remaining work

Source and tests changed:

- `scripts/diagnostic/run_diagnostic.py`
- `verification/claim_verifier.py`
- `tests/diagnostic/test_run_diagnostic.py`
- `tests/unit/test_generation_verification.py`
- `docs/evaluation/final-engineering-remediation.md`
- `docs/README.md` (index link to this report)

The canonical offline rescore artifacts are the four files in `data/diagnostic/posthoc-remediation-20261107T072431Z-663cec3c/`. Earlier timestamped post-hoc directories from this remediation turn are intermediate scorer snapshots; use the canonical directory above for cited results.

No application ACL, retrieval, answer-generation, model, or prompt behavior was changed. Remaining work is human claim-level review, determining why claim references are omitted in some model outputs without weakening verification, rerunning the focused integration test in a stable test environment, and testing the punctuation fix through a new controlled request only when approved. Live corpus-mode comparisons and oracle-context experiments require the dedicated 1,467-point evaluation stack.

This work does not complete deployment or production readiness.
