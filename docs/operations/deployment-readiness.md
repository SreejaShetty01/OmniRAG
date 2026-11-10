# OmniRAG Deployment Readiness and Hardening

**Date:** 2026-11-05
**Scope:** Static repository review and non-mutating local validation. No container start/stop/recreate, image build, live deployment, inference, backup, restore, or secret-value inspection was performed.

## Classification

**Local development ready** — source-level tests and frontend build pass, and all three Compose configurations render. Container runtime state could not be inspected because access to the Docker Desktop Linux engine pipe was denied. This is not a claim that a running local stack, single-user deployment, small-team deployment, or production deployment is ready.

## 1. Docker and deployment configuration

| Check | Finding | Status |
|---|---|---|
| Build context and Dockerfile | API build context is `.`. `Dockerfile` uses a Node 22 Alpine frontend build stage and Python 3.11 slim runtime, installs Python package dependencies, then uses `COPY . .`; its final `COPY --from=frontend` includes the built UI. | PARTIAL — static configuration only |
| Build context secret exclusion | `.dockerignore` excludes `.env`, `.env.*`, `*.env`, and `*.env.*`, while explicitly retaining `.env.example`. The protected `.env`, `.env.phase8`, and `.env.phase8-rw` filenames are also ignored by Git. No values were read into output. Because the Dockerfile copies the whole context, exclusion depends on keeping these rules correct. | PASS — rule inspection |
| Compose rendering | `docker compose config --quiet`, `docker compose -f docker-compose.yml -f docker-compose.phase8.yml config --quiet`, and `docker compose -f docker-compose.yml -f docker-compose.phase8-rw.yml config --quiet` all exited successfully. These checks validate syntax/rendering only. | PASS |
| Running container comparison | `docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' --filter name=omnirag` and targeted `docker inspect` could not connect to `npipe:////./pipe/dockerDesktopLinuxEngine` (`permission denied while trying to connect to the Docker API`). Current image IDs/tags, labels, effective env, health, restart policy, resource enforcement, and running-versus-rendered differences are therefore unknown. No container was changed. | UNKNOWN |
| Image references | Dockerfile bases are mutable tags `node:22-alpine` and `python:3.11-slim`; Compose references `qdrant/qdrant:latest`. API images use mutable local names such as `omnirag-api-eval` and `omnirag-api-eval-rw`. No immutable digest pin is defined. | FAIL — reproducible release pinning is absent |
| Health and readiness | API Compose healthchecks call `/health/ready`; base and Phase 8 API wait for healthy `qdrant`. Readiness always checks Qdrant and checks Ollama only when `OMNIRAG_OLLAMA_HEALTH_URL` is configured. The `api-eval-rw` Compose service has a healthcheck but no `depends_on: qdrant-rw`; its startup order is documented as a manual prerequisite. | PARTIAL — configured, not live-verified; RW start ordering is not Compose-enforced |
| Restart and resource policies | Base API/Qdrant have no restart policy or explicit CPU/memory limits. Phase 8 evaluation API specifies 2 CPUs and 4 GiB memory. RW API specifies 2 CPUs/4 GiB and RW Qdrant specifies 1 CPU/2 GiB. Enforcement and fit are unknown without Docker access; restart policies are absent in all inspected files. | PARTIAL |
| Network and persistence | Base API and Qdrant host ports bind to `127.0.0.1`; API and Qdrant use named volumes. Evaluation stacks have separate named API/Qdrant volumes in their configuration. Current volume identities and attachments could not be inspected. | PARTIAL |
| Runtime user | Dockerfile does not set a non-root `USER`; the runtime defaults to the Python base image's configured user (normally root). Volume ownership and a non-root deployment have not been validated. | FAIL — no non-root runtime configuration |

### Compose/documentation discrepancy corrected

`docs/operations/deployment.md` said Qdrant ports were not published to the host. The base Compose file maps REST/gRPC to `127.0.0.1:6333` and `127.0.0.1:6334`. The documentation now says they are reachable by host-local processes but not bound to external interfaces, and recommends removing these mappings in a production-specific configuration if host access is unnecessary.

No current running-container versus Compose diff is available: Docker daemon access failed before container metadata could be read. The Phase 12 report contains historical Docker findings, but they are not substituted for current runtime evidence.

## 2. Environment and secret handling

- `Settings` uses the `OMNIRAG_` prefix; the checked `.env` and isolated env files use that prefix. Environment-variable **names only** were inspected. Secret values were not displayed.
- Compose injects `.env` into the base API and `.env` plus `.env.phase8` into the Phase 8 API; the RW API uses `.env.phase8-rw`. These are runtime environment files, not Docker Secret objects. They are excluded from the build context, but Docker administrators can inspect container environment values when the engine is available.
- `SecretStr` is used for provider/auth API keys in settings. Application log formatting and experiment artifact serialization have configured-secret redaction tests, included in the recent focused backend run. This does not prove redaction by Docker, provider SDKs, external log sinks, or every unexpected exception.
- No secret literals were found in the reviewed Compose/Dockerfile configuration; secret values are supplied through env files. This is a source inspection, not a scan of their contents or the built image.
- Frontend `VITE_*` values are build-time client-visible values and must not contain secrets. No frontend secret-bearing build argument is defined in the reviewed Dockerfile.

**Status:** PARTIAL — protected files are excluded and app-level redaction is tested; runtime injection, image contents, rotation, and external log handling are unverified.

## 3. Application security assessment

| Control | Finding | Status |
|---|---|---|
| Signed proxy identity | HMAC-SHA256 signature, constant-time compare, 60-second timestamp window, length/newline checks, optional identity allowlist. Unit tests cover fresh, tampered, missing, and stale signatures. Reverse-proxy header stripping and production signer are not present in the Compose stack. | PARTIAL |
| Authentication failure | The trusted identity dependency fails closed outside development. Development mode has a fixed local identity. Current effective runtime environment could not be inspected. | PARTIAL |
| Rate limiting | Per-principal fixed-window 429 with `Retry-After` is implemented and unit-tested. It is process-local; one worker per instance is assumed. No edge/global limit was verified. | PARTIAL |
| CORS and proxy | CORS allowlist is settings-driven. API uses Uvicorn proxy headers in the Dockerfile, but no active trusted proxy, forwarded-IP policy, TLS listener, or proxy configuration is deployed in the repository Compose files. `docs/release/Caddyfile.example` is illustrative and depends on a separate auth gateway; it is not active deployment evidence. | PARTIAL |
| Upload and API validation | Upload filename/extension/path/size validation is present. Prompt 3 changed failed ingestion responses to 422 for `SourceLoadError` and 503 for other failures while preserving failure records. The frontend rejects non-2xx responses and now displays the returned safe `error_message`. | PASS — deterministic compatibility test; live HTTP not tested |
| Tenant isolation | Tenant/principal filters, active-point filter, denied-principal conditions, and scoped product-store queries are implemented and locally covered. No live cross-tenant API request was performed. | PARTIAL |
| API errors | Request validation and unexpected errors use normalized response envelopes; unexpected errors return generic client text with a request ID. Provider/ingestion errors may have more specific safe fields. Live proxy/error logging has not been exercised. | PARTIAL |

### Prompt 3 status-code compatibility

`frontend/src/api.ts` already rejects HTTP 4xx/5xx, so the 422/503 changes do not turn failed uploads into success. It previously ignored the top-level `error_message` field in the returned `DocumentResponse`, producing only a generic “Request failed” message. It now includes that field after the existing structured `error.message` and `detail` choices. A mocked 503 client regression test checks that the ingestion message reaches the caller. HTTP transport and actual browser display were not tested live.

## 4. Dependency vulnerability assessment

| Check | Result | Status |
|---|---|---|
| `npm audit --omit=dev` from `frontend/` | Could not retrieve advisory data: DNS lookup for `registry.npmjs.org` failed with `getaddrinfo ENOTFOUND`. No vulnerability count can be inferred. No dependency was installed or changed by the audit command. | UNKNOWN |
| `npm ls --omit=dev --depth=0` | Local production tree resolved: `lucide-react@0.468.0`, `react@19.3.1`, `react-dom@19.3.1`, `react-markdown@10.1.0`, `remark-gfm@4.0.1`. This reports installed tree, not security advisories. | PASS — tree inspection only |
| Python vulnerability tool | `pip-audit` is not installed (`No module named pip_audit`); `safety` was not found. No package was installed. | NOT TESTED |
| Python dependency consistency | `.venv\Scripts\python.exe -m pip check` → `No broken requirements found.` This checks dependency metadata compatibility, not CVEs. | PASS — consistency only |
| Dependency pinning | Python `requirements.txt` and `pyproject.toml` use version ranges; no Python lockfile was found. npm uses `package-lock.json`. | PARTIAL |

Minimal compatible dependency fixes cannot be recommended from this evidence because the advisory databases were unreachable and Python vulnerability scanning was unavailable. Re-run audits in a network-permitted CI environment before release; do not upgrade major versions based on this failed audit.

## 5. Operational reliability

- API liveness is `/health`; readiness calls Qdrant and optionally Ollama. A mocked Qdrant connection failure and mocked Ollama failure both return sanitized HTTP 503 and are covered by deterministic tests. This does not validate actual restart/recovery behavior.
- Docker healthchecks exist for the API and Qdrant. The base/Phase 8 API has health-gated Qdrant dependency ordering. RW API readiness is checked, but Compose does not encode its dependency on `qdrant-rw`.
- API logs are structured JSON to stdout; application logger redaction is tested. `/metrics` is authenticated and process-local; no monitoring backend, alerting, retention, or external tracing service is configured in the reviewed deployment files.
- No explicit base-stack CPU/memory caps, restart policy, ingress request limit, or concurrency limit is configured. Evaluation Compose resource numbers are declarations only until runtime inspection verifies enforcement.
- `docs/release/Caddyfile.example` shows an optional Caddy/forward-auth pattern, but no active TLS/reverse-proxy service or real auth gateway configuration exists in Compose. Certificate issuance, header stripping, CORS, and direct API reachability remain deployment tasks.
- Provider and Qdrant outage handling is unit/mock-tested at selected boundaries. Live outage and recovery are untested and were not injected.

**Status:** PARTIAL.

## 6. Backup and recovery

`docs/operations/backup-restore-drill.md` is a prepared, approval-gated procedure. `scripts/ops/prepare_backup_restore_plan.ps1` validates distinct source/restore identities and prints a plan; it performs no backup, restore, Docker, SQLite, Qdrant, filesystem, or network operation.

The documented data set is SQLite/WAL metadata, uploaded source files, and a Qdrant collection snapshot. Experiment runs, dataset/snapshot manifests, configuration identifiers, and checksums should also be retained for reproducibility; secret env files should be restored from the approved secret manager, not copied into image artifacts. The current Compose volumes persist application and Qdrant data but are not backups.

No coordinated SQLite online backup, Qdrant snapshot, off-host copy, restore, integrity check, point-count comparison, ACL comparison, RPO, or RTO measurement was performed. Restore safety and recoverability are **NOT TESTED**. The existing drill correctly calls for a distinct project, volumes, and collection and requires approval before execution.

## 7. Prompt 3 API compatibility assessment

- Backend response mapping: failed source load → 422; other failed ingestion → 503; successful first upload → 201; successful existing upload/reingestion → 200.
- Persisted failed document records remain available and response body retains the sanitized `error_message`.
- Frontend API client treats 422/503 as errors and now extracts `error_message` from the document response.
- Focused backend tests after Prompt 3: **79 passed, 2 warnings** (reported in `docs/security/quality-security-reliability.md`). Frontend compatibility test added in this prompt verifies a mocked 503 body. No live route/browser test was performed.

**Status:** PASS — deterministic contract compatibility; PARTIAL for live HTTP/browser compatibility.

## 8. Commands and results actually executed

| Command/check | Result |
|---|---|
| `docker compose config --quiet` | Exit 0 (`BASE_COMPOSE_CONFIG=PASS`) |
| `docker compose -f docker-compose.yml -f docker-compose.phase8.yml config --quiet` | Exit 0 (`PHASE8_COMPOSE_CONFIG=PASS`) |
| `docker compose -f docker-compose.yml -f docker-compose.phase8-rw.yml config --quiet` | Exit 0 (`RW_COMPOSE_CONFIG=PASS`) |
| `docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' --filter name=omnirag` | Failed: `permission denied while trying to connect to the Docker API at npipe:////./pipe/dockerDesktopLinuxEngine` |
| `docker inspect --format '{{.Name}} {{.Image}} {{.State.Status}} {{.State.StartedAt}} {{.HostConfig.RestartPolicy.Name}} {{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}' ...` | Failed with the same Docker API pipe permission error; no container fields were retrieved |
| `npm audit --omit=dev --json` (`frontend/`) | Exit 1; registry DNS failure: `getaddrinfo ENOTFOUND registry.npmjs.org` |
| `npm ls --omit=dev --depth=0` (`frontend/`) | Exit 0; production dependency tree listed above |
| `.venv\Scripts\python.exe -m pip_audit --version` | Exit 1: `No module named pip_audit` |
| `.venv\Scripts\python.exe -m pip check` | Exit 0: `No broken requirements found.` |
| `npm test` (`frontend/`) | **2 test files passed; 4 tests passed** |
| `npm run build` (`frontend/`) | Exit 0; TypeScript and Vite production build passed |
| `git check-ignore -v .env .env.phase8 .env.phase8-rw` | All three files matched their `.gitignore` entries; values not displayed |

No install/upgrade, image build, runtime operation, backup/restore, or external service request was run. Compose render checks did not start services.

## 9. Minimal changes made

- `frontend/src/api.ts` — read top-level `error_message` when handling non-2xx responses, preserving existing structured error and detail precedence.
- `frontend/src/api.test.ts` — mocked 503 response regression test for ingestion error text.
- `docs/operations/deployment.md` — corrected the Qdrant host-port statement to match loopback mappings in `docker-compose.yml`.
- `docs/operations/deployment-readiness.md` — this audit.

No `.env` file, Compose runtime setting, dependency, application backend behavior, container, collection, volume, or historical artifact was modified.

## 10. Remaining deployment blockers

1. Docker Desktop access is unavailable in this session; inspect actual container/image/environment/volume/resource state in an authorized host session and compare it to the rendered Compose config.
2. Pin Python/Node/Qdrant/API images to reviewed immutable versions/digests and validate upgrades in CI.
3. Re-run npm advisory audit and Python vulnerability audit from a network-permitted CI environment; current CVE status is unknown.
4. Configure and verify an actual TLS/authentication proxy that strips incoming identity headers, signs authenticated tenant identities, limits ingress, and blocks direct API access.
5. Decide production Qdrant host exposure; loopback is accessible to local host processes. Add Qdrant authentication/network policy if required.
6. Define/verify restart policy, CPU/memory fit, worker count, non-root volume ownership, disk limits, and service startup ordering.
7. Execute an approved isolated backup and restore drill; verify SQLite/WAL, uploaded sources, Qdrant schema/count/indexes, ACL behavior, hashes, and measured RPO/RTO.
8. Add an operational monitoring/alerting and log retention destination, with access controls and redaction validation.
9. Perform isolated live authenticated API/profile verification and controlled dependency recovery only after explicit approval.

## Exact remaining actions for Prompt 5

1. In a network-permitted CI job, run `npm audit --omit=dev` and an approved Python dependency audit tool; review advisories and propose only compatible, necessary fixes.
2. With Docker Desktop access restored, capture read-only labels, image digests, health, restart policy, mounts, resource limits, and effective env **with all secret values masked**; compare to the three Compose renders.
3. Prepare a deployment-specific immutable image pin and production Compose/proxy draft for review; do not apply it until separately approved.
4. Obtain explicit approval for one isolated deployment validation and one isolated backup/restore drill, then produce measured evidence without touching development data.

**Prompt 4 status:** **PARTIAL** — static configuration and deterministic checks completed; dependency advisories and current runtime deployment state remain unknown, and production blockers remain open.
