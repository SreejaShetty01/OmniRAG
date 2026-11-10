# OmniRAG Phase 12 â€” Production Readiness Audit

**Date:** 2026-11-04
**Recommendation:** **NO-GO for production deployment** until the release gates below are addressed and verified in the target environment. The repository has useful application controls, but live infrastructure, restore, rate limiting, and production deployment behavior have not been verified.

## Architecture and deployment assessment

The application is a FastAPI service with the React production bundle built into the Python image. Compose runs the API and Qdrant; SQLite metadata, uploaded source files, and Qdrant data use named volumes. API and Qdrant ports in the base Compose file bind to host loopback, so they are not published on external interfaces by this configuration. The Phase 8 overlay is an evaluation deployment and is not a production deployment recipe.

The Docker build uses a multi-stage frontend build and Python 3.11 runtime. `docker-compose.yml` selects `qdrant/qdrant:latest`, which is mutable and prevents reproducible Qdrant deployment. No CPU/memory limits or restart policies are defined. The API runs under the image's default user; non-root execution has not been validated against volume ownership. These should be resolved and tested in a deployment-specific change before production.

### Safe improvement made

The Dockerfile copies the build context into the runtime image (`COPY . .`). The prior `.dockerignore` excluded `.env` but did not exclude `.env.phase8` or other environment variants. The ignore rules now exclude `.env`, `.env.*`, `*.env`, and `*.env.*`, while retaining the documented non-secret `.env.example` template. This prevents environment files from being included through the broad context copy. The environment filenames were checked, but their contents were not read or printed. No `.env` file was modified.

## Security assessment

### Implemented controls

- Production identity is accepted only after a trusted proxy signature passes HMAC-SHA256 validation, constant-time comparison, identity length/newline checks, and a 60-second timestamp window (`api/trusted_proxy.py`). The API dependency fails closed outside development if a verified identity is unavailable (`api/dependencies.py`).
- Chat, document, collection, and conversation routes use identity and tenant-scoped stores/scopes. Retrieval filters include active points, ACL principals, and tenant conditions. The Phase 7c integration tests exercise allowed and denied synthetic access cases.
- Secrets use Pydantic `SecretStr`; settings responses are designed to expose readiness rather than secret values. CORS uses an explicit configured origin list. The API binds to loopback in the base Compose configuration.
- Upload routes enforce a configured byte cap while streaming, normalize and allowlist file extensions, constrain destination paths, and partition upload directories by tenant and user.
- Request validation and unexpected server failures return normalized error envelopes; unexpected exceptions are logged with a request ID while the response body stays generic.

### Required deployment controls / gaps

- Authentication depends on a separately operated reverse proxy. Production must strip client-supplied `X-OmniRAG-*` identity headers, authenticate users, sign verified identities, use TLS, and restrict direct API reachability. This proxy configuration has not been inspected or verified.
- Rate limiting is not implemented in the API. Deployment documentation assigns it to the reverse proxy; a production proxy and its limits have not been verified.
- The base Compose Qdrant service has no API key configured. Its host port is loopback-bound and its service is on the Compose network, but host-local processes can reach the published port. Confirm host access is acceptable or remove the host publication in the production-specific deployment configuration.
- `qdrant/qdrant:latest` is not pinned. Pin and test an approved immutable version/digest before release.
- The Docker image's default runtime user is root. Non-root operation and data-volume permissions need a controlled deployment validation.
- The API exposes `/docs`/OpenAPI by default through FastAPI. Confirm whether public production access is permitted; restrict or disable it in the deployment if policy requires.

## API, validation, and reliability

Request models validate profile names, lengths, and extra fields. Chat execution preserves shared Lite/Core/Research profile routing. Generation, retrieval, claim verification, citation resolution, abstention, ACL, tenant filters, token budgets, and Research deadlines remain application-level controls; this audit did not weaken them. `BudgetExceeded` is mapped to HTTP 429 and unexpected errors return HTTP 500 with a request ID.

The application has generation, verification, reranker, and Research time bounds. These are provider-call/workflow limits rather than proof of a global ingress deadline. Proxy request body/time limits, connection limits, and rate limits are deployment responsibilities and remain unverified. No live provider/API load or concurrency test was run.

## Logging, monitoring, and tracing

The application writes JSON-formatted logs to stdout, including request ID, method, path, status, and exception details for unhandled failures. Phase 7b experiment tracing is available for explicitly configured runs. There is no confirmed production metrics backend, alerting, log retention/rotation, distributed tracing collector, or dashboard in the repository configuration. Configure and verify these before production; ensure logs and traces have access controls and retention policies appropriate for user/document metadata.

## Health, persistence, backup, and recovery

`/health` is liveness. `/health/ready` checks Qdrant connectivity and returns 503 when unavailable. Compose health checks use these endpoints. The API performs payload-index initialization for an existing Qdrant collection on startup; startup/recovery behavior has not been exercised against live production volumes in this audit.

Compose persists Qdrant storage and API data in named volumes. SQLite uses WAL mode, and uploads share the API data volume. Documentation says SQLite/WAL data, uploads, and Qdrant data must be backed up together. The repository does not provide a production backup schedule, coordinated snapshot procedure, retention policy, off-host copy, restore runbook, or verified restore evidence. A backup is not production-ready until a restore is tested and data consistency checked. Never treat Docker named-volume persistence as a backup.

No live containers, volumes, databases, or collections were inspected or changed for this audit.

## Frontend production readiness

The Dockerfile builds frontend assets and copies `frontend/dist` into the API image; FastAPI serves the static bundle. `npm test` and `npm run build` passed in this workspace. A production domain, TLS termination, cache/security headers, content security policy, and browser-to-production API routing have not been verified. Do not place secrets in `VITE_*` build variables because they are client-visible.

## Test and static-check results

| Check | Result |
|---|---|
| Backend: `.venv\Scripts\python.exe -m pytest tests -q` | **169 passed, 1 skipped, 2 warnings in 13.94 s** |
| Skipped backend test | `tests/integration/test_live_qdrant_gates.py` was skipped because `OMNIRAG_7A_QDRANT_URL` was not configured. |
| Backend warnings | Two existing FastAPI `on_event` deprecation warnings. |
| Frontend: `npm test` | **3 tests passed**. |
| Frontend: `npm run build` | **Passed**; TypeScript and Vite production bundle completed. |
| Compose syntax: `docker compose config --quiet` | **Passed**. This validates configuration syntax, not container health or effective deployment secrets. |
| Docker context ignore policy | **Passed**; rules exclude environment files while retaining `.env.example`. No secret values were inspected. |
| Dockerfile static check: `docker buildx build --check .` | **Not run to completion**; Docker returned `open C:\\Users\\sreej\\.docker\\buildx\\.lock: Access is denied`. No image build was started. |
| `git diff --check` for changed files | **Passed**. |

No production deployment, container rebuild, live API request, inference, benchmark, Qdrant mutation, or restore test was performed.

## Production deployment checklist

### Required before release

- [ ] Pin API base image and Qdrant image to reviewed immutable versions/digests.
- [ ] Use a production-specific Compose/deployment file with explicit CPU, memory, process, restart, and health/recovery policies validated against the host.
- [ ] Verify API runs non-root with correct persistent-volume ownership, or document an approved exception.
- [ ] Configure the trusted reverse proxy: TLS, authenticated principal/tenant derivation, inbound identity-header stripping, signing-secret handling/rotation, direct API network restriction, request/body/concurrency limits, and rate limiting.
- [ ] Decide whether Qdrant host-loopback publication is necessary; if not, omit host port publication in production. Configure Qdrant authentication/network policy as appropriate.
- [ ] Supply secrets through the deployment secret manager. Confirm build context/image contains no environment or secret files and that no secrets enter frontend assets or logs.
- [ ] Establish encrypted, access-controlled, off-host backups for SQLite/WAL, uploads, and Qdrant. Document recovery-point/recovery-time objectives and retention.
- [ ] Perform a restore drill into an isolated environment and validate SQLite records, upload files, Qdrant collection schemas/counts, and application readiness.
- [ ] Configure and test metrics, alerting, log retention, disk/RAM monitoring, and incident escalation.
- [ ] Run the live Qdrant gates and authenticated endpoint checks against a production-like isolated environment; validate Lite/Core/Research deadlines and failure responses without relaxing safeguards.
- [ ] Verify TLS, CORS allowlist, docs/OpenAPI exposure policy, frontend routes/assets, and proxy-to-API behavior on the actual production domain.
- [ ] Define release/rollback steps, schema compatibility checks, change approval, and post-deploy health verification.

## Known risks and limitations

1. Live infrastructure, proxy trust boundaries, TLS, secrets injection, Qdrant access policy, and resource behavior were not verified.
2. Backups and restoration are documented only at a high level; no operational runbook or restore drill exists in this evidence set.
3. No API-level rate limiting, metrics/alerting configuration, resource limits, or immutable Qdrant pin is present in the inspected Compose configuration.
4. Production startup, migrations, restart/recovery, and non-root volume ownership remain unverified.
5. Live Qdrant integration was skipped; local tests use isolated fixtures/fakes for many boundaries.
6. The Phase 8 baseline is incomplete and Phase 10 changes have no measured before/after performance results. No performance or capacity claim is made.

## Go / no-go

**NO-GO for production today.** The application and frontend automated tests pass, and a build-context secret-exclusion gap was corrected. Production approval should wait for the required deployment controls above, especially trusted proxy/TLS/rate limiting, pinned images, resource/recovery policy, and a verified backup restore. Production infrastructure must then be checked in its actual target environment.

## Files changed

- `.dockerignore` â€” exclude local environment/secret files from Docker build context while retaining `.env.example`.
- `docs/operations/production-readiness.md` â€” this audit and release checklist.

## M1-P5 addendum â€” 2026-11-05

The original 2026-11-04 audit and its NO-GO recommendation above are preserved as historical findings. M1-P5 adds API process-local per-principal rate limiting, authenticated aggregate `/metrics`, optional configured Ollama readiness probing, configured-secret redaction for application logs and Phase 7b JSON artifacts, evaluation Compose resource declarations, and release examples. Focused unit tests cover the new limiter, metrics aggregation and redaction paths. No services were started or restarted, no live provider or collection was called, and no backup/restore or full regression was run.

These changes do **not** close the production blockers. TLS/proxy, resource enforcement and fit, Qdrant authentication, image pinning, backup/restore, operational monitoring, global ingress limits and deployment recovery remain unverified or incomplete. The detailed M1-P5 closeout ledger is internal and is not included in this public release. The current target description remains **production-oriented, with measured limits**; no measured capacity claim is made here.
