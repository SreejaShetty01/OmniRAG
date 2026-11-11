# Final release and portable deployment package

**Status:** Release candidate is not ready for deployment. The files here are a single-host deployment candidate and an operational runbook, not evidence of a tested deployment.

## Package contents

- [`Dockerfile`](../../docker/Dockerfile) builds the frontend and installs the Python project, then runs the API as UID/GID 10001.
- [`docker/compose.production.yml`](../../docker/compose.production.yml) deploys an API image by immutable reference and defines authenticated private Qdrant, health checks, restart/resource policy, read-only API root filesystem, writable temporary storage, and separate named data volumes.
- [`docker/env.production.example`](../../docker/env.production.example) is a placeholder-only template. Copy it to the ignored `docker/env.production`, replace every placeholder, and provide the file through a protected deployment workflow. Do not commit it or copy secrets into build arguments.
- [`pyproject.toml`](../../pyproject.toml) is the authoritative Python dependency declaration. `pyproject.toml` is the authoritative Python dependency declaration; install development dependencies with `pip install -e ".[dev]`. The frontend uses `frontend/package-lock.json` with `npm ci`.

The image builds the frontend from its lockfile and packages the Python project from the repository. Application data is mounted at `/app/data`; Qdrant data uses its own volume. Qdrant has no published host port in this candidate. API port 8000 binds to host loopback for a separately managed same-host TLS/authentication proxy. No proxy, identity signer, TLS certificate, or secret manager is provisioned by this Compose file.

## Configuration gates

Before a deployment review, replace the API, Qdrant, Node build-base, and Python runtime-base digest placeholders with verified immutable digests. Build and publish the API image separately, then use its registry digest in Compose. Supply a unique HMAC secret of at least 32 UTF-8 bytes, a distinct Qdrant API key, a non-empty JSON principal/tenant allowlist, exact browser origins, collection name, and provider/model configuration. Production settings now reject a missing/short authentication secret or an empty identity allowlist. The API uses the Qdrant key and the Qdrant service requires that same key.

An approved release builder can produce the image with `docker build --build-arg NODE_IMAGE=<verified-node-digest> --build-arg PYTHON_IMAGE=<verified-python-digest> -f docker/Dockerfile -t <registry>/omnirag-api:<candidate> .`; after publishing, record the registry's immutable digest in `OMNIRAG_API_IMAGE`. The Dockerfile's mutable defaults are for local development only.

Compose environment injection is not a secret manager: Docker administrators can inspect container environment values. Use an approved secret-management mechanism and review its operational integration before handling sensitive production credentials. The current example is intentionally a placeholder template, not a secure secret delivery implementation.

## Review and deployment commands

These commands are for an operator after a deployment target is approved and the private env file has been populated. First confirm the project name, container names, and the explicitly named volumes do not collide with any existing data. Never run `down -v`, `volume prune`, or collection reset commands.

```powershell
docker compose -p omnirag-production --env-file docker/env.production -f docker/compose.production.yml config --quiet
docker compose -p omnirag-production --env-file docker/env.production -f docker/compose.production.yml pull api
docker compose -p omnirag-production --env-file docker/env.production -f docker/compose.production.yml up -d --wait qdrant api
Invoke-WebRequest http://127.0.0.1:8000/health/ready -TimeoutSec 5 -UseBasicParsing
```

The commands above have **not** been run against a target environment. Before starting services, inspect the rendered configuration without printing it to an untrusted log, verify all image digests and volume identities, and establish the trusted proxy/signing path. After startup, verify signed settings, one approved upload and query, ACL allow/deny cases, frontend routes, logs, budget counters, and persistence. Do not treat readiness alone as end-to-end validation.

## Persistence, recovery, and operations

The API SQLite/upload state and Qdrant vectors live in separate named volumes. They must be backed up as a coordinated set using the existing [backup and restore drill](backup-restore-drill.md), then restored into distinct temporary volumes and validated before a recovery claim. No production backup schedule, off-host backup, restore rehearsal, measured RPO/RTO, or restart-recovery test has been completed. Volume persistence alone is not a backup.

The service exposes health/readiness and authenticated process-local `/metrics`. There is no configured monitoring backend, alerting, log retention policy, distributed quota store, or queue-depth measurement. Rate limiting is per process; keep the one-worker default unless an edge quota is added and tested. CPU/memory declarations are starting limits, not measured capacity recommendations. Ollama is external to this Compose stack; model availability, GPU/CPU capacity, concurrency, and cold-load behavior must be verified on the target host.

## Reproducibility and known release blockers

- Application/package metadata currently reports version `0.6.0`. The observed Git HEAD is `645ce65c688760a8a4eced985bd911b740b16700`, but the worktree is dirty (113 status entries), so HEAD alone does not identify the source tree reviewed here. No release tag or source/archive checksum was created.
- Python dependency ranges are not a complete cross-platform lock; no verified Linux image build was possible in this workspace.
- The Dockerfile defaults to convenient Node and Python tags for local builds; release builds must pass verified immutable digests using `--build-arg NODE_IMAGE=... --build-arg PYTHON_IMAGE=...`. The example requires immutable API and Qdrant image references, but no digest has been selected or verified.
- Docker engine access was denied in the current environment; Compose build, container health, volume initialization under UID 10001, Qdrant key compatibility, and deployment smoke tests are unverified.
- No current live Lite/Core/Research request on one identified image was performed. Historical diagnostic data used a small 18-point snapshot and cannot establish general quality or production behavior. Human faithfulness/citation review remains incomplete.
- The official Phase 8 600-unit baseline did not complete. Its interrupted prefix is not a baseline.
- Frontend dependency audit history reports advisories; current advisory status and Python CVE status remain unverified.
- Trusted proxy/TLS/authentication, Qdrant network policy, secret-manager integration, monitoring, backup/restore, recovery, and actual target resource fit remain deployment gates.

## Release decision

**NOT READY TO DEPLOY.** The deployment candidate and safer production defaults are prepared, but the image and runtime have not been built or exercised, and material quality, identity-proxy, dependency, recovery, and monitoring gates are still open. A successful local test suite or Compose render does not close those gates.

## 2026-11-07 deployment verification attempt

The requested deployment preflight was read-only. `docker info --format "CPUs={{.NCPU}} MemBytes={{.MemTotal}} DockerRoot={{.DockerRootDir}}"` failed with: `permission denied while trying to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`. The private `docker/env.production` file was absent; only the placeholder `docker/env.production.example` was present. No image digest or credential values were supplied for deployment verification. No image was built or pulled, no Compose service was started, and no API/Qdrant readiness, end-to-end, persistence, or backup/restore check was run. The existing Compose configuration and user data were left untouched.

Minimum next action: make the Docker Desktop Linux engine pipe accessible to this workspace and provide a protected `docker/env.production` containing reviewed immutable image references and valid deployment secrets. Then repeat the read-only configuration review before any service creation.

## 2026-11-07 follow-up verification

Docker was checked outside the sandbox after the operator reported engine recovery. `docker version --format "Client={{.Client.Version}} Server={{.Server.Version}}"` returned Client `29.5.3`, Server `29.5.3`. The required configuration check, `docker compose --env-file docker/env.production -p omnirag-production -f docker/compose.production.yml config --quiet`, stopped with `couldn't find env file: C:\Users\sreej\OneDrive\Documents\OmniRag\deploy\env.production`. The checked-in example still contains placeholders and is not deployment configuration. Image digests and credentials therefore could not be validated. No image was built or pulled; no container, network, volume, or collection was inspected or changed; no health, authentication, ACL, ingestion, generation, persistence, or backup/restore test was run.

Minimum next action: place the reviewed production env file at `docker/env.production` (or explicitly provide the approved env-file path, without sending its contents in chat), then rerun config validation. Do not substitute the development `.env` implicitly.
