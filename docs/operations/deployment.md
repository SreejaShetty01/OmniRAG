# Deployment and runtime validation

## Local development

1. Copy `.env.example` to `.env` and configure an embedding provider/model before ingestion.
2. Start Qdrant with `docker compose up -d qdrant` (the Compose service is internal-only; use a local Qdrant instance or temporary port mapping for development tooling).
3. In `frontend/`, run `npm ci` and `npm run dev`.
4. From the repository root, run `uvicorn omnirag.api.main:app --reload`.

The Vite proxy sends `/api` and `/health` to the API. `VITE_API_BASE_URL` can point
the typed client at another backend; avoid embedding secrets in frontend variables.

## Production container

`docker compose up --build -d` builds the React assets into the FastAPI image, starts
the API and shared Qdrant, and persists SQLite/uploads and Qdrant data in named volumes.
The API is bound to host loopback on port 8000 by default; place a TLS reverse proxy
in front. The base Compose file publishes Qdrant REST/gRPC on host loopback ports
6333/6334, so host-local processes can reach them; these ports are not exposed on
external interfaces by this mapping. Remove the host port mappings in a production-
specific deployment if host-local access is not required. Set provider credentials,
`OMNIRAG_AUTH_SHARED_SECRET`, and `OMNIRAG_CORS_ORIGINS` in a secret manager or deployment
environment. Do not commit `.env`.

The reverse proxy must authenticate users, remove incoming `X-OmniRAG-*` headers, then
sign `user_id + newline + tenant_id + newline + unix_timestamp` using HMAC-SHA256 with
the same shared secret. It sends those values in `X-OmniRAG-User-ID`,
`X-OmniRAG-Tenant-ID`, `X-OmniRAG-Auth-Timestamp`, and
`X-OmniRAG-Auth-Signature` (lowercase hex). Signatures expire after 60 seconds and are
compared in constant time. Without a valid identity, protected endpoints fail closed.
Use HTTPS between proxy and API when they are on separate hosts, rotate the secret,
and apply request rate limits at the proxy. The fixed development identity is enabled
only when `OMNIRAG_ENVIRONMENT=development`.

`/health` is liveness; `/health/ready` checks Qdrant. SQLite WAL data and uploads must
be backed up together with the Qdrant volume for a consistent restore. Scale the API
only with shared persistent volumes or move product metadata into a managed database;
SQLite is intended for a single-node deployment. No streaming endpoint is implemented.

### M1-P5 operational controls

`OMNIRAG_RATE_LIMIT_REQUESTS_PER_MINUTE` configures the in-process per-principal fixed-window request limit (default `120`). The Phase 8 isolated evaluation Compose overlays set `1200` for their allowlisted workloads. The limiter is per worker process; keep one worker per instance or enforce an additional quota at the trusted edge before scaling. `/metrics` returns aggregate, authenticated process-local request/error counts, latency buckets, provider calls observed through run-budget ledgers, and latest budget usage; it is not a distributed metrics backend and has no queue-depth measurement. Set `OMNIRAG_OLLAMA_HEALTH_URL` to an operator-controlled Ollama health URL (for example `/api/tags`) only when Ollama is a required readiness dependency; Qdrant remains checked unconditionally. None of these deployment controls has been validated against a live production target by M1-P5.

## Linux async/provider validation

The Windows environment can run the current unit suite, but it does not have a running
Docker Desktop Linux daemon here, so a Linux image or end-to-end run could not be made.
On Linux, use:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
cd frontend && npm ci && npm run build && npm test
docker compose config
docker compose up --build --wait
```

Standard tests use fakes and require no provider credentials. No live embedding,
generation, verification, reranker, or hosted Qdrant integration was configured or
claimed in this workspace. Benchmark data is empty and no persisted evaluation runner
exists, so the evaluation page reports unavailable and does not display sample scores.
