# OmniRAG API

Run the API with `uvicorn omnirag.api.main:app --reload`. OpenAPI is available at `/docs`.
Product routes use `/api/v1`. Development uses a documented fixed identity. Production
requires `OMNIRAG_AUTH_SHARED_SECRET` and a trusted reverse proxy that strips inbound
`X-OmniRAG-*` identity headers and signs the user ID, tenant ID and Unix timestamp
(HMAC-SHA256, newline separated, 60 second freshness window). Unsigned or stale identity
headers are ignored; arbitrary identity headers are never trusted. Responses do not
expose provider secrets or local paths.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness |
| GET | `/health/ready` | Qdrant readiness |
| GET | `/profiles` | Compatibility profile list |
| GET | `/api/v1/profiles` | Profile policies and configured model availability |
| GET/PUT | `/api/v1/preferences` | Durable per-identity preferences |
| POST | `/api/v1/chat` | Execute Lite, Core or Research through the shared execution service |
| GET | `/api/v1/conversations` | List owned conversations |
| GET/DELETE | `/api/v1/conversations/{id}` | Read or delete an owned conversation |
| GET | `/api/v1/documents` | List owned document metadata |
| PUT | `/api/v1/documents/upload?filename=...` | Upload and ingest (raw request body; configured byte cap) |
| GET/DELETE | `/api/v1/documents/{id}` | Read or delete an owned document; `POST /{id}/reingest` retries ingestion |
| GET/POST | `/api/v1/collections` | List or create logical collections |
| PUT | `/api/v1/collections/documents/{id}` | Assign a document to a collection |
| DELETE | `/api/v1/collections/{id}` | Delete a collection and unassign its documents |
| GET | `/api/v1/settings` | Safe model readiness and supported limits |
| GET/POST | `/api/v1/evaluations/status`, `/runs` | Evaluation availability; run only when runner and data exist |

Upload is a raw byte stream with the original filename in the query string, avoiding
an extra multipart dependency. CORS origins are allowlisted by `OMNIRAG_CORS_ORIGINS`.
The API emits request IDs and normalized validation/internal error envelopes.
