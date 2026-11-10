# Changelog

## Unreleased — M1-P5 preparation (2026-11-05)

- Added configurable per-principal process-local API rate limiting with HTTP 429 and `Retry-After`.
- Added authenticated aggregate runtime metrics and optional Ollama readiness probing.
- Redacted configured credentials from structured application logs and Phase 7b JSON artifacts.
- Declared static CPU/memory limits for isolated evaluation services and prepared production proxy, backup/restore, demo, release, and limitations documentation.
- No live service operations, backup/restore, full regression, inference, benchmark, or release-candidate comparison was performed.
