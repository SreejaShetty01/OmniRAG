# Limitations

- The Phase 8 600-unit baseline did not complete. The partial B7 run is not a frozen baseline; B5 values are small historical pilot samples only.
- No validated performance winner, local CPU capacity ceiling, or resource peak is established by M1-P5.
- Research previously hit a 120-second finalization failure; the underlying provider/node cause remains unresolved in historical evidence.
- MiniCheck's optional model has not been downloaded or executed; verifier-catch behavior is unproven.
- Rate limiting and metrics are process-local. Multi-worker/shared quotas require a trusted proxy or a future distributed design. Queue depth is not available.
- Ollama readiness is checked only when `OMNIRAG_OLLAMA_HEALTH_URL` is configured; no live health check was performed in M1-P5.
- Evaluation container resource limits are declared but not applied to running services or measured. Ollama is external and its effective keep-alive, queue and concurrency settings are not established here.
- TLS, production reverse-proxy signing, direct API network restriction, Qdrant authentication, non-root volume ownership, backup/restore, alerting, log retention and disaster recovery need target-environment verification.
- The final release candidate now includes a production Compose example with Qdrant API-key configuration, non-empty signed-identity allowlist and strong-secret validation, and an API image configured for a non-root UID. These controls have not been validated in a built container or live target deployment.
- Qdrant image references remain mutable in current Compose files and must be pinned to a reviewed immutable digest for a release deployment.
- `npm audit` (2026-11-05) reports six vulnerabilities in the frontend development/test dependency tree, including two critical and one high. The available remediation requires a major Vitest upgrade and was not applied pending isolated compatibility validation. `pip-audit` was unavailable, so Python dependency advisory status is unknown.
- Secret redaction is tested for configured credentials in application logging and Phase 7b JSON artifact persistence. Other custom output channels, shell history, crash dumps, and third-party logging are outside that test scope.
- Small-team deployment is not ready. This application is described as **production-oriented, with measured limits** only where the limit has actually been measured.
