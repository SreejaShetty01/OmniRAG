# Release and rollback procedure — prepared

No release candidate or tag has been created by M1-P5. Obtain explicit approval before the full regression and candidate/control comparison.

## Candidate gates

1. Review the complete source diff and lock/dependency changes; do not include local `.env*`, data, snapshots, or historical experiment records.
2. Pin API and Qdrant images by immutable digest; review a production-specific Compose file and secret-manager injection.
3. Approve and execute the isolated backup/restore drill. Require SQLite integrity, Qdrant schema/count/index validation, five reference queries, exact top-K ID comparison, and zero ACL/tenant violations.
4. Approve the full backend/frontend suites and same-dataset/offline-scorer regression check. The comparison is not a frozen Phase 8 baseline comparison.
5. Review measured limitations, dependency audit, TLS/authenticated proxy, rate limits, resource controls, logs/metrics/alerts, and recovery runbook.
6. Record the exact Git commit, dirty diff status/hash before packaging, dataset/snapshot checksums, test outputs, deployment configuration fingerprint, and rollback image/digest.

## Version and tag

Select the version only after the gates pass. Use annotated tags with a release note and exact commit; do not force-move or reuse tags. Example after approval:

```powershell
git status --short
git diff --check
git tag -a vX.Y.Z -m "OmniRAG X.Y.Z"
git show --no-patch --decorate vX.Y.Z
```

Tag creation/push is not authorized by this preparation task. Never commit `.env`, `.env.phase8`, database files, volumes, corpus content with restricted terms, or historical run artifacts.

## Deployment and rollback

Deploy a separately reviewed pinned image/configuration. Keep the previous immutable image and compatible configuration available. Verify readiness, signed identity, tenant isolation, a small approved smoke flow, and logs/metrics after deployment. Roll back by redeploying the prior immutable image/configuration; do not reset volumes or collections as a rollback step. Validate schema compatibility before rolling back across migrations. No deployment or rollback was performed here.
