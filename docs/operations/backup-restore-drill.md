# Isolated backup/restore drill — PREPARED, NOT RUN

**Approval required before execution.** The procedure must use a separate, disposable restore project, separate volumes, a separately named Qdrant collection, and a separate API data directory. Never point restore commands at the development or source evaluation volume. Do not use Compose `down -v`, prune, volume deletion, or existing production collection names.

The plan-only PowerShell generator is [`scripts/ops/prepare_backup_restore_plan.ps1`](../../scripts/ops/prepare_backup_restore_plan.ps1). It checks that the supplied restore identities differ and prints a plan; it performs no Docker, SQLite, Qdrant, filesystem, or network operation. Its output is not a substitute for runtime identity verification or approval.

## Preconditions

- Obtain explicit approval for one drill window and identify the isolated source evaluation API, Qdrant service, named volumes, collection, data directory, and five approved non-sensitive reference queries.
- Confirm that the target restore project and all target volume names are new and absent. Abort if any target already exists.
- Record source container IDs, collection name, collection schema/count, SQLite database path, SQLite integrity result, current dataset/snapshot identifiers, and source volume identities.
- Ensure sufficient separate disk capacity and an encrypted, access-controlled output directory. Record operator, time, host, code revision, and command logs with secrets redacted.

## Capture (approved execution only)

1. Use Qdrant's snapshot API for the evaluation collection. Save the snapshot in a new run-specific directory and record its SHA-256. Do not use a snapshot from or write to another collection.
2. Create a transaction-consistent SQLite backup using SQLite's online backup API from the evaluation API container (or a read-only copied database plus WAL after a cleanly coordinated checkpoint). Include associated upload/source files and record file hashes. Never copy only the main SQLite file while WAL writes are active.
3. Record start/end times and bytes for each item. RPO is the age of the newest included committed database and Qdrant state at capture time; if cross-store writes were not coordinated, report RPO consistency as unknown rather than infer atomicity.

## Restore into separate isolated stack (approved execution only)

1. Render and inspect a separate Compose project with a unique project name, API port, Qdrant port, API-data volume, and Qdrant volume. Do not attach development/evaluation volumes.
2. Restore Qdrant snapshot to a new collection name in the new Qdrant volume, preserving vector schema and payload indexes. Restore SQLite through a new data volume and restore associated source files.
3. Verify SQLite `PRAGMA integrity_check`, schema/version, document counts, and source-file hashes; verify Qdrant collection config, point count, required payload indexes, and snapshot checksum.
4. Run the same five approved reference queries against source and restored stacks with identical identity signatures, profile/config, query text, and cache policy. Compare ordered top-K point IDs and evidence metadata. Confirm allow and deny cases, including that denied evidence is absent from citations and metadata. No generation is required unless a separately approved query explicitly needs it.
5. Measure RTO from restore start to all readiness/integrity/reference checks passing. Report measured RPO from captured timestamps, measured RTO, snapshot hash, DB integrity, per-query top-K equivalence, and ACL/tenant results.
6. Stop only the temporary restore stack after collecting evidence. Preserve the drill artifacts. Do not remove its volumes without a separate explicit approval.

## Current status

No snapshot, SQLite backup, restore, reference query, or service operation has been executed for this drill. RPO, RTO, snapshot integrity, restored database consistency, top-K equivalence, and ACL behavior are **NOT RUN / UNKNOWN**. Exact live command lines must be generated from the approved runtime labels and paths; this repository does not guess project/container/volume names.
