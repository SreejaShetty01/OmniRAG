# ADR 0001: Shared indexing infrastructure

- Status: Accepted
- Decision: All profiles share one ingestion/indexing infrastructure and one Qdrant knowledge store.
- Rationale: Profiles express execution policy, not isolated data systems. Shared indexing reduces duplicated behavior and keeps data access consistent.
- Consequences: Profile-specific storage is out of scope; future isolation requirements must be expressed with access controls or explicit tenancy design.
