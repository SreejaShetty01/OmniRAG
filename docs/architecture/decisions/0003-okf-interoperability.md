# ADR 0003: OKF is an interoperability boundary

- Status: Accepted
- Decision: OKF (Open Knowledge Format) is treated as an optional interoperability representation at ingestion/export boundaries, not as a duplicate canonical model layer.
- Rationale: Core document and chunk contracts remain provider- and storage-neutral while external structured knowledge can be mapped at system edges.
- Consequences: The OKF mapping document describes the planned relationship. No converter or OKF-specific pipeline is implemented in Phase 1.
