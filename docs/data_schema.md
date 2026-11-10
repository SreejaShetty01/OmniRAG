# Data contracts

The contracts are Pydantic v2 models in `models/`. They validate transport/domain data without embedding provider or database behavior.

- **Document**: required `document_id`, `title`, `source`, `document_type`, and `content_hash`; timestamps default to UTC and metadata defaults to an empty mapping. Extra fields are allowed for source-specific metadata.
- **Chunk**: required identifiers, non-empty content, type, hash, and source; optional parent/version; section path and links; access-control metadata with allow/deny principals and optional sensitivity. Extra fields are allowed.
- **Query**: non-empty text, optional ID, selected profile (Core by default), filters, conversation context, and escalation preference.
- **OmniRAGResponse**: answer, citations, evidence references, profile used, verification state, abstention flag, and execution metadata.

Document type, profile, escalation, and verification state use enums. Database-specific payload formats and model-provider structures are intentionally excluded.

## Ingestion metadata

`ingestion/structure/schema.py` defines parser-neutral `NormalizedDocument` and `ContentBlock`. Blocks retain parent IDs, section path, heading, page number, content kind, language, source lines, links, and parser metadata. Indexed chunk payloads carry document and chunk IDs, parent, source/source type, title, section/heading/page, content type, language, source and chunk hashes, version, creation time, links, arbitrary metadata, and explicit ACL principal fields.

Chunk `content` is source-derived text. Deterministic contextual prefixes are stored under `metadata.context_prefix` and are only added to embedding input; they do not replace or mutate the source text. Token size estimates use whitespace/punctuation counting until a tokenizer is selected through benchmarking.
