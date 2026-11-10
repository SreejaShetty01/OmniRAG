# Ingestion and indexing

## Data flow

`source → loader → parser → normalized blocks → structure/reference extraction → structure-aware chunking → deterministic context metadata → dense embedding + sparse encoding → Qdrant`

`IngestionPipeline.ingest(source)` returns `indexed`, `unchanged`, or `failed` per document. Exceptions are represented by a typed error name and message in the result and logged without logging document contents. Use `delete_document(document_id)` for lifecycle removal.

## Loaders and parsers

- Local document files: PDF, DOCX, TXT, Markdown, and HTML. Extension, byte limit, emptiness, and UTF-8 text validity are checked before parsing.
- Web: only HTTP(S) HTML is accepted. Redirects are bounded and every target is checked against non-public IP resolution to reduce SSRF exposure.
- Code: Python, JavaScript, TypeScript, JSX, and TSX. Python uses AST/tokenization; JavaScript and TypeScript use Tree-sitter grammars. Other recognized source extensions use a paragraph/line-boundary fallback with no claimed symbol extraction. Source is never executed.
- Docling is an optional layout-aware conversion path for PDF/DOCX; native parsers remain the fallback. PDF fallback preserves page numbers but does not provide OCR.

## Chunking

`ChunkingConfig` is independent of profile configuration. Defaults are configurable starting values, not measured optima. Sections remain separate; adjacent blocks pack together up to an estimated token bound. Code/table/list content splits on line boundaries, prose on sentence boundaries, and a final word-boundary fallback handles oversized units. Overlap is configurable. Parent IDs point to the document or source structural block. Context prefixes are deterministic and derived only from known title and section metadata.

## Embeddings and sparse representation

`EmbeddingProvider` isolates provider implementations. The included HTTP adapter expects an OpenAI-compatible JSON request/response shape at the configured endpoint and uses the configured model ID and optional secret. Batches are validated for count and dimensions; duplicate inputs are cached within the service instance. This cache is process-local and capacity-bounded, so it is not a durable embedding cache.

Sparse vectors use stable 32-bit hashed term indices with sublinear term frequency. Qdrant's IDF modifier supplies corpus IDF during sparse search. Hash collisions are possible, and the representation does not implement BM25 field-length normalization.

`evaluation/ingestion_benchmark.py` runs explicit structure-aware and paragraph-baseline chunking cases and reports chunk count and estimated size distributions. It does not score retrieval or claim a best configuration; those measurements belong to later benchmark datasets.

## Qdrant lifecycle

One collection uses named dense and sparse vectors. Payloads include source, structure, hashes, versions, links, and ACL-ready values. Points for a new source hash are staged inactive, then activated after successful upsert; older generations are removed afterward. Failures before activation trigger cleanup of staged IDs. Qdrant does not offer a transaction over activation and retirement, so lifecycle cleanup failures are surfaced to the caller and require reconciliation. Unchanged source hashes bypass parsing and embedding after loading.
