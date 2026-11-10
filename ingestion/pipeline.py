"""Single ingestion orchestration path from source bytes to published index points."""

import hashlib
import inspect
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse

from config.settings import get_settings
from indexing.embeddings import EmbeddingError, EmbeddingService, EmbeddingProvider, provider_from_config
from indexing.provenance import new_point_provenance
from indexing.sparse_encoder import SparseEncoder
from infrastructure.provider_budget import BudgetExceeded
from indexing.qdrant_indexer import QdrantIndexer
from ingestion.chunking.contextualizer import DeterministicContextualizer
from ingestion.chunking.structure_chunker import ChunkingConfig, StructureChunker
from ingestion.loaders.base import SourceLoadError
from ingestion.loaders.code_loader import CODE_LANGUAGES, GENERIC_CODE_EXTENSIONS, CodeLoader
from ingestion.loaders.document_loader import DocumentLoader
from ingestion.loaders.web_loader import WebLoader
from ingestion.parsers.code_parser import CodeParser
from ingestion.parsers.document_parser import DocumentParser
from ingestion.structure.schema import LoadedSource
from ingestion.structure.structure_extractor import StructureExtractor
from ingestion.structure.normalize import stable_id
from models.chunk import AccessControlMetadata

logger = logging.getLogger(__name__)


class IngestionStatus(StrEnum):
    INDEXED = "indexed"
    UNCHANGED = "unchanged"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    status: IngestionStatus
    source: str
    document_id: str | None = None
    source_hash: str | None = None
    chunk_count: int = 0
    error_type: str | None = None
    error_message: str | None = None


class IngestionPipeline:
    def __init__(self, *, document_loader=None, code_loader=None, web_loader=None,
                 document_parser=None, code_parser=None, structure_extractor=None,
                 chunker=None, contextualizer=None, embedder: EmbeddingService | None = None,
                 embedding_provider: EmbeddingProvider | None = None, indexer: QdrantIndexer | None = None,
                 contextualization_enabled: bool | None = None):
        settings = get_settings()
        self.document_loader = document_loader or DocumentLoader()
        self.code_loader = code_loader or CodeLoader()
        self.web_loader = web_loader or WebLoader()
        self.document_parser = document_parser or DocumentParser()
        self.code_parser = code_parser or CodeParser()
        self.structure_extractor = structure_extractor or StructureExtractor()
        self.chunker = chunker or StructureChunker(ChunkingConfig(settings.chunk_max_tokens,
                                                                   settings.chunk_overlap_tokens))
        self.contextualizer = contextualizer or DeterministicContextualizer()
        if embedder is not None:
            self.embedder: EmbeddingService | None = embedder
        elif embedding_provider is not None:
            self.embedder = EmbeddingService(embedding_provider)
        else:
            self.embedder = None
        self.indexer = indexer or QdrantIndexer()
        self.contextualization_enabled = (settings.chunk_contextualization_enabled
                                          if contextualization_enabled is None
                                          else contextualization_enabled)

    def ingest(self, source: str | Path, *, version: str | None = None,
               access_control: AccessControlMetadata | None = None,
               tenant_id: str | None = None, collection_id: str | None = None) -> IngestionResult:
        source_value = str(source)
        try:
            loaded = self._load(source_value)
            source_hash = hashlib.sha256(loaded.data).hexdigest()
            document_id = stable_id("doc", loaded.source)
            if self.indexer.is_unchanged(document_id, source_hash):
                logger.info("Ingestion unchanged", extra={"document_id": document_id})
                return IngestionResult(IngestionStatus.UNCHANGED, loaded.source, document_id,
                                       source_hash, 0)
            parser = self.code_parser if loaded.source_type == "code" else self.document_parser
            normalized = parser.parse(loaded)
            normalized.document.version = version
            normalized = self.structure_extractor.extract(normalized)
            chunks = self.chunker.chunk(normalized)
            for chunk in chunks:
                if tenant_id:
                    chunk.metadata["tenant_id"] = tenant_id
                if collection_id:
                    chunk.metadata["collection_id"] = collection_id
            if access_control is not None:
                for chunk in chunks:
                    chunk.access_control = access_control.model_copy(deep=True)
            if self.contextualization_enabled:
                chunks = self.contextualizer.enrich(chunks)
            if not chunks:
                raise SourceLoadError("Document produced no indexable chunks")
            inputs = []
            for chunk in chunks:
                prefix = chunk.metadata.get("context_prefix", "")
                inputs.append(f"{prefix}\n\n{chunk.content}" if prefix else chunk.content)
            if self.embedder is None:
                self.embedder = EmbeddingService(provider_from_config())
            cache_scope = None
            if tenant_id:
                cache_scope = hashlib.sha256(
                    f"{tenant_id}\0{self.indexer.collection}\0{document_id}\0{version or ''}\0{source_hash}".encode()
                ).hexdigest()
            vectors = self.embedder.embed(inputs, cache_scope=cache_scope)
            settings = get_settings()
            provider = getattr(self.embedder.provider, "provider_name", None) or settings.embedding_provider or "custom"
            model = getattr(self.embedder.provider, "model_id", None) or settings.embedding_model_id or "unknown"
            sparse_encoder = getattr(self.indexer, "sparse_encoder", None) or SparseEncoder()
            provenance = new_point_provenance(
                provider=provider, model=model, dimension=len(vectors[0]),
                sparse_method=sparse_encoder.method,
                sparse_tokenizer_version=sparse_encoder.tokenizer_version,
                chunker_version=settings.chunker_version,
                enrichment_mode="deterministic" if self.contextualization_enabled else "none",
                ingestion_schema_version=settings.ingestion_schema_version,
            )
            index_parameters = inspect.signature(self.indexer.index_document).parameters
            if "provenance" in index_parameters:
                self.indexer.index_document(chunks, vectors, source_hash, provenance=provenance)
            else:
                self.indexer.index_document(chunks, vectors, source_hash)
            logger.info("Ingestion indexed", extra={"document_id": document_id,
                                                     "chunk_count": len(chunks)})
            return IngestionResult(IngestionStatus.INDEXED, loaded.source, document_id,
                                   source_hash, len(chunks))
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.error("Ingestion failed", extra={"error_type": type(exc).__name__})
            safe_message = (str(exc) if isinstance(exc, (SourceLoadError, ValueError, EmbeddingError))
                            else "An external service operation failed")
            return IngestionResult(IngestionStatus.FAILED, source_value,
                                   error_type=type(exc).__name__, error_message=safe_message)

    def delete_document(self, document_id: str) -> None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        self.indexer.delete_document(document_id)

    def _load(self, source: str) -> LoadedSource:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            return self.web_loader.load(source)
        suffix = Path(source).suffix.lower()
        if suffix in CODE_LANGUAGES or suffix in GENERIC_CODE_EXTENSIONS:
            return self.code_loader.load(source)
        return self.document_loader.load(source)
