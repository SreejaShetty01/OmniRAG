"""Deterministic query characterization and explicit metadata extraction."""

import re

from models.query import Query
from retrieval.contracts import QueryAnalysis, QueryCharacteristics, RetrievalFilters

_DIRECTIVE = re.compile(
    r"\b(document_id|doc_id|source|source_type|document_type|content_type|language|version|section|path|collection_id)\s*:\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"(?<!\w)(?:[A-Za-z_$][\w$]*(?:(?:::|[./#])|[_-])[\w$./#:-]*|[A-Z][A-Z0-9_-]{2,})(?!\w)")
_TOKEN = re.compile(r"[\w$]+", re.UNICODE)


class QueryAnalyzer:
    def analyze(self, query: Query) -> QueryAnalysis:
        text = query.query_text.strip()
        extracted: dict[str, list[str]] = {}
        warnings: list[str] = []
        for match in _DIRECTIVE.finditer(text):
            key = match.group(1).lower()
            value = next(part for part in match.groups()[1:] if part is not None).strip().strip(",;")
            mapped = {
                "document_id": "document_ids", "doc_id": "document_ids", "source": "sources",
                "source_type": "source_types", "document_type": "document_types",
                "content_type": "content_types", "language": "languages", "version": "versions",
                "section": "section_paths", "path": "sources", "collection_id": "collection_ids",
            }[key]
            extracted.setdefault(mapped, []).append(value)
        explicit = {key: value for key, value in query.filters.items()}
        mapped_filters = {
            "document_id": "document_ids", "document_ids": "document_ids", "source": "sources",
            "sources": "sources", "source_type": "source_types", "source_types": "source_types",
            "document_type": "document_types", "document_types": "document_types",
            "content_type": "content_types", "content_types": "content_types",
            "language": "languages", "languages": "languages", "version": "versions",
            "versions": "versions", "section_path": "section_paths", "section_paths": "section_paths",
            "collection_id": "collection_ids", "collection_ids": "collection_ids",
        }
        accepted: dict[str, list] = {key: list(value) for key, value in extracted.items()}
        for key, value in explicit.items():
            target = mapped_filters.get(key)
            if target is None:
                warnings.append(f"Unsupported metadata filter ignored: {key}")
                continue
            values = value if isinstance(value, list) else [value]
            if target == "section_paths":
                accepted.setdefault(target, []).extend(
                    [item] if isinstance(item, list) else [part.strip() for part in str(item).split(">") if part.strip()]
                    for item in values
                )
            else:
                accepted.setdefault(target, []).extend(str(item) for item in values)
        if accepted.get("section_paths"):
            accepted["section_paths"] = [
                [part.strip() for part in path.split(">") if part.strip()] if isinstance(path, str) else path
                for path in accepted["section_paths"]
            ]
        try:
            filters = RetrievalFilters.model_validate(accepted)
        except Exception as exc:
            raise ValueError(f"Invalid metadata filter: {exc}") from exc

        normalized = _DIRECTIVE.sub(" ", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        identifiers = list(dict.fromkeys(_IDENTIFIER.findall(text)))
        tokens = _TOKEN.findall(normalized)
        lower = normalized.casefold()
        code_related = bool(re.search(r"\b(function|class|method|symbol|import|code|file|line|module|api)\b", lower))
        section_specific = bool(filters.section_paths or re.search(r"\b(section|chapter|heading)\s+", lower))
        version_sensitive = bool(filters.versions or re.search(r"\b(version|release|revision|v\d+(?:\.\d+)*)\b", lower))
        characteristics = QueryCharacteristics(
            exact_identifier=bool(identifiers), natural_language=bool(re.search(r"\b(what|why|how|when|where|explain|describe)\b", lower)),
            technical=code_related or bool(re.search(r"[.:/#_$]", normalized)),
            section_specific=section_specific, multi_concept=bool(re.search(r"\b(and|or|also|plus)\b", lower)),
            document_specific=bool(filters.document_ids or filters.sources or filters.collection_ids),
            version_sensitive=version_sensitive, code_related=code_related or bool(filters.languages),
        )
        return QueryAnalysis(
            original_query=query.query_text, normalized_query=normalized or text,
            characteristics=characteristics, exact_terms=identifiers,
            extracted_filters=filters, warnings=warnings,
        )
