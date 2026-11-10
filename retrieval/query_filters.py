"""Qdrant filter construction with mandatory active-generation and ACL clauses."""

from qdrant_client import models

from retrieval.contracts import RetrievalFilters, RetrievalScope


class FilterBuilder:
    def build(self, filters: RetrievalFilters, scope: RetrievalScope) -> models.Filter:
        if not scope.principal_ids:
            raise ValueError("Retrieval scope is required")
        if not scope.tenant_id:
            raise ValueError("Tenant scope is required; unattributed points are not retrievable")
        must: list = [
            models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        ]
        must_not: list = [
            models.FieldCondition(key="acl_denied_principals",
                                  match=models.MatchAny(any=list(scope.principal_ids))),
        ]
        allowed = models.Filter(
            should=[
                models.IsEmptyCondition(is_empty=models.PayloadField(key="acl_allowed_principals")),
                models.FieldCondition(key="acl_allowed_principals",
                                      match=models.MatchAny(any=list(scope.principal_ids))),
            ],
        )
        must.append(allowed)
        must.append(models.FieldCondition(key="metadata.tenant_id",
                                          match=models.MatchValue(value=scope.tenant_id)))
        if scope.workspace_id is not None:
            must.append(models.FieldCondition(key="metadata.workspace_id",
                                              match=models.MatchValue(value=scope.workspace_id)))

        self._match_any(must, "document_id", filters.document_ids)
        self._match_any(must, "source", filters.sources)
        self._match_any(must, "source_type", filters.source_types)
        self._match_any(must, "document_type", filters.document_types)
        self._match_any(must, "content_type", filters.content_types)
        self._match_any(must, "language", filters.languages)
        self._match_any(must, "version", filters.versions)
        self._match_any(must, "metadata.collection_id", filters.collection_ids)
        for path in filters.section_paths:
            for part in path:
                must.append(models.FieldCondition(key="section_path", match=models.MatchValue(value=part)))
        return models.Filter(must=must, must_not=must_not)

    @staticmethod
    def _match_any(conditions: list, key: str, values: list[str]) -> None:
        if values:
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
