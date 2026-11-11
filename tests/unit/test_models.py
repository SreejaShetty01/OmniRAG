from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from omnirag.models import Chunk, Document, DocumentType, Query
from omnirag.models.response import OmniRAGResponse, VerificationStatus


def test_document_defaults_timestamp_and_metadata():
    document = Document(
        document_id="doc-1", title="Guide", source="guide.md",
        document_type=DocumentType.MARKDOWN, content_hash="abc",
    )
    assert document.created_at.tzinfo is not None
    assert document.metadata == {}


def test_chunk_rejects_empty_content():
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="chunk-1", document_id="doc-1", content="",
            document_type="text", content_hash="abc", source="guide.txt",
        )


def test_query_uses_core_profile_and_validates_text():
    query = Query(query_text="What is OmniRAG?")
    assert query.selected_profile.value == "core"
    with pytest.raises(ValidationError):
        Query(query_text=" ")


def test_response_contains_typed_verification_state():
    response = OmniRAGResponse(answer="Answer", profile_used="lite")
    assert response.verification_status is VerificationStatus.NOT_RUN
    assert response.abstained is False
