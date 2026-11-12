from omnirag.api.main import app, health, profiles
from omnirag.api.dependencies import RequestIdentity
from omnirag.api.routes.preferences import get_preferences, update_preferences
from omnirag.api.schemas import PreferencesUpdate
from omnirag.infrastructure.product_store import ProductStore


def test_health_endpoint():
    assert health()["status"] == "ok"


def test_profiles_endpoint_lists_shared_profiles():
    assert profiles()["profiles"] == ["lite", "core", "research"]


def test_application_import_exposes_startup_routes():
    route_paths = set(app.openapi()["paths"])
    assert app.title == "OmniRAG"
    assert {"/health", "/profiles", "/api/v1/chat", "/api/v1/preferences",
            "/api/v1/documents/upload", "/api/v1/collections"}.issubset(route_paths)


def test_preferences_api_persists_preferred_name(tmp_path):
    store = ProductStore(tmp_path / "api.sqlite3")
    identity = RequestIdentity(user_id="user-test", tenant_id="tenant-test")
    update_preferences(PreferencesUpdate(preferred_name="Sreeja"), identity, store)
    assert get_preferences(identity, store)["preferred_name"] == "Sreeja"


def test_upload_filename_validation_rejects_invalid_extension():
    import pytest
    from fastapi import HTTPException
    from omnirag.api.routes.documents import _safe_filename

    with pytest.raises(HTTPException) as exc:
        _safe_filename("payload.exe")
    assert exc.value.status_code == 415


def test_upload_filename_validation_accepts_corpus_proto_and_rst_types():
    from omnirag.api.routes.documents import _doc_type, _safe_filename, _source_type

    assert _safe_filename("points.proto") == "points.proto"
    assert _source_type("points.proto") == "code"
    assert _doc_type("points.proto") == "code"
    assert _safe_filename("guide.rst") == "guide.rst"
    assert _source_type("guide.rst") == "document"
    assert _doc_type("guide.rst") == "text"


def test_upload_size_guard_rejects_chunk_that_crosses_limit():
    from omnirag.api.routes.documents import _upload_exceeds_limit
    from omnirag.config.settings import get_settings

    maximum = get_settings().max_upload_bytes
    assert _upload_exceeds_limit(maximum - 1, 2, maximum)
    assert not _upload_exceeds_limit(maximum - 1, 1, maximum)


def test_failed_ingestion_is_not_reported_as_successful_upload_or_reingestion():
    from omnirag.ingestion.pipeline import IngestionResult, IngestionStatus
    from omnirag.api.routes.documents import _ingestion_response_status

    failed = IngestionResult(
        status=IngestionStatus.FAILED,
        source="notes.md",
        error_type="EmbeddingError",
        error_message="embedding provider unavailable",
    )
    assert _ingestion_response_status(failed, already_exists=False) == 503
    assert _ingestion_response_status(failed, already_exists=True) == 503

    unsupported_source = IngestionResult(
        status=IngestionStatus.FAILED,
        source="scan.pdf",
        error_type="SourceLoadError",
        error_message="No extractable text was found",
    )
    assert _ingestion_response_status(unsupported_source, already_exists=False) == 422

    indexed = IngestionResult(status=IngestionStatus.INDEXED, source="notes.md")
    unchanged = IngestionResult(status=IngestionStatus.UNCHANGED, source="notes.md")
    assert _ingestion_response_status(indexed, already_exists=False) == 201
    assert _ingestion_response_status(unchanged, already_exists=True) == 200
