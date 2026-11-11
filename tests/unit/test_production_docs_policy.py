from omnirag.api.main import _interactive_docs_enabled


def test_interactive_openapi_docs_are_disabled_only_in_production():
    assert not _interactive_docs_enabled("production")
    assert not _interactive_docs_enabled("Production")
    assert _interactive_docs_enabled("development")
    assert _interactive_docs_enabled("test")
