"""Shared pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def clear_settings_cache():
    from config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
