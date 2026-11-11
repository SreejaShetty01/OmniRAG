"""Loader contracts and shared input validation."""

from pathlib import Path
from typing import Protocol

from omnirag.ingestion.structure.schema import LoadedSource


class SourceLoader(Protocol):
    def load(self, source: str) -> LoadedSource: ...


class SourceLoadError(ValueError):
    """Raised when a source cannot be read or is unsafe/unsupported."""


def validate_file(path: Path, *, max_bytes: int, allowed_extensions: set[str]) -> bytes:
    if not path.exists() or not path.is_file():
        raise SourceLoadError(f"Source file does not exist or is not a file: {path}")
    if path.suffix.lower() not in allowed_extensions:
        raise SourceLoadError(f"Unsupported file type: {path.suffix or '(no extension)'}")
    size = path.stat().st_size
    if size == 0:
        raise SourceLoadError("Source file is empty")
    if size > max_bytes:
        raise SourceLoadError(f"Source exceeds maximum size of {max_bytes} bytes")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceLoadError(f"Unable to read source file: {exc}") from exc
