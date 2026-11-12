"""Safe local document loading for supported office and text formats."""

import mimetypes
from pathlib import Path

from omnirag.config.settings import get_settings
from omnirag.ingestion.loaders.base import SourceLoadError, validate_file
from omnirag.ingestion.structure.schema import LoadedSource

DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".rst"}


class DocumentLoader:
    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes if max_bytes is not None else get_settings().max_upload_bytes

    def load(self, source: str | Path) -> LoadedSource:
        path = Path(source).expanduser().resolve()
        data = validate_file(path, max_bytes=self.max_bytes, allowed_extensions=DOCUMENT_EXTENSIONS)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown", ".html", ".htm", ".rst"}:
            try:
                data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise SourceLoadError("Text document is not valid UTF-8") from exc
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return LoadedSource(str(path), path.name, media_type, data, "document")
