"""Safe source-code file loader. Code is read as data and is never executed."""

from pathlib import Path

from config.settings import get_settings
from ingestion.loaders.base import SourceLoadError, validate_file
from ingestion.structure.schema import LoadedSource

CODE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
}
GENERIC_CODE_EXTENSIONS = {".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".rb", ".php", ".cs", ".sh", ".sql", ".proto"}


class CodeLoader:
    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = max_bytes if max_bytes is not None else get_settings().max_upload_bytes

    def load(self, source: str | Path) -> LoadedSource:
        path = Path(source).expanduser().resolve()
        supported_extensions = set(CODE_LANGUAGES) | GENERIC_CODE_EXTENSIONS
        if path.suffix.lower() not in supported_extensions:
            raise SourceLoadError(f"Unsupported code language: {path.suffix or '(no extension)'}")
        data = validate_file(path, max_bytes=self.max_bytes, allowed_extensions=supported_extensions)
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise SourceLoadError("Source code is not valid UTF-8") from exc
        return LoadedSource(str(path), path.name, "text/plain", data, "code")
