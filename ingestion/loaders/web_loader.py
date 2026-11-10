"""HTTP(S) HTML loader with response size and timeout limits."""

from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import ipaddress
import socket

from config.settings import get_settings
from ingestion.loaders.base import SourceLoadError
from ingestion.structure.schema import LoadedSource


class WebLoader:
    def __init__(self, max_bytes: int | None = None, timeout_seconds: float = 15.0):
        self.max_bytes = max_bytes if max_bytes is not None else get_settings().max_upload_bytes
        self.timeout_seconds = timeout_seconds

    def load(self, source: str) -> LoadedSource:
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        current = source
        opener = build_opener(NoRedirect)
        for _ in range(6):
            self._validate_public_url(current)
            request = Request(current, headers={"User-Agent": "OmniRAG-ingestion/1.0"})
            try:
                response = opener.open(request, timeout=self.timeout_seconds)
                break
            except HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                    current = urljoin(current, exc.headers["Location"])
                    continue
                raise SourceLoadError(f"Web source returned HTTP {exc.code}") from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise SourceLoadError(f"Unable to fetch web source: {exc}") from exc
        else:
            raise SourceLoadError("Web source exceeded redirect limit")
        try:
            with response:
                media_type = response.headers.get_content_type()
                if media_type not in {"text/html", "application/xhtml+xml"}:
                    raise SourceLoadError(f"Unsupported web content type: {media_type}")
                data = response.read(self.max_bytes + 1)
        except OSError as exc:
            raise SourceLoadError(f"Unable to read web response: {exc}") from exc
        if len(data) > self.max_bytes:
            raise SourceLoadError(f"Web source exceeds maximum size of {self.max_bytes} bytes")
        if not data:
            raise SourceLoadError("Web source is empty")
        parsed = urlparse(current)
        name = parsed.path.rsplit("/", 1)[-1] or parsed.hostname or "web-page"
        return LoadedSource(current, name, media_type, data, "web")

    @staticmethod
    def _validate_public_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise SourceLoadError("Only absolute HTTP(S) URLs without credentials are supported")
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))}
        except OSError as exc:
            raise SourceLoadError(f"Unable to resolve web host: {parsed.hostname}") from exc
        if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
            raise SourceLoadError("Web sources may not resolve to private or non-public network addresses")
