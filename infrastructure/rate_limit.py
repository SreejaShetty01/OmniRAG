"""Single-process, per-principal fixed-window request limiting."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi.responses import JSONResponse


class PrincipalRateLimiter:
    """Fixed windows are intentionally simple; use one API worker per instance."""

    def __init__(self, requests_per_minute: int, *, clock=time.monotonic):
        self.limit = requests_per_minute
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[tuple[str | tuple[str, str], int], int] = defaultdict(int)

    def check(self, principal: str | tuple[str, str]) -> tuple[bool, int]:
        now = self._clock()
        window = int(now // 60)
        key = (principal, window)
        with self._lock:
            # Bound memory by dropping old windows on each request.
            for old_key in [key for key in self._windows if key[1] < window]:
                del self._windows[old_key]
            self._windows[key] += 1
            allowed = self._windows[key] <= self.limit
        retry_after = max(1, int(60 - (now % 60)))
        return allowed, retry_after


def too_many_requests_response(request_id: str, retry_after: int) -> JSONResponse:
    return JSONResponse(status_code=429, content={"error": {
        "code": "rate_limit_exceeded", "message": "Request rate limit exceeded.",
        "request_id": request_id,
    }}, headers={"Retry-After": str(retry_after)})
