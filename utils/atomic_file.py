"""Small bounded retry for Windows antivirus/sync locks during atomic replace."""

import os
import time
from pathlib import Path

IS_WINDOWS = os.name == "nt"
MAX_REPLACE_ATTEMPTS = 4


def atomic_replace(source: str | Path, destination: str | Path) -> None:
    """Atomically replace a file, retrying only transient Windows permission locks."""
    for attempt in range(MAX_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if not IS_WINDOWS or attempt == MAX_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.025 * (2 ** attempt))
