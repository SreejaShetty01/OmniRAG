"""Thread-safe TTL cache; caller supplies a fully scoped corpus/config cache key."""

import threading
import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class RetrievalCache(Generic[T]):
    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("Cache capacity must be positive")
        self.capacity = capacity
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, ttl_seconds: int) -> T | None:
        if ttl_seconds <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created, value = item
            if now - created > ttl_seconds:
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
