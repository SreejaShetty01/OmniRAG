"""Minimal in-process API counters and fixed-bucket latency histogram."""

from __future__ import annotations

import threading
from collections import Counter, OrderedDict


class RuntimeMetrics:
    BUCKETS_MS = (10, 50, 100, 250, 500, 1000, 5000, 30000)

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.latency_buckets: Counter[str] = Counter()
        self.provider_calls = 0
        self.budget_usage: dict[str, int | float] = {}
        self._run_calls: OrderedDict[str, int] = OrderedDict()

    def record(self, status: int, elapsed_ms: float, usage: dict | None) -> None:
        with self._lock:
            self.requests[str(status)] += 1
            if status >= 400:
                self.errors[str(status)] += 1
            bucket = next((str(bound) for bound in self.BUCKETS_MS if elapsed_ms <= bound), "+Inf")
            self.latency_buckets[bucket] += 1
            if usage:
                run_id = str(usage.get("run_id", ""))
                calls = int(usage.get("calls", 0))
                if run_id:
                    self.provider_calls += max(0, calls - self._run_calls.get(run_id, 0))
                    self._run_calls[run_id] = calls
                    self._run_calls.move_to_end(run_id)
                    while len(self._run_calls) > 512:
                        self._run_calls.popitem(last=False)
                self.budget_usage = {
                    key: usage[key] for key in
                    ("input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd")
                    if isinstance(usage.get(key), (int, float))
                }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "request_count_by_status": dict(self.requests),
                "error_count_by_status": dict(self.errors),
                "latency_histogram_ms": {
                    "buckets": dict(self.latency_buckets),
                    "bounds": list(self.BUCKETS_MS),
                },
                "provider_calls_observed": self.provider_calls,
                "latest_budget_usage": dict(self.budget_usage),
                "queue_depth": None,
            }


runtime_metrics = RuntimeMetrics()
