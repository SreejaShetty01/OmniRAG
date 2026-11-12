"""Run-scoped provider spend and token guard; inactive unless explicitly configured."""

from __future__ import annotations

import contextvars
import json
import math
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def estimate_tokens(text: str) -> int:
    """Conservative character-count estimate used when the provider has no tokenizer."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


@dataclass(frozen=True)
class BudgetTicket:
    ledger: "RunBudget"
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    input_rate: float
    output_rate: float


class RunBudget:
    def __init__(self, run_id: str, *, max_cost_usd: float, max_total_tokens: int,
                 max_calls: int, price_table: dict[str, dict[str, float]]):
        self.run_id = run_id
        self.max_cost_usd = max_cost_usd
        self.max_total_tokens = max_total_tokens
        self.max_calls = max_calls
        self.price_table = price_table
        self._lock = threading.RLock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self._next_id = 0
        self._pending: dict[int, BudgetTicket] = {}

    def reserve(self, provider: str, model: str, input_tokens: int,
                output_tokens: int) -> tuple[int, BudgetTicket]:
        key = f"{provider}/{model}"
        rates = self.price_table.get(key)
        if rates is None:
            raise BudgetExceeded(f"No price configured for provider/model {key}")
        input_rate = float(rates.get("input_per_million_usd", 0.0))
        output_rate = float(rates.get("output_per_million_usd", 0.0))
        estimated_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
        with self._lock:
            if self.calls + 1 > self.max_calls:
                raise BudgetExceeded("Provider call-count budget exceeded")
            if self.input_tokens + self.output_tokens + input_tokens + output_tokens > self.max_total_tokens:
                raise BudgetExceeded("Provider token budget exceeded")
            if self.cost_usd + estimated_cost > self.max_cost_usd:
                raise BudgetExceeded("Estimated provider cost budget exceeded")
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cost_usd += estimated_cost
            self._next_id += 1
            ticket = BudgetTicket(self, provider, model, input_tokens, output_tokens,
                                  estimated_cost, input_rate, output_rate)
            self._pending[self._next_id] = ticket
            return self._next_id, ticket

    def record(self, ticket_id: int, ticket: BudgetTicket, *, input_tokens: int,
               output_tokens: int, cost_usd: float | None) -> None:
        with self._lock:
            if self._pending.pop(ticket_id, None) is None:
                return
            actual_cost = cost_usd if cost_usd is not None else (
                input_tokens * ticket.input_rate + output_tokens * ticket.output_rate) / 1_000_000
            self.input_tokens += input_tokens - ticket.input_tokens
            self.output_tokens += output_tokens - ticket.output_tokens
            self.cost_usd += actual_cost - ticket.estimated_cost
            if self.input_tokens + self.output_tokens > self.max_total_tokens:
                raise BudgetExceeded("Provider token budget exceeded after reported usage")
            if self.cost_usd > self.max_cost_usd:
                raise BudgetExceeded("Provider cost budget exceeded after reported usage")

    def snapshot(self) -> dict[str, int | float | str]:
        with self._lock:
            return {"run_id": self.run_id, "calls": self.calls,
                    "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_tokens": self.input_tokens + self.output_tokens,
                    "estimated_cost_usd": round(self.cost_usd, 8),
                    "max_calls": self.max_calls, "max_total_tokens": self.max_total_tokens,
                    "max_cost_usd": self.max_cost_usd}


_active_budget: contextvars.ContextVar[RunBudget | None] = contextvars.ContextVar(
    "omnirag_provider_budget", default=None)
_registry_lock = threading.RLock()
_registry: OrderedDict[str, RunBudget] = OrderedDict()


def _configured_budget(run_id: str, settings) -> RunBudget | None:
    configured = (settings.validation_max_cost_usd is not None or
                  settings.validation_max_total_tokens is not None or
                  settings.validation_max_calls is not None)
    if not configured:
        return None
    if (settings.validation_max_cost_usd is None or settings.validation_max_total_tokens is None or
            settings.validation_max_calls is None):
        raise ValueError("Set all three validation budget limits: max USD, total tokens, and calls")
    try:
        prices = json.loads(settings.provider_price_table_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Provider price table must be valid JSON") from exc
    with _registry_lock:
        budget = _registry.get(run_id)
        if budget is None:
            budget = RunBudget(run_id, max_cost_usd=settings.validation_max_cost_usd,
                               max_total_tokens=settings.validation_max_total_tokens,
                               max_calls=settings.validation_max_calls, price_table=prices)
            _registry[run_id] = budget
            while len(_registry) > 512:
                _registry.popitem(last=False)
        _registry.move_to_end(run_id)
        return budget


@contextmanager
def activate_run_budget(run_id: str, settings=None) -> Iterator[RunBudget | None]:
    if settings is None:
        from omnirag.config.settings import get_settings
        settings = get_settings()
    budget = _configured_budget(run_id, settings)
    token = _active_budget.set(budget)
    try:
        yield budget
    finally:
        _active_budget.reset(token)


def before_provider_call(provider: str, model: str, input_tokens: int,
                         output_tokens: int = 0) -> tuple[int, BudgetTicket] | None:
    budget = _active_budget.get()
    return budget.reserve(provider, model, input_tokens, output_tokens) if budget else None


def after_provider_call(ticket_data: tuple[int, BudgetTicket] | None, *, input_tokens: int,
                        output_tokens: int, cost_usd: float | None = None) -> None:
    if ticket_data is not None:
        ticket_data[1].ledger.record(ticket_data[0], ticket_data[1], input_tokens=input_tokens,
                                     output_tokens=output_tokens, cost_usd=cost_usd)


def current_budget_snapshot() -> dict | None:
    budget = _active_budget.get()
    return budget.snapshot() if budget else None
