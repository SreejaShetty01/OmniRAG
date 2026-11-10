from concurrent.futures import ThreadPoolExecutor

import pytest

from infrastructure.provider_budget import BudgetExceeded, RunBudget


def budget(*, usd=1.0, tokens=100, calls=10):
    return RunBudget("test-run", max_cost_usd=usd, max_total_tokens=tokens,
                     max_calls=calls,
                     price_table={"fake/model": {"input_per_million_usd": 10.0,
                                                  "output_per_million_usd": 20.0}})


def test_provider_budget_trips_token_call_and_cost_caps():
    with pytest.raises(BudgetExceeded, match="token budget"):
        budget(tokens=4).reserve("fake", "model", 4, 1)
    ledger = budget(calls=1)
    ledger.reserve("fake", "model", 1, 0)
    with pytest.raises(BudgetExceeded, match="call-count"):
        ledger.reserve("fake", "model", 1, 0)
    with pytest.raises(BudgetExceeded, match="cost budget"):
        budget(usd=0.00001).reserve("fake", "model", 0, 1)


def test_unpriced_model_fails_closed_when_a_usd_budget_is_active():
    with pytest.raises(BudgetExceeded, match="No price configured"):
        budget().reserve("fake", "unpriced", 1, 1)


def test_concurrent_reservations_cannot_exceed_call_or_token_caps():
    ledger = budget(tokens=20, calls=3)

    def reserve_one(_):
        try:
            ledger.reserve("fake", "model", 4, 0)
            return "reserved"
        except BudgetExceeded:
            return "blocked"

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(reserve_one, range(12)))
    assert outcomes.count("reserved") == 3
    assert ledger.snapshot()["calls"] == 3
    assert ledger.snapshot()["total_tokens"] == 12


def test_reported_usage_replaces_pre_call_reservation():
    ledger = budget(tokens=20)
    ticket_id, ticket = ledger.reserve("fake", "model", 8, 10)
    ledger.record(ticket_id, ticket, input_tokens=5, output_tokens=2, cost_usd=0.0002)
    usage = ledger.snapshot()
    assert usage["total_tokens"] == 7
    assert usage["estimated_cost_usd"] == 0.0002
