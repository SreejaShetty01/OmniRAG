import json

import pytest

from omnirag.config.settings import Settings
from eval.phase8_safety import require_evaluation_collection
from omnirag.infrastructure.provider_budget import BudgetExceeded, activate_run_budget, before_provider_call


def test_collection_guard_accepts_only_phase8_prefixed_targets():
    assert require_evaluation_collection("omnirag_eval_phase8_run1") == "omnirag_eval_phase8_run1"
    for value in ("omnirag", "other_eval_collection", "omnirag_eval_", "omnirag_eval_../omnirag"):
        with pytest.raises(ValueError):
            require_evaluation_collection(value)


def test_zero_cost_budget_blocks_positive_price_before_provider_call():
    settings = Settings(
        validation_max_cost_usd=0,
        validation_max_total_tokens=100,
        validation_max_calls=5,
        provider_price_table_json=json.dumps({
            "paid/model": {"input_per_million_usd": 1, "output_per_million_usd": 1},
        }),
    )
    with activate_run_budget("phase8-budget-zero-cost-test", settings) as budget:
        assert budget is not None
        with pytest.raises(BudgetExceeded, match="Estimated provider cost"):
            before_provider_call("paid", "model", 10, 10)
        assert budget.snapshot()["calls"] == 0


def test_call_cap_rejects_without_model_inference():
    settings = Settings(
        validation_max_cost_usd=0,
        validation_max_total_tokens=100,
        validation_max_calls=1,
        provider_price_table_json=json.dumps({
            "local/model": {"input_per_million_usd": 0, "output_per_million_usd": 0},
        }),
    )
    with activate_run_budget("phase8-budget-call-cap-test", settings) as budget:
        assert budget is not None
        before_provider_call("local", "model", 1, 1)
        with pytest.raises(BudgetExceeded, match="call-count"):
            before_provider_call("local", "model", 1, 1)
        assert budget.snapshot()["calls"] == 1


def test_unpriced_provider_model_is_rejected_by_active_budget():
    settings = Settings(
        validation_max_cost_usd=0,
        validation_max_total_tokens=100,
        validation_max_calls=5,
        provider_price_table_json="{}",
    )
    with activate_run_budget("phase8-budget-unpriced-test", settings):
        with pytest.raises(BudgetExceeded, match="No price configured"):
            before_provider_call("unexpected", "model", 1, 1)
