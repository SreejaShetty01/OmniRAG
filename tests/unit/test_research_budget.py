from agent.graph import ResearchGraph, ResearchLimits
from config.profiles import ProfileName
from models.query import Query


def test_research_budget_termination_routes_directly_to_finalize():
    graph = ResearchGraph(object(), object())
    assert graph._route({"research_termination_reason": "budget_exceeded",
                         "round_number": 1, "executed_queries": [],
                         "pending_queries": ["should not run"], "limits": ResearchLimits()}) == "finalize"


def test_research_budget_exceeded_response_is_explicit_and_abstains():
    response = ResearchGraph._budget_exceeded_response("question")
    assert response.abstained
    assert response.abstention_reason == "budget_exceeded"
    assert "configured provider budget" in response.answer
    assert "provider_budget_exceeded" in response.warnings


def test_research_assessment_preserves_provider_budget_termination_without_results():
    from models.query import Query

    graph = ResearchGraph(object(), object())
    result = graph._assess_and_ledger({
        "query": Query(query_text="question"),
        "limits": ResearchLimits(),
        "round_number": 1,
        "round_results": [],
        "all_results": [],
        "ledger": [],
        "research_termination_reason": "budget_exceeded",
    })

    assert result["research_termination_reason"] == "budget_exceeded"
