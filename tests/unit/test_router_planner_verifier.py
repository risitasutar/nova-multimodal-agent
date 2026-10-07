import httpx
import pytest

from nova.agent.classifier import QueryClassification, apply_policy_guards, classify, heuristic_classify
from nova.agent.planner import Plan, PlanStep, deterministic_plan, plan_query, validate_plan
from nova.agent.verifier import check_numbers, extract_numbers, is_abstention, number_supported
from nova.errors import LLMUnavailableError
from tests.fakes import ScriptedChatModel


# ------------------------------------------------------------- classifier
def test_rule_fast_paths_skip_the_llm():
    llm = ScriptedChatModel(structured_fn=lambda *a: pytest.fail("LLM should not be called"))
    c, method, _ = classify(llm, "What is 9283 * 47?", "", [])
    assert (c.query_type, c.expression, method) == ("calculation", "9283 * 47", "rule")
    c, method, _ = classify(llm, "hello!", "", [])
    assert (c.query_type, method) == ("general", "rule")
    c, method, _ = classify(llm, "What is the current stock price of AAPL?", "", [])
    assert (c.query_type, c.tickers, method) == ("finance", ["AAPL"], "rule")


def test_llm_structured_classification_is_used_for_ambiguous_queries():
    llm = ScriptedChatModel(structured_fn=lambda schema, msgs: {"query_type": "document",
                                                               "document_query": "net income 2025"})
    c, method, _ = classify(llm, "What was Northwind's net income?", "", ["report.pdf"])
    assert method == "llm" and c.query_type == "document" and c.document_query == "net income 2025"


def test_llm_parse_failure_falls_back_to_heuristics():
    def broken(schema, msgs):
        raise ValueError("model returned garbage")

    c, method, _ = classify(ScriptedChatModel(structured_fn=broken), "latest news on chip exports", "", ["r.pdf"])
    assert method == "fallback" and c.query_type == "web"


def test_llm_connection_error_is_not_swallowed():
    def down(schema, msgs):
        raise httpx.ConnectError("Connection refused")

    with pytest.raises(LLMUnavailableError):
        classify(ScriptedChatModel(structured_fn=down), "Tell me about Northwind", "", ["r.pdf"])


def test_policy_guard_time_sensitive_never_general():
    c, notes = apply_policy_guards(QueryClassification(query_type="general"), "What happened in markets today?", False)
    assert c.query_type == "web" and c.search_query and notes


def test_policy_guard_finance_without_ticker_asks_for_clarification():
    c, _ = apply_policy_guards(QueryClassification(query_type="finance"), "what is the share price?", False)
    assert c.query_type == "clarification_required" and c.clarification_question


def test_policy_guard_document_reference():
    c, _ = apply_policy_guards(QueryClassification(query_type="general"), "Summarize my report", True)
    assert c.query_type == "document" and c.document_query == "Summarize my report"


def test_heuristic_multi_tool():
    c = heuristic_classify("Compare today's AAPL stock price with my uploaded report", True)
    assert c.query_type == "multi_tool" and "AAPL" in c.tickers and c.document_query


def test_ticker_cleanup():
    c = QueryClassification(query_type="finance", tickers=["$aapl", "nvda", "not a ticker!", "AAPL"])
    assert c.tickers == ["AAPL", "NVDA"]


# ------------------------------------------------------------- planner
def test_deterministic_finance_plan():
    c = QueryClassification(query_type="finance", tickers=["NVDA"], finance_needs=["history", "news"], period="1mo")
    plan = deterministic_plan(c, "Analyze NVDA", False)
    assert [s.tool for s in plan.steps] == ["price_history", "news_search"]
    assert plan.steps[0].period == "1mo"


def test_validate_plan_drops_doc_steps_without_documents_and_dedupes():
    plan = Plan(steps=[PlanStep(tool="document_search", input="x"), PlanStep(tool="stock_quote", input="aapl"),
                       PlanStep(tool="stock_quote", input="AAPL")])
    out, notes = validate_plan(plan, has_documents=False, max_steps=5)
    assert [(s.tool, s.input) for s in out.steps] == [("stock_quote", "AAPL")]
    assert any("no document" in n for n in notes)


def test_plan_truncated_to_budget():
    plan = Plan(steps=[PlanStep(tool="web_search", input=f"q{i}") for i in range(9)])
    out, _ = validate_plan(plan, True, max_steps=3)
    assert len(out.steps) == 3


def test_multi_tool_uses_llm_planner_and_falls_back():
    c = QueryClassification(query_type="multi_tool", tickers=["AAPL"], document_query="revenue")
    good = ScriptedChatModel(structured_fn=lambda s, m: {"steps": [
        {"tool": "document_search", "input": "revenue 2025"}, {"tool": "stock_quote", "input": "AAPL"}],
        "needs_calculation": True, "summary": "x"})
    plan, method, _ = plan_query(good, c, "compare", ["r.pdf"], 5)
    assert method == "llm" and plan.needs_calculation and len(plan.steps) == 2

    def broken(s, m):
        raise ValueError("bad json")

    plan, method, _ = plan_query(ScriptedChatModel(structured_fn=broken), c, "compare", ["r.pdf"], 5)
    assert method == "fallback" and {s.tool for s in plan.steps} == {"document_search", "stock_quote"}


# ------------------------------------------------------------- verifier
def test_extract_numbers_ignores_citations_and_dates():
    nums = [v for _, v in extract_numbers("Revenue was $412.6 million [S1] on 2025-12-31 [Source: a.pdf, Page 3].")]
    assert nums == [412.6]


@pytest.mark.parametrize(
    "token,value,evidence,ok",
    [
        ("412.6", 412.6, [412.6], True),
        ("22.0%", 22.0, [0.21999], True),
        ("22%", 22.0, [21.9994], True),
        ("$1.2", 1.2, [1_200_000_000.0], True),
        ("4.2", 4.2, [4200.0], True),
        ("999.9", 999.9, [412.6, 338.2], False),
        ("3", 3.0, [], True),
        ("2025", 2025.0, [], True),
    ],
)
def test_number_support(token, value, evidence, ok):
    assert number_supported(token, value, evidence) is ok


def test_check_numbers_flags_fabricated_figures():
    state = {"user_query": "q", "sources": [{"text": "Net income was $28.1 million."}], "tool_results": []}
    assert check_numbers("Net income was $28.1 million [S1].", state) == []
    assert check_numbers("Net income was $31.4 million [S1].", state) == ["$31.4"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The sources do not contain information about dividends.", True),
        ("I couldn't find sufficient evidence in the uploaded document.", True),
        ("The dividend is not mentioned in the report.", True),
        ("Revenue was $412.6 million [S1].", False),
    ],
)
def test_abstention_detection(text, expected):
    assert is_abstention(text) is expected


# ------------------------------------------------------------- claim support
def _src(sid, kind, text):
    return {"id": sid, "type": kind, "text": text}


def test_claim_supported_by_its_cited_source_passes():
    from nova.agent.verifier import check_claim_support

    sources = [_src("S1", "document", "Net income was $28.1 million in 2025, compared with $14.6 million in 2024.")]
    assert check_claim_support("Northwind's net income in 2025 was $28.1 million [S1].", sources) == []


def test_claim_citing_an_unrelated_chunk_is_flagged():
    from nova.agent.verifier import check_claim_support

    sources = [_src("S1", "document", "Net income was $28.1 million in 2025."),
               _src("S2", "document", "The company opened a robotics laboratory in Lisbon staffed by engineers.")]
    flagged = check_claim_support("The company opened a robotics laboratory in Lisbon [S1].", sources)
    assert flagged and "Lisbon" in flagged[0]
    assert check_claim_support("The company opened a robotics laboratory in Lisbon [S2].", sources) == []


def test_claim_check_skips_web_finance_abstentions_and_short_sentences():
    from nova.agent.verifier import check_claim_support

    sources = [_src("W1", "web", "Chip export rules."), _src("F1", "finance", "AAPL 230 USD."),
               _src("V1", "video", "We agreed the cloud budget.")]
    draft = ("Apple stock is trading higher this morning [F1]. Regulators announced sweeping new semiconductor "
             "restrictions [W1]. The sources do not mention the hiring freeze [V1]. Budget agreed [V1].")
    assert check_claim_support(draft, sources) == []


def test_provenance_sentence_is_not_flagged():
    """Regression (post-verifier smoke eval, cite-03): a correct 'mentioned on page N' sentence was flagged."""
    from nova.agent.verifier import check_claim_support

    sources = [_src("S1", "document", "Construction of a new assembly plant in Monterrey, Mexico will begin in 2026.")]
    draft = ("Northwind's new assembly plant will be located in Monterrey, Mexico. "
             "This information is mentioned on page 10 of the report [S1].")
    assert check_claim_support(draft, sources) == []
    # A substantive claim on the same citation is still checked.
    assert check_claim_support("Northwind is opening a chocolate factory in Lisbon next spring [S1].", sources)
