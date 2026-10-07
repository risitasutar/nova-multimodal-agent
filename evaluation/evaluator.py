"""Grades raw case runs and aggregates them into the metrics reported in results/report.md."""

from __future__ import annotations

from typing import Any

from evaluation import metrics as m


def grade(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Attach per-case scores to a raw run record (see runners for the record shape)."""
    failed = bool(run.get("error")) or not (run.get("answer") or "").strip()
    answer = run.get("answer") or ""
    gold = case.get("gold_pages") or []
    ranked = run.get("retrieved_pages") or []
    correct = m.answer_correct(case, answer)
    if failed and correct is not None:
        correct = False  # a failed run never counts as a correct answer
    tool_ok = m.tool_selection_correct(case, run.get("tools", []))
    cite_ok = m.citation_correct(case, run.get("cited_pages", []), ranked) if not failed else (
        False if gold and not case.get("expect_insufficient") else None)
    grounded = None if failed else m.lexical_groundedness(answer, run.get("evidence", ""))
    passed = (not failed) and tool_ok and (correct is not False) and (
        cite_ok is not False if case.get("requires_citation") else True)
    return {
        **run,
        "failed": failed,
        "tool_correct": tool_ok,
        "answer_correct": correct,
        "citation_correct": cite_ok,
        "groundedness": grounded,
        "hit_at_1": m.hit_at_k(ranked, gold, 1) if gold else None,
        "hit_at_4": m.hit_at_k(ranked, gold, 4) if gold else None,
        "mrr": m.reciprocal_rank(ranked, gold) if gold else None,
        "passed": passed,
    }


def aggregate(graded: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = len(graded)
    lat = [g["latency_ms"] for g in graded if not g["failed"]]
    objective = [g for g in graded if g["answer_correct"] is not None]
    multi = [g for g in graded if cases[g["id"]]["category"] == "multi_tool"]
    absent = [g for g in graded if cases[g["id"]].get("expect_insufficient")]
    summary = {
        "cases": n,
        "pass_rate": m.mean([float(g["passed"]) for g in graded]),
        "tool_selection_accuracy": m.mean([float(g["tool_correct"]) for g in graded]),
        "answer_accuracy_objective": m.mean([float(g["answer_correct"]) for g in objective]),
        "objective_cases": len(objective),
        "retrieval_hit_at_1": m.mean([g["hit_at_1"] for g in graded]),
        "retrieval_hit_at_4": m.mean([g["hit_at_4"] for g in graded]),
        "retrieval_mrr": m.mean([g["mrr"] for g in graded]),
        "retrieval_cases": sum(1 for g in graded if g["hit_at_1"] is not None),
        "citation_accuracy": m.mean([None if g["citation_correct"] is None else float(g["citation_correct"])
                                     for g in graded]),
        "citation_cases": sum(1 for g in graded if g["citation_correct"] is not None),
        "groundedness": m.mean([g["groundedness"] for g in graded]),
        "groundedness_cases": sum(1 for g in graded if g["groundedness"] is not None),
        "abstention_accuracy": m.mean([float(bool(g["answer_correct"])) for g in absent]),
        "multi_tool_success_rate": m.mean([
            float(g["tool_correct"] and not g["failed"] and g["answer_correct"] is not False) for g in multi]),
        "failure_rate": m.mean([float(g["failed"]) for g in graded]),
        **{f"latency_{k}": v for k, v in m.summarize_latency(lat).items()},
    }
    by_cat: dict[str, dict[str, Any]] = {}
    for g in graded:
        cat = cases[g["id"]]["category"]
        bucket = by_cat.setdefault(cat, {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += int(g["passed"])
    summary["by_category"] = {k: {**v, "pass_rate": round(v["passed"] / v["cases"], 3)} for k, v in sorted(by_cat.items())}
    return summary
