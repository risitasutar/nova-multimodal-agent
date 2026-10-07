"""
Metric primitives. Pure functions — no model calls — so every number in the report
is reproducible from the saved per-case records.
"""

from __future__ import annotations

import re
from typing import Any

from nova.agent.verifier import extract_numbers, is_abstention, number_supported
from nova.observability.metrics import percentile

# Broader than the agent's own abstention check, and applied identically to both systems.
_EVAL_ABSTAIN = re.compile(
    r"(don'?t|do not|doesn'?t|does not) have (access|information|data|that)"
    r"|(can'?t|cannot|unable to) (predict|know|provide|answer|determine|verify|confirm)"
    r"|no way to know|not possible to (know|determine)|speculat"
    r"|(there is|there's) no (page|information|mention|data|record)"
    r"|only (has|contains|includes) \d+ pages|no page 12|does not have a page",
    re.I,
)
_STOP = frozenset(
    "the a an and or of to in on for with by at from as is are was were be been this that these those it its "
    "which who what when where how than then there their they them we our you your has have had not no but "
    "also into about over under more most less such can could would should will may might".split()
)
_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace(",", "").lower())


def abstained(answer: str) -> bool:
    return is_abstention(answer) or bool(_EVAL_ABSTAIN.search(answer or ""))


# ---------------------------------------------------------------- retrieval
def hit_at_k(ranked_pages: list[int], gold: list[int], k: int) -> float:
    return 1.0 if any(p in gold for p in ranked_pages[:k]) else 0.0


def reciprocal_rank(ranked_pages: list[int], gold: list[int]) -> float:
    for i, p in enumerate(ranked_pages, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------- answers
def number_match(answer: str, value: float, tolerance: float) -> bool:
    for _token, v in extract_numbers(answer):
        for candidate in (v, v * 100, v / 100):
            if abs(candidate - value) <= tolerance:
                return True
    return False


def answer_correct(case: dict[str, Any], answer: str) -> bool | None:
    """True/False for objectively gradable cases, None when not objectively gradable."""
    checks: list[bool] = []
    a = norm(answer)
    if case.get("answer_any"):
        checks.append(any(norm(x) in a for x in case["answer_any"]))
    if case.get("answer_all"):
        checks.append(all(norm(x) in a for x in case["answer_all"]))
    if case.get("answer_number"):
        checks.append(number_match(answer, case["answer_number"]["value"], case["answer_number"]["tolerance"]))
    if case.get("expect_insufficient"):
        checks.append(abstained(answer))
    if case.get("expect_clarification"):
        checks.append("?" in answer)
    for pattern in case.get("must_not_regex", []):
        checks.append(not re.search(pattern, answer, re.I))
    return all(checks) if checks else None


def tool_selection_correct(case: dict[str, Any], used: list[str]) -> bool:
    used_set = set(used)
    return set(case.get("expected_tools", [])) <= used_set and not (used_set & set(case.get("forbidden_tools", [])))


def citation_correct(case: dict[str, Any], cited_pages: list[int], retrieved_pages: list[int]) -> bool | None:
    """Only for cases with gold pages that expect an answer. None = not applicable."""
    gold = case.get("gold_pages")
    if not gold or case.get("expect_insufficient"):
        return None
    if not cited_pages:
        return False
    # Every cited page must be one that was actually retrieved, and at least one must be gold.
    return all(p in retrieved_pages for p in cited_pages) and any(p in gold for p in cited_pages)


# ---------------------------------------------------------------- groundedness
def lexical_groundedness(answer: str, evidence: str) -> float | None:
    """
    Deterministic groundedness proxy: share of answer sentences whose material numbers
    all appear in the evidence AND whose content words are ≥60% present in the evidence.
    Abstentions are grounded by definition. None when there is no evidence to check.
    """
    if not evidence.strip():
        return None
    if abstained(answer):
        return 1.0
    ev_words = set(_WORD.findall(evidence.lower()))
    ev_nums = [v for _, v in extract_numbers(evidence)]
    clean = re.sub(r"\[Source:[^\n]*?\](?:\([^)]*\))?\]?|\[[SWF]\d+(?:,\s*[SWF]\d+)*\]", " ", answer)
    sentences = [s for s in _SENT.split(clean) if len(_WORD.findall(s.lower())) >= 3]
    if not sentences:
        return None
    supported = 0
    for s in sentences:
        nums_ok = all(number_supported(t, v, ev_nums) for t, v in extract_numbers(s))
        words = [w for w in _WORD.findall(s.lower()) if w not in _STOP]
        overlap = sum(w in ev_words for w in words) / len(words) if words else 1.0
        supported += nums_ok and overlap >= 0.6
    return supported / len(sentences)


def summarize_latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "avg_ms": round(sum(values) / len(values), 1),
        "p50_ms": round(percentile(values, 50), 1),
        "p95_ms": round(percentile(values, 95), 1),
    }


def mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None
