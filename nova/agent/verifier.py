"""
Answer verification (deterministic by default; optional LLM groundedness judge).

Checks:
  * citations   – every marker refers to a real source; grounded answers cite at least one
  * numbers     – every material number in the answer appears in the evidence
                  (sources, tool data, calculator results, the question itself)
  * claims      – each sentence citing a document/transcript source shares enough content
                  words with *the sources it cites* (catches citations attached to the wrong chunk)
  * coverage    – non-empty answer
Outcomes: VERIFIED | RETRY (bounded) | FAILED.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

_NUMBER = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_CITATION_BLOCK = re.compile(
    r"\[(?:Source|PDF|VIDEO|AUDIO|WEB|FINANCE):[^\n]*?\](?:\([^)]*\))?\]?|\[[SVWF]\d+(?:\s*,\s*[SVWF]\d+)*\]")
_TIMESTAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # locators, validated by the citation layer
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


ABSTENTION = re.compile(
    r"(couldn'?t|could not|cannot|can'?t|unable to) (find|locate|determine|answer)"
    r"|(sources?|documents?|reports?|context|evidence|information)[^.\n]{0,40}\b(do(es)?n'?t|do(es)? not)\s+"
    r"(contain|mention|include|provide|specify|state|say|disclose|cover)"
    r"|\bnot (mentioned|provided|included|specified|stated|disclosed|available|found)\b"
    r"|\bno (information|mention|data|details|evidence)\b"
    r"|insufficient (evidence|information)",
    re.I,
)


def is_abstention(text: str) -> bool:
    """True when the answer explicitly says the evidence does not contain the answer."""
    return bool(ABSTENTION.search(text or ""))


def _to_float(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("$", "").replace("%", "").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float]]:
    text = _CITATION_BLOCK.sub(" ", text)
    text = _DATE.sub(" ", text)
    text = _TIMESTAMP.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        value = _to_float(m.group(0))
        if value is not None:
            out.append((m.group(0), value))
    return out


def _decimals(token: str) -> int:
    core = token.replace("%", "").replace(",", "")
    return len(core.split(".")[1]) if "." in core else 0


def number_supported(token: str, value: float, evidence: list[float]) -> bool:
    if abs(value) <= 10 and float(value).is_integer():
        return True  # small counts/ordinals ("3 risks", "Q2") are not material claims
    if 1900 <= value <= 2100 and float(value).is_integer():
        return True  # years are dates, checked by grounding rather than arithmetic
    nd = _decimals(token)
    for ev in evidence:
        for candidate in (ev, ev * 100, ev / 100):  # 0.052 vs 5.2%
            if abs(candidate - value) <= max(0.5 * 10 ** (-nd), abs(value) * 0.005):
                return True
            if candidate != 0 and abs(value) >= 1000 and abs(candidate - value) / abs(candidate) < 0.01:
                return True  # "$1.2 billion" style rounding of large values
        # scaled units: "4.2 million" vs 4,200,000 ; "4.2 billion" vs 4200 (millions)
        for scale in (1e3, 1e6, 1e9):
            if ev and abs(ev / scale - value) <= max(0.05 * 10 ** (-nd + 1), abs(value) * 0.01):
                return True
    return False


def collect_evidence_numbers(state: dict[str, Any]) -> list[float]:
    blobs = [state.get("user_query", "")]
    blobs += [s.get("text", "") + " " + s.get("label", "") for s in state.get("sources", [])]
    blobs += [json.dumps(t.get("data", {})) for t in state.get("tool_results", []) if t.get("status") == "success"]
    nums: list[float] = []
    for blob in blobs:
        nums.extend(v for _, v in extract_numbers(blob))
    return nums


def check_numbers(answer: str, state: dict[str, Any]) -> list[str]:
    evidence = collect_evidence_numbers(state)
    unsupported = [tok for tok, val in extract_numbers(answer) if not number_supported(tok, val, evidence)]
    return list(dict.fromkeys(unsupported))


STOPWORDS = frozenset(
    "the a an and or of to in on for with by at from as is are was were be been this that these those it its "
    "which who what when where how than then there their they them we our you your has have had not no but "
    "also into about over under more most less such can could would should will may might".split()
)
CONTENT_WORD = re.compile(r"[a-z][a-z0-9\-]{3,}")
_CLAIM_MARKER = re.compile(r"\[((?:[SVWF]\d{1,3})(?:\s*[,;]\s*[SVWF]\d{1,3})*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Calibrated on the saved evaluation answers: 1 of 73 correctly cited sentences falls below 0.2.
CLAIM_MIN_OVERLAP = 0.2
CLAIM_MIN_WORDS = 4
# Words that say *where* a fact is ("this is mentioned on page 10 of the report") rather than the fact.
# They are not expected in the cited text, so they are not scored; citation validity covers the locator.
_PROVENANCE = frozenset(
    "information mentioned mentions page pages report document documents source sources stated states "
    "according section specified indicated noted transcript recording".split()
)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in CONTENT_WORD.findall(text.lower()) if w not in STOPWORDS}


def check_claim_support(draft: str, sources: list[dict[str, Any]]) -> list[str]:
    """
    Sentences whose document/transcript citations do not support them.

    For each sentence citing [S#] or [V#], the share of its content words (5-char stems)
    found in the cited sources' text must reach CLAIM_MIN_OVERLAP. Web and market-data
    citations are skipped: their snippets are too short for a lexical check to be fair.
    """
    by_id = {s["id"]: s for s in sources}
    unsupported: list[str] = []
    for sentence in _SENTENCE.split(draft or ""):
        cited = [by_id[i] for m in _CLAIM_MARKER.findall(sentence) for i in re.findall(r"[SVWF]\d{1,3}", m)
                 if i in by_id and by_id[i]["type"] in ("document", "video")]
        if not cited or is_abstention(sentence):
            continue
        words = [w[:5] for w in CONTENT_WORD.findall(_CLAIM_MARKER.sub(" ", sentence).lower())
                 if w not in STOPWORDS and w not in _PROVENANCE]
        if len(words) < CLAIM_MIN_WORDS:
            continue
        evidence = set().union(*(_stems(s.get("text", "")) for s in cited))
        if sum(w in evidence for w in words) / len(words) < CLAIM_MIN_OVERLAP:
            unsupported.append(_CLAIM_MARKER.sub("", sentence).strip()[:120])
    return unsupported


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(description="True if every factual claim is supported by the evidence")
    unsupported_claims: list[str] = Field(default_factory=list)


JUDGE_SYSTEM = """You are a strict fact-checker. Decide whether every factual claim in the ANSWER is supported by the EVIDENCE.
Ignore style, greetings, citations formatting and statements that the evidence is insufficient. List any unsupported claims."""


def judge_groundedness(llm: BaseChatModel, question: str, evidence: str, answer: str) -> GroundednessVerdict:
    structured = llm.with_structured_output(GroundednessVerdict, method="json_schema")
    raw = structured.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence[:12000]}\n\nANSWER:\n{answer}"),
        ]
    )
    return raw if isinstance(raw, GroundednessVerdict) else GroundednessVerdict.model_validate(raw)
