"""Runs evaluation cases through the BASELINE (pre-upgrade ReAct) agent."""

from __future__ import annotations

import ast
import json
import re
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from evaluation.fixtures.documents import fixture_path
from nova.agent.baseline import BaselineAgent
from nova.config import get_settings
from nova.llm import get_chat_model, get_embeddings

TOOL_CATEGORY = {"rag_tool": "document", "duckduckgo_search": "web", "get_stock_price": "finance",
                 "calculator": "calculator"}
_THINK = re.compile(r"<think>.*?</think>", re.S)
_PAGE = re.compile(r"\bpages?\s*:?\s*(\d{1,3})", re.I)


def _parse(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (ValueError, SyntaxError):
            continue
    return content


class BaselineRunner:
    name = "baseline"

    def __init__(self) -> None:
        s = get_settings()
        # Original embedding usage: no nomic task prefixes.
        self.agent = BaselineAgent(get_chat_model(s), get_embeddings(s, task_prefixes=False))
        self.graph = self.agent.build(InMemorySaver())

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        tid = f"base-{case['id']}-{uuid.uuid4().hex[:6]}"
        cfg = {"configurable": {"thread_id": tid}, "recursion_limit": 25}
        if case.get("document"):
            self.agent.ingest_pdf(fixture_path(case["document"]).read_bytes(), tid, case["document"])
        for turn in case.get("setup_turns", []):
            self.graph.invoke({"messages": [HumanMessage(content=turn)]}, cfg)
        before = len(self.graph.get_state(cfg).values.get("messages", []))

        record: dict[str, Any] = {"id": case["id"], "system": self.name, "error": None}
        start = time.perf_counter()
        try:
            result = self.graph.invoke({"messages": [HumanMessage(content=case["question"])]}, cfg)
            new = result["messages"][before:]
        except Exception as exc:  # noqa: BLE001 - record any failure as a failed case
            record.update(answer="", tools=[], raw_tools=[], retrieved_pages=[], cited_pages=[], evidence="",
                          error=repr(exc)[:300], latency_ms=round((time.perf_counter() - start) * 1000, 1))
            return record
        latency = round((time.perf_counter() - start) * 1000, 1)

        tool_msgs = [m for m in new if isinstance(m, ToolMessage)]
        raw_tools = [m.name for m in tool_msgs]
        retrieved: list[int] = []
        for m in tool_msgs:
            if m.name == "rag_tool" and not retrieved:
                payload = _parse(m.content)
                if isinstance(payload, dict):
                    # Baseline stores PyPDF's 0-based page index; convert to real page numbers.
                    retrieved = [int(p) + 1 for p in payload.get("pages", []) if p is not None]
        final = next((m for m in reversed(new) if isinstance(m, AIMessage) and m.content), None)
        answer = _THINK.sub("", str(final.content)).strip() if final else ""
        record.update(
            answer=answer,
            tools=sorted({TOOL_CATEGORY.get(n, n) for n in raw_tools}),
            raw_tools=raw_tools,
            retrieved_pages=retrieved,
            # The baseline has no citation mechanism; credit any explicit "page N" it writes.
            cited_pages=[int(p) for p in _PAGE.findall(answer)],
            evidence="\n".join(str(m.content) for m in tool_msgs),
            latency_ms=latency,
            route=None,
            verification_status=None,
        )
        return record
