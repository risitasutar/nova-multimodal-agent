"""Runs evaluation cases through the ENHANCED Nova agent (the shipped NovaService)."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from evaluation.fixtures.documents import fixture_path
from nova.service import NovaService
from nova.tools.registry import tool_category


class EnhancedRunner:
    name = "enhanced"

    def __init__(self) -> None:
        self.svc = NovaService.from_settings()

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        tid = f"enh-{case['id']}-{uuid.uuid4().hex[:6]}"
        if case.get("document"):
            self.svc.ingest_document(tid, case["document"], fixture_path(case["document"]).read_bytes())
        for turn in case.get("setup_turns", []):
            self.svc.chat(turn, tid)

        record: dict[str, Any] = {"id": case["id"], "system": self.name, "error": None}
        start = time.perf_counter()
        try:
            r = self.svc.chat(case["question"], tid, approval_required=False)
        except Exception as exc:  # noqa: BLE001 - record any failure as a failed case
            record.update(answer="", tools=[], raw_tools=[], retrieved_pages=[], cited_pages=[], evidence="",
                          error=repr(exc)[:300], latency_ms=round((time.perf_counter() - start) * 1000, 1))
            return record
        latency = round((time.perf_counter() - start) * 1000, 1)

        state = self.svc.graph.get_state({"configurable": {"thread_id": tid}}).values
        evidence_parts = [s.get("text", "") for s in state.get("sources", [])]
        evidence_parts += [json.dumps(t["data"]) for t in state.get("tool_results", [])
                           if t["tool"] == "calculator" and t["status"] == "success"]
        record.update(
            answer=r.answer,
            tools=sorted({c for c in (tool_category(t) for t in r.tools_used) if c}),
            raw_tools=r.tools_used,
            retrieved_pages=[c["page"] for c in state.get("retrieved_documents", [])],
            cited_pages=[c["page"] for c in r.citations if c["type"] == "document"],
            evidence="\n".join(evidence_parts),
            latency_ms=latency,
            route=r.route,
            verification_status=r.verification_status,
        )
        return record
