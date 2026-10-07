"""
Video/audio evaluation of the Nova agent (live LLM over deterministic transcript fixtures).

    python -m evaluation.video_evaluation                 # all cases, per-case timeout 600 s
    python -m evaluation.video_evaluation --resume
    python -m evaluation.video_evaluation --report-only

Each case runs in a child process with a hard timeout (same harness as the PDF evaluation).
Outputs: results/video_latest.json, results/video_results.csv, results/video_report.md;
raw records in results/runs/video.jsonl.

Temporal metrics: a transcript chunk or a citation is "correct" when its [start, end]
overlaps a gold span. Hit@k / MRR rank the agent's retrieved transcript chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation import metrics as m
from evaluation import utf8_stdout, worker_env
from evaluation.run_evaluation import RESULTS, RUNS, _failure_record, _setup_env

EVAL_DIR = Path(__file__).resolve().parent
MARKER = "__NOVA_VIDEO_RECORD__"


def load_cases() -> list[dict[str, Any]]:
    return json.loads((EVAL_DIR / "video_dataset.json").read_text(encoding="utf-8"))["cases"]


# ------------------------------------------------------------------ running
def run_case(case: dict[str, Any]) -> dict[str, Any]:
    from evaluation.fixtures.media import load_transcript, media_filename, strategy_pdf
    from nova.service import NovaService
    from nova.tools.registry import tool_category

    svc = NovaService.from_settings()
    assert svc.media is not None
    tid = f"vid-{case['id']}-{uuid.uuid4().hex[:6]}"
    for doc in case.get("documents", []):
        svc.ingest_document(tid, doc, strategy_pdf())
    setup_ms: dict[str, float] = {}
    for name in case.get("media", []):
        asset = svc.media.create_from_transcript(tid, media_filename(name), load_transcript(name))
        done = svc.media.process(asset.media_id, insights=True)
        if done.status.value != "COMPLETED":
            raise RuntimeError(f"media processing failed: {done.error}")
        setup_ms.update({f"{name}.{k}": v for k, v in done.timings_ms.items()})
    for turn in case.get("setup_turns", []):
        svc.chat(turn, tid)

    record: dict[str, Any] = {"id": case["id"], "system": "enhanced", "error": None, "setup_timings_ms": setup_ms}
    start = time.perf_counter()
    r = svc.chat(case["question"], tid, approval_required=False)
    latency = round((time.perf_counter() - start) * 1000, 1)
    state = svc.graph.get_state({"configurable": {"thread_id": tid}}).values
    evidence = [s.get("text", "") for s in state.get("sources", [])]
    evidence += [json.dumps(t["data"]) for t in state.get("tool_results", []) if t["tool"] == "calculator" and t["status"] == "success"]
    evidence += state.get("evidence_notes", [])
    record.update(
        answer=r.answer, route=r.route, verification_status=r.verification_status,
        tools=sorted({c for c in (tool_category(t) for t in r.tools_used) if c}), raw_tools=r.tools_used,
        retrieved_spans=[[c["start_time"], c["end_time"]] for c in state.get("retrieved_media", [])],
        cited_spans=[[c["start_time"], c["end_time"]] for c in r.citations if c["type"] == "video"],
        citation_types=[c["type"] for c in r.citations], evidence="\n".join(evidence), latency_ms=latency,
        retrieval_ms=sum(t["latency_ms"] for t in state.get("tool_results", []) if t["tool"] == "video_search"),
    )
    return record


def _isolated(case: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "evaluation.video_evaluation", "--worker", case["id"]]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, cwd=str(EVAL_DIR.parent), env=worker_env())
    except subprocess.TimeoutExpired:
        return _failure_record(case["id"], "enhanced", f"timeout after {timeout_s:.0f}s",
                               (time.perf_counter() - start) * 1000, timed_out=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        return _failure_record(case["id"], "enhanced", f"worker exit {proc.returncode}: {tail[0][:200]}",
                               (time.perf_counter() - start) * 1000, timed_out=False)
    return json.loads(lines[-1][len(MARKER):])


def run_all(cases: list[dict[str, Any]], resume: bool, timeout_s: float) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "video.jsonl"
    done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines()} if (resume and out.exists()) else set()
    if not resume and out.exists():
        out.unlink()
    todo = [c for c in cases if c["id"] not in done]
    print(f"[video] {len(todo)} case(s) to run ({len(done)} done), per-case timeout {timeout_s:.0f}s", flush=True)
    for i, case in enumerate(todo, start=1):
        rec = _isolated(case, timeout_s)
        rec["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status = ("TIMEOUT " if rec.get("timed_out") else "ERROR ") + rec["error"][:80] if rec.get("error") \
            else f"route={rec.get('route')} tools={rec['tools']}"
        line = f"[video] {i}/{len(todo)} {case['id']:<10} {rec['latency_ms'] / 1000:6.1f}s {status}"
        print(line, flush=True)
        if rec.get("error"):
            with (RUNS / "events.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{rec['finished_at']} {line}\n")


# ------------------------------------------------------------------ grading
def overlaps(span: list[float], gold: list[list[float]]) -> bool:
    return any(span[0] <= g[1] and span[1] >= g[0] for g in gold)


def grade(case: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    failed = bool(run.get("error")) or not (run.get("answer") or "").strip()
    answer = run.get("answer") or ""
    gold = case.get("gold_spans") or []
    ranked = [overlaps(s, gold) for s in run.get("retrieved_spans", [])] if gold else []
    cited = run.get("cited_spans", [])
    correct = m.answer_correct(case, answer)
    timestamp_ok = None
    if gold and not case.get("expect_insufficient"):
        timestamp_ok = (not failed) and bool(cited) and any(overlaps(s, gold) for s in cited)
    if case.get("timestamp_question"):
        correct = timestamp_ok  # the answer to "when" is the cited time span
    if failed and correct is not None:
        correct = False
    citation_ok = None
    if case.get("requires_citation") and not case.get("expect_insufficient"):
        retrieved = run.get("retrieved_spans", [])
        valid = all(any(c[0] <= r[1] and c[1] >= r[0] for r in retrieved) or not retrieved for c in cited)
        citation_ok = (not failed) and bool(run.get("citation_types")) and valid and (
            not gold or any(overlaps(c, gold) for c in cited))
    tool_ok = m.tool_selection_correct(case, run.get("tools", []))
    passed = (not failed) and tool_ok and correct is not False and citation_ok is not False
    return {
        **run, "failed": failed, "tool_correct": tool_ok, "answer_correct": correct, "timestamp_correct": timestamp_ok,
        "citation_correct": citation_ok,
        "hit_at_1": (1.0 if ranked[:1] and ranked[0] else 0.0) if gold else None,
        "hit_at_4": (1.0 if any(ranked[:4]) else 0.0) if gold else None,
        "mrr": (next((1 / (i + 1) for i, ok in enumerate(ranked) if ok), 0.0)) if gold else None,
        "groundedness": None if failed else m.lexical_groundedness(answer, run.get("evidence", "")),
        "passed": passed,
    }


def aggregate(graded: list[dict[str, Any]], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    lat = [g["latency_ms"] for g in graded if not g["failed"]]
    multi = [g for g in graded if len(cases[g["id"]].get("expected_tools", [])) > 1]
    by_cat: dict[str, dict[str, int]] = {}
    for g in graded:
        b = by_cat.setdefault(cases[g["id"]]["category"], {"cases": 0, "passed": 0})
        b["cases"] += 1
        b["passed"] += int(g["passed"])

    def rate(key: str) -> float | None:
        return m.mean([None if g[key] is None else float(g[key]) for g in graded])

    return {
        "cases": len(graded),
        "completed": sum(1 for g in graded if not g["failed"]),
        "timed_out": sum(1 for g in graded if g.get("timed_out")),
        "pass_rate": rate("passed"),
        "routing_accuracy": rate("tool_correct"),
        "answer_accuracy_objective": rate("answer_correct"),
        "objective_cases": sum(1 for g in graded if g["answer_correct"] is not None),
        "retrieval_hit_at_1": rate("hit_at_1"),
        "retrieval_hit_at_4": rate("hit_at_4"),
        "retrieval_mrr": rate("mrr"),
        "retrieval_cases": sum(1 for g in graded if g["hit_at_1"] is not None),
        "timestamp_accuracy": rate("timestamp_correct"),
        "timestamp_cases": sum(1 for g in graded if g["timestamp_correct"] is not None),
        "citation_accuracy": rate("citation_correct"),
        "citation_cases": sum(1 for g in graded if g["citation_correct"] is not None),
        "groundedness": rate("groundedness"),
        "multi_tool_success_rate": m.mean([float(g["tool_correct"] and not g["failed"] and g["answer_correct"] is not False)
                                           for g in multi]),
        "failure_rate": rate("failed"),
        **{f"latency_{k}": v for k, v in m.summarize_latency(lat).items()},
        "video_retrieval_latency": m.summarize_latency([g.get("retrieval_ms", 0) for g in graded
                                                        if not g["failed"] and g.get("retrieval_ms")]),
        "by_category": {k: {**v, "pass_rate": round(v["passed"] / v["cases"], 3)} for k, v in sorted(by_cat.items())},
    }


ROWS = [("Pass rate", "pass_rate", "pct"), ("Routing / tool selection accuracy", "routing_accuracy", "pct"),
        ("Answer accuracy (objective cases)", "answer_accuracy_objective", "pct"),
        ("Retrieval Hit@1 (temporal)", "retrieval_hit_at_1", "pct"), ("Retrieval Hit@4 (temporal)", "retrieval_hit_at_4", "pct"),
        ("Retrieval MRR", "retrieval_mrr", "num"), ("Timestamp accuracy (cited span overlaps gold)", "timestamp_accuracy", "pct"),
        ("Citation accuracy", "citation_accuracy", "pct"), ("Groundedness (lexical proxy)", "groundedness", "pct"),
        ("Multi-tool success rate", "multi_tool_success_rate", "pct"), ("Failure rate", "failure_rate", "pct"),
        ("Average latency (s)", "latency_avg_ms", "sec"), ("P95 latency (s)", "latency_p95_ms", "sec")]


def report(cases_list: list[dict[str, Any]]) -> dict[str, Any]:
    from nova.config import get_settings

    cases = {c["id"]: c for c in cases_list}
    runs = []
    path = RUNS / "video.jsonl"
    if path.exists():
        latest: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            latest[rec["id"]] = rec
        runs = [r for r in latest.values() if r["id"] in cases]
    graded = [grade(cases[r["id"]], r) for r in runs]
    summary = aggregate(graded, cases) if graded else {}
    s = get_settings()
    meta = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "dataset_cases": len(cases_list),
            "model": s.chat_model_name, "embedding_model": s.embedding_model_name,
            "hardware": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()} (CPU-only Ollama)"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "video_latest.json").write_text(json.dumps({"meta": meta, "summary": summary, "cases": graded}, indent=2,
                                                          ensure_ascii=False), encoding="utf-8")
    cols = ["id", "category", "passed", "failed", "timed_out", "route", "tools", "tool_correct", "answer_correct",
            "timestamp_correct", "citation_correct", "hit_at_1", "hit_at_4", "mrr", "groundedness", "latency_ms",
            "verification_status", "error", "answer"]
    with (RESULTS / "video_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for g in graded:
            w.writerow({**{k: g.get(k) for k in cols}, "category": cases[g["id"]]["category"],
                        "tools": "|".join(g.get("tools", [])), "answer": (g.get("answer") or "")[:500]})

    def fmt(v: Any, kind: str) -> str:
        if v is None:
            return "n/a"
        return f"{v * 100:.1f}%" if kind == "pct" else f"{v / 1000:.1f}" if kind == "sec" else f"{v:.3f}"

    lines = ["# Nova Video Evaluation Report", "",
             f"Generated: {meta['generated_at']} · {len(cases_list)} cases · model `{meta['model']}` · "
             f"embeddings `{meta['embedding_model']}` · {meta['hardware']}", "",
             "Live LLM over deterministic transcript fixtures (no Whisper in the loop: transcription accuracy is "
             "measured separately). Generated from `results/runs/video.jsonl`.", ""]
    if summary:
        lines += [f"Completed {summary['completed']}/{summary['cases']} · timed out {summary['timed_out']}", "",
                  "| Metric | Value |", "|---|---|"]
        lines += [f"| {label} | {fmt(summary.get(key), kind)} |" for label, key, kind in ROWS]
        vr = summary["video_retrieval_latency"]
        lines += ["", f"Video retrieval latency (video_search tool): avg {vr['avg_ms']:.0f} ms, P95 {vr['p95_ms']:.0f} ms.",
                  f"Denominators: objective {summary['objective_cases']}, retrieval {summary['retrieval_cases']}, "
                  f"timestamp {summary['timestamp_cases']}, citation {summary['citation_cases']}.",
                  "", "## Pass rate by category", "", "| Category | Passed |", "|---|---|"]
        lines += [f"| {k} | {v['passed']}/{v['cases']} |" for k, v in summary["by_category"].items()]
        failed = [g for g in graded if not g["passed"]]
        lines += ["", f"## Cases not passed ({len(failed)})", ""]
        for g in failed:
            why = []
            if g["failed"]:
                why.append(f"error: {g.get('error')}")
            if not g["tool_correct"]:
                why.append(f"tools {g['tools']} (expected {cases[g['id']].get('expected_tools')})")
            if g["answer_correct"] is False:
                why.append("answer check failed")
            if g["citation_correct"] is False:
                why.append("citation incorrect/missing")
            lines.append(f"- `{g['id']}` — {'; '.join(why)}")
    (RESULTS / "video_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--case-timeout", type=float, default=600.0)
    ap.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    _setup_env()
    cases = load_cases()
    if args.worker:
        rec = run_case(next(c for c in cases if c["id"] == args.worker))
        utf8_stdout()
        print(MARKER + json.dumps(rec, ensure_ascii=False), flush=True)
        return 0
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]
    if not args.report_only:
        from nova.llm import check_llm

        ok, msg = check_llm()
        if not ok:
            print(f"LLM not ready: {msg}", file=sys.stderr)
            return 1
        run_all(cases, args.resume, args.case_timeout)
    summary = report(load_cases())
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
