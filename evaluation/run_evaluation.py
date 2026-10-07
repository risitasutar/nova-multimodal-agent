"""
Nova evaluation: baseline (pre-upgrade ReAct agent) vs enhanced (Nova graph).

    python -m evaluation.run_evaluation                      # both systems, all cases
    python -m evaluation.run_evaluation --systems enhanced   # one system
    python -m evaluation.run_evaluation --resume             # continue an interrupted run
    python -m evaluation.run_evaluation --case-timeout 900   # per-case hard timeout (default 600 s)
    python -m evaluation.run_evaluation --report-only        # re-grade saved runs

Outputs (evaluation/results/): latest.json, results.csv, report.md.
Raw per-case records are appended to results/runs/<system>.jsonl as each case finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation import utf8_stdout, worker_env

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
RUNS = RESULTS / "runs"


def _setup_env() -> None:
    # Isolated storage: evaluation never touches the user's conversations or indexes.
    os.environ.setdefault("NOVA_DATA_DIR", tempfile.mkdtemp(prefix="nova-eval-"))
    os.environ.setdefault("NOVA_LOG_LEVEL", "WARNING")
    os.environ.setdefault("NOVA_ENV", "test")


def load_dataset() -> list[dict[str, Any]]:
    return json.loads((EVAL_DIR / "dataset.json").read_text(encoding="utf-8"))["cases"]


def _make_runner(name: str) -> Any:
    from evaluation.baseline_runner import BaselineRunner
    from evaluation.enhanced_runner import EnhancedRunner

    return BaselineRunner() if name == "baseline" else EnhancedRunner()


def _failure_record(case_id: str, system: str, error: str, latency_ms: float, timed_out: bool) -> dict[str, Any]:
    """An explicit failed record. Nothing about the answer is invented: it is empty and graded as failed."""
    return {"id": case_id, "system": system, "error": error, "timed_out": timed_out, "answer": "", "tools": [],
            "raw_tools": [], "retrieved_pages": [], "cited_pages": [], "evidence": "",
            "latency_ms": round(latency_ms, 1), "route": None, "verification_status": None}


def _run_case_isolated(name: str, case: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """
    Run one case in a child process with a hard timeout. A thread cannot abort an
    in-flight Ollama request; killing the child closes its HTTP connection, which makes
    Ollama cancel the generation, so a stuck case cannot slow down the next one.
    """
    import subprocess

    cmd = [sys.executable, "-m", "evaluation.run_evaluation", "--worker", name, "--cases", case["id"]]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout_s, cwd=str(EVAL_DIR.parent), env=worker_env())
    except subprocess.TimeoutExpired:
        return _failure_record(case["id"], name, f"timeout after {timeout_s:.0f}s",
                               (time.perf_counter() - start) * 1000, timed_out=True)
    marker = "__NOVA_EVAL_RECORD__"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(marker)]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        return _failure_record(case["id"], name, f"worker exit {proc.returncode}: {tail[0][:200]}",
                               (time.perf_counter() - start) * 1000, timed_out=False)
    return json.loads(lines[-1][len(marker):])


def _worker(name: str, case_id: str) -> int:
    """Child-process entry point: run exactly one case and print its record."""
    case = next(c for c in load_dataset() if c["id"] == case_id)
    rec = _make_runner(name).run_case(case)
    utf8_stdout()
    print("__NOVA_EVAL_RECORD__" + json.dumps(rec, ensure_ascii=False), flush=True)
    return 0


def run_system(name: str, cases: list[dict[str, Any]], resume: bool, case_timeout_s: float = 0) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{name}.jsonl"
    events = RUNS / "events.log"
    # Resume skips cases already recorded, except those whose latest record is a harness crash
    # ("worker exit ..."): those are retried and the new record supersedes the old line (latest wins).
    # Timeouts and wrong answers are genuine results and are not retried.
    done = {r["id"] for r in load_runs(name) if not str(r.get("error") or "").startswith("worker exit")} \
        if (resume and out.exists()) else set()
    if not resume and out.exists():
        out.unlink()
    runner = None if case_timeout_s > 0 else _make_runner(name)
    todo = [c for c in cases if c["id"] not in done]
    print(f"[{name}] {len(todo)} case(s) to run ({len(done)} already done)"
          + (f", per-case timeout {case_timeout_s:.0f}s" if case_timeout_s > 0 else ""), flush=True)
    for i, case in enumerate(todo, start=1):
        rec = _run_case_isolated(name, case, case_timeout_s) if runner is None else runner.run_case(case)
        rec["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        with out.open("a", encoding="utf-8") as fh:  # saved before moving on: nothing is lost on interruption
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("timed_out"):
            status = f"TIMEOUT {rec['error']}"
        elif rec.get("error"):
            status = "ERROR " + rec["error"][:80]
        else:
            status = f"tools={rec['tools']}"
        line = f"[{name}] {i}/{len(todo)} {case['id']:<10} {rec['latency_ms'] / 1000:6.1f}s {status}"
        print(line, flush=True)
        if rec.get("error"):
            with events.open("a", encoding="utf-8") as fh:
                fh.write(f"{rec['finished_at']} {line}\n")


def load_runs(name: str) -> list[dict[str, Any]]:
    path = RUNS / f"{name}.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        latest[rec["id"]] = rec
    return list(latest.values())


# ------------------------------------------------------------------ component retrieval benchmark
def retrieval_benchmark(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Ranking quality of each retriever in isolation, on the raw question (no agent, no LLM)."""
    from evaluation import metrics as m
    from evaluation.fixtures.documents import fixture_path
    from nova.agent.baseline import BaselineAgent
    from nova.config import get_settings
    from nova.llm import get_chat_model, get_embeddings
    from nova.rag.ingest import parse_pdf
    from nova.rag.retrieve import Retriever
    from nova.rag.store import DocumentStore

    s = get_settings()
    base = BaselineAgent(get_chat_model(s), get_embeddings(s, task_prefixes=False))
    emb = get_embeddings(s)
    store = DocumentStore(s.vector_dir / "benchmark", emb, s.embedding_model_name)
    ranker = Retriever(store, emb, top_k=4, candidates=s.retrieval_candidates, min_relevance=-1.0)
    docs = sorted({c["document"] for c in cases if c.get("document")})
    for d in docs:
        data = fixture_path(d).read_bytes()
        base.ingest_pdf(data, d, d)
        store.add_document(d.replace(".", "_"), parse_pdf(data, d))

    rows: dict[str, list[dict[str, float]]] = {"baseline": [], "enhanced": []}
    absent_rejected, answerable_rejected, answerable = 0, 0, 0
    absent_cases = [c for c in cases if c.get("document") and c.get("expect_insufficient")]
    for c in cases:
        if not c.get("document"):
            continue
        tid = c["document"].replace(".", "_")
        enh = ranker.retrieve(tid, c["question"])
        if c.get("gold_pages"):
            gold = c["gold_pages"]
            base_pages = [int(d.metadata.get("page", -1)) + 1 for d in base._retrievers[c["document"]].invoke(c["question"])]
            enh_pages = [ch["page"] for ch in enh.chunks]
            for name, ranked in (("baseline", base_pages), ("enhanced", enh_pages)):
                rows[name].append({"hit1": m.hit_at_k(ranked, gold, 1), "hit4": m.hit_at_k(ranked, gold, 4),
                                   "mrr": m.reciprocal_rank(ranked, gold)})
            answerable += 1
            answerable_rejected += enh.max_relevance < s.min_relevance
        elif c.get("expect_insufficient"):
            absent_rejected += enh.max_relevance < s.min_relevance

    def avg(name: str, key: str) -> float:
        vals = [r[key] for r in rows[name]]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "questions": len(rows["enhanced"]),
        "baseline": {"hit_at_1": avg("baseline", "hit1"), "hit_at_4": avg("baseline", "hit4"), "mrr": avg("baseline", "mrr"),
                     "config": "LangChain FAISS L2, 1000/200 chunks, no task prefixes, top-4"},
        "enhanced": {"hit_at_1": avg("enhanced", "hit1"), "hit_at_4": avg("enhanced", "hit4"), "mrr": avg("enhanced", "mrr"),
                     "config": f"cosine FAISS + task prefixes + hybrid rerank, {s.chunk_size}/{s.chunk_overlap} chunks, top-4"},
        "threshold": s.min_relevance,
        "absent_questions_rejected_by_threshold": f"{absent_rejected}/{len(absent_cases)}",
        "answerable_questions_wrongly_rejected": f"{answerable_rejected}/{answerable}",
    }


# ------------------------------------------------------------------ reporting
METRIC_ROWS = [
    ("Pass rate (all cases)", "pass_rate", "pct"),
    ("Tool selection accuracy", "tool_selection_accuracy", "pct"),
    ("Answer accuracy (objective cases)", "answer_accuracy_objective", "pct"),
    ("Retrieval Hit@1 (end-to-end)", "retrieval_hit_at_1", "pct"),
    ("Retrieval Hit@4 (end-to-end)", "retrieval_hit_at_4", "pct"),
    ("Retrieval MRR (end-to-end)", "retrieval_mrr", "num"),
    ("Citation correctness", "citation_accuracy", "pct"),
    ("Groundedness (lexical proxy)", "groundedness", "pct"),
    ("Abstention accuracy (absent answers)", "abstention_accuracy", "pct"),
    ("Multi-tool success rate", "multi_tool_success_rate", "pct"),
    ("Failure rate", "failure_rate", "pct_low"),
    ("Average latency (s)", "latency_avg_ms", "sec"),
    ("P95 latency (s)", "latency_p95_ms", "sec"),
]


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind.startswith("pct"):
        return f"{v * 100:.1f}%"
    if kind == "sec":
        return f"{v / 1000:.1f}"
    return f"{v:.3f}"


def _delta(b: Any, e: Any, kind: str) -> str:
    if b is None or e is None:
        return "n/a"
    d = e - b
    if kind.startswith("pct"):
        return f"{d * 100:+.1f} pp"
    if kind == "sec":
        return f"{d / 1000:+.1f}"
    return f"{d:+.3f}"


def write_report(summary: dict[str, Any], graded: dict[str, list[dict[str, Any]]], cases: dict[str, dict], meta: dict) -> None:
    lines = [
        "# Nova Evaluation Report",
        "",
        f"Generated: {meta['generated_at']} · Dataset: {meta['dataset_cases']} cases · "
        f"Model: `{meta['model']}` · Embeddings: `{meta['embedding_model']}`",
        f"Hardware: {meta['hardware']}",
        "",
        "All numbers below were produced by `python -m evaluation.run_evaluation` from the per-case records in",
        "`results/runs/*.jsonl`. Live web/market answers vary over time; those cases are scored on tool use,",
        "grounding and failures only. See `evaluation/README.md` for metric definitions and caveats.",
        "",
    ]
    systems = [s for s in ("baseline", "enhanced") if s in summary]
    if len(systems) == 2:
        b, e = summary["baseline"], summary["enhanced"]
        lines += ["## Baseline vs Enhanced (end-to-end agent)", "", "| Metric | Baseline | Enhanced | Delta |", "|---|---|---|---|"]
        for label, key, kind in METRIC_ROWS:
            lines.append(f"| {label} | {_fmt(b.get(key), kind)} | {_fmt(e.get(key), kind)} | {_delta(b.get(key), e.get(key), kind)} |")
        lines.append("")
        lines.append(f"Retrieval metrics cover {e['retrieval_cases']} cases with gold pages; citation correctness "
                     f"{e['citation_cases']}; groundedness {e['groundedness_cases']} (enhanced) / "
                     f"{b['groundedness_cases']} (baseline) cases with evidence.")
    else:
        s = summary[systems[0]]
        lines += [f"## {systems[0].title()} results", "", "| Metric | Value |", "|---|---|"]
        for label, key, kind in METRIC_ROWS:
            lines.append(f"| {label} | {_fmt(s.get(key), kind)} |")
    if "retrieval_component" in summary:
        rc = summary["retrieval_component"]
        lines += ["", "## Retrieval component benchmark (raw question, no agent)", "",
                  "| Retriever | Hit@1 | Hit@4 | MRR | Configuration |", "|---|---|---|---|---|"]
        for name in ("baseline", "enhanced"):
            r = rc[name]
            lines.append(f"| {name} | {r['hit_at_1'] * 100:.1f}% | {r['hit_at_4'] * 100:.1f}% | {r['mrr']:.3f} | {r['config']} |")
        lines += ["", f"{rc['questions']} questions with gold pages. Relevance threshold {rc['threshold']}: "
                  f"absent-answer questions rejected before generation: {rc['absent_questions_rejected_by_threshold']}; "
                  f"answerable questions wrongly rejected: {rc['answerable_questions_wrongly_rejected']}."]
    lines += ["", "## Pass rate by category", "", "| Category | " + " | ".join(systems) + " |",
              "|---|" + "---|" * len(systems)]
    cats = sorted({c["category"] for c in cases.values()})
    for cat in cats:
        cells = []
        for s in systems:
            bc = summary[s]["by_category"].get(cat)
            cells.append(f"{bc['passed']}/{bc['cases']}" if bc else "–")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    for s in systems:
        failed = [g for g in graded[s] if not g["passed"]]
        lines += ["", f"## {s.title()}: cases not passed ({len(failed)})", ""]
        for g in failed:
            reasons = []
            if g["failed"]:
                reasons.append(f"error: {g.get('error') or 'empty answer'}")
            if not g["tool_correct"]:
                reasons.append(f"tools {g['tools']} (expected {cases[g['id']].get('expected_tools')})")
            if g["answer_correct"] is False:
                reasons.append("answer check failed")
            if g["citation_correct"] is False and cases[g["id"]].get("requires_citation"):
                reasons.append("citation incorrect/missing")
            lines.append(f"- `{g['id']}` — {'; '.join(reasons)}")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(cases_list: list[dict[str, Any]], systems: list[str], retrieval: dict | None, started: float) -> dict:
    from evaluation.evaluator import aggregate, grade
    from nova.config import get_settings

    cases = {c["id"]: c for c in cases_list}
    graded: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for s in systems:
        runs = [r for r in load_runs(s) if r["id"] in cases]
        if not runs:
            continue
        graded[s] = [grade(cases[r["id"]], r) for r in runs]
        summary[s] = aggregate(graded[s], cases)
    if retrieval:
        summary["retrieval_component"] = retrieval
    settings = get_settings()
    meta = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_cases": len(cases_list),
        "model": settings.chat_model_name,
        "embedding_model": settings.embedding_model_name,
        "hardware": f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, "
                    f"Python {platform.python_version()} (CPU-only Ollama inference)",
        "eval_wall_time_s": round(time.time() - started, 1),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latest.json").write_text(json.dumps({"meta": meta, "summary": summary, "cases": graded}, indent=2,
                                                    ensure_ascii=False), encoding="utf-8")
    with (RESULTS / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["system", "id", "category", "passed", "failed", "tool_correct", "answer_correct", "citation_correct",
                "groundedness", "hit_at_1", "hit_at_4", "mrr", "latency_ms", "tools", "retrieved_pages", "cited_pages",
                "verification_status", "error", "answer"]
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for rows in graded.values():
            for g in rows:
                writer.writerow({**{k: g.get(k) for k in cols}, "category": cases[g["id"]]["category"],
                                 "tools": "|".join(g.get("tools", [])), "answer": (g.get("answer") or "")[:500]})
    write_report(summary, graded, cases, meta)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="baseline,enhanced")
    parser.add_argument("--cases", default="", help="comma-separated case ids or categories")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--skip-retrieval-benchmark", action="store_true")
    parser.add_argument("--case-timeout", type=float, default=600.0,
                        help="hard per-case timeout in seconds (each case runs in a child process); 0 = in-process, no timeout")
    parser.add_argument("--results-dir", default="",
                        help="write records and reports here instead of evaluation/results (e.g. for smoke runs)")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.results_dir:
        global RESULTS, RUNS
        RESULTS = Path(args.results_dir).resolve()
        RUNS = RESULTS / "runs"

    _setup_env()
    if args.worker:
        return _worker(args.worker, args.cases)
    from nova.llm import check_llm

    started = time.time()
    cases = load_dataset()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    if not args.report_only:
        ok, msg = check_llm()
        if not ok:
            print(f"LLM not ready: {msg}", file=sys.stderr)
            return 1
        for s in systems:
            run_system(s, cases, args.resume, args.case_timeout)
    retrieval = None
    if not args.skip_retrieval_benchmark:
        retrieval = retrieval_benchmark(cases)
    summary = write_outputs(load_dataset() if not args.cases else cases, systems, retrieval, started)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_category"} if isinstance(v, dict) else v
                      for k, v in summary.items()}, indent=2))
    print(f"\nReport: {RESULTS / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
