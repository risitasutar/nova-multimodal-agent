"""
Evaluation dashboard — renders ONLY what `evaluation/results/latest.json` contains.

    streamlit run evaluation/dashboard.py          (standalone)
    or open the "Evaluation" page inside the Nova app.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

RESULTS = Path(__file__).resolve().parent / "results" / "latest.json"

CARDS = [
    ("Pass rate", "pass_rate", "pct"),
    ("Tool accuracy", "tool_selection_accuracy", "pct"),
    ("Retrieval Hit@4", "retrieval_hit_at_4", "pct"),
    ("MRR", "retrieval_mrr", "num"),
    ("Citation accuracy", "citation_accuracy", "pct"),
    ("Groundedness", "groundedness", "pct"),
    ("Avg latency", "latency_avg_ms", "sec"),
    ("P95 latency", "latency_p95_ms", "sec"),
    ("Failure rate", "failure_rate", "pct"),
]


def _fmt(v, kind: str) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if kind == "pct" else f"{v / 1000:.1f}s" if kind == "sec" else f"{v:.3f}"


def render() -> None:
    st.title("📊 Nova evaluation")
    if not RESULTS.exists():
        st.info("No results yet. Run `python -m evaluation.run_evaluation` to generate them.")
        return
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    meta, summary = data["meta"], data["summary"]
    st.caption(f"Generated {meta['generated_at']} · {meta['dataset_cases']} cases · model `{meta['model']}` · "
               f"{meta['hardware']}")
    systems = [s for s in ("enhanced", "baseline") if s in summary]
    system = st.radio("System", systems, horizontal=True)
    s = summary[system]
    other = summary.get("baseline" if system == "enhanced" else "enhanced")
    cols = st.columns(3)
    for i, (label, key, kind) in enumerate(CARDS):
        delta = None
        if other and s.get(key) is not None and other.get(key) is not None:
            d = s[key] - other[key]
            delta = f"{d * 100:+.1f} pp" if kind == "pct" else f"{d / 1000:+.1f}s" if kind == "sec" else f"{d:+.3f}"
        cols[i % 3].metric(label, _fmt(s.get(key), kind), delta,
                           delta_color="inverse" if key in ("failure_rate", "latency_avg_ms", "latency_p95_ms") else "normal")
    st.caption(f"Total cases: {s['cases']} · objective answer accuracy {_fmt(s['answer_accuracy_objective'], 'pct')} "
               f"over {s['objective_cases']} cases · deltas are vs the other system")

    st.subheader("Pass rate by category")
    st.bar_chart({cat: v["pass_rate"] for cat, v in s["by_category"].items()}, horizontal=True, height=320)

    if "retrieval_component" in summary:
        st.subheader("Retrieval component benchmark")
        rc = summary["retrieval_component"]
        st.table({name: {"Hit@1": f"{rc[name]['hit_at_1']:.1%}", "Hit@4": f"{rc[name]['hit_at_4']:.1%}",
                         "MRR": f"{rc[name]['mrr']:.3f}"} for name in ("baseline", "enhanced")})

    st.subheader("Cases")
    rows = [{"id": c["id"], "passed": c["passed"], "tools": ", ".join(c.get("tools", [])),
             "answer_correct": c["answer_correct"], "citation": c["citation_correct"],
             "latency_s": round(c["latency_ms"] / 1000, 1), "answer": (c.get("answer") or "")[:160]}
            for c in data["cases"][system]]
    st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":  # `streamlit run evaluation/dashboard.py`
    st.set_page_config(page_title="Nova evaluation", page_icon="📊", layout="wide")
    render()
