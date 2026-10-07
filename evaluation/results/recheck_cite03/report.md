# Nova Evaluation Report

Generated: 2026-10-06T16:42:38+00:00 · Dataset: 1 cases · Model: `qwen3:8b` · Embeddings: `nomic-embed-text`
Hardware: Windows 11, Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.0 (CPU-only Ollama inference)

All numbers below were produced by `python -m evaluation.run_evaluation` from the per-case records in
`results/runs/*.jsonl`. Live web/market answers vary over time; those cases are scored on tool use,
grounding and failures only. See `evaluation/README.md` for metric definitions and caveats.

## Enhanced results

| Metric | Value |
|---|---|
| Pass rate (all cases) | 100.0% |
| Tool selection accuracy | 100.0% |
| Answer accuracy (objective cases) | 100.0% |
| Retrieval Hit@1 (end-to-end) | 100.0% |
| Retrieval Hit@4 (end-to-end) | 100.0% |
| Retrieval MRR (end-to-end) | 1.000 |
| Citation correctness | 100.0% |
| Groundedness (lexical proxy) | 50.0% |
| Abstention accuracy (absent answers) | n/a |
| Multi-tool success rate | n/a |
| Failure rate | 0.0% |
| Average latency (s) | 15.7 |
| P95 latency (s) | 15.7 |

## Pass rate by category

| Category | enhanced |
|---|---|
| citation | 1/1 |

## Enhanced: cases not passed (0)

