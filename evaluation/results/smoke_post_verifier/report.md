# Nova Evaluation Report

Generated: 2026-10-06T16:38:59+00:00 · Dataset: 17 cases · Model: `qwen3:8b` · Embeddings: `nomic-embed-text`
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
| Retrieval Hit@1 (end-to-end) | 90.0% |
| Retrieval Hit@4 (end-to-end) | 100.0% |
| Retrieval MRR (end-to-end) | 0.933 |
| Citation correctness | 100.0% |
| Groundedness (lexical proxy) | 75.0% |
| Abstention accuracy (absent answers) | 100.0% |
| Multi-tool success rate | 100.0% |
| Failure rate | 0.0% |
| Average latency (s) | 102.2 |
| P95 latency (s) | 241.7 |

## Retrieval component benchmark (raw question, no agent)

| Retriever | Hit@1 | Hit@4 | MRR | Configuration |
|---|---|---|---|---|
| baseline | 50.0% | 80.0% | 0.617 | LangChain FAISS L2, 1000/200 chunks, no task prefixes, top-4 |
| enhanced | 90.0% | 100.0% | 0.933 | cosine FAISS + task prefixes + hybrid rerank, 600/100 chunks, top-4 |

10 questions with gold pages. Relevance threshold 0.56: absent-answer questions rejected before generation: 0/4; answerable questions wrongly rejected: 0/10.

## Pass rate by category

| Category | enhanced |
|---|---|
| absent_answer | 4/4 |
| citation | 3/3 |
| direct_document | 6/6 |
| multi_tool | 4/4 |

## Enhanced: cases not passed (0)

