# Nova Evaluation Report

Generated: 2026-10-06T01:54:40+00:00 · Dataset: 42 cases · Model: `qwen3:8b` · Embeddings: `nomic-embed-text`
Hardware: Windows 11, Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, Python 3.12.0 (CPU-only Ollama inference)

All numbers below were produced by `python -m evaluation.run_evaluation` from the per-case records in
`results/runs/*.jsonl`. Live web/market answers vary over time; those cases are scored on tool use,
grounding and failures only. See `evaluation/README.md` for metric definitions and caveats.

## Baseline vs Enhanced (end-to-end agent)

| Metric | Baseline | Enhanced | Delta |
|---|---|---|---|
| Pass rate (all cases) | 40.5% | 90.5% | +50.0 pp |
| Tool selection accuracy | 69.0% | 95.2% | +26.2 pp |
| Answer accuracy (objective cases) | 60.6% | 90.9% | +30.3 pp |
| Retrieval Hit@1 (end-to-end) | 31.6% | 94.7% | +63.2 pp |
| Retrieval Hit@4 (end-to-end) | 47.4% | 100.0% | +52.6 pp |
| Retrieval MRR (end-to-end) | 0.386 | 0.965 | +0.579 |
| Citation correctness | 0.0% | 94.7% | +94.7 pp |
| Groundedness (lexical proxy) | 63.9% | 85.6% | +21.7 pp |
| Abstention accuracy (absent answers) | 83.3% | 83.3% | +0.0 pp |
| Multi-tool success rate | 25.0% | 75.0% | +50.0 pp |
| Failure rate | 0.0% | 0.0% | +0.0 pp |
| Average latency (s) | 73.9 | 55.2 | -18.7 |
| P95 latency (s) | 150.3 | 144.4 | -5.9 |

Retrieval metrics cover 19 cases with gold pages; citation correctness 19; groundedness 27 (enhanced) / 32 (baseline) cases with evidence.

## Retrieval component benchmark (raw question, no agent)

| Retriever | Hit@1 | Hit@4 | MRR | Configuration |
|---|---|---|---|---|
| baseline | 52.6% | 79.0% | 0.632 | LangChain FAISS L2, 1000/200 chunks, no task prefixes, top-4 |
| enhanced | 89.5% | 100.0% | 0.930 | cosine FAISS + task prefixes + hybrid rerank, 600/100 chunks, top-4 |

19 questions with gold pages. Relevance threshold 0.56: absent-answer questions rejected before generation: 0/6; answerable questions wrongly rejected: 0/19.

## Pass rate by category

| Category | baseline | enhanced |
|---|---|---|
| absent_answer | 1/4 | 4/4 |
| adversarial | 2/4 | 2/4 |
| ambiguous | 1/2 | 2/2 |
| calculator | 4/4 | 4/4 |
| citation | 0/3 | 3/3 |
| direct_document | 0/6 | 6/6 |
| finance | 3/4 | 4/4 |
| memory | 2/2 | 2/2 |
| multi_hop_document | 0/3 | 2/3 |
| multi_page_document | 0/3 | 3/3 |
| multi_tool | 1/4 | 3/4 |
| web | 3/3 | 3/3 |

## Baseline: cases not passed (25)

- `doc-01` — tools ['web'] (expected ['document']); answer check failed; citation incorrect/missing
- `doc-02` — tools [] (expected ['document']); answer check failed; citation incorrect/missing
- `doc-03` — citation incorrect/missing
- `doc-04` — citation incorrect/missing
- `doc-05` — citation incorrect/missing
- `doc-06` — tools ['web'] (expected ['document']); answer check failed; citation incorrect/missing
- `hop-01` — answer check failed; citation incorrect/missing
- `hop-02` — tools ['web'] (expected ['document']); answer check failed; citation incorrect/missing
- `hop-03` — answer check failed; citation incorrect/missing
- `page-01` — citation incorrect/missing
- `page-02` — tools ['web'] (expected ['document']); answer check failed; citation incorrect/missing
- `page-03` — tools [] (expected ['document']); answer check failed; citation incorrect/missing
- `absent-01` — tools [] (expected ['document'])
- `absent-02` — tools [] (expected ['document'])
- `absent-03` — answer check failed
- `amb-02` — tools ['finance'] (expected []); answer check failed
- `cite-01` — citation incorrect/missing
- `cite-02` — answer check failed; citation incorrect/missing
- `cite-03` — citation incorrect/missing
- `fin-02` — tools ['web'] (expected ['finance'])
- `multi-02` — answer check failed
- `multi-03` — tools ['finance'] (expected ['finance', 'calculator'])
- `multi-04` — tools ['web'] (expected ['finance', 'web'])
- `adv-02` — answer check failed
- `adv-04` — tools [] (expected ['document'])

## Enhanced: cases not passed (4)

- `hop-03` — answer check failed; citation incorrect/missing
- `multi-03` — tools ['finance'] (expected ['finance', 'calculator'])
- `adv-02` — answer check failed
- `adv-04` — tools [] (expected ['document']); answer check failed
