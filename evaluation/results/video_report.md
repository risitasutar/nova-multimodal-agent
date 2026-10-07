# Nova Video Evaluation Report

Generated: 2026-10-06T01:54:43+00:00 · 27 cases · model `qwen3:8b` · embeddings `nomic-embed-text` · Windows 11, Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (CPU-only Ollama)

Live LLM over deterministic transcript fixtures (no Whisper in the loop: transcription accuracy is measured separately). Generated from `results/runs/video.jsonl`.

Completed 27/27 · timed out 0

| Metric | Value |
|---|---|
| Pass rate | 81.5% |
| Routing / tool selection accuracy | 92.6% |
| Answer accuracy (objective cases) | 96.2% |
| Retrieval Hit@1 (temporal) | 86.4% |
| Retrieval Hit@4 (temporal) | 86.4% |
| Retrieval MRR | 0.864 |
| Timestamp accuracy (cited span overlaps gold) | 90.9% |
| Citation accuracy | 88.2% |
| Groundedness (lexical proxy) | 58.1% |
| Multi-tool success rate | 71.4% |
| Failure rate | 0.0% |
| Average latency (s) | 162.9 |
| P95 latency (s) | 321.5 |

Video retrieval latency (video_search tool): avg 2255 ms, P95 5100 ms.
Denominators: objective 26, retrieval 22, timestamp 22, citation 17.

## Pass rate by category

| Category | Passed |
|---|---|
| absent_evidence | 2/2 |
| action_items | 0/2 |
| adversarial_injection | 2/2 |
| direct_video | 4/4 |
| key_decisions | 2/2 |
| memory | 1/1 |
| multi_tool | 1/1 |
| open_questions | 1/2 |
| summary | 1/1 |
| thread_isolation | 1/1 |
| timestamp | 3/3 |
| video_calculator | 2/2 |
| video_finance | 1/1 |
| video_pdf | 1/3 |

## Cases not passed (5)

- `v-act-01` — citation incorrect/missing
- `v-act-02` — answer check failed
- `v-oq-01` — citation incorrect/missing
- `v-x-02` — tools ['document', 'video', 'web'] (expected ['video', 'document'])
- `v-x-03` — tools ['document', 'finance', 'video', 'web'] (expected ['video', 'document'])
