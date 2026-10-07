# Final project audit

Audit date: 2026-10-06. Method: the code was read for every capability below, the deterministic suite and the
live end-to-end tests were run, and the published evaluation reports were regenerated from their saved
per-case records with `--report-only`; the regenerated metrics were identical. Nothing here is taken from
earlier documents without re-checking.

Legend: **Yes** = verified in code and exercised · **Partial** = present but with a stated gap · **No** = absent.

| Capability | Implemented | Integrated | Tested | Evidence | Resume Claim |
|------------|-------------|------------|--------|----------|--------------|
| LangGraph orchestration (10-node state machine) | Yes | Yes | Yes | `nova/agent/graph.py`; `tests/agent/test_graph.py` | Yes |
| Query classifier (8 routes, rules + structured LLM + fallback + guards) | Yes | Yes | Yes | `nova/agent/classifier.py`; `tests/unit/test_router_planner_verifier.py`; eval tool-selection 95.2% (docs) / 92.6% (video) | Yes |
| Planner (deterministic + validated LLM plans) | Yes | Yes | Yes | `nova/agent/planner.py`; unit + graph tests | Yes |
| PDF / document RAG | Yes | Yes | Yes | `nova/rag/`; `tests/rag/`; live e2e `test_live_document_answer_is_cited`; eval Hit@1 94.7% | Yes |
| Pickle-free FAISS store with checksums and ownership | Yes | Yes | Yes | `nova/rag/store.py`; `tests/rag/test_store_retrieval.py` | Yes |
| Video/audio ingestion (validate, PyAV, job service, cache) | Yes | Yes | Yes | `nova/video/`; `tests/video/`; `tests/integration/test_video_pipeline.py` | Yes |
| Whisper transcription (faster-whisper) | Yes | Yes | Yes (live) | `nova/video/transcription.py`; live `tests/e2e/test_live_video.py` passed 2026-10-06 (97 s) | Yes |
| Sarvam transcription (Hinglish) | Yes | Yes | Partial (mocked HTTP only) | `nova/video/transcription.py`; `tests/video/test_transcription_chunking.py` | "Implemented, not exercised live" only |
| Timestamp-aware chunks and retrieval | Yes | Yes | Yes | `nova/video/chunking.py`, `retrieval.py`; video eval temporal Hit@1 86.4%, timestamp accuracy 90.9% | Yes |
| Meeting intelligence (summary, decisions, actions, open questions; item verification) | Yes | Yes | Yes | `nova/video/summarization.py`; `tests/video/test_video_agent.py`; eval action-item category 0/2 | Yes, noting action items are the weakest category |
| Cross-modal (video + PDF + calculator) | Yes | Yes | Yes | `tests/integration/test_cross_modal.py`; video eval `video_pdf` 1/3 passed | Partial: works, weak in eval |
| Web / news search with URLs | Yes | Yes | Yes (fakes) + live eval | `nova/tools/web.py`; eval web 3/3 | Yes |
| Finance tools (quote, history analytics, 2 providers) | Yes | Yes | Yes | `nova/tools/finance/`; unit tests; live e2e `test_live_market_quote` | Yes |
| Safe calculator (AST, no eval) | Yes | Yes | Yes | `nova/tools/calculator.py`; `tests/unit/test_calculator.py`; live e2e | Yes |
| Unified evidence model with typed locators | Yes | Yes | Yes | `nova/evidence.py`; `tests/rag/test_citations.py` (web `retrieved_at` and finance `metric` added 2026-10-06) | Yes |
| Citation rendering and validation (incl. time spans) | Yes | Yes | Yes | `nova/rag/citations.py`; citation correctness 94.7% (docs), 88.2% (video) | Yes |
| Verifier: citation validity, numbers, abstention, bounded retry | Yes | Yes | Yes | `nova/agent/verifier.py`; graph tests | Yes |
| Verifier: claim-to-cited-source support (lexical) | Yes (added 2026-10-06) | Yes | Yes (unit + graph + regression); 17-case live smoke eval below | `check_claim_support`; threshold calibrated on 73 saved cited sentences | Yes, described as lexical |
| LLM groundedness judge | Code only | No | No | `judge_groundedness` exists but is not called | No |
| Insufficient-evidence fallback | Yes | Yes | Yes | `check_evidence` node; graph tests; abstention accuracy 83.3% | Yes |
| Persistent memory (SQLite checkpoints) | Yes | Yes | Yes | `nova/memory.py`; `test_conversation_survives_restart` | Yes |
| Thread isolation (documents, media, evidence) | Yes | Yes | Yes | `test_threads_are_isolated`, `test_document_isolation_between_threads`, `test_cross_modal_evidence_stays_in_thread`; video eval isolation 1/1 | Yes |
| Human-in-the-loop approval | Yes | Yes | Yes | `approval_gate` with `interrupt()`; approve/reject/blocked tests; API endpoint | Yes |
| Prompt-injection defences | Yes | Yes | Yes | `nova/safety.py`; injection tests for PDF and transcript; live e2e `test_live_prompt_injection_is_ignored` | Yes, as mitigation, not a guarantee |
| Input / file / path validation | Yes | Yes | Yes | `safety.py`, `rag/ingest.py`, `video/ingest.py`, `store._safe_dir` | Yes |
| Observability (structured logs, metrics, optional LangSmith) | Yes | Yes | Yes | `nova/observability/`; `/metrics`; unit tests | Yes |
| FastAPI | Yes | Yes | Yes | `api/`; `tests/api/test_api.py` | Yes |
| Streamlit UI | Yes | Yes | Yes (headless AppTest) | `streamlit_app_pro.py`; `tests/integration/test_ui.py` | Yes |
| Evaluation framework | Yes | Yes | Yes | `evaluation/`; reports regenerate identically from saved records | Yes |
| Docker / Compose | Yes | n/a | No (never built) | `Dockerfile`, `docker-compose.yml` | "Dockerfile provided" only |
| Public deployment | No | No | No | none | No |
| Speaker diarization | No | No | No | speaker is always `null` by design | No |

## Status summary

**COMPLETED** (implemented, integrated, tested):
- LangGraph orchestration, classifier, planner.
- Document RAG on a pickle-free FAISS store.
- Video/audio ingestion and faster-whisper transcription (verified live); timestamp-preserving chunks and
  temporal retrieval; meeting intelligence.
- Web/news search, finance tools, AST calculator.
- Unified evidence model and citations; verifier (citation validity, claim support, numbers, abstention,
  bounded retry); insufficient-evidence fallback.
- SQLite memory, thread isolation, human-in-the-loop approval.
- Prompt-injection mitigations, input/path validation.
- FastAPI and Streamlit (both started live); test suite; evaluation harness.

**PARTIALLY COMPLETED:**
- Sarvam (Hinglish) transcription: implemented, tested with mocked HTTP only.
- Mistral insights and the hosted-LLM provider: implemented, mocked only.
- Cross-modal video + PDF comparison: works and is tested, but weak in evaluation (1/3).
- Action-item extraction: 0/2 in the video evaluation.
- Docker/Compose: files written, image never built.
- Full-run metrics: they predate the 2026-10-06 verifier change; only a 17-case smoke run covers the current
  verifier.
- multi-01: verification `FAILED` both before and after the change.

**NOT IMPLEMENTED:**
- LLM groundedness judge in the loop (code exists, not called).
- Speaker diarization.
- UI authentication.
- Public deployment.
- CI.

## Test results

Deterministic suite (`pytest`, 2026-10-06): **251 passed, 0 failed, 0 skipped**, 5 deselected (the opt-in
`e2e` tests). `ruff check .` and `mypy`: clean.

Live end-to-end tests (`pytest -m e2e`, 2026-10-06, real `qwen3:8b` on CPU-only Ollama, after the verifier
change): **5 passed, 0 failed**.

| Test | Result | Time |
|---|---|---|
| `test_live_video.py::test_spoken_video_end_to_end` (TTS MP4 → PyAV → faster-whisper `small` → FAISS → answer with video timestamp citation → deletion) | passed | 97 s |
| `test_live_agent.py` (document answer cited to p.3; PDF prompt injection ignored; exact calculation; live market quote) | 4 passed | 526 s |

Re-run on the final code (after the provenance fix to the claim check), 2026-10-06: `pytest -m e2e` → **5 passed** in 209 s.

`python smoke_test.py` on the final code (2026-10-06/07): **12/12 checks passed, 0 failures**.
- **Checks:** document RAG + citation; insufficient-evidence fallback; calculator; live market data; memory;
  PDF prompt-injection resistance; human approval (reject path); media pipeline + insights (14 chunks); video
  retrieval with a timestamp citation; cross-source video + PDF; transcript injection resistance; media
  deletion.
- **Timing caveat:** the cross-source step logged 27,836 s of wall-clock time because the machine was
  apparently suspended overnight during it (the other steps took 11–262 s). That figure is not a latency
  measurement.

## Application verification (2026-10-06)

Live checks run against the real apps, using a throw-away data directory:

| Check | Result |
|---|---|
| FastAPI starts (`uvicorn api.main:app`) | `/health` → ok; `/ready` → LLM `qwen3:8b` ready, database ok; `/docs` → 200 |
| API validation | empty body → 422; `thread_id="../etc"` → 422 (pattern `^[A-Za-z0-9_\-]+$`) |
| Calculator via API | `What is 9283 * 47?` → `**9283 * 47 = 436,301**`, route `calculation`, no LLM call |
| Streamlit starts (`streamlit run streamlit_app_pro.py`) | `/_stcore/health` → ok; page → 200; no errors in the log |
| Web search tool (live) | 4 results with URLs and a retrieval timestamp |
| Finance tool (live) | AAPL quote from Yahoo Finance with a retrieval timestamp |

The remaining behaviours are covered by existing tests rather than re-run by hand:

- **PDF ingestion and RAG:** `tests/rag/*`, plus live `test_live_document_answer_is_cited`.
- **Video ingestion, retrieval and timestamp citations:** `tests/video/*`, `tests/integration/test_video_pipeline.py`,
  plus live `test_live_video.py`.
- **Memory and thread isolation:** `tests/integration/test_service.py`.
- **Human approval:** `tests/agent/test_graph.py::test_approval_*`.
- **Insufficient-evidence fallback:** `test_no_evidence_means_no_confident_answer`, plus 4/4 absent-answer
  smoke cases.

## Security status

- **Repository scan**: no API keys, tokens or private keys in the files being committed. The scan test
  (`tests/unit/test_security_scan.py`) covers key patterns, machine-specific paths and, by SHA-256, the
  upstream tutorial's leaked Alpha Vantage key.
- **Leaked upstream key**: the leaked key was removed from code in an earlier pass. On 2026-10-06 its literal
  value was also redacted from the docs and the test. It remains in the CampusX repository's public history,
  which is **not** pushed here: this repository's history starts from its own initial commit.
- **`.env`**: gitignored and not tracked; `.env.example` contains placeholders only (empty keys).
- **Unsafe code**: no `eval`/`exec`, no pickle loading in Nova, no `shell=True`. `subprocess` is used only by
  the evaluation runners (fixed argument lists) and a Windows TTS fixture builder.
- **Logs**: `logs/nova.log` checked for key-like strings: none. Log redaction is unit-tested.
- **Known residual risk**: prompt-injection defences are mitigations (delimiting, flagging, tools chosen
  outside retrieved text), not a guarantee. The UI has no authentication. The API key is optional.

## Evaluation status

- **Full runs (2026-10-05, `qwen3:8b`, CPU-only)**: 42-case documents/finance/web (baseline vs Nova) and 27-case
  video. Reports in `evaluation/results/`; per-case records in `evaluation/results/runs/*.jsonl` (now
  committed). Regenerated with `--report-only` on 2026-10-06: identical.
- **Caveat**: these full runs predate the 2026-10-06 claim-support check in the verifier.

- **POST-VERIFIER SMOKE EVALUATION (17 cases, 2026-10-06)**: enhanced agent only, on the categories
  `direct_document`, `citation`, `absent_answer` and `multi_tool`; records in
  `evaluation/results/smoke_post_verifier/`. The run was interrupted by a shutdown at 14/17. 2 harness crashes
  (Windows encoding) were fixed, and the run was resumed to 17/17 with no records deleted.

  | Metric | Same 17 cases, before change | Post-verifier smoke |
  |---|---|---|
  | Pass rate | 94.1% | 100% |
  | Tool selection | 94.1% | 100% |
  | Answer accuracy (15) | 100% | 100% |
  | Hit@1 / Hit@4 / MRR (10) | 90.0% / 100% / 0.933 | 90.0% / 100% / 0.933 |
  | Citation accuracy (10) | 100% | 100% |
  | Groundedness (lexical) | 86.7% (15 scored) | 75.0% (17 scored; 85.0% on the same 15) |
  | Abstention (4) | 100% | 100% |
  | Multi-tool success (4) | 75% | 100% |
  | Failure rate | 0% | 0% |
  | Avg / P50 / P95 latency | 79.2 / 70.7 / 295.1 s | 102.2 / 100.3 / 241.7 s |

  Verification statuses: 11 VERIFIED, 4 INSUFFICIENT_EVIDENCE (the absent-answer cases), 2 FAILED.
  - **cite-03 (FAILED):** a false positive of the new claim check on a provenance sentence. Fixed by excluding
    provenance words, with a regression test; a live re-run on the fixed code is `VERIFIED`
    (`evaluation/results/recheck_cite03/`).
  - **multi-01 (FAILED):** also `FAILED` in the pre-change run, so not caused by the new check; known limitation.
  - **Material effect on claims:** none of the headline full-run metrics are contradicted. The full-run figures
    still describe the code before the check and are labelled that way. The latency comparison is not
    controlled.

- **Metric coverage**: tool selection, answer accuracy, Hit@1, Hit@4, MRR, citation accuracy, groundedness
  (lexical proxy), abstention accuracy, multi-tool success, average/P50/P95 latency are all computed and reported.

## Deployment status

Not deployed. The Docker image has never been built (Docker is not installed on the development machine).
Runs locally on Windows 11 with Python 3.12 and Ollama.

## Documentation status

Rewritten README (architecture, flow and multimodal diagrams in Mermaid, measured results only, limitations,
attribution); `docs/architecture.md`, eight ADRs, `docs/SECURITY.md`, `docs/ENGINEERING_CONTRIBUTIONS.md`,
`docs/VIDEO_ASSISTANT_INTEGRATION.md`, `evaluation/README.md`. No screenshots exist and none were fabricated.
`docs/PROJECT_AUDIT.md` is the pre-upgrade audit and is kept as a historical record.

## GitHub status

- Remote `origin` = `https://github.com/risitasutar/nova-multimodal-agent.git`. The CampusX repository is kept
  locally as `upstream` (fetch only).
- One commit, `feat: complete Nova multimodal agent`, is built on top of the remote's existing
  `Initial commit`: no force push and no history rewrite. The local pre-existing upstream history is preserved
  on the local branch `upstream-tutorial`.
- The repository description and topics can't be set from this machine (GitHub CLI not installed); they are
  manual steps.

## Known limitations

- CPU-only latency: about 55 s average per LLM-backed answer in the document evaluation, about 163 s in the
  video evaluation.
- Small, synthetic evaluation sets; groundedness is a lexical proxy.
- Verifier claim check is lexical: it catches citations attached to unrelated passages, not subtle
  paraphrase errors.
- Weakest areas in the evaluation: action items (0/2), video+PDF comparison (1/3), adversarial (2/4),
  multi-hop documents (2/3).
- multi-01 (web + calculator: 'how many years before 2026 was the Eiffel Tower completed') answers correctly
  but ends verification `FAILED` both before and after the verifier change; the caveat is shown to the user.
- Sarvam, Mistral and hosted-LLM paths are implemented but not exercised live.
- No diarization, no UI authentication, single-node storage, Docker untested, not deployed.

## Resume Safe Claims

Each claim below is backed by code in this repository and by the tests or saved evaluation records named.

- Built a multimodal agent in LangGraph: a 10-node state machine with a query classifier (8 routes), a
  planner, tool execution, an evidence gate and a verifier, over PDF documents, meeting recordings, web
  search, market data and a safe calculator. *(graph tests; live e2e)*
- Integrated video/audio retrieval: PyAV audio extraction → faster-whisper transcription → timestamp-preserving
  chunks → FAISS → answers cited to time ranges (e.g. `[VIDEO: meeting.mp4, 12:23–14:06]`); verified end to
  end on a real spoken recording. *(`tests/e2e/test_live_video.py`)*
- Redesigned the tutorial's PDF RAG (real page numbers, embedding task prefixes, hybrid rerank, calibrated
  relevance threshold, pickle-free checksummed FAISS store). On a 42-case evaluation against the reproduced
  tutorial agent: end-to-end retrieval Hit@1 31.6% → 94.7%, citation correctness 0% → 94.7%, tool-selection
  accuracy 69.0% → 95.2% (single run, `qwen3:8b`, synthetic data). *(`evaluation/results/report.md`)*
- Designed a unified evidence model (page / time span / URL / symbol + retrieval time) with citations rendered
  only from stored locators, and a deterministic verifier: citation validity, claim-to-cited-source support,
  numeric checks against evidence, bounded retry, and explicit insufficient-evidence answers.
- Implemented human-in-the-loop approval for external tool calls with LangGraph `interrupt()`/resume, exposed
  in both the UI and the API.
- Implemented persistent conversation memory (SQLite checkpoints) with per-thread isolation of documents,
  recordings and evidence. *(integration tests)*
- Hardened the system: removed a hard-coded API key inherited from the tutorial, replaced pickle-based index
  loading, AST-only calculator, prompt-injection delimiting and flagging for PDFs, transcripts and web text,
  input/path validation, secret-redacting logs, and a repository secret-scan test.
- Exposed the agent through FastAPI and Streamlit sharing one service layer; 251 deterministic tests plus 5
  live end-to-end tests.
- Built an evaluation harness with per-case process isolation, hard timeouts, resumable incremental records
  and reports regenerated from saved records; 42-case document/finance/web set with a baseline, 27-case video
  set with temporal metrics (temporal Hit@1 86.4%, timestamp accuracy 90.9%).

## Claims Not Yet Safe

- "Production-ready", "deployed", "scalable", or "serves users": it is not deployed, the Docker image was never
  built, storage is single-node SQLite + local FAISS, and the UI has no authentication.
- "Eliminates hallucinations" or "guarantees grounded answers": the verifier is lexical and numeric; it can
  miss a wrong paraphrase of the right passage. The groundedness metric is a lexical proxy.
- "Prevents prompt injection": the defences are mitigations, tested on a handful of cases (adversarial
  category 2/4 in the full evaluation).
- Any Sarvam / Hinglish, Mistral or hosted-LLM claim beyond "implemented with mocked tests": none were run
  live.
- Speaker identification / diarization: not implemented.
- Any claim that the 42-case and 27-case full-run metrics describe the current verifier: those runs predate
  the claim-support check. The 17-case post-verifier smoke run is a small regression check, not a benchmark.
- Statistical claims beyond what is measured: each figure comes from a single run over small synthetic
  datasets.
- Latency claims such as "fast" or "real-time": answers average tens of seconds on CPU.
