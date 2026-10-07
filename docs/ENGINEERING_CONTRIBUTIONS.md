# Engineering contributions

This document separates **what came from source projects** from **what was engineered in this repository**.
Everything listed under "Built here" exists in the code and is covered by tests or evaluation as noted.

## Source projects (not claimed as original work)

| Source | What it provided | Where it lives now |
|---|---|---|
| CampusX `chatbot-in-langgraph` (no license) | Staged LangGraph tutorial: in-memory chat graph, streaming, threads, SQLite checkpointer, ReAct tool loop (DuckDuckGo, Alpha Vantage, calculator), MCP client, FAISS PDF RAG, simple Streamlit UIs | `examples/upstream_tutorial/` (reference only); the RAG stage is reproduced as the evaluation baseline in `nova/agent/baseline.py` |
| `risitasutar/AI-Video-Assistant` (MIT per its README) | Whisper/Sarvam transcription routing (Sarvam endpoint, model variable, 25 s pieces), yt-dlp YouTube ingestion, map-reduce meeting summary, action items / key decisions / open questions prompts, Chroma transcript Q&A | Reused and refactored into `nova/video/` — see `docs/VIDEO_ASSISTANT_INTEGRATION.md` and `THIRD_PARTY_NOTICES.md` |

## Built here

### Agent architecture (`nova/agent/`)
- Explicit LangGraph state machine (validate → classify → plan → approval gate → execute → compute → evidence gate → generate → verify → finalize) replacing the free-running ReAct loop; typed per-turn state reset; bounded retries.
- **Router** (`classifier.py`): deterministic fast paths, structured LLM classification over 8 routes (incl. `video`), heuristic fallback, policy guards (time-sensitive → live data, explicit document/media references, finance without ticker → clarification), retrieval probes over documents and transcripts.
- **Planner** (`planner.py`): deterministic plans per capability, validated LLM plans for multi-tool requests, binding of router evidence requirements (a model plan cannot drop a referenced source).
- **Verifier**: citation-id validation, claim-to-cited-source lexical support check (calibrated on saved evaluation answers), number checks against evidence/calculator output (timestamps excluded), abstention detection, bounded retry, explicit caveat.
- **Compute step**: calculator over retrieved evidence with a gate rejecting numbers that are not in the evidence.
- **Human-in-the-loop**: LangGraph `interrupt()`/resume approval for external calls.

### Retrieval and evidence
- Document RAG (`nova/rag/`): per-page chunking with real page numbers, nomic task prefixes, multi-query + hybrid rerank, calibrated relevance threshold, corrective retrieval, pickle-free FAISS store with SHA-256 manifests and thread ownership checks.
- **Unified evidence model** (`nova/evidence.py`): one schema (kind, source type, source id/name, content, locator, actual relevance) across PDF, video/audio, web and finance.
- **Unified citations** (`nova/rag/citations.py`): `[PDF: x, p.n]`, `[VIDEO: x, mm:ss–mm:ss]`, `[AUDIO: …]`, `[WEB: …]`, `[FINANCE: SYM, provider, retrieved …]` rendered only from stored locators; model-written citations (incl. time ranges) validated against what was actually retrieved.

### Video/audio integration (`nova/video/`)
- Backend service with job status (QUEUED/PROCESSING/COMPLETED/FAILED), per-step progress, crash recovery of interrupted jobs.
- Timestamp preservation end-to-end (the source project discarded Whisper segment times): segments → time-bounded chunks (`VIDEO_CHUNK_MAX_SECONDS`) → vectors → evidence → citations.
- PyAV audio extraction (no FFmpeg binary), media validation (extension, container probe, size, duration), hardened YouTube ingestion (host allow-list, pre-download duration check, fixed filenames).
- faster-whisper default backend with the original openai-whisper selectable; Sarvam pieces mapped to real offsets; SRT/VTT import.
- Content-addressed artifact cache (audio, transcript, embeddings, insights keyed by content hash + configuration) with reference-counted deletion.
- Meeting intelligence via Nova's LLM abstraction (Mistral optional), one structured pass per window, **verification of every item against cited transcript chunks** (unsupported items dropped; owners/deadlines not in the transcript → "Not identified in the transcript").
- Single vector store: transcripts reuse Nova's FAISS store (per-thread, per-media collections) instead of a second database; the source project's global Chroma collection (no isolation) was retired.
- Agent tools `video_search` (thread/media scope from run config, never from model input) and `media_insights`; multi-recording disambiguation and conversational memory of the active recording.
- Cross-modal reasoning: video + document + calculator + finance plans, prompt structure that separates **Video evidence** from **Document evidence**.

### Product surfaces
- `NovaService` façade shared by the FastAPI API and the Streamlit UI (no duplicated business logic).
- FastAPI: chat, approval, documents, threads, media (upload, YouTube, process with background jobs, status, summary, search, delete), health/readiness, metrics, optional API key, typed errors.
- Streamlit: streaming answers, execution timeline, source cards, verification/evidence badges, finance panels, approval card, media upload/processing status, meeting-intelligence panels, ▶ timestamp seeking with the real player (`st.video`/`st.audio` `start_time`), evaluation dashboard.

### Quality, security, operations
- Test pyramid: deterministic unit/RAG/agent/video/integration/API/UI tests with fakes (no network, no Ollama), opt-in live e2e tests.
- Evaluation harnesses: 42-case document/finance/web golden set with a reproduced baseline; 27-case video golden set with temporal metrics; per-case process isolation with hard timeouts and resumable incremental saving; threshold calibration on dev sets disjoint from test sets.
- Security: removal of the hard-coded key, pickle-free indexes, untrusted-source delimiting and injection flagging (documents, web **and transcripts**), input/file/media limits, secret-redacting structured logs, path containment for all media/index paths.
- Observability, centralised configuration, Docker/Compose, ADRs and architecture docs.
