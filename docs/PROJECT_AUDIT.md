# Nova — Project Audit (pre-upgrade)

Audit date: 2026-10-05. Scope: every tracked and untracked file in the repository
(excluding `.venv/`), git history, git remote, the local runtime (Ollama, Python
environment) and the on-disk data produced by the app.

## 1. Provenance

| Item | Finding |
|---|---|
| Remote | `origin = https://github.com/campusx-official/chatbot-in-langgraph.git` |
| History | 10 commits (2025-08-01 → 2025-11-22) by CampusX / Nitish — all upstream. No local commits. |
| License | The upstream repository publishes **no license** (GitHub API `license: null`). Upstream code is therefore not re-licensable by us. |
| Local work (uncommitted) | 5 modified upstream files (OpenAI → Ollama swap), 8 untracked files (`llm_config.py`, `streamlit_app_pro.py`, `smoke_test.py`, `run.bat`, `requirements_app.txt`, `.env.example`, `.gitignore`, `.streamlit/config.toml`). |

## 2. Current architecture

The repository is a staged LangGraph tutorial. Each stage is a backend + Streamlit
frontend pair; the most capable pair is `langraph_rag_backend.py` + `streamlit_app_pro.py`.

```
Streamlit (streamlit_app_pro.py)
   │  chatbot.stream(stream_mode="messages")
   ▼
LangGraph: START → chat_node ⇄ ToolNode (tools_condition) → END       (ReAct loop)
   │            │
   │            ├─ duckduckgo_search   (DuckDuckGoSearchRun, string output)
   │            ├─ get_stock_price     (Alpha Vantage GLOBAL_QUOTE)
   │            ├─ calculator          (AST evaluator)
   │            └─ rag_tool            (FAISS similarity k=4, per thread)
   ▼
SqliteSaver checkpointer (chatbot.db, WAL)      FAISS indexes (data/vectorstores/<thread>/, pickle)
   ▲
ChatOllama qwen3:8b  +  OllamaEmbeddings nomic-embed-text   (llm_config.py)
```

Tutorial stages still present: `langgraph_backend.py` (in-memory), `langgraph_database_backend.py`
(SQLite), `langgraph_tool_backend.py` (tools), `langgraph_mcp_backend.py` (MCP), and six
`streamlit_frontend*.py` / `streamlit_rag_frontend.py` UIs.

## 3. Existing capabilities (verified by reading code)

- Local LLM + embeddings through Ollama, configurable by `.env`; explicit `reasoning=` flag to stop Qwen3 `<think>` leakage.
- Single ReAct agent with 4 tools; the LLM alone decides tool use (no explicit router/planner).
- Persistent multi-thread conversations (`SqliteSaver`), thread listing sorted by recency, thread deletion.
- PDF upload → PyPDFLoader → RecursiveCharacterTextSplitter(1000/200) → FAISS → similarity top-4.
- Per-thread index persisted to disk and lazily reloaded.
- Ollama health check (`/api/tags`) with actionable "ollama pull" message.
- Streaming answers, tool status widget, chat export to Markdown, polished dark UI.
- `smoke_test.py`: 6 live end-to-end checks against the real model.

## 4. Weaknesses and technical debt

| # | Area | Finding |
|---|---|---|
| W1 | Agent design | No input validation, classification, planning, evidence check or verification — the LLM decides everything in one loop. Tool choice is not inspectable or testable. |
| W2 | Grounding | No citations. The model is never asked to attribute claims; nothing stops it answering a document question from parametric memory. |
| W3 | RAG | No relevance threshold: top-4 chunks are always returned even when irrelevant, so "absent" questions still get confident answers. |
| W4 | RAG | Page metadata is PyPDF's 0-based index and is shown to the model as-is → any page reference is off by one. |
| W5 | RAG | `nomic-embed-text` is used without its required task prefixes (`search_query:` / `search_document:`), degrading retrieval quality. |
| W6 | RAG | No query rewriting; follow-up questions ("what about its risks?") are embedded verbatim. |
| W7 | RAG | One index per thread; uploading a second PDF overwrites the first. |
| W8 | Tools | Web search returns one concatenated string with no URLs → sources cannot be cited. |
| W9 | Tools | Tool outputs are not normalised (different shapes, errors as free text). |
| W10 | Finance | Quote only; no history, returns, volatility, drawdown. Single provider with no fallback. |
| W11 | Errors | Exceptions surface as raw text (`⚠️ Something went wrong: <exception>`). |
| W12 | Config | Settings scattered (`llm_config.py`, module constants, hard-coded `chatbot.db`, region, k=4, chunk sizes). |
| W13 | Performance | `retrieve_all_threads()` scans every checkpoint on each page load (O(checkpoints)). |
| W14 | Code | Six near-duplicate tutorial frontends/backends at repo root; `requirements.txt` is a UTF-16 pip-freeze of the original OpenAI stack and does not match what the app imports. |
| W15 | Ops | No API, no structured logging, no metrics, no tracing, no Docker, no README. |

## 5. Security issues

| # | Severity | Finding |
|---|---|---|
| S1 | High | Alpha Vantage key (value redacted; it is an upstream free-tier key) hard-coded as a fallback in `langraph_rag_backend.py`, `langgraph_tool_backend.py`, `langgraph_mcp_backend.py`. It is also present in **upstream git history** (commits by CampusX), i.e. publicly exposed. It belongs to the upstream author, not to this project; it must be removed from code and must not be relied on. History is not rewritten (remote history belongs to upstream). |
| S2 | High | `FAISS.load_local(..., allow_dangerous_deserialization=True)` unpickles `index.pkl` from disk with no integrity or ownership check — anyone able to write into `data/vectorstores/` gets code execution. |
| S3 | Medium | Retrieved PDF text and web snippets are concatenated into the prompt with no data/instruction boundary → document prompt-injection. |
| S4 | Medium | No limits on message size, PDF size, page count or chunk count (memory/CPU exhaustion). |
| S5 | Medium | `langgraph_mcp_backend.py` hard-codes `an absolute path on the upstream author's machine` and a third-party public MCP URL; MCP failures are silently swallowed (`except Exception: return []`). `langchain_mcp_adapters` is not even installed, so the module cannot import. |
| S6 | Low | Upload filename rendered into HTML with `unsafe_allow_html=True` (HTML injection in sidebar). |
| S7 | Low | `.env` and `.env.example` are identical; `.gitignore` correctly ignores `.env`, `chatbot.db*`, `data/`. |

## 6. Testing gaps

- Only test: `smoke_test.py` — requires a live Ollama and internet; non-deterministic; no assertions on citations, isolation, deletion or errors.
- No unit tests (calculator, routing, retrieval, citations), no API tests, no thread/document isolation tests, no malformed-PDF tests, no evaluation dataset or metrics.

## 7. Runtime observations

- Ollama reachable with `qwen3:8b`, `qwen2.5:7b`, `llama3.2`, `nomic-embed-text`.
- No NVIDIA GPU: a structured `qwen3:8b` call takes ~5–9 s warm, ~27 s cold. **Every extra LLM call costs seconds**, which drives the target design (deterministic steps wherever an LLM adds nothing).
- Alpha Vantage `demo` key is rejected for real symbols; the Yahoo Finance chart endpoint answers without a key.
- Docker is not installed on this machine (Docker files can be written, not built here).

## 8. Target architecture (summary — implemented in this upgrade)

```
validate_input → classify_query ─┬─ general/clarify ───────────────────────────► generate_response
                                 └─ tools ─► planner ─► approval_gate (interrupt) ─► execute_tools
execute_tools ─► compute (calculator over gathered data) ─► check_evidence ─┬─ insufficient ─► finalize
                                                                            └─► generate_response
generate_response ─► verify_response ─┬─ RETRY (bounded) ─► generate_response
                                      └─► finalize_response (validated citations, metadata)
```

- `nova/` package: `config` (one Settings object), `llm`, `errors`, `observability`, `tools` (normalised `ToolResult`), `tools/finance` (providers + analytics), `rag` (ingest/store/retrieve/citations), `agent` (state, classifier, planner, nodes, verifier, graph, baseline), `memory`, `service` (single façade used by UI **and** API).
- FAISS indexes stored as raw `index.faiss` + JSON chunk store + SHA-256 manifest — **no pickle**.
- FastAPI app, Streamlit UI, evaluation harness (baseline vs enhanced), deterministic test pyramid, Docker, docs/ADRs.
