# Nova architecture — multimodal agent

This document describes the runtime architecture of Nova. The design rationale is in [`adr/`](adr/), and
the measured behaviour is in [`../evaluation/results/report.md`](../evaluation/results/report.md).

## 1. High-level architecture

```mermaid
flowchart LR
    U[User] --> UI[Streamlit UI<br/>streamlit_app_pro.py]
    C[API client] --> API[FastAPI<br/>api/main.py]
    UI --> SVC[NovaService<br/>nova/service.py]
    API --> SVC
    SVC --> G[LangGraph agent<br/>nova/agent/graph.py]
    G --> LLM[(LLM endpoint<br/>Ollama qwen3:8b<br/>or OpenAI-compatible)]
    G --> RAG[RAG<br/>nova/rag]
    G --> T[Tools<br/>nova/tools]
    G --> VID[Video/audio<br/>nova/video]
    SVC --> VID
    VID --> MVS[(FAISS transcript indexes<br/>data/media_vectors)]
    VID --> MC[(Artifact cache<br/>audio · transcript · vectors · insights)]
    VID --> STT[(Whisper local<br/>Sarvam API)]
    RAG --> VS[(FAISS indexes<br/>data/vectorstores)]
    RAG --> EMB[(Embeddings<br/>nomic-embed-text)]
    VID --> EMB
    T --> WEB[(DuckDuckGo)]
    T --> MKT[(Alpha Vantage / Yahoo)]
    SVC --> DB[(SQLite<br/>checkpoints + thread registry)]
    G -. optional .-> LS[(LangSmith)]
    SVC --> OBS[JSON logs + /metrics]
```

The UI and the API are thin transports. **Both call the same `NovaService`, which wraps the same compiled
graph.** No agent logic is duplicated between them.

## 2. LangGraph flow

```mermaid
stateDiagram-v2
    [*] --> validate_input
    validate_input --> finalize_response: invalid input
    validate_input --> classify_query
    classify_query --> generate_response: general
    classify_query --> finalize_response: clarification_required
    classify_query --> planner: document / video / web / finance / calculation / multi_tool
    planner --> finalize_response: document requested, none uploaded
    planner --> approval_gate: external tools and approval required
    planner --> execute_tools
    approval_gate --> execute_tools: approved (or local steps remain)
    approval_gate --> finalize_response: rejected, nothing local left
    execute_tools --> compute
    compute --> check_evidence
    check_evidence --> finalize_response: insufficient evidence / all tools failed
    check_evidence --> generate_response
    generate_response --> verify_response
    verify_response --> generate_response: RETRY (bounded)
    verify_response --> finalize_response: VERIFIED / INSUFFICIENT_EVIDENCE / FAILED
    finalize_response --> [*]
```

| Node | Calls the LLM? | Responsibility |
|---|---|---|
| `validate_input` | no | Normalise and size-check the message; reset per-turn state. |
| `classify_query` | only if ambiguous | Use rule fast paths, then structured output (`QueryClassification`), then policy guards and a retrieval probe. |
| `planner` | only for `multi_tool` | Build a deterministic plan from the classifier's entities, or a validated LLM plan (`Plan`). |
| `approval_gate` | no | Call LangGraph `interrupt()` with the external operations, then resume or prune them. |
| `execute_tools` | no | Run the plan in order with per-tool timeouts and normalised `ToolResult`s; document and **temporal video** retrieval, stored meeting insights; build the unified evidence registry. |
| `compute` | only if needed | Extract calculations from the evidence, reject invented numbers, run the AST calculator. |
| `check_evidence` | no | Apply the relevance gate, run corrective retrieval, score evidence, fall back gracefully when tools fail. |
| `generate_response` | yes, except pure arithmetic | Write a grounded answer that cites `[S#]`, `[W#]` and `[F#]` ids (streamed to the UI). |
| `verify_response` | no | Validate citations, check numbers, detect abstention, decide on a bounded retry. |
| `finalize_response` | no | Render the final answer and attach metadata (route, plan, tools, citations, timeline, latency, finance panels). |

## 3. RAG pipeline

```mermaid
flowchart LR
    PDF[PDF upload] --> V{validate<br/>type · size · pages}
    V --> P[parse pypdf]
    P --> CL[clean<br/>de-hyphenate · strip page furniture]
    CL --> CH[chunk per page<br/>600/100 chars]
    CH --> MD[metadata<br/>document_id · filename · page · chunk_id · injection flags]
    MD --> E[embed<br/>search_document: prefix]
    E --> IDX[(index.faiss + chunks.json<br/>+ manifest SHA-256)]
    Q[question] --> RW[rewrite<br/>classifier document_query]
    RW --> R[dense search<br/>raw + rewritten, merged]
    IDX --> R
    R --> RR[hybrid rerank<br/>0.75 cosine + 0.25 lexical]
    RR --> F{relevance ≥ 0.56?}
    F -- no --> CR[corrective keyword retrieval] --> F2{found?}
    F2 -- no --> INS[insufficient-evidence answer]
    F -- yes --> CTX[context assembly<br/>untrusted source blocks S1..S4]
    F2 -- yes --> CTX
    CTX --> GEN[grounded generation]
    GEN --> VER[citation + number verification]
    VER --> ANS[answer with validated page citations]
```

## 4. Memory architecture

```mermaid
flowchart TB
    subgraph SQLite [data/nova.db · WAL]
        CP[LangGraph checkpoints<br/>thread_id → AgentState]
        REG[threads registry<br/>title · created · updated · turns<br/>idx_threads_updated]
    end
    subgraph Disk [data/vectorstores]
        T1[thread A / doc 1..n]
        T2[thread B / doc 1..n]
    end
    SVC[NovaService] -->|per-thread lock| CP
    SVC --> REG
    SVC --> Disk
    DEL[DELETE /threads/id] --> CP & REG & Disk
```

- **Conversation memory:** the `messages` list in the checkpoint, the only field that accumulates across turns.
  Before each LLM call it is filtered to plain user and assistant text and truncated
  (`NOVA_HISTORY_TURNS`, `NOVA_HISTORY_MESSAGE_CHARS`).
- **Per-turn working state:** plan, tool results, sources, verification and so on. `validate_input` resets it
  every turn, so nothing stale carries over.
- **Isolation:** checkpoints are keyed by `thread_id`, and vector indexes live under `<thread_id>/` with
  manifest ownership checks. The tests prove thread A cannot retrieve thread B's documents.
- **Restart persistence:** everything is on disk. The legacy `chatbot.db` is migrated once, without
  destroying it.

## 5. API architecture

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant S as NovaService
    participant G as LangGraph
    C->>A: POST /chat {thread_id, message, approval_required}
    A->>A: request-id middleware · optional X-API-Key
    A->>S: chat() in threadpool
    S->>G: stream(updates, messages)
    alt approval required
        G-->>S: interrupt(operations)
        S-->>A: status=awaiting_approval
        A-->>C: 200 {approval}
        C->>A: POST /chat/{id}/approval {approved}
        A->>S: resume()
        S->>G: Command(resume=...)
    end
    G-->>S: final state
    S-->>A: TurnResult
    A-->>C: 200 {answer, citations, tools_used, latency_ms, ...}
```

Errors are typed (`nova/errors.py`) and mapped to HTTP codes with safe messages: 401, 404, 409, 413, 415, 422
and 503. Tracebacks go to the logs only.

## 6. Deployment architecture

```mermaid
flowchart LR
    subgraph Host / VM
        subgraph nova-api [container: nova api :8000]
            API2[uvicorn api.main:app]
        end
        subgraph nova-ui [container: nova ui :8501]
            UI2[streamlit]
        end
        VOL[(volume nova-data<br/>SQLite + indexes + logs)]
        API2 --- VOL
        UI2 --- VOL
    end
    API2 --> LLM2[(LLM endpoint<br/>Ollama on host / ollama container /<br/>hosted OpenAI-compatible)]
    UI2 --> LLM2
```

The application image never contains a model. The LLM endpoint is configuration (`OLLAMA_BASE_URL` or
`NOVA_LLM_PROVIDER=openai_compatible`). The `/health` (liveness) and `/ready` (LLM and database) endpoints
drive orchestration health checks.

## 7. Evaluation pipeline

```mermaid
flowchart LR
    DS[dataset.json<br/>42 cases · 12 categories] --> BR[baseline_runner<br/>original ReAct agent]
    DS --> ER[enhanced_runner<br/>NovaService]
    FX[fixtures<br/>fictional report + injection PDF] --> BR & ER
    BR --> RUNS[(results/runs/*.jsonl)]
    ER --> RUNS
    DS --> RB[retrieval component benchmark]
    RUNS --> GR[evaluator.grade<br/>deterministic metrics]
    GR --> AG[aggregate]
    RB --> AG
    AG --> OUT[latest.json · results.csv · report.md]
    OUT --> DB2[Streamlit evaluation page]
```


## 8. Video / audio ingestion

```mermaid
flowchart LR
    IN[upload MP4/MOV/AVI/MKV/WEBM/MP3/WAV<br/>· YouTube URL · SRT/VTT] --> V{validate<br/>extension · size · container probe · duration · URL allow-list}
    V --> Q[MediaAsset QUEUED<br/>SQLite media_assets]
    Q --> P[PROCESSING]
    P --> C1{cached transcript<br/>for this content + engine?}
    C1 -- yes --> SEG
    C1 -- no --> A[PyAV → 16 kHz mono WAV<br/>cached]
    A --> TR{language}
    TR -- english / auto --> W[Whisper<br/>faster-whisper default]
    TR -- hinglish --> SV[Sarvam STT-translate<br/>25 s pieces]
    W --> SEG[timestamped segments<br/>segment_id · start · end · text · language · speaker=null]
    SV --> SEG
    SEG --> CH[time-bounded chunks<br/>≤500 chars · ≤120 s · 1-segment overlap]
    CH --> EM[embeddings<br/>cached .npy]
    EM --> IX[(per-thread FAISS collection<br/>media_vectors/thread/media)]
    CH --> MI[meeting intelligence<br/>1 structured LLM pass per window]
    MI --> VF[verify items vs cited chunks<br/>drop unsupported · blank owners/deadlines]
    VF --> DONE[COMPLETED<br/>steps · timings · counts]
```

Every step updates `MediaAsset.steps` (`done` / `cached` / `skipped`) and the UI/API progress. A job
interrupted by a crash is marked FAILED on the next start. API processing runs in a FastAPI background task;
the design does not need a broker for a single node.

## 9. Temporal RAG

```mermaid
sequenceDiagram
    participant G as execute_tools
    participant M as MediaService.search
    participant R as Retriever (shared code)
    participant S as FAISS media store
    G->>M: query + rewrite, thread_id (from run config), media_ids (from router)
    M->>R: retrieve(thread, queries, document_ids=media_ids)
    R->>S: cosine top-k over this thread's collections only
    S-->>R: chunks {media_id, start_time, end_time, source_name, text}
    R-->>M: hybrid-reranked, relevance ≥ NOVA_VIDEO_MIN_RELEVANCE
    M-->>G: chunks with actual scores and time spans
    G->>G: Evidence(kind=video, locator={start_time, end_time})
```

Timestamps are carried in chunk metadata from transcription to citation; nothing interpolates or guesses a
time. Retrieval scope: the store only loads collections whose manifest names the requesting thread; `media_id`
filters come from the router (filename match, "all meetings", or the conversation's active recording).

## 10. Cross-modal reasoning

```mermaid
flowchart TB
    Q["Is the Q4 target in the meeting consistent with the strategy PDF?"] --> CL[router: multi_tool<br/>media_query + document_query + needs_calculation]
    CL --> PL[planner: video_search + document_search<br/>router requirements are binding]
    PL --> VS[video_search → V1..Vn<br/>with time spans]
    PL --> DS[document_search → S1..Sn<br/>with pages]
    VS --> CMP[compute: calculator over evidenced numbers<br/>12 − 10 · 2 / 10 × 100]
    DS --> CMP
    CMP --> GEN[generation: Video evidence / Document evidence / Comparison]
    GEN --> VER[verifier: ids valid · numbers evidenced]
    VER --> OUT["…10 crore [VIDEO: meeting.mp4, 12:23–14:06] … 12 crore [PDF: strategy.pdf, p.1] … 20% higher"]
```

## 11. Unified evidence and citations

```mermaid
classDiagram
    class Evidence {
        id: S#|V#|W#|F#
        kind: document|video|web|finance
        source_type: pdf|video|audio|youtube|web|finance
        source_id
        source_name
        content
        locator
        relevance  (actual score or null)
        metadata
    }
    class PDF { locator: page }
    class Video { locator: start_time, end_time }
    class Web { locator: url }
    class Finance { locator: symbol, retrieved_at }
    Evidence <|-- PDF
    Evidence <|-- Video
    Evidence <|-- Web
    Evidence <|-- Finance
```

| Kind | Rendered citation (from the locator) |
|---|---|
| PDF | `[PDF: strategy_2026.pdf, p.1]` |
| Video / YouTube | `[VIDEO: quarterly_review_meeting.mp4, 12:23–14:06]` (H:MM:SS for ≥ 1 h) |
| Audio | `[AUDIO: vendor_call.mp3, 00:18–00:44]` |
| Web | `[WEB: [domain](url)]` |
| Finance | `[FINANCE: AAPL, Yahoo Finance, retrieved 2026-10-05T…]` |

## 12. Video evaluation

```mermaid
flowchart LR
    VD[video_dataset.json<br/>27 cases · 14 categories] --> W[per-case worker process<br/>hard timeout]
    FX[transcript fixtures + strategy PDF] --> W
    W --> SVC2[NovaService<br/>real LLM · real retrieval]
    SVC2 --> RUN[(runs/video.jsonl)]
    RUN --> GR[grade: routing · Hit@k/MRR on gold time spans ·<br/>timestamp accuracy · citations · answers · groundedness]
    GR --> OUT2[video_latest.json · video_results.csv · video_report.md]
```
