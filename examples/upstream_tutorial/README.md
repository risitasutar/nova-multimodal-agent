# Upstream tutorial stages (CampusX)

These files are the staged learning path from the CampusX
[`chatbot-in-langgraph`](https://github.com/campusx-official/chatbot-in-langgraph) repository
that Nova grew out of. They are kept for attribution and reference. **They are not part of the
Nova product** (`nova/`, `api/`, `streamlit_app_pro.py`).

| Stage | Backend | Frontend | Concept |
|---|---|---|---|
| 1 | `langgraph_backend.py` | `streamlit_frontend.py` | single-node graph, in-memory checkpointer |
| 2 | `langgraph_backend.py` | `streamlit_frontend_streaming.py` | token streaming |
| 3 | `langgraph_tool_backend.py` | `streamlit_frontend_threading.py` | multiple threads |
| 4 | `langgraph_database_backend.py` | `streamlit_frontend_database.py` | SQLite persistence |
| 5 | `langgraph_tool_backend.py` | `streamlit_frontend_tool.py` | ReAct tool loop |
| 6 | `langgraph_mcp_backend.py` | `streamlit_frontend_mcp.py` | MCP tools (optional) |
| 7 | `langraph_rag_backend.py` | `streamlit_rag_frontend.py` | PDF RAG (pre-upgrade version) |

Changes made here during the Nova upgrade (behaviour otherwise unchanged):

- Models come from `nova.config` via the `llm_config.py` shim (OpenAI → local Ollama).
- The hard-coded Alpha Vantage key was removed; set `ALPHAVANTAGE_API_KEY` if you want stock quotes.
- The MCP stage no longer hard-codes a personal path or a third-party server: configure
  `NOVA_MCP_MATH_SERVER` (path to a stdio MCP server script) and/or `NOVA_MCP_EXPENSE_URL`,
  and `pip install -r requirements-optional.txt`. Failures are logged, not silently swallowed.
- Tutorial databases are written next to these files (`tutorial_chatbot.db`, git-ignored),
  separate from Nova's `data/nova.db`.

Stage 7 still uses LangChain's pickle-based `FAISS.load_local(..., allow_dangerous_deserialization=True)`.
Nova replaced this with a pickle-free, checksummed store (see `docs/adr/ADR-002-faiss-vector-store.md`);
do not point this stage at index files you did not create yourself.

Run from the repository root, e.g.:

```bash
python -m streamlit run examples/upstream_tutorial/streamlit_frontend_tool.py
```
