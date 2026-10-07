"""
Multi-tool LangGraph chatbot backend running fully on local Ollama models.

Features
  • Chat model + embeddings served by Ollama (see llm_config.py / .env)
  • Tools: web search, stock price, calculator, PDF retrieval (RAG)
  • Conversations persisted in SQLite (tutorial_chatbot.db)
  • PDF indexes persisted per thread on disk (data/vectorstores/<thread_id>)
"""

from __future__ import annotations

import ast
import json
import operator
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_MODEL,
    get_embeddings,
    get_llm,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "data" / "vectorstores"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "tutorial_chatbot.db"

# -------------------
# 1. LLM + embeddings
# -------------------
llm = get_llm()
embeddings = get_embeddings()


def check_ollama() -> tuple[bool, str]:
    """Verify the Ollama server is up and both required models are pulled."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException:
        return False, (
            f"Can't reach Ollama at {OLLAMA_BASE_URL}. "
            "Start the Ollama app (or run `ollama serve`) and refresh."
        )

    installed = {m["name"] for m in resp.json().get("models", [])}
    installed |= {name.split(":")[0] for name in installed if name.endswith(":latest")}
    missing = [m for m in (OLLAMA_MODEL, OLLAMA_EMBED_MODEL) if m not in installed]
    if missing:
        cmds = " && ".join(f"ollama pull {m}" for m in missing)
        return False, f"Missing Ollama model(s): {', '.join(missing)}. Run: {cmds}"
    return True, f"{OLLAMA_MODEL} ready"


# -------------------
# 2. PDF retriever store (per thread, persisted to disk)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _thread_dir(thread_id: str) -> Path:
    return VECTOR_DIR / str(thread_id)


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread, loading it from disk if needed."""
    if not thread_id:
        return None
    thread_id = str(thread_id)
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    path = _thread_dir(thread_id)
    if (path / "index.faiss").exists():
        store = FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )
        _THREAD_RETRIEVERS[thread_id] = store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Build a FAISS index for the uploaded PDF, save it, and attach it to the thread."""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    thread_id = str(thread_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        docs = PyPDFLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            raise ValueError(
                "No text found in this PDF (it may be a scanned image). "
                "Try a PDF with selectable text."
            )

        vector_store = FAISS.from_documents(chunks, embeddings)
        path = _thread_dir(thread_id)
        path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(path))

        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        (path / "meta.json").write_text(json.dumps(summary), encoding="utf-8")

        _THREAD_RETRIEVERS[thread_id] = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        _THREAD_METADATA[thread_id] = summary
        return summary
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed")


@tool
def calculator(expression: str) -> dict:
    """
    Evaluate an arithmetic expression and return the exact result.
    Pass the WHOLE calculation as one expression, e.g. "1849 * 37 / 4" or "(2 + 3) ** 2".
    Supports + - * / // % ** and parentheses.
    """
    try:
        expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
        result = _safe_eval(ast.parse(expr, mode="eval"))
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return {"expression": expression, "result": result}
    except ZeroDivisionError:
        return {"error": "Division by zero is not allowed"}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA', 'NVDA').
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey={api_key}"
    )
    try:
        data = requests.get(url, timeout=15).json()
    except requests.RequestException as e:
        return {"error": f"Stock API request failed: {e}"}

    quote = data.get("Global Quote")
    if not quote:
        return {"error": data.get("Note") or data.get("Information") or f"No data for {symbol}"}
    return {
        "symbol": quote.get("01. symbol"),
        "price": quote.get("05. price"),
        "change": quote.get("09. change"),
        "change_percent": quote.get("10. change percent"),
        "latest_trading_day": quote.get("07. latest trading day"),
    }


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant passages from the PDF the user uploaded in this chat.
    Use this for any question about the user's document.
    """
    # thread_id is injected from the run config, so the model never has to pass it.
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Ask the user to upload a PDF.",
            "query": query,
        }

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "pages": [doc.metadata.get("page") for doc in result],
        "source_file": thread_document_metadata(str(thread_id)).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
SYSTEM_PROMPT = (
    "You are Nova, a helpful assistant with tools.\n"
    "- For questions about the user's uploaded PDF, call `rag_tool`.\n"
    "- For current events or facts you are unsure about, call `duckduckgo_search`.\n"
    "- For stock quotes, call `get_stock_price` with the ticker symbol.\n"
    "- For ANY arithmetic, call `calculator` with the full expression in one call. "
    "Never do arithmetic in your head.\n"
    "- For greetings and general questions, answer directly without tools.\n"
    "Report tool results faithfully. Use markdown formatting where helpful."
)


def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads() -> list[str]:
    """All thread ids, most recently active first."""
    latest: Dict[str, str] = {}
    for checkpoint in checkpointer.list(None):
        tid = str(checkpoint.config["configurable"]["thread_id"])
        ts = checkpoint.checkpoint.get("ts", "")
        if ts > latest.get(tid, ""):
            latest[tid] = ts
    return sorted(latest, key=latest.get, reverse=True)


def delete_thread(thread_id: str) -> None:
    """Remove a conversation and its PDF index."""
    thread_id = str(thread_id)
    checkpointer.delete_thread(thread_id)
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)
    shutil.rmtree(_thread_dir(thread_id), ignore_errors=True)


def thread_has_document(thread_id: str) -> bool:
    return _get_retriever(thread_id) is not None


def thread_document_metadata(thread_id: str) -> dict:
    thread_id = str(thread_id)
    if thread_id not in _THREAD_METADATA:
        meta_file = _thread_dir(thread_id) / "meta.json"
        if not meta_file.exists():
            return {}
        _THREAD_METADATA[thread_id] = json.loads(meta_file.read_text(encoding="utf-8"))
    return _THREAD_METADATA[thread_id]
