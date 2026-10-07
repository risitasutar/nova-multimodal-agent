"""
BASELINE agent — a faithful reproduction of the pre-upgrade `langraph_rag_backend.py`
(single ReAct loop: chat_node ⇄ ToolNode), kept ONLY for the baseline-vs-enhanced
evaluation (evaluation/baseline_runner.py).

Deliberately unchanged from the original: system prompt, tool set and docstrings,
PyPDFLoader + RecursiveCharacterTextSplitter(1000/200), LangChain FAISS L2 similarity
top-4 without nomic task prefixes or relevance threshold, 0-based page metadata, no
citations, no verification.

Changes required for safety/fairness only (documented in evaluation/README.md):
  * the hard-coded Alpha Vantage key is gone — the stock tool calls Nova's market-data
    provider layer, so both systems see the same data source;
  * indexes are kept in memory (no pickle written to disk).
"""

from __future__ import annotations

import ast
import operator
import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from nova.tools.base import ToolError
from nova.tools.finance.providers import get_provider, normalize_symbol

BASELINE_SYSTEM_PROMPT = (
    "You are Nova, a helpful assistant with tools.\n"
    "- For questions about the user's uploaded PDF, call `rag_tool`.\n"
    "- For current events or facts you are unsure about, call `duckduckgo_search`.\n"
    "- For stock quotes, call `get_stock_price` with the ticker symbol.\n"
    "- For ANY arithmetic, call `calculator` with the full expression in one call. "
    "Never do arithmetic in your head.\n"
    "- For greetings and general questions, answer directly without tools.\n"
    "Report tool results faithfully. Use markdown formatting where helpful."
)

_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> Any:
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


class BaselineState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class BaselineAgent:
    def __init__(self, llm: BaseChatModel, embeddings: Embeddings) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self._retrievers: Dict[str, Any] = {}
        self._metadata: Dict[str, dict] = {}

    # ---- ingestion (original pipeline) ----
    def ingest_pdf(self, file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        try:
            docs = PyPDFLoader(path).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
                                                      separators=["\n\n", "\n", " ", ""])
            chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
            store = FAISS.from_documents(chunks, self.embeddings)
            self._retrievers[str(thread_id)] = store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            self._metadata[str(thread_id)] = {"filename": filename, "documents": len(docs), "chunks": len(chunks)}
            return self._metadata[str(thread_id)]
        finally:
            os.remove(path)

    # ---- graph (original ReAct loop) ----
    def build(self, checkpointer: BaseCheckpointSaver) -> Any:
        agent = self
        search_tool = DuckDuckGoSearchRun(region="us-en")  # type: ignore[call-arg]  # verbatim from the original

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
            except Exception as e:  # noqa: BLE001 - original behaviour
                return {"error": f"Could not evaluate '{expression}': {e}"}

        @tool
        def get_stock_price(symbol: str) -> dict:
            """
            Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA', 'NVDA').
            """
            try:
                q = get_provider().quote(normalize_symbol(symbol))
            except ToolError as e:
                return {"error": e.user_message}
            return {"symbol": q.symbol, "price": q.price, "change": q.change,
                    "change_percent": q.change_percent, "latest_trading_day": q.as_of}

        @tool
        def rag_tool(query: str, config: RunnableConfig) -> dict:
            """
            Retrieve relevant passages from the PDF the user uploaded in this chat.
            Use this for any question about the user's document.
            """
            thread_id = str(config.get("configurable", {}).get("thread_id"))
            retriever = agent._retrievers.get(thread_id)
            if retriever is None:
                return {"error": "No document indexed for this chat. Ask the user to upload a PDF.", "query": query}
            result = retriever.invoke(query)
            return {
                "query": query,
                "context": [doc.page_content for doc in result],
                "pages": [doc.metadata.get("page") for doc in result],
                "source_file": agent._metadata.get(thread_id, {}).get("filename"),
            }

        tools = [search_tool, get_stock_price, calculator, rag_tool]
        llm_with_tools = self.llm.bind_tools(tools)

        def chat_node(state: BaselineState, config: RunnableConfig) -> dict:
            messages = [SystemMessage(content=BASELINE_SYSTEM_PROMPT), *state["messages"]]
            return {"messages": [llm_with_tools.invoke(messages, config=config)]}

        graph = StateGraph(BaselineState)
        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")
        return graph.compile(checkpointer=checkpointer)
