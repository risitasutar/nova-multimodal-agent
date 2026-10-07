"""
Deterministic test doubles: a scripted chat model and lexical hashing embeddings.

They let the whole LangGraph agent run in unit/integration tests with no Ollama and
no network, while still exercising the real graph, retrieval, citations and verifier.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

StructuredFn = Callable[[type[BaseModel], list[BaseMessage]], Any]
AnswerFn = Callable[[list[BaseMessage]], str]


class ScriptedChatModel(BaseChatModel):
    """`answer_fn(messages) -> str` for free text; `structured_fn(schema, messages)` for JSON."""

    answer_fn: Any = None
    structured_fn: Any = None
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _text(self, messages: list[BaseMessage]) -> str:
        self.calls.append(("text", messages))
        return self.answer_fn(messages) if self.answer_fn else "ok"

    def _generate(self, messages: list[BaseMessage], stop: Any = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._text(messages)))])

    def _stream(self, messages: list[BaseMessage], stop: Any = None,
                run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        text = self._text(messages)
        for i in range(0, len(text), 12):
            piece = text[i : i + 12]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any):  # type: ignore[override]
        def _run(messages: Any) -> Any:
            msgs = messages if isinstance(messages, list) else messages.to_messages()
            self.calls.append(("structured", schema.__name__, msgs))
            if self.structured_fn is None:
                raise ValueError("no structured output scripted")
            out = self.structured_fn(schema, msgs)
            return out if isinstance(out, BaseModel) else schema.model_validate(out)

        return RunnableLambda(_run)


_WORD = re.compile(r"[a-z0-9]+")


class HashEmbeddings(Embeddings):
    """Bag-of-words hashed into a fixed vector: similar wording → high cosine."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            if len(w) < 3 and not w.isdigit():
                continue
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_WINDOW_LINE = re.compile(r"\[(C\d+)\] \(([^)]*)\) (.*)")


def fake_meeting_analyst(schema: Any, msgs: list[BaseMessage]) -> Any:
    """
    Deterministic stand-in for the LLM meeting analyst (RawInsights schema). Picks lines by
    keywords and cites their [C#] ids. It also proposes one fabricated owner ("Rahul") and
    one item citing a non-existent chunk, so tests can prove the verifier removes them.
    """
    lines = [m.groups() for m in _WINDOW_LINE.finditer(str(msgs[-1].content))]
    if not lines:  # reduce step: echo the first partial unchanged
        import json as _json

        return _json.loads(str(msgs[-1].content).split("\n\n")[0])

    def pick(*words: str) -> list[tuple[str, str]]:
        return [(cid, text) for cid, _, text in lines if any(w in text.lower() for w in words)]

    actions = []
    for cid, text in pick(" will ", "needs to", "can you"):
        owner = next((n for n in ("Arjun", "Meera", "Kavya", "Ravi") if n in text), "Rahul")
        deadline = next((d for d in ("Friday", "next week", "Thursday", "next Monday") if d in text), None)
        actions.append({"task": text[:90], "owner": owner, "deadline": deadline, "chunk_ids": [cid]})
    return {
        "title": "Business review",
        "summary_points": [{"text": t[:120], "chunk_ids": [c]} for c, _, t in lines[:3]],
        "decisions": [{"text": t[:120], "chunk_ids": [c]} for c, t in pick("agreed", "decided", "raising", "we will hire")]
        + [{"text": "Unsupported invented decision", "chunk_ids": ["C999"]}],
        "action_items": actions,
        "open_questions": [{"text": t[:120], "chunk_ids": [c]} for c, t in pick("do not know", "still need", "nobody", "no decision")],
    }
