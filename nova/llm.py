"""
LLM and embedding factories.

Two providers are supported behind the same interface:
  * ollama             – local models (default; qwen3:8b + nomic-embed-text)
  * openai_compatible  – any OpenAI-compatible hosted endpoint (requires `langchain-openai`)
"""

from __future__ import annotations

from typing import Any

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from nova.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None, **overrides: Any) -> BaseChatModel:
    s = settings or get_settings()
    if s.llm_provider == "openai_compatible":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "NOVA_LLM_PROVIDER=openai_compatible requires `pip install langchain-openai`"
            ) from exc
        return ChatOpenAI(
            model=s.chat_model_name,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value() if s.llm_api_key else None,
            temperature=overrides.pop("temperature", s.llm_temperature),
            seed=s.llm_seed,
            timeout=s.llm_timeout_s,
            max_retries=1,
            **overrides,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=overrides.pop("temperature", s.llm_temperature),
        seed=s.llm_seed,
        # Always send an explicit think flag: left unset, Qwen3 leaks <think> tags into the
        # reply; True keeps reasoning out of the content. Reasoning is never shown to users.
        reasoning=s.ollama_think,
        client_kwargs={"timeout": s.llm_timeout_s},
        **overrides,
    )


class TaskPrefixedEmbeddings(Embeddings):
    """
    nomic-embed-text is trained with task prefixes; omitting them measurably hurts
    retrieval. This wrapper adds `search_document:` / `search_query:` transparently.
    """

    def __init__(self, inner: Embeddings, doc_prefix: str, query_prefix: str) -> None:
        self.inner = inner
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_prefix + text)


def get_embeddings(settings: Settings | None = None, *, task_prefixes: bool = True) -> Embeddings:
    s = settings or get_settings()
    if s.llm_provider == "openai_compatible":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai_compatible embeddings require langchain-openai") from exc
        return OpenAIEmbeddings(
            model=s.embedding_model_name,
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value() if s.llm_api_key else None,
        )

    from langchain_ollama import OllamaEmbeddings

    base: Embeddings = OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_base_url)
    if task_prefixes and "nomic" in s.ollama_embed_model:
        return TaskPrefixedEmbeddings(base, "search_document: ", "search_query: ")
    return base


def check_llm(settings: Settings | None = None) -> tuple[bool, str]:
    """Readiness probe: is the model server reachable and are the models available?"""
    s = settings or get_settings()
    if s.llm_provider == "openai_compatible":
        if not s.llm_base_url and not s.llm_api_key:
            return False, "NOVA_LLM_BASE_URL / NOVA_LLM_API_KEY are not configured."
        try:
            headers = {"Authorization": f"Bearer {s.llm_api_key.get_secret_value()}"} if s.llm_api_key else {}
            base = (s.llm_base_url or "https://api.openai.com/v1").rstrip("/")
            requests.get(f"{base}/models", headers=headers, timeout=5).raise_for_status()
        except requests.RequestException:
            return False, "The hosted model endpoint is not reachable."
        return True, f"{s.chat_model_name} ready"

    try:
        resp = requests.get(f"{s.ollama_base_url}/api/tags", timeout=3)
        resp.raise_for_status()
    except requests.RequestException:
        return False, (
            f"Nova cannot reach the local model server at {s.ollama_base_url}. "
            "Please start Ollama (or run `ollama serve`) and try again."
        )
    installed = {m["name"] for m in resp.json().get("models", [])}
    installed |= {name.split(":")[0] for name in installed if name.endswith(":latest")}
    missing = [m for m in (s.ollama_model, s.ollama_embed_model) if m not in installed]
    if missing:
        cmds = " && ".join(f"ollama pull {m}" for m in missing)
        return False, f"Missing Ollama model(s): {', '.join(missing)}. Run: {cmds}"
    return True, f"{s.ollama_model} ready"
