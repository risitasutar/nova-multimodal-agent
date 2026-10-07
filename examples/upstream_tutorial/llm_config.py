"""
Compatibility shim for the upstream tutorial stages: model configuration now lives in
`nova.config` / `nova.llm` (single source of truth). Run examples from the repo root:

    python -m streamlit run examples/upstream_tutorial/streamlit_frontend.py
"""

from nova.config import get_settings
from nova.llm import get_chat_model, get_embeddings as _get_embeddings

_s = get_settings()
OLLAMA_MODEL = _s.ollama_model
OLLAMA_THINK = _s.ollama_think
OLLAMA_EMBED_MODEL = _s.ollama_embed_model
OLLAMA_BASE_URL = _s.ollama_base_url


def get_llm(**kwargs):
    return get_chat_model(**kwargs)


def get_embeddings():
    # The tutorial RAG stage predates task prefixes; keep its original behaviour.
    return _get_embeddings(task_prefixes=False)
