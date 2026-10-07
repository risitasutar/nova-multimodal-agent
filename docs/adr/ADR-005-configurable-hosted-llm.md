# ADR-005: Configurable hosted LLM endpoint

**Status:** Accepted · 2026-10-05

## Context
A cloud deployment usually cannot run an 8B model on CPU with acceptable latency.

## Decision
- `nova/llm.py` exposes one factory for chat models and one for embedding models.
- `NOVA_LLM_PROVIDER=openai_compatible` switches to any OpenAI-compatible endpoint, configured with
  `NOVA_LLM_BASE_URL`, `NOVA_LLM_API_KEY`, `NOVA_LLM_MODEL` and `NOVA_EMBED_MODEL`. This uses the optional
  `langchain-openai` dependency.
- The application container never bundles a model (see the Dockerfile).

## Consequences
- The same graph, prompts and evaluation apply to local and hosted models.
- Document indexes record their embedding model, and switching embedding providers forces re-indexing.
- The hosted path is implemented and its configuration is validated, but it was not run against a paid
  endpoint in this repository because no credentials were available. Evaluation results cover the local
  model only.
