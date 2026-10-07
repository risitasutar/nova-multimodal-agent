# ADR-004: Local Ollama as the default model runtime

**Status:** Accepted · 2026-10-05

## Context
The project must run without paid API keys and must keep documents on the user's machine.

## Decision
- Default to Ollama with `qwen3:8b` (supports tool calling and structured output) and `nomic-embed-text`.
- Always send Qwen3 an explicit `think` flag so reasoning tokens never leak into answers. Reasoning is never shown.
- Use `nomic-embed-text` with its `search_document:` / `search_query:` task prefixes.

## Measured constraint
On the development machine (CPU only, no GPU offload), Ollama's own `/api/chat` timing fields show:
- prompt processing: about 13 tokens/s;
- generation: about 5 tokens/s.

Prompt length therefore dominates latency. That drove these choices:
- short prompts;
- 600-character chunks;
- truncated conversation history;
- deterministic routing for unambiguous queries;
- template answers for pure arithmetic, with no LLM call.

## Consequences
Private and free, but slow on CPU: tens of seconds per answer. ADR-005 covers hosted inference.
