# ADR-001: LangGraph for agent orchestration

**Status:** Accepted · 2026-10-05

## Context
The upstream code used one LangGraph ReAct loop (`chat_node ⇄ ToolNode`) in which the LLM alone decided
which tools to call, in which order, and when to stop. Routing was invisible and untestable, there was no place
to enforce grounding, and the loop could run until the recursion limit.

## Decision
Keep LangGraph, but model the agent as an **explicit state machine** with typed state and conditional edges
(`nova/agent/graph.py`):

`validate_input → classify_query → planner → approval_gate → execute_tools → compute → check_evidence →
generate_response → verify_response → finalize_response`

- Deterministic nodes do everything that does not need language understanding: validation, tool execution,
  evidence thresholds, citation validation, number checks.
- The LLM is called only to classify ambiguous queries, plan multi-tool requests, extract calculations and
  write the answer.
- Every loop is bounded (`NOVA_MAX_PLAN_STEPS`, `NOVA_MAX_VERIFY_ATTEMPTS`, `recursion_limit`).

## Consequences
- Routing, planning and verification are unit-tested individually with a scripted model.
- Fewer LLM calls than a free-running ReAct loop, which matters on CPU (ADR-004).
- LangGraph checkpointing provides persistence (ADR-003), and `interrupt()` provides human approval (ADR-008).
- Less emergent flexibility than an open ReAct loop: a new capability needs an explicit tool spec and route.
