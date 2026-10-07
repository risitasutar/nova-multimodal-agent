# ADR-008: Human-in-the-loop approval

**Status:** Accepted · 2026-10-05

## Context
Nova's tools are read-only, but web search and market-data calls still send queries derived from user input to
third parties. Some deployments need a person to approve that first.

## Decision
- An `approval_gate` node calls LangGraph `interrupt()` with the planned external operations whenever approval
  is required.
- Approval is set per request (UI toggle or the API's `approval_required` field). The default comes from
  `NOVA_APPROVAL_REQUIRED`.
- The paused state is checkpointed. The turn resumes with `Command(resume={"approved": bool})`, sent by the UI
  buttons or `POST /chat/{thread_id}/approval`.
- A rejection removes the external steps. Nova continues with local evidence only, or explains that it could
  not answer.
- While an approval is pending, new messages are refused (HTTP 409).

## Consequences
- Pause and resume work across processes because the checkpoint survives restarts.
- Nova has no trade execution or other side effects, and none are simulated.
