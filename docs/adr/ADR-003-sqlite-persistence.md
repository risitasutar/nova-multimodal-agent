# ADR-003: SQLite for conversation persistence

**Status:** Accepted · 2026-10-05

## Context
Nova needs conversations that survive restarts, thread listing and deletion, and strict isolation between threads.

## Decision
- Conversations use LangGraph's `SqliteSaver` checkpointer (`data/nova.db`).
- A small indexed `threads` registry table (`nova/memory.py`) lists conversations without scanning checkpoints.
- Connections run in WAL mode with `busy_timeout=5000` and `synchronous=NORMAL`.
- Writes are serialised by SqliteSaver's own lock plus a per-thread lock in `NovaService`.
- Deleting a thread removes its checkpoints, registry row and vector indexes together.

## Why not PostgreSQL now
A single-node demo or portfolio deployment has one writer process per container and modest traffic. SQLite
needs no operations work, is transactional, and is fast here. Postgres would add infrastructure with no
current need.

## When to switch
Move to `langgraph-checkpoint-postgres`, with pgvector for embeddings, when any of these hold:
- more than one API replica writes concurrently;
- the deployment spans several hosts;
- you need point-in-time recovery or replication;
- write concurrency stays above what SQLite's single writer can handle.

The checkpointer is injected into `NovaService`, so the swap touches one place.
