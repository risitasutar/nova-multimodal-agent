# Production readiness checklist

Status as of 2026-10-05. ✅ = implemented and verified in this repository. ⚠️ = implemented with a caveat.

| | Item | Where / evidence |
|---|---|---|
| ✅ | Configuration centralised | `nova/config.py` (one pydantic `Settings`; no other module reads the environment) |
| ✅ | Secrets externalised | `.env` (git-ignored); `SecretStr`; `tests/unit/test_security_scan.py` |
| ✅ | Error handling | `nova/errors.py` typed errors → safe UI/API messages; tools never raise into the graph |
| ✅ | Logging | JSON logs with request/thread ids and secret redaction (`nova/observability/logging.py`, `logs/nova.log`) |
| ✅ | Observability | Per-node latency in every answer's metadata; `/metrics` counters and latency percentiles; optional LangSmith |
| ✅ | Retries | Bounded exponential backoff for HTTP providers and web search (`nova/tools/base.py`, `nova/tools/web.py`) |
| ✅ | Timeouts | HTTP timeouts, per-tool executor timeouts (`TOOL_SPECS`), LLM client timeout (`NOVA_LLM_TIMEOUT_S`) |
| ✅ | Input validation | Message size and control characters (`nova/safety.py`); Pydantic API schemas; thread-id validation |
| ✅ | File limits | PDF type, magic bytes, size, pages, chunks, documents per thread (`nova/rag/ingest.py`, `NovaService`) |
| ✅ | Prompt-injection defence | `docs/SECURITY.md`; test `test_prompt_injection_in_document_is_quoted_as_data`; eval cases `adv-01`/`adv-02` |
| ✅ | Thread isolation | Checkpoints keyed by thread; `tests/integration/test_service.py::test_threads_are_isolated` |
| ✅ | Document isolation | Per-thread index directories plus manifest ownership checks; `test_document_isolation_between_threads`, `test_foreign_manifest_is_rejected` |
| ✅ | Media limits | Size (`MEDIA_MAX_SIZE_MB`), duration (probe + decode guard), transcript length, chunk count, media per thread |
| ✅ | Media job status | QUEUED/PROCESSING/COMPLETED/FAILED in SQLite; per-step progress; interrupted jobs marked FAILED on restart; background processing via FastAPI `BackgroundTasks` |
| ✅ | Media isolation and deletion | Per-thread collections, ownership checks, reference-counted cache deletion (`tests/video/test_media_service.py`) |
| ✅ | Tests | 164 deterministic tests (unit, rag, agent, integration, api, headless UI) plus opt-in live e2e (`pytest -m e2e`) and `smoke_test.py` |
| ✅ | Evaluation | `evaluation/` golden set, baseline vs enhanced, `results/report.md` |
| ✅ | API | FastAPI `/health /ready /chat /chat/{id}/approval /documents /threads /threads/{id} /metrics` |
| ⚠️ | Docker | `Dockerfile`, `.dockerignore` and `docker-compose.yml` are written and the compose YAML validates, but the image was **not built here** (Docker is not installed on the dev machine) |
| ✅ | Health checks | `/health` (liveness), `/ready` (LLM and DB), Docker `HEALTHCHECK` |
| ✅ | Documentation | `README.md`, `docs/architecture.md`, `docs/adr/`, `docs/SECURITY.md`, `evaluation/README.md` |
| ⚠️ | Deployment readiness | Single-node ready (SQLite, local volume). Multi-replica deployments need Postgres checkpoints and a shared vector store (ADR-003) and a real metrics backend |
| ✅ | Reproducible setup | Pinned `requirements.txt`; `run.bat`; README steps verified on Windows 11 / Python 3.12 |

## Before exposing Nova to the internet

- [ ] Set `NOVA_API_KEY` (or put the API behind an authenticating gateway) and restrict `NOVA_CORS_ORIGINS`.
- [ ] Configure `ALPHAVANTAGE_API_KEY`, or another licensed market-data source, instead of the keyless fallback.
- [ ] Use a GPU or hosted LLM endpoint (CPU-only latency is tens of seconds per answer).
- [ ] Put Streamlit behind authentication; it has no user accounts, and every visitor sees every thread.
- [ ] Add per-client rate limiting at the gateway.
- [ ] Ship logs and `/metrics` to a central system (for example Loki/Prometheus) and back up the data volume.
