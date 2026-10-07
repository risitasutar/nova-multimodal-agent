"""
Nova HTTP API (FastAPI).

The API is a thin transport over `NovaService` — the exact same LangGraph agent the
Streamlit UI uses. No agent logic lives here.

    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ApprovalRequest,
    ChatRequest,
    ChatResponse,
    DocumentResponse,
    ErrorResponse,
    MediaSearchRequest,
    MediaSearchResponse,
    ThreadDetail,
    ThreadSummary,
    YouTubeRequest,
)
from nova import __version__
from nova.config import get_settings
from nova.errors import DocumentTooLargeError, InputValidationError, NovaError
from nova.observability import log_event, metrics, new_request_id, request_context
from nova.service import NovaService, get_service
from nova.video.models import MediaStatus
from nova.video.retrieval import to_results
from nova.video.service import MediaNotReadyError


def service_dep() -> NovaService:
    return get_service()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional shared-secret auth: enforced only when NOVA_API_KEY is configured."""
    expected = get_settings().api_key
    if expected is None:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        raise _Unauthorized()


class _Unauthorized(NovaError):
    code = "unauthorized"
    http_status = 401
    user_message = "Missing or invalid API key."


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log_event("api_startup", version=__version__)
    yield


app = FastAPI(
    title="Nova API",
    version=__version__,
    description="Nova: multimodal LangGraph agent over documents, recordings, web and market data, with verified citations.",
    lifespan=lifespan,
)
_settings = get_settings()
if _settings.api_cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=_settings.api_cors_origins,
                       allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = (request.headers.get("x-request-id") or new_request_id())[:64]
    request.state.request_id = rid
    with request_context(request_id=rid):
        with metrics.timer(f"http.{request.method}.{request.url.path.split('/')[1] or 'root'}"):
            response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.exception_handler(NovaError)
async def nova_error_handler(request: Request, exc: NovaError) -> JSONResponse:
    log_event("api_error", level=30 if exc.http_status < 500 else 40, code=exc.code, detail=exc.detail,
              path=request.url.path)
    body = ErrorResponse(error=exc.to_dict(), request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=exc.http_status, content=body.model_dump())


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event("api_unhandled_error", level=40, error=repr(exc), path=request.url.path, exc_info=True)
    body = ErrorResponse(error=NovaError().to_dict(), request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=500, content=body.model_dump())


# ------------------------------------------------------------------ health
@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness: the process is up (does not check dependencies)."""
    return {"status": "ok", "version": __version__}


@app.get("/ready", tags=["ops"])
def ready(svc: NovaService = Depends(service_dep)) -> JSONResponse:
    """Readiness: model server reachable, models pulled, database usable."""
    report = svc.readiness()
    return JSONResponse(status_code=200 if report["ready"] else 503, content=report)


@app.get("/metrics", tags=["ops"], dependencies=[Depends(require_api_key)])
def get_metrics() -> dict:
    return metrics.snapshot()


# ------------------------------------------------------------------ chat
@app.post("/chat", response_model=ChatResponse, tags=["chat"], dependencies=[Depends(require_api_key)],
          responses={409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def chat(req: ChatRequest, svc: NovaService = Depends(service_dep)) -> ChatResponse:
    result = await run_in_threadpool(svc.chat, req.message, req.thread_id, approval_required=req.approval_required)
    return ChatResponse.model_validate(result.to_dict())


@app.post("/chat/{thread_id}/approval", response_model=ChatResponse, tags=["chat"],
          dependencies=[Depends(require_api_key)], responses={409: {"model": ErrorResponse}})
async def approve(thread_id: str, req: ApprovalRequest, svc: NovaService = Depends(service_dep)) -> ChatResponse:
    """Resume a turn paused by the human-in-the-loop approval gate."""
    result = await run_in_threadpool(svc.resume, thread_id, req.approved)
    return ChatResponse.model_validate(result.to_dict())


# ------------------------------------------------------------------ documents
@app.post("/documents", response_model=DocumentResponse, tags=["documents"],
          dependencies=[Depends(require_api_key)],
          responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(..., max_length=64),
    svc: NovaService = Depends(service_dep),
) -> DocumentResponse:
    limit = int(get_settings().max_pdf_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        from nova.errors import DocumentTooLargeError

        raise DocumentTooLargeError(user_message=f"This PDF is larger than the {get_settings().max_pdf_mb:g} MB limit.")
    if not data:
        raise InputValidationError(user_message="The uploaded file is empty.")
    info = await run_in_threadpool(svc.ingest_document, thread_id, file.filename, data)
    return DocumentResponse.model_validate(info | {"thread_id": thread_id})


# ------------------------------------------------------------------ threads
@app.get("/threads", response_model=list[ThreadSummary], tags=["threads"], dependencies=[Depends(require_api_key)])
def list_threads(limit: int = 100, svc: NovaService = Depends(service_dep)) -> list[dict]:
    return svc.list_threads(min(max(limit, 1), 500))


@app.get("/threads/{thread_id}", response_model=ThreadDetail, tags=["threads"],
         dependencies=[Depends(require_api_key)], responses={404: {"model": ErrorResponse}})
def get_thread(thread_id: str, svc: NovaService = Depends(service_dep)) -> dict:
    return svc.get_thread(thread_id)


@app.delete("/threads/{thread_id}", status_code=204, tags=["threads"],
            dependencies=[Depends(require_api_key)], responses={404: {"model": ErrorResponse}})
def delete_thread(thread_id: str, svc: NovaService = Depends(service_dep)) -> None:
    svc.delete_thread(thread_id)


# ------------------------------------------------------------------ media (video / audio)
def _media(svc: NovaService) -> Any:
    if svc.media is None:
        raise NovaError("media disabled", user_message="The video/audio capability is not enabled.")
    return svc.media


@app.post("/media/upload", tags=["media"], status_code=201, dependencies=[Depends(require_api_key)],
          responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
async def upload_media(
    file: UploadFile = File(...),
    thread_id: str = Form(..., max_length=64),
    language: str = Form("auto"),
    svc: NovaService = Depends(service_dep),
) -> dict:
    """Store a video/audio file (or an .srt/.vtt transcript) for a conversation. Status: QUEUED."""
    media = _media(svc)
    if language not in ("auto", "english", "hinglish"):
        raise InputValidationError(user_message="language must be auto, english or hinglish.")
    limit = int(get_settings().media_max_size_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise DocumentTooLargeError(
            user_message=f"This file is larger than the {get_settings().media_max_size_mb:g} MB limit.")
    name = file.filename or "media"
    if name.lower().endswith((".srt", ".vtt")):
        asset = await run_in_threadpool(media.create_subtitle_import, thread_id, name, data.decode("utf-8", "replace"))
    else:
        asset = await run_in_threadpool(media.create_upload, thread_id, name, data, language)
    svc.registry.touch(thread_id, f"🎬 {asset.filename}", count_turn=False)
    return asset.model_dump()


@app.post("/media/youtube", tags=["media"], status_code=201, dependencies=[Depends(require_api_key)])
def add_youtube(req: YouTubeRequest, svc: NovaService = Depends(service_dep)) -> dict:
    asset = _media(svc).create_youtube(req.thread_id, req.url, req.language)
    svc.registry.touch(req.thread_id, f"🎬 {asset.source_uri}", count_turn=False)
    return asset.model_dump()


@app.post("/media/{media_id}/process", tags=["media"], dependencies=[Depends(require_api_key)],
          responses={202: {"description": "Accepted; poll GET /media/{media_id}"}})
async def process_media(media_id: str, background: BackgroundTasks, wait: bool = False, insights: bool = True,
                        svc: NovaService = Depends(service_dep)) -> JSONResponse:
    """Run the pipeline. Default: in the background (poll GET /media/{id}); `?wait=true` blocks."""
    media = _media(svc)
    media.get(media_id)  # 404 early
    if wait:
        asset = await run_in_threadpool(lambda: media.process(media_id, insights=insights))
        return JSONResponse(status_code=200, content=asset.model_dump())
    background.add_task(media.process, media_id, insights=insights)
    return JSONResponse(status_code=202, content={"media_id": media_id, "status": "QUEUED",
                                                  "poll": f"/media/{media_id}"})


@app.get("/media/{media_id}", tags=["media"], dependencies=[Depends(require_api_key)],
         responses={404: {"model": ErrorResponse}})
def get_media(media_id: str, svc: NovaService = Depends(service_dep)) -> dict:
    return _media(svc).get(media_id).model_dump()


@app.get("/media/{media_id}/summary", tags=["media"], dependencies=[Depends(require_api_key)],
         responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def media_summary(media_id: str, svc: NovaService = Depends(service_dep)) -> dict:
    insights = _media(svc).insights(media_id)
    if insights is None:
        raise MediaNotReadyError(media_id, user_message="No summary yet: process the media with insights enabled.")
    return {"media_id": media_id, **insights.model_dump()}


@app.post("/media/{media_id}/search", response_model=MediaSearchResponse, tags=["media"],
          dependencies=[Depends(require_api_key)], responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
def media_search(media_id: str, req: MediaSearchRequest, svc: NovaService = Depends(service_dep)) -> dict:
    media = _media(svc)
    asset = media.get(media_id)
    if asset.status != MediaStatus.COMPLETED:
        raise MediaNotReadyError(media_id)
    # Scope: the media's own thread and this media id only.
    result = media.search(asset.thread_id, req.query, media_ids=[media_id], top_k=req.top_k)
    return {"media_id": media_id, "results": to_results(result), "latency_ms": result.latency_ms}


@app.delete("/media/{media_id}", status_code=204, tags=["media"], dependencies=[Depends(require_api_key)],
            responses={404: {"model": ErrorResponse}})
def delete_media(media_id: str, svc: NovaService = Depends(service_dep)) -> None:
    _media(svc).delete(media_id)
