"""Pydantic request/response models for the Nova HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str | None = Field(None, description="Existing conversation id; omit to start a new one.",
                                  max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    message: str = Field(..., min_length=1, max_length=20000)
    approval_required: bool | None = Field(
        None, description="Pause for human approval before external data calls (default from config)."
    )


class ApprovalRequest(BaseModel):
    approved: bool


class Citation(BaseModel):
    id: str
    type: Literal["document", "video", "web", "finance"]
    display: str
    filename: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    preview: str | None = None
    title: str | None = None
    url: str | None = None
    domain: str | None = None
    label: str | None = None
    media_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    source_type: str | None = None
    source_name: str | None = None
    locator: dict[str, Any] | None = None
    relevance: float | None = None


class ChatResponse(BaseModel):
    thread_id: str
    request_id: str
    status: Literal["completed", "awaiting_approval"]
    answer: str
    route: str | None = None
    plan: list[dict[str, Any]] = []
    plan_summary: str = ""
    tools_used: list[str] = []
    citations: list[Citation] = []
    verification_status: str | None = None
    evidence_score: float = 0.0
    latency_ms: float = Field(0.0, description="Sum of agent node execution time.")
    wall_time_ms: float = 0.0
    timeline: list[str] = []
    finance: list[dict[str, Any]] = []
    approval: dict[str, Any] | None = None


class DocumentResponse(BaseModel):
    thread_id: str
    document_id: str
    filename: str
    pages: int
    chunks: int
    status: str
    duplicate: bool = False
    injection_flagged_chunks: int = 0
    ingest_ms: float | None = None


class ThreadSummary(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    turns: int


class ThreadMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, Any] | None = None


class ThreadDetail(ThreadSummary):
    messages: list[ThreadMessage]
    documents: list[dict[str, Any]]
    media: list[dict[str, Any]] = []
    pending_approval: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: dict[str, str]
    request_id: str | None = None


class MediaSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(4, ge=1, le=20)


class MediaSearchHit(BaseModel):
    media_id: str
    chunk_id: str
    source_name: str
    text: str
    start_time: float
    end_time: float
    relevance: float = Field(description="Actual cosine similarity from the vector index.")
    rank: int | None = None


class MediaSearchResponse(BaseModel):
    status: Literal["success"] = "success"
    media_id: str
    results: list[MediaSearchHit]
    latency_ms: float


class YouTubeRequest(BaseModel):
    thread_id: str = Field(..., max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    url: str = Field(..., max_length=500)
    language: Literal["auto", "english", "hinglish"] = "auto"
