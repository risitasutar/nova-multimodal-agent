"""Data models for the media (video/audio) capability."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["video", "audio", "youtube"]
LanguageMode = Literal["auto", "english", "hinglish"]

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("validated", "Media validated"),
    ("audio_extracted", "Audio extracted"),
    ("transcribed", "Transcription completed"),
    ("indexed", "Transcript indexed"),
    ("summarized", "Summary generated"),
    ("insights", "Insights extracted"),
]


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MediaStatus(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TranscriptSegment(BaseModel):
    segment_id: str
    start_time: float
    end_time: float
    text: str
    language: str | None = None
    speaker: str | None = None  # stays None: no diarization is performed, speakers are never invented


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str | None = None
    engine: str  # faster-whisper | openai-whisper | sarvam | subtitle-import
    model: str | None = None
    duration_seconds: float | None = None

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)


class MediaAsset(BaseModel):
    media_id: str
    thread_id: str
    filename: str
    source_type: SourceType
    source_uri: str  # sanitised filename or canonical YouTube URL — never a local absolute path
    mime_type: str | None = None
    duration_seconds: float | None = None
    language: LanguageMode = "auto"
    detected_language: str | None = None
    status: MediaStatus = MediaStatus.QUEUED
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    source_key: str | None = None  # content-addressed cache key (sha256 of bytes / YouTube id)
    transcript_path: str | None = None  # relative to the data dir
    summary_path: str | None = None
    vector_collection_id: str | None = None
    steps: dict[str, str] = Field(default_factory=dict)  # step -> done | cached | skipped | failed
    segments: int | None = None
    chunks: int | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    transcription_engine: str | None = None
    imported: bool = False  # transcript supplied (subtitle import / fixture) rather than transcribed
    error: str | None = None

    @property
    def media_kind(self) -> Literal["video", "audio"]:
        return "audio" if self.source_type == "audio" else "video"


class InsightItem(BaseModel):
    text: str
    owner: str | None = None  # only kept when the name appears in the cited transcript text
    deadline: str | None = None  # likewise
    chunk_ids: list[str] = Field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None


class MeetingInsights(BaseModel):
    title: str | None = None
    summary: list[InsightItem] = Field(default_factory=list)
    decisions: list[InsightItem] = Field(default_factory=list)
    action_items: list[InsightItem] = Field(default_factory=list)
    open_questions: list[InsightItem] = Field(default_factory=list)
    model: str | None = None
    windows: int = 0
    generated_at: str = Field(default_factory=utcnow)
    dropped_items: int = 0  # items removed because their evidence could not be verified
