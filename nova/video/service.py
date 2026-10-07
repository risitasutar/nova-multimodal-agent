"""
MediaService — backend service for video/audio. No UI code: the Streamlit app, the API
and the agent's tools all call this.

Pipeline (each step recorded on the asset and reported to an optional progress callback):

    validate → extract audio → transcribe → chunk → embed → index → summarize → insights

Status: QUEUED → PROCESSING → COMPLETED | FAILED. Processing is synchronous within the
calling thread (the API runs it in a background task); artifacts are cached by content
hash + configuration, so a re-upload of the same recording, or a re-run, skips finished work.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel

from nova.config import Settings
from nova.errors import DocumentTooLargeError, InputValidationError, NovaError, is_connection_error
from nova.observability import log_event, metrics
from nova.rag.retrieve import RetrievalResult, Retriever
from nova.rag.store import DocumentStore
from nova.video import ingest
from nova.video.chunking import chunk_segments
from nova.video.models import (
    PIPELINE_STEPS,
    LanguageMode,
    MediaAsset,
    MediaStatus,
    MeetingInsights,
    Transcript,
)
from nova.video.retrieval import search_media
from nova.video.storage import MediaFiles, MediaRegistry, config_hash
from nova.video.summarization import generate_insights, get_media_llm
from nova.video.transcription import engine_signature, parse_subtitles, transcribe

Progress = Callable[[str, str], None]  # (step, label)
Transcriber = Callable[[Path, LanguageMode, Settings], Transcript]


class MediaNotFoundError(NovaError):
    code = "media_not_found"
    http_status = 404
    user_message = "That media item does not exist."


class MediaNotReadyError(NovaError):
    code = "media_not_ready"
    http_status = 409
    user_message = "This media item has not finished processing."


class MediaService:
    def __init__(
        self,
        settings: Settings,
        registry: MediaRegistry,
        store: DocumentStore,
        retriever: Retriever,
        llm: BaseChatModel,
        transcriber: Transcriber = transcribe,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.store = store
        self.retriever = retriever
        self.files = MediaFiles(settings.media_dir, settings.media_cache_dir, settings.data_dir)
        self._llm = llm
        self.transcriber = transcriber
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        for asset in registry.interrupted():  # a crash mid-processing must not leave a stuck job
            asset.status, asset.error = MediaStatus.FAILED, "Processing was interrupted; process it again."
            registry.save(asset)

    # ------------------------------------------------------------ helpers
    def _lock(self, media_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(media_id, threading.Lock())

    def get(self, media_id: str, thread_id: str | None = None) -> MediaAsset:
        asset = self.registry.get(media_id) if media_id else None
        if asset is None or (thread_id is not None and asset.thread_id != thread_id):
            raise MediaNotFoundError(media_id)  # never reveal another thread's media
        return asset

    def list_assets(self, thread_id: str) -> list[MediaAsset]:
        return self.registry.list_assets(thread_id)

    def completed(self, thread_id: str) -> list[MediaAsset]:
        return [a for a in self.list_assets(thread_id) if a.status == MediaStatus.COMPLETED]

    def _new_asset(self, thread_id: str, **fields: Any) -> MediaAsset:
        self.files._check(thread_id)
        if len(self.list_assets(thread_id)) >= self.settings.max_media_per_thread:
            raise InputValidationError("media limit", user_message=(
                f"A conversation can hold at most {self.settings.max_media_per_thread} media items."))
        asset = MediaAsset(media_id=uuid.uuid4().hex, thread_id=thread_id, **fields)
        return self.registry.save(asset)

    # ------------------------------------------------------------ creation
    def create_upload(self, thread_id: str, filename: str | None, data: bytes, language: LanguageMode = "auto") -> MediaAsset:
        safe, kind, mime = ingest.validate_upload(filename, len(data), self.settings)
        asset = self._new_asset(thread_id, filename=safe, source_type=kind, source_uri=safe, mime_type=mime,
                                language=language)
        try:
            dest = self.files.asset_dir(thread_id, asset.media_id) / f"source{Path(safe).suffix.lower()}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            asset.source_key = hashlib.sha256(data).hexdigest()
        except Exception:
            self.delete(asset.media_id)
            raise
        return self.registry.save(asset)

    def create_youtube(self, thread_id: str, url: str, language: LanguageMode = "auto") -> MediaAsset:
        if not self.settings.youtube_enabled:
            raise InputValidationError("youtube disabled", user_message="YouTube ingestion is disabled.")
        vid = ingest.youtube_video_id(url)
        return self._new_asset(thread_id, filename=f"youtube-{vid}", source_type="youtube",
                               source_uri=f"https://www.youtube.com/watch?v={vid}", language=language,
                               source_key=f"yt-{vid}")

    def create_from_transcript(self, thread_id: str, filename: str, transcript: Transcript,
                               source_type: str | None = None) -> MediaAsset:
        """Register media whose transcript already exists (subtitle import, evaluation fixtures)."""
        if source_type is None:
            source_type = "audio" if Path(filename).suffix.lower() in ingest.AUDIO_EXTENSIONS else "video"
        payload = transcript.model_dump_json()
        asset = self._new_asset(thread_id, filename=filename, source_type=source_type, source_uri=filename,
                                imported=True, transcription_engine=transcript.engine, source_key=hashlib.sha256(payload.encode()).hexdigest(),
                                duration_seconds=transcript.duration_seconds or
                                (transcript.segments[-1].end_time if transcript.segments else None))
        cache = self.files.cache_dir(asset.source_key or "")
        self.files.write_json(cache / f"transcript-{self._transcript_cfg(asset)}.json", json.loads(payload))
        return asset

    def create_subtitle_import(self, thread_id: str, filename: str | None, text: str) -> MediaAsset:
        from nova.rag.ingest import sanitize_filename

        transcript = parse_subtitles(text)
        return self.create_from_transcript(thread_id, sanitize_filename(filename or "transcript.srt"), transcript)

    # ------------------------------------------------------------ processing
    def _transcript_cfg(self, asset: MediaAsset) -> str:
        """Cache-key component for the transcript: the exact engine/model/language, or 'import'."""
        return "import" if asset.imported else config_hash(engine_signature(self.settings, asset.language))

    def process(self, media_id: str, *, insights: bool = True, progress: Progress | None = None) -> MediaAsset:
        with self._lock(media_id):
            asset = self.get(media_id)
            if asset.status == MediaStatus.PROCESSING:
                raise InputValidationError("already processing", user_message="This media item is already being processed.")
            asset.status, asset.error = MediaStatus.PROCESSING, None
            asset.steps = {}
            self.registry.save(asset)
            started = time.perf_counter()
            try:
                self._run_pipeline(asset, insights, progress or (lambda step, label: None))
                asset.status = MediaStatus.COMPLETED
                metrics.incr("media.completed")
            except NovaError as exc:
                asset.status, asset.error = MediaStatus.FAILED, exc.user_message
                metrics.incr("media.failed")
                log_event("media_failed", level=40, media_id=media_id, code=exc.code, detail=exc.detail)
            except Exception as exc:  # noqa: BLE001 - record any failure on the job, keep details in logs
                asset.status = MediaStatus.FAILED
                asset.error = ("Nova cannot reach the language model server." if is_connection_error(exc)
                               else "Media processing failed unexpectedly.")
                metrics.incr("media.failed")
                log_event("media_failed", level=40, media_id=media_id, error=repr(exc), exc_info=True)
            asset.timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.registry.save(asset)
            log_event("media_processed", media_id=media_id, status=asset.status.value, timings=asset.timings_ms,
                      segments=asset.segments, chunks=asset.chunks)
            return asset

    def _step(self, asset: MediaAsset, step: str, state: str, progress: Progress) -> None:
        asset.steps[step] = state
        self.registry.save(asset)
        progress(step, dict(PIPELINE_STEPS)[step] + ("" if state == "done" else f" ({state})"))

    def _timed(self, asset: MediaAsset, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        out = fn()
        ms = round((time.perf_counter() - t0) * 1000, 1)
        asset.timings_ms[name] = ms
        metrics.observe(f"media.{name}", ms)
        return out

    def _run_pipeline(self, asset: MediaAsset, want_insights: bool, progress: Progress) -> None:
        s = self.settings
        asset_dir = self.files.asset_dir(asset.thread_id, asset.media_id)
        cache = self.files.cache_dir(asset.source_key or asset.media_id)
        transcript_file = cache / f"transcript-{self._transcript_cfg(asset)}.json"

        # 1. validate (+ 2. audio) — skipped entirely when a cached transcript exists
        if transcript_file.exists():
            transcript = Transcript.model_validate_json(transcript_file.read_text(encoding="utf-8"))
            for step in ("validated", "audio_extracted", "transcribed"):
                self._step(asset, step, "cached", progress)
        else:
            audio = cache / "audio.wav"
            if asset.source_type == "youtube":
                vid = (asset.source_key or "")[3:]
                meta = self._timed(asset, "metadata_ms", lambda: ingest.youtube_metadata(vid))
                ingest.check_duration(meta["duration_seconds"], s)
                asset.filename = meta["title"][:120]
                asset.duration_seconds = meta["duration_seconds"]
                self._step(asset, "validated", "done", progress)
                if not audio.exists():
                    src = self._timed(asset, "download_ms", lambda: ingest.download_youtube_audio(vid, asset_dir))
                else:
                    src = None
            else:
                src = next(asset_dir.glob("source.*"), None)
                if src is None:
                    raise InputValidationError("source missing", user_message="The uploaded file is missing; upload it again.")
                info = ingest.probe(src)
                ingest.check_duration(info["duration_seconds"], s)
                asset.duration_seconds = info["duration_seconds"]
                self._step(asset, "validated", "done", progress)
            if audio.exists():
                self._step(asset, "audio_extracted", "cached", progress)
            else:
                assert src is not None
                decoded = self._timed(asset, "audio_extraction_ms",
                                      lambda: ingest.extract_audio(src, audio, s.media_max_duration_seconds))
                asset.duration_seconds = asset.duration_seconds or decoded
                self._step(asset, "audio_extracted", "done", progress)
            transcript = self._timed(asset, "transcription_ms", lambda: self.transcriber(audio, asset.language, s))
            transcript.duration_seconds = asset.duration_seconds
            self.files.write_json(transcript_file, transcript.model_dump())
            self._step(asset, "transcribed", "done", progress)

        if not transcript.segments:
            raise NovaError("empty transcript", user_message="No speech was detected in this recording.")
        if sum(len(x.text) for x in transcript.segments) > s.media_max_transcript_chars:
            raise DocumentTooLargeError("transcript too long", user_message="This transcript is too long to index.")
        asset.duration_seconds = asset.duration_seconds or transcript.duration_seconds or transcript.segments[-1].end_time
        asset.detected_language = transcript.language
        asset.transcription_engine = transcript.engine
        asset.segments = len(transcript.segments)
        asset.transcript_path = self.files.relative(transcript_file)

        # 3. chunk + embed (cached per chunking/embedding config) + index into this thread
        index_cfg = config_hash(transcript_file.name, s.video_chunk_size, s.video_chunk_overlap,
                                s.video_chunk_max_seconds, self.store.embedding_model)
        chunks = chunk_segments(transcript.segments, asset, s.video_chunk_size, s.video_chunk_overlap,
                                asset.duration_seconds, s.video_chunk_max_seconds)
        if len(chunks) > s.media_max_chunks:
            raise DocumentTooLargeError("too many chunks", user_message="This transcript is too long to index.")
        vec_file = cache / f"vectors-{index_cfg}.npy"
        vectors = None
        if vec_file.exists():
            cached = np.load(vec_file)
            vectors = cached if cached.shape[0] == len(chunks) else None
        if vectors is None:
            vectors = self._timed(asset, "embedding_ms", lambda: self.store.embed_texts([c["text"] for c in chunks]))
            np.save(vec_file, vectors)
        t0 = time.perf_counter()
        self.store.add_collection(asset.thread_id, asset.media_id, chunks,
                                  {"media_id": asset.media_id, "filename": asset.filename,
                                   "source_type": asset.source_type}, vectors=vectors)
        asset.timings_ms["indexing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        asset.chunks, asset.vector_collection_id = len(chunks), asset.media_id
        self._step(asset, "indexed", "done", progress)

        # 4. meeting intelligence (cached per transcript + model)
        if not want_insights:
            self._step(asset, "summarized", "skipped", progress)
            self._step(asset, "insights", "skipped", progress)
            return
        llm = get_media_llm(s, self._llm)
        model_name = s.mistral_model if s.media_llm_provider == "mistral" else s.chat_model_name
        ins_file = cache / f"insights-{config_hash(transcript_file.name, model_name, s.media_insight_window_chars, s.video_chunk_size, s.video_chunk_overlap, s.video_chunk_max_seconds)}.json"
        if ins_file.exists():
            insights = MeetingInsights.model_validate_json(ins_file.read_text(encoding="utf-8"))
            state = "cached"
        else:
            t0 = time.perf_counter()
            insights, sub = generate_insights(llm, chunks, s, asset.duration_seconds, model_name)
            asset.timings_ms["insights_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            asset.timings_ms.update({f"insights_{k}": v for k, v in sub.items()})
            self.files.write_json(ins_file, insights.model_dump())
            state = "done"
        # Chunk ids are per-asset (media_id prefix); remap cached insights onto this asset's ids.
        prefix = asset.media_id[:8]
        for group in (insights.summary, insights.decisions, insights.action_items, insights.open_questions):
            for item in group:
                item.chunk_ids = [f"{prefix}-{cid.split('-', 1)[1]}" if "-" in cid else cid for cid in item.chunk_ids]
        summary_file = asset_dir / "insights.json"
        self.files.write_json(summary_file, insights.model_dump())
        asset.summary_path = self.files.relative(summary_file)
        self._step(asset, "summarized", state, progress)
        self._step(asset, "insights", state, progress)

    # ------------------------------------------------------------ queries
    def insights(self, media_id: str, thread_id: str | None = None) -> MeetingInsights | None:
        asset = self.get(media_id, thread_id)
        if not asset.summary_path:
            return None
        path = self.files.absolute(asset.summary_path)
        return MeetingInsights.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def transcript(self, media_id: str, thread_id: str | None = None) -> Transcript | None:
        asset = self.get(media_id, thread_id)
        if not asset.transcript_path:
            return None
        path = self.files.absolute(asset.transcript_path)
        return Transcript.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def search(self, thread_id: str, query: str, *, media_ids: list[str] | None = None, rewritten: str | None = None,
               top_k: int | None = None, min_relevance: float | None = None) -> RetrievalResult:
        # min_relevance=None → the retriever's configured floor (NOVA_VIDEO_MIN_RELEVANCE in production).
        result = search_media(self.retriever, thread_id, query, rewritten, media_ids, top_k, min_relevance)
        metrics.observe("media.retrieval", result.latency_ms)
        return result

    def source_path(self, media_id: str, thread_id: str | None = None) -> Path | None:
        """Local source file for playback (None for YouTube or when not stored)."""
        asset = self.get(media_id, thread_id)
        if asset.source_type == "youtube":
            return None
        return next(self.files.asset_dir(asset.thread_id, media_id).glob("source.*"), None)

    # ------------------------------------------------------------ deletion
    def delete(self, media_id: str, thread_id: str | None = None) -> None:
        """Remove metadata, source/audio files, vectors and — if unreferenced — cached artifacts."""
        asset = self.get(media_id, thread_id)
        with self._lock(media_id):
            self.store.delete_collection(asset.thread_id, media_id)
            self.files.remove_asset_dir(asset.thread_id, media_id)
            self.registry.delete(media_id)
            if asset.source_key and self.registry.references(asset.source_key) == 0:
                self.files.remove_cache(asset.source_key)
        log_event("media_deleted", media_id=media_id)

    def delete_thread(self, thread_id: str) -> None:
        for asset in self.list_assets(thread_id):
            self.delete(asset.media_id)
        self.store.delete_thread(thread_id)
