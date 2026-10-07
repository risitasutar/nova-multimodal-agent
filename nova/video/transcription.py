"""
Unified, timestamp-preserving transcription.

    transcribe(audio_wav, language="auto" | "english" | "hinglish") -> Transcript

* english / auto → Whisper (local). Backend `faster` (faster-whisper, default) or
  `openai` (the original AI Video Assistant's openai-whisper). Whisper's own segment
  timestamps are kept — the original project discarded them (`result["text"]`).
* hinglish → Sarvam `speech-to-text-translate` (same endpoint/model/25 s piece size as the
  original). Sarvam's sync API returns no timings, so each 25 s piece becomes one segment
  with its true offset — the most precise timestamp available without inventing one.
* Subtitle import (.srt / .vtt) → segments from the cue timings.

Speaker is always None: no diarization is performed, and speakers are never guessed.
"""

from __future__ import annotations

import re
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

import requests

from nova.config import Settings
from nova.errors import NovaError
from nova.video.models import LanguageMode, Transcript, TranscriptSegment

SARVAM_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_PIECE_SECONDS = 25  # sync API rejects audio longer than 30 s (from the original project)

_whisper_lock = threading.Lock()
_whisper_models: dict[tuple[str, str, str], Any] = {}


class TranscriptionError(NovaError):
    code = "transcription_failed"
    http_status = 502
    user_message = "Transcription failed."


def _segment(i: int, start: float, end: float, text: str, language: str | None) -> TranscriptSegment:
    return TranscriptSegment(segment_id=f"seg_{i:04d}", start_time=round(float(start), 2),
                             end_time=round(float(end), 2), text=" ".join(text.split()), language=language)


def engine_signature(settings: Settings, language: LanguageMode) -> str:
    """Identifies the exact transcription configuration (part of the cache key)."""
    if language == "hinglish":
        return f"sarvam:{settings.sarvam_stt_model}"
    return f"whisper-{settings.whisper_backend}:{settings.whisper_model}:{language}"


# ---------------------------------------------------------------- Whisper
def _load_whisper(settings: Settings) -> Any:
    key = (settings.whisper_backend, settings.whisper_model, settings.whisper_compute_type)
    with _whisper_lock:
        if key not in _whisper_models:
            if settings.whisper_backend == "openai":
                try:
                    import whisper  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise TranscriptionError("openai-whisper not installed",
                                             user_message="NOVA_WHISPER_BACKEND=openai needs `pip install openai-whisper` and FFmpeg.") from exc
                _whisper_models[key] = whisper.load_model(settings.whisper_model)
            else:
                from faster_whisper import WhisperModel

                _whisper_models[key] = WhisperModel(settings.whisper_model, device="cpu",
                                                    compute_type=settings.whisper_compute_type)
        return _whisper_models[key]


def load_wav_float32(path: Path) -> Any:
    """16 kHz mono 16-bit WAV (what ingest.extract_audio writes) → float32 samples in [-1, 1].

    Passing samples instead of a path means neither Whisper backend decodes the file itself:
    faster-whisper 1.2.1's own PyAV call is incompatible with PyAV 19, and openai-whisper
    would otherwise need the FFmpeg binary.
    """
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise TranscriptionError(f"unexpected WAV format {w.getparams()}")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_whisper(audio: Path, language: LanguageMode, settings: Settings) -> Transcript:
    model = _load_whisper(settings)
    lang = "en" if language == "english" else None  # auto → Whisper language detection
    samples = load_wav_float32(audio)
    if settings.whisper_backend == "openai":
        result = model.transcribe(samples, task="transcribe", language=lang)
        detected = result.get("language")
        raw = [(s["start"], s["end"], s["text"]) for s in result.get("segments", [])]
    else:
        segments, info = model.transcribe(samples, language=lang, beam_size=5, vad_filter=True)
        raw = [(s.start, s.end, s.text) for s in segments]  # generator: decoding happens here
        detected = info.language
    segs = [_segment(i, a, b, t, detected) for i, (a, b, t) in enumerate(raw, start=1) if t.strip()]
    return Transcript(segments=segs, language=detected, engine=f"{settings.whisper_backend}-whisper",
                      model=settings.whisper_model)


# ---------------------------------------------------------------- Sarvam
def transcribe_sarvam(audio: Path, settings: Settings, session: requests.Session | None = None) -> Transcript:
    if not settings.sarvam_api_key:
        raise TranscriptionError("SARVAM_API_KEY missing",
                                 user_message="Hinglish transcription needs SARVAM_API_KEY to be configured.")
    http = session or requests.Session()
    headers = {"api-subscription-key": settings.sarvam_api_key.get_secret_value()}
    segs: list[TranscriptSegment] = []
    with wave.open(str(audio), "rb") as src, tempfile.TemporaryDirectory(prefix="nova-sarvam-") as tmpdir:
        rate, width, channels = src.getframerate(), src.getsampwidth(), src.getnchannels()
        piece_frames = SARVAM_PIECE_SECONDS * rate
        total = src.getnframes()
        for i, start in enumerate(range(0, total, piece_frames)):
            frames = src.readframes(piece_frames)
            piece = Path(tmpdir) / f"piece_{i}.wav"
            with wave.open(str(piece), "wb") as out:
                out.setnchannels(channels)
                out.setsampwidth(width)
                out.setframerate(rate)
                out.writeframes(frames)
            text = _sarvam_request(http, headers, piece, settings.sarvam_stt_model, settings.http_max_retries)
            end = min(total, start + piece_frames) / rate
            if text.strip():
                segs.append(_segment(len(segs) + 1, start / rate, end, text, "en"))
    return Transcript(segments=segs, language="en", engine="sarvam", model=settings.sarvam_stt_model)


def _sarvam_request(http: requests.Session, headers: dict[str, str], piece: Path, model: str, retries: int) -> str:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with piece.open("rb") as fh:
                resp = http.post(SARVAM_URL, headers=headers, timeout=120,
                                 files={"file": (piece.name, fh, "audio/wav")},
                                 data={"model": model, "with_diarization": "false"})
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                raise requests.HTTPError(f"retryable {resp.status_code}", response=resp)
            resp.raise_for_status()
            return str(resp.json().get("transcript", ""))
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in (429, 500, 502, 503, 504):
                break
            if attempt < retries:
                time.sleep(min(8.0, 0.5 * 2**attempt))
    raise TranscriptionError(repr(last), user_message="Sarvam transcription is unavailable right now.")


# ---------------------------------------------------------------- subtitles
_TS = r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}"
_CUE = re.compile(rf"(?P<start>{_TS})\s*-->\s*(?P<end>{_TS})")


def _ts_to_seconds(ts: str) -> float:
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    secs = float(parts[-1])
    mins = int(parts[-2])
    hours = int(parts[-3]) if len(parts) == 3 else 0
    return hours * 3600 + mins * 60 + secs


def parse_subtitles(text: str, language: str | None = None) -> Transcript:
    """Parse SRT or WebVTT cues into timestamped segments (cue settings and tags stripped)."""
    segs: list[TranscriptSegment] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _CUE.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = _ts_to_seconds(m.group("start")), _ts_to_seconds(m.group("end"))
        i += 1
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(re.sub(r"<[^>]+>", "", lines[i]).strip())
            i += 1
        content = " ".join(b for b in body if b)
        if content and end >= start:
            segs.append(_segment(len(segs) + 1, start, end, content, language))
    if not segs:
        raise TranscriptionError("no cues", user_message="No subtitle cues were found in this file.")
    return Transcript(segments=segs, language=language, engine="subtitle-import", model=None,
                      duration_seconds=segs[-1].end_time)


def transcribe(audio: Path, language: LanguageMode, settings: Settings) -> Transcript:
    """Route to Sarvam for Hinglish, Whisper otherwise (mirrors the original routing)."""
    if language == "hinglish":
        return transcribe_sarvam(audio, settings)
    return transcribe_whisper(audio, language, settings)
