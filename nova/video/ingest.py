"""
Media validation, audio extraction and YouTube download.

Audio extraction uses PyAV (bundled FFmpeg libraries) instead of the original project's
pydub + FFmpeg binary: decode the first audio stream of any supported container,
resample to 16 kHz mono 16-bit PCM and write a WAV — the format every transcription
backend accepts.
"""

from __future__ import annotations

import mimetypes
import re
import wave
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from nova.config import Settings
from nova.errors import DocumentTooLargeError, InputValidationError, UnsupportedFileError
from nova.rag.ingest import sanitize_filename
from nova.video.models import SourceType

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
SUBTITLE_EXTENSIONS = {".srt", ".vtt"}
YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
_YT_ID = re.compile(r"^[A-Za-z0-9_\-]{11}$")
SAMPLE_RATE = 16_000


class MediaError(InputValidationError):
    code = "media_error"
    user_message = "This media file could not be processed."


def classify_extension(filename: str) -> SourceType:
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    raise UnsupportedFileError(
        f"rejected extension {ext}",
        user_message="Supported media: MP4, MOV, AVI, MKV, WEBM, MP3, WAV, M4A (or a YouTube URL).",
    )


def validate_upload(filename: str | None, size_bytes: int, settings: Settings) -> tuple[str, SourceType, str | None]:
    """Name/extension/size checks before anything is written. Returns (safe_name, kind, mime)."""
    safe = sanitize_filename(filename or "media")
    kind = classify_extension(safe)
    if size_bytes <= 0:
        raise MediaError("empty upload", user_message="The uploaded file is empty.")
    limit = int(settings.media_max_size_mb * 1024 * 1024)
    if size_bytes > limit:
        raise DocumentTooLargeError(
            f"{size_bytes} > {limit}", user_message=f"This file is larger than the {settings.media_max_size_mb:g} MB limit."
        )
    mime, _ = mimetypes.guess_type(safe)
    return safe, kind, mime


def probe(path: Path) -> dict[str, Any]:
    """Open the container (real content validation) and read duration/stream info."""
    import av

    try:
        with av.open(str(path)) as container:
            audio = [s for s in container.streams if s.type == "audio"]
            video = [s for s in container.streams if s.type == "video"]
            duration = None
            if container.duration:
                duration = container.duration / 1_000_000  # av.time_base
            elif audio and audio[0].duration and audio[0].time_base:
                duration = float(audio[0].duration * audio[0].time_base)
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise MediaError(repr(exc), user_message="This file is not a readable audio/video file.") from exc
    if not audio:
        raise MediaError("no audio stream", user_message="This file has no audio track to transcribe.")
    return {"duration_seconds": duration, "has_video": bool(video), "has_audio": True}


def check_duration(duration: float | None, settings: Settings) -> None:
    if duration is not None and duration > settings.media_max_duration_seconds:
        raise DocumentTooLargeError(
            f"duration {duration:.0f}s",
            user_message=f"This recording is {duration / 60:.0f} min long; the limit is "
            f"{settings.media_max_duration_seconds / 60:.0f} min.",
        )


def extract_audio(src: Path, dst: Path, max_seconds: float) -> float:
    """Decode → 16 kHz mono s16 WAV. Returns the decoded duration in seconds."""
    import av

    samples_written = 0
    max_samples = int(max_seconds * SAMPLE_RATE) + SAMPLE_RATE
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    try:
        with av.open(str(src)) as container, wave.open(str(tmp), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(stream):
                for res in resampler.resample(frame):  # type: ignore[arg-type]  # audio stream → AudioFrame
                    data = res.to_ndarray().tobytes()
                    out.writeframes(data)
                    samples_written += len(data) // 2
                if samples_written > max_samples:
                    raise DocumentTooLargeError("decoded duration over limit",
                                                user_message="This recording exceeds the maximum duration.")
            for res in resampler.resample(None):  # flush
                out.writeframes(res.to_ndarray().tobytes())
    except (av.error.FFmpegError, StopIteration) as exc:
        tmp.unlink(missing_ok=True)
        raise MediaError(repr(exc), user_message="The audio track could not be decoded.") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dst)
    with wave.open(str(dst), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ---------------------------------------------------------------- YouTube
def youtube_video_id(url: str) -> str:
    """Validate a YouTube URL against an allow-list and return its 11-character id."""
    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        raise MediaError("bad url", user_message="That is not a valid URL.") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in YOUTUBE_HOSTS:
        raise MediaError(f"host {host!r} not allowed", user_message="Only YouTube URLs are supported.")
    vid = None
    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        vid = (parse_qs(parsed.query).get("v") or [""])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        vid = parsed.path.split("/")[2]
    if not vid or not _YT_ID.match(vid):
        raise MediaError("no video id", user_message="Could not find a video id in that YouTube URL.")
    return vid


def youtube_metadata(video_id: str) -> dict[str, Any]:
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
        raise MediaError(repr(exc)[:300], user_message="YouTube metadata could not be retrieved.") from exc
    return {"title": info.get("title") or video_id, "duration_seconds": info.get("duration")}


def download_youtube_audio(video_id: str, dest_dir: Path) -> Path:
    """Download best audio with a FIXED filename (the original used the video title as path)."""
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except Exception as exc:  # noqa: BLE001
        raise MediaError(repr(exc)[:300], user_message="The YouTube audio could not be downloaded.") from exc
    files = [p for p in dest_dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise MediaError("download produced no file", user_message="The YouTube audio could not be downloaded.")
    return files[0]
