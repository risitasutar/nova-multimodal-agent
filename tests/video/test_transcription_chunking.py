import pytest
import requests

from evaluation.fixtures.media import load_transcript
from nova.config import get_settings
from nova.video import transcription
from nova.video.chunking import chunk_segments
from nova.video.models import MediaAsset

SRT = """1
00:00:01,000 --> 00:00:04,500
Revenue target is <b>10 crore</b>.

2
00:12:23,200 --> 00:13:31,000
We agreed to hire 12 engineers.
"""

VTT = """WEBVTT

00:01.000 --> 00:04.500 align:start
Hello team.

01:02:03.000 --> 01:02:09.500
Long meeting cue.
"""


def test_parse_srt_keeps_timestamps_and_strips_tags():
    t = transcription.parse_subtitles(SRT)
    assert [(s.start_time, s.end_time) for s in t.segments] == [(1.0, 4.5), (743.2, 811.0)]
    assert t.segments[0].text == "Revenue target is 10 crore."
    assert all(s.speaker is None for s in t.segments)  # no diarization → no invented speakers
    assert t.segments[0].segment_id == "seg_0001"


def test_parse_vtt_hours():
    t = transcription.parse_subtitles(VTT)
    assert t.segments[1].start_time == 3723.0 and t.segments[1].end_time == 3729.5


def test_parse_rejects_empty():
    with pytest.raises(transcription.TranscriptionError):
        transcription.parse_subtitles("nothing here")


def test_language_routing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(transcription, "transcribe_whisper", lambda a, lang, s: calls.append(("whisper", lang)))
    monkeypatch.setattr(transcription, "transcribe_sarvam", lambda a, s: calls.append(("sarvam", None)))
    s = get_settings()
    transcription.transcribe(tmp_path / "a.wav", "english", s)
    transcription.transcribe(tmp_path / "a.wav", "auto", s)
    transcription.transcribe(tmp_path / "a.wav", "hinglish", s)
    assert calls == [("whisper", "english"), ("whisper", "auto"), ("sarvam", None)]


def test_engine_signature_changes_with_config(monkeypatch):
    a = transcription.engine_signature(get_settings(), "english")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    get_settings.cache_clear()
    assert transcription.engine_signature(get_settings(), "english") != a
    assert transcription.engine_signature(get_settings(), "hinglish").startswith("sarvam:")


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


class _FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, headers=None, timeout=None, files=None, data=None):
        self.calls.append((url, headers, data))
        return self.responses.pop(0)


def test_sarvam_pieces_get_real_offsets(monkeypatch, tmp_path):
    from tests.video.conftest import write_tone_wav

    monkeypatch.setenv("SARVAM_API_KEY", "sk-test-sarvam")
    get_settings.cache_clear()
    wav = write_tone_wav(tmp_path / "long.wav", seconds=60, rate=16000)
    session = _FakeSession([_FakeResp(200, {"transcript": "first part"}), _FakeResp(200, {"transcript": "second"}),
                            _FakeResp(200, {"transcript": "third"})])
    t = transcription.transcribe_sarvam(wav, get_settings(), session=session)
    assert [(s.start_time, s.end_time) for s in t.segments] == [(0.0, 25.0), (25.0, 50.0), (50.0, 60.0)]
    assert t.engine == "sarvam" and len(session.calls) == 3
    assert session.calls[0][1] == {"api-subscription-key": "sk-test-sarvam"}
    assert session.calls[0][2]["model"] == get_settings().sarvam_stt_model


def test_sarvam_requires_key(tmp_path):
    with pytest.raises(transcription.TranscriptionError, match="SARVAM_API_KEY"):
        transcription.transcribe_sarvam(tmp_path / "x.wav", get_settings())


def test_sarvam_http_failure_is_reported(monkeypatch, tmp_path):
    from tests.video.conftest import write_tone_wav

    monkeypatch.setenv("SARVAM_API_KEY", "k")
    monkeypatch.setattr(transcription.time, "sleep", lambda s: None)
    get_settings.cache_clear()
    wav = write_tone_wav(tmp_path / "a.wav", seconds=3, rate=16000)
    session = _FakeSession([_FakeResp(503, {}), _FakeResp(503, {})])
    with pytest.raises(transcription.TranscriptionError):
        transcription.transcribe_sarvam(wav, get_settings(), session=session)


def _asset():
    return MediaAsset(media_id="abcdef0123456789", thread_id="t", filename="meeting.mp4", source_type="video",
                      source_uri="meeting.mp4")


def test_chunks_preserve_timestamps_and_metadata():
    segs = load_transcript("quarterly_review_meeting").segments
    chunks = chunk_segments(segs, _asset(), chunk_size=300, overlap=1, duration=1205.4)
    assert chunks[0]["start_time"] == 0.0 and chunks[-1]["end_time"] == 1205.4
    for c in chunks:
        assert {"media_id", "chunk_id", "text", "start_time", "end_time", "source_type", "source_name",
                "language", "segment_ids"} <= set(c)
        assert c["start_time"] <= c["end_time"] and c["source_name"] == "meeting.mp4"
        first, last = c["segment_ids"][0], c["segment_ids"][-1]
        by_id = {s.segment_id: s for s in segs}
        assert c["start_time"] == by_id[first].start_time and c["end_time"] == by_id[last].end_time
    # the target decision keeps its true time span
    target = next(c for c in chunks if "10 crore" in c["text"])
    assert target["start_time"] <= 743.2 <= target["end_time"]


def test_chunk_overlap_and_progress():
    segs = load_transcript("quarterly_review_meeting").segments
    chunks = chunk_segments(segs, _asset(), chunk_size=200, overlap=1)
    assert len({s for c in chunks for s in c["segment_ids"]}) == len(segs)  # every segment indexed
    assert any(set(a["segment_ids"]) & set(b["segment_ids"]) for a, b in zip(chunks, chunks[1:]))
    huge_overlap = chunk_segments(segs, _asset(), chunk_size=200, overlap=50)
    assert len(huge_overlap) <= len(segs)  # terminates, always advances


def test_injection_in_transcript_is_flagged():
    chunks = chunk_segments(load_transcript("vendor_call_injection").segments, _asset(), 2000, 0)
    assert any("ignore previous instructions" in c["injection_flags"] for c in chunks)


def test_whisper_receives_decoded_samples(monkeypatch, tmp_path):
    """Whisper gets a float32 array (no file decoding inside Whisper → no FFmpeg/PyAV version coupling)."""
    import numpy as np

    from nova.video import ingest
    from tests.video.conftest import write_tone_wav

    wav16 = tmp_path / "a16.wav"
    ingest.extract_audio(write_tone_wav(tmp_path / "src.wav", seconds=1), wav16, 60)
    seen = {}

    class FakeModel:
        def transcribe(self, audio, language=None, beam_size=None, vad_filter=None):
            seen["audio"] = audio

            class Seg:
                start, end, text = 0.0, 1.0, " hello "

            class Info:
                language = "en"

            return iter([Seg()]), Info()

    monkeypatch.setattr(transcription, "_load_whisper", lambda s: FakeModel())
    t = transcription.transcribe_whisper(wav16, "english", get_settings())
    assert isinstance(seen["audio"], np.ndarray) and seen["audio"].dtype == np.float32
    assert len(seen["audio"]) == 16000 and float(np.abs(seen["audio"]).max()) <= 1.0
    assert t.segments[0].text == "hello" and t.language == "en"
