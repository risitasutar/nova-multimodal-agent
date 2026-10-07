import pytest

from nova.config import get_settings
from nova.errors import DocumentTooLargeError, UnsupportedFileError
from nova.video import ingest


@pytest.mark.parametrize("name,kind", [("a.mp4", "video"), ("b.MOV", "video"), ("c.avi", "video"), ("d.mkv", "video"),
                                        ("e.webm", "video"), ("f.mp3", "audio"), ("g.wav", "audio")])
def test_supported_extensions(name, kind):
    safe, k, _ = ingest.validate_upload(name, 100, get_settings())
    assert k == kind and safe == name


@pytest.mark.parametrize("name", ["notes.txt", "x.exe", "movie.flv", "noext"])
def test_rejects_unsupported(name):
    with pytest.raises(UnsupportedFileError):
        ingest.validate_upload(name, 100, get_settings())


def test_size_limit_and_empty(monkeypatch):
    monkeypatch.setenv("MEDIA_MAX_SIZE_MB", "1")
    get_settings.cache_clear()
    with pytest.raises(DocumentTooLargeError):
        ingest.validate_upload("big.mp4", 2 * 1024 * 1024, get_settings())
    with pytest.raises(ingest.MediaError):
        ingest.validate_upload("empty.mp4", 0, get_settings())


def test_unsafe_filename_is_sanitised():
    safe, _, _ = ingest.validate_upload("../../etc/<evil>.mp4", 10, get_settings())
    assert "/" not in safe and ".." not in safe.replace("..mp4", "") and safe.endswith(".mp4")


def test_probe_and_extract_real_wav(tone_wav, tmp_path):
    info = ingest.probe(tone_wav)
    assert info["has_audio"] and not info["has_video"] and info["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    out = tmp_path / "out.wav"
    seconds = ingest.extract_audio(tone_wav, out, max_seconds=60)
    assert seconds == pytest.approx(2.0, abs=0.1)
    import wave

    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)


def test_probe_and_extract_real_mp4(tone_mp4, tmp_path):
    info = ingest.probe(tone_mp4)
    assert info["has_video"] and info["has_audio"]
    assert ingest.extract_audio(tone_mp4, tmp_path / "a.wav", 60) == pytest.approx(2.0, abs=0.2)


def test_corrupt_media_rejected(tmp_path):
    bad = tmp_path / "fake.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(ingest.MediaError):
        ingest.probe(bad)


def test_duration_limit(tone_wav, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_MAX_DURATION_SECONDS", "1")
    get_settings.cache_clear()
    with pytest.raises(DocumentTooLargeError):
        ingest.check_duration(ingest.probe(tone_wav)["duration_seconds"], get_settings())
    with pytest.raises(DocumentTooLargeError):
        ingest.extract_audio(tone_wav, tmp_path / "x.wav", max_seconds=0.5)


@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s", "_Q-e_nczWqM"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
])
def test_youtube_ids(url, vid):
    assert ingest.youtube_video_id(url) == vid


@pytest.mark.parametrize("url", ["https://evil.com/watch?v=dQw4w9WgXcQ", "file:///etc/passwd", "ftp://youtube.com/x",
                                 "https://www.youtube.com/watch?v=short", "https://youtube.com.evil.io/watch?v=dQw4w9WgXcQ",
                                 "not a url"])
def test_youtube_rejects_bad_urls(url):
    with pytest.raises(ingest.MediaError):
        ingest.youtube_video_id(url)
