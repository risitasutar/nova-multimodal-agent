"""Media pipeline end to end through the HTTP API (real decoding, fake transcription/LLM)."""

import pytest
from fastapi.testclient import TestClient

from tests.video.conftest import write_tone_wav
from tests.video.helpers import structured


@pytest.fixture
def client(make_service, monkeypatch):
    from api import main

    svc = make_service(structured_fn=structured(), answer_fn=lambda m: "ok", video_min_relevance=0.0)
    main.app.dependency_overrides[main.service_dep] = lambda: svc
    monkeypatch.setattr("nova.service.check_llm", lambda s: (True, "test-model ready"))
    with TestClient(main.app, raise_server_exceptions=False) as c:
        c.svc = svc
        yield c
    main.app.dependency_overrides.clear()


def test_upload_process_summary_search_delete(client, tmp_path):
    from evaluation.fixtures.media import _wav_to_mp4

    wav = write_tone_wav(tmp_path / "t.wav", seconds=3)
    mp4 = tmp_path / "meeting.mp4"
    _wav_to_mp4(wav, mp4)
    up = client.post("/media/upload", data={"thread_id": "vt1", "language": "english"},
                     files={"file": ("meeting.mp4", mp4.read_bytes(), "video/mp4")})
    assert up.status_code == 201, up.text
    asset = up.json()
    assert asset["status"] == "QUEUED" and asset["source_type"] == "video" and asset["filename"] == "meeting.mp4"
    mid = asset["media_id"]

    done = client.post(f"/media/{mid}/process?wait=true").json()
    assert done["status"] == "COMPLETED", done
    assert list(done["steps"]) == ["validated", "audio_extracted", "transcribed", "indexed", "summarized", "insights"]
    assert done["duration_seconds"] == pytest.approx(3.0, abs=0.3)

    summary = client.get(f"/media/{mid}/summary").json()
    assert summary["decisions"] and all(d["start_time"] is not None for d in summary["decisions"])

    hits = client.post(f"/media/{mid}/search", json={"query": "fourth quarter revenue target", "top_k": 2}).json()
    assert hits["status"] == "success" and len(hits["results"]) == 2
    first = hits["results"][0]
    assert {"start_time", "end_time", "relevance", "source_name"} <= set(first) and first["media_id"] == mid

    thread = client.get("/threads/vt1").json()
    assert thread["media"][0]["media_id"] == mid

    assert client.delete(f"/media/{mid}").status_code == 204
    assert client.get(f"/media/{mid}").status_code == 404


def test_background_processing_and_status_poll(client, tmp_path):
    wav = write_tone_wav(tmp_path / "a.wav", seconds=1)
    mid = client.post("/media/upload", data={"thread_id": "vt2"},
                      files={"file": ("call.wav", wav.read_bytes(), "audio/wav")}).json()["media_id"]
    accepted = client.post(f"/media/{mid}/process")
    assert accepted.status_code == 202 and accepted.json()["poll"] == f"/media/{mid}"
    # TestClient runs background tasks before returning; the job is finished by now.
    assert client.get(f"/media/{mid}").json()["status"] == "COMPLETED"


def test_subtitle_import(client):
    srt = "1\n00:00:01,000 --> 00:00:05,000\nWe agreed to raise the target to 10 crore.\n"
    up = client.post("/media/upload", data={"thread_id": "vt3"}, files={"file": ("meeting.srt", srt.encode(), "text/plain")})
    assert up.status_code == 201 and up.json()["imported"] is True
    done = client.post(f"/media/{up.json()['media_id']}/process?wait=true&insights=false").json()
    assert done["status"] == "COMPLETED" and done["segments"] == 1


def test_media_errors(client):
    bad = client.post("/media/upload", data={"thread_id": "vt4"}, files={"file": ("x.exe", b"MZ", "application/x")})
    assert bad.status_code == 415
    corrupt = client.post("/media/upload", data={"thread_id": "vt4"},
                          files={"file": ("broken.mp4", b"junk" * 100, "video/mp4")}).json()
    failed = client.post(f"/media/{corrupt['media_id']}/process?wait=true").json()
    assert failed["status"] == "FAILED" and "Traceback" not in failed["error"]
    assert client.post(f"/media/{corrupt['media_id']}/search", json={"query": "x"}).status_code == 409
    assert client.get("/media/doesnotexist").status_code == 404
    assert client.post("/media/youtube", json={"thread_id": "vt4", "url": "https://evil.example/watch?v=dQw4w9WgXcQ"}
                       ).status_code == 422


def test_youtube_registration_is_validated_without_download(client):
    r = client.post("/media/youtube", json={"thread_id": "vt5", "url": "https://youtu.be/dQw4w9WgXcQ"})
    assert r.status_code == 201
    body = r.json()
    assert body["source_type"] == "youtube" and body["source_key"] == "yt-dQw4w9WgXcQ" and body["status"] == "QUEUED"


def test_chat_over_http_uses_video_evidence(client):
    from tests.video.helpers import add_fixture_media

    add_fixture_media(client.svc, "vt6", insights=False)
    client.svc.deps.llm.answer_fn = lambda msgs: "The target is 10 crore [V1]."
    r = client.post("/chat", json={"thread_id": "vt6", "message": "What revenue target was set in the meeting?"}).json()
    assert r["route"] == "video" and r["citations"][0]["type"] == "video"
    assert r["citations"][0]["start_time"] is not None and "[VIDEO: quarterly_review_meeting.mp4," in r["answer"]
