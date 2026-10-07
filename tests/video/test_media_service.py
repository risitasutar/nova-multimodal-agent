import pytest

from nova.errors import NovaError
from nova.video.models import MediaStatus
from nova.video.service import MediaNotFoundError
from tests.video.helpers import add_fixture_media, structured


def test_imported_transcript_pipeline(make_service):
    svc = make_service(structured_fn=structured())
    asset = add_fixture_media(svc, "t1")
    assert asset.status == MediaStatus.COMPLETED and asset.error is None
    assert asset.segments == 20 and asset.chunks and asset.chunks >= 4
    assert asset.steps == {"validated": "cached", "audio_extracted": "cached", "transcribed": "cached",
                           "indexed": "done", "summarized": "done", "insights": "done"}
    assert asset.duration_seconds == 1205.4 and asset.vector_collection_id == asset.media_id
    assert not asset.transcript_path.startswith(("/", "C:"))  # stored relative, no absolute paths


def test_insights_are_verified_against_transcript(make_service):
    svc = make_service(structured_fn=structured())
    asset = add_fixture_media(svc, "t1")
    ins = svc.media.insights(asset.media_id, "t1")
    assert ins.decisions and all(d.start_time is not None and d.chunk_ids for d in ins.decisions)
    assert not any("Unsupported invented decision" in d.text for d in ins.decisions)  # cited C999: dropped
    assert ins.dropped_items >= 1
    owners = {a.owner for a in ins.action_items}
    assert "Arjun" in owners  # named in the transcript → kept
    assert "Rahul" not in owners  # invented by the model → removed (shown as "Not identified")
    arjun = next(a for a in ins.action_items if a.owner == "Arjun")
    assert arjun.deadline == "Friday" and arjun.start_time <= 431.0 <= arjun.end_time
    assert ins.open_questions


def test_upload_pipeline_extracts_audio_and_caches_transcription(make_service, tone_wav):
    calls = []

    def counting_transcriber(audio, language, settings):
        from tests.conftest import fake_transcriber

        calls.append(audio)
        return fake_transcriber(audio, language, settings)

    svc = make_service(structured_fn=structured(), transcriber=counting_transcriber)
    data = tone_wav.read_bytes()
    a = svc.media.create_upload("t1", "tone.wav", data)
    a = svc.media.process(a.media_id)
    assert a.status == MediaStatus.COMPLETED and a.steps["audio_extracted"] == "done"
    assert a.steps["transcribed"] == "done" and len(calls) == 1
    assert {"audio_extraction_ms", "transcription_ms", "indexing_ms", "total_ms"} <= set(a.timings_ms)
    # Same bytes in another thread: transcript, vectors and insights all come from the cache.
    b = svc.media.process(svc.media.create_upload("t2", "copy.wav", data).media_id)
    assert len(calls) == 1 and b.steps["transcribed"] == "cached" and b.steps["insights"] == "cached"
    # Re-processing the same asset also reuses the cache.
    svc.media.process(a.media_id)
    assert len(calls) == 1


def test_failure_is_recorded_not_raised(make_service, tone_wav):
    def broken(audio, language, settings):
        raise NovaError("boom", user_message="Transcription failed.")

    svc = make_service(structured_fn=structured(), transcriber=broken)
    a = svc.media.process(svc.media.create_upload("t1", "x.wav", tone_wav.read_bytes()).media_id)
    assert a.status == MediaStatus.FAILED and a.error == "Transcription failed."
    assert a.steps["validated"] == "done" and "transcribed" not in a.steps


def test_corrupt_upload_fails_cleanly(make_service):
    svc = make_service(structured_fn=structured())
    a = svc.media.process(svc.media.create_upload("t1", "bad.mp4", b"garbage" * 50).media_id)
    assert a.status == MediaStatus.FAILED and "not a readable" in a.error


def test_insights_can_be_skipped(make_service):
    svc = make_service(structured_fn=structured())
    a = add_fixture_media(svc, "t1", insights=False)
    assert a.steps["insights"] == "skipped" and svc.media.insights(a.media_id) is None


def test_search_returns_timestamped_scored_evidence(make_service):
    svc = make_service(structured_fn=structured())
    add_fixture_media(svc, "t1")
    r = svc.media.search("t1", "fourth quarter revenue target 10 crore")
    top = r.chunks[0]
    assert top["start_time"] <= 743.2 <= top["end_time"] and "10 crore" in top["text"]
    assert top["source_name"] == "quarterly_review_meeting.mp4" and top["dense_score"] > 0


def test_thread_isolation(make_service):
    svc = make_service(structured_fn=structured())
    a = add_fixture_media(svc, "thread-a", "quarterly_review_meeting")
    b = add_fixture_media(svc, "thread-b", "product_sync_meeting")
    hits_a = svc.media.search("thread-a", "Orbit mobile app launch Flutter budget lakh", min_relevance=0.0)
    assert hits_a.chunks and all(c["media_id"] == a.media_id for c in hits_a.chunks)
    hits_b = svc.media.search("thread-b", "fourth quarter revenue target crore", min_relevance=0.0)
    assert all(c["media_id"] == b.media_id for c in hits_b.chunks)
    with pytest.raises(MediaNotFoundError):  # cannot read another thread's media by id
        svc.media.get(b.media_id, "thread-a")
    with pytest.raises(MediaNotFoundError):
        svc.media.insights(b.media_id, "thread-a")


def test_delete_removes_everything(make_service, tone_wav):
    svc = make_service(structured_fn=structured())
    s = svc.settings
    data = tone_wav.read_bytes()
    a = svc.media.process(svc.media.create_upload("t1", "a.wav", data).media_id)
    b = svc.media.process(svc.media.create_upload("t2", "b.wav", data).media_id)  # shares the cache
    key = a.source_key
    svc.media.delete(a.media_id)
    assert svc.media.registry.get(a.media_id) is None
    assert not (s.media_dir / "t1" / a.media_id).exists()
    assert not svc.media.store.collection_exists("t1", a.media_id)  # no orphaned vectors
    assert (s.media_cache_dir / key).exists()  # still referenced by b
    svc.media.delete(b.media_id)
    assert not (s.media_cache_dir / key).exists()  # last reference gone → transcript/audio/insights removed
    assert svc.media.search("t2", "anything", min_relevance=0.0).chunks == []


def test_thread_deletion_cascades_to_media(make_service):
    svc = make_service(structured_fn=structured(), answer_fn=lambda m: "ok")
    a = add_fixture_media(svc, "gone")
    svc.chat("hello", "gone")
    svc.delete_thread("gone")
    assert svc.media.list_assets("gone") == [] and svc.media.registry.get(a.media_id) is None


def test_media_limit_and_bad_ids(make_service, monkeypatch):
    from nova.config import get_settings
    from nova.errors import InputValidationError

    monkeypatch.setenv("NOVA_MAX_MEDIA_PER_THREAD", "1")
    get_settings.cache_clear()
    svc = make_service(structured_fn=structured())
    add_fixture_media(svc, "t1", insights=False)
    with pytest.raises(InputValidationError):
        add_fixture_media(svc, "t1", "product_sync_meeting", insights=False)
    with pytest.raises(InputValidationError):
        svc.media.create_upload("../etc", "a.wav", b"x")


def test_interrupted_jobs_are_marked_failed_on_restart(make_service):
    from nova.video.service import MediaService

    svc = make_service(structured_fn=structured())
    a = svc.media.create_from_transcript("t1", "m.mp4",
                                         __import__("evaluation.fixtures.media", fromlist=["x"]).load_transcript(
                                             "product_sync_meeting"))
    a.status = MediaStatus.PROCESSING
    svc.media.registry.save(a)
    MediaService(svc.settings, svc.media.registry, svc.media.store, svc.media.retriever, svc.deps.llm)
    assert svc.media.registry.get(a.media_id).status == MediaStatus.FAILED


def test_uncited_items_are_aligned_to_supporting_chunk_or_dropped():
    from nova.video.extraction import RawAction, RawItem, verify_items

    chunks = {
        "C0": {"chunk_id": "C0", "text": "Arjun said he will have the revised pricing sheet ready by Friday.",
               "start_time": 431.0, "end_time": 466.4},
        "C1": {"chunk_id": "C1", "text": "We still do not know whether the Pune office lease will be renewed.",
               "start_time": 960.0, "end_time": 994.7},
    }
    actions, dropped = verify_items(
        [RawAction(task="Prepare the revised pricing sheet", owner="Arjun", deadline="Friday", chunk_ids=[]),
         RawAction(task="Book the company offsite in Goa", owner="Rahul", deadline=None, chunk_ids=[])],
        chunks, action=True)
    assert dropped == 1 and len(actions) == 1
    a = actions[0]
    assert (a.chunk_ids, a.start_time, a.end_time, a.owner, a.deadline) == (["C0"], 431.0, 466.4, "Arjun", "Friday")
    items, _ = verify_items([RawItem(text="Pune office lease renewal is unresolved", chunk_ids=["C9"])], chunks)
    assert items[0].chunk_ids == ["C1"]  # invalid id → aligned to the chunk that actually says it
