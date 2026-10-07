"""
Live video e2e (opt-in: pytest -m e2e): real speech (Windows TTS → MP4) → PyAV → real Whisper →
real embeddings/FAISS → real LLM answer with a timestamp citation → deletion.

Skipped when Ollama is not ready, when not on Windows (TTS fixture), or when the Whisper model is not
available locally (it is downloaded on first use; pre-fetch it to run this test offline).
"""

import os
import sys

import pytest

from nova.llm import check_llm

pytestmark = pytest.mark.e2e


def _whisper_available(model: str) -> bool:
    try:
        from faster_whisper.utils import download_model

        download_model(model, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(autouse=True)
def isolated_settings():  # keep the live service's settings
    yield


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    ok, msg = check_llm()
    if not ok:
        pytest.skip(f"Ollama not ready: {msg}")
    if sys.platform != "win32":
        pytest.skip("spoken fixture uses Windows text-to-speech")
    from nova.config import get_settings

    os.environ["NOVA_DATA_DIR"] = str(tmp_path_factory.mktemp("live-video"))
    get_settings.cache_clear()
    if not _whisper_available(get_settings().whisper_model):
        pytest.skip("Whisper model not available locally")
    from nova.service import NovaService

    return NovaService.from_settings(), tmp_path_factory.mktemp("spoken")


def test_spoken_video_end_to_end(live):
    from evaluation.fixtures.media import build_spoken_media

    svc, tmp = live
    _, mp4 = build_spoken_media(tmp)
    asset = svc.media.create_upload("e2e-video", "budget_review.mp4", mp4.read_bytes(), "english")
    done = svc.media.process(asset.media_id, insights=False)
    assert done.status.value == "COMPLETED", done.error
    transcript = svc.media.transcript(asset.media_id).text.lower()
    # Whisper small hears the TTS "lakh" as "lock" (observed); assert on words it recognises reliably.
    assert "30" in transcript and "cloud budget" in transcript and "ravi" in transcript
    r = svc.chat("What cloud budget was agreed in the budget review video?", "e2e-video")
    assert "30" in r.answer and any(c["type"] == "video" for c in r.citations)
    svc.media.delete(asset.media_id)
    assert svc.media.list_assets("e2e-video") == []
