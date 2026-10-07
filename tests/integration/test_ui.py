"""
Headless Streamlit UI test (streamlit.testing.AppTest) against the real NovaService
wired to fakes: the page must render, run a turn, and show answer + metadata.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.agent.test_graph import router

APP = str(Path(__file__).resolve().parents[2] / "streamlit_app_pro.py")


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    import streamlit as st

    st.cache_data.clear()
    st.cache_resource.clear()
    yield


@pytest.fixture
def ui(make_service, monkeypatch):
    svc = make_service(structured_fn=router({}), answer_fn=lambda m: "Hello from Nova.")
    monkeypatch.setattr("nova.service.get_service", lambda: svc)
    monkeypatch.setattr("nova.service.check_llm", lambda s: (True, "test-model ready"))
    return svc


def test_ui_renders_home(ui):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("How can I help you today?" in m.value for m in at.markdown)
    assert len(at.button) >= 7  # new chat + demo queries


def test_ui_runs_a_calculation_turn(ui):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is 9283 * 47?").run()
    assert not at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "436,301" in texts and "Evidence verified" in texts


def test_ui_shows_friendly_error_when_model_down(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr("nova.service.get_service", lambda: svc)
    monkeypatch.setattr("nova.service.check_llm", lambda s: (False, "Nova cannot reach the local model server."))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Nova isn't ready" in m.value for m in at.markdown)


def test_ui_media_flow_shows_processing_and_meeting_intelligence(make_service, monkeypatch):
    from tests.video.helpers import structured

    svc = make_service(structured_fn=structured(), answer_fn=lambda m: "ok")
    monkeypatch.setattr("nova.service.get_service", lambda: svc)
    monkeypatch.setattr("nova.service.check_llm", lambda s: (True, "test-model ready"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if "Load sample meeting" in b.label).click().run()
    assert not at.exception
    tid = at.session_state["thread_id"]
    assets = svc.media.list_assets(tid)
    assert assets and assets[0].status.value == "COMPLETED"
    labels = [e.label for e in at.expander]
    assert any("Meeting intelligence" in lbl for lbl in labels)
    text = " ".join(m.value for m in at.markdown)
    assert "Key decisions" in text and "quarterly_review_meeting.mp4" in text
