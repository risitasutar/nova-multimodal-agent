"""Shared helpers for video tests."""

from evaluation.fixtures.media import load_transcript, media_filename
from tests.agent.test_graph import router
from tests.fakes import fake_meeting_analyst


def structured(mapping: dict | None = None):
    """Route RawInsights to the fake analyst, everything else to the routing script."""
    route = router(mapping or {})

    def _fn(schema, msgs):
        if schema.__name__ == "RawInsights":
            return fake_meeting_analyst(schema, msgs)
        return route(schema, msgs)

    return _fn


def add_fixture_media(svc, thread_id: str, name: str = "quarterly_review_meeting", insights: bool = True):
    asset = svc.media.create_from_transcript(thread_id, media_filename(name), load_transcript(name))
    return svc.media.process(asset.media_id, insights=insights)
