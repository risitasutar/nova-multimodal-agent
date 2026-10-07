import math
import struct
import wave
from pathlib import Path

import pytest


def write_tone_wav(path: Path, seconds: float = 2.0, rate: int = 22050) -> Path:
    """A tiny real WAV (sine tone) for decoding tests — no external tools needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                          for i in range(int(seconds * rate)))
        w.writeframes(frames)
    return path


@pytest.fixture
def tone_wav(tmp_path) -> Path:
    return write_tone_wav(tmp_path / "tone.wav")


@pytest.fixture
def tone_mp4(tmp_path, tone_wav) -> Path:
    from evaluation.fixtures.media import _wav_to_mp4

    out = tmp_path / "tone.mp4"
    _wav_to_mp4(tone_wav, out)
    return out
