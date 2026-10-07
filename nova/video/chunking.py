"""
Timestamp-aware transcript chunking.

Consecutive segments are grouped until `chunk_size` characters; each chunk keeps the
start of its first segment and the end of its last, plus the segment ids it covers.
A chunk also closes once it would span more than `max_seconds`, so a citation's time
window stays tight. `overlap` segments are repeated at the start of the next chunk so a sentence split across
a boundary is still retrievable. Timestamps are never interpolated or invented.
"""

from __future__ import annotations

from typing import Any

from nova.safety import find_injection_markers
from nova.video.models import MediaAsset, TranscriptSegment


def chunk_segments(
    segments: list[TranscriptSegment],
    asset: MediaAsset,
    chunk_size: int,
    overlap: int,
    duration: float | None = None,
    max_seconds: float | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    i = 0
    n = len(segments)
    while i < n:
        group: list[TranscriptSegment] = []
        size = 0
        j = i
        while j < n and (not group or (
            size + len(segments[j].text) + 1 <= chunk_size
            and (max_seconds is None or segments[j].end_time - group[0].start_time <= max_seconds)
        )):
            group.append(segments[j])
            size += len(segments[j].text) + 1
            j += 1
        text = " ".join(s.text for s in group)
        chunks.append(
            {
                "chunk_id": f"{asset.media_id[:8]}-c{len(chunks)}",
                "media_id": asset.media_id,
                "text": text,
                "start_time": group[0].start_time,
                "end_time": group[-1].end_time,
                "source_type": asset.media_kind if asset.source_type != "youtube" else "video",
                "source_name": asset.filename,
                "language": group[0].language,
                "duration_seconds": duration,
                "segment_ids": [s.segment_id for s in group],
                "injection_flags": find_injection_markers(text),
            }
        )
        if j >= n:
            break
        i = max(j - overlap, i + 1)  # always advance, even with a large overlap
    return chunks
