"""
Tiny in-process metrics registry (no external service required).

Counters and latency samples are kept in memory and exposed by `GET /metrics`.
For multi-replica deployments this would be replaced by a Prometheus client.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100). Returns 0.0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class MetricsRegistry:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._started = time.time()

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, ms: float) -> None:
        with self._lock:
            self._latencies[name].append(float(ms))

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000)

    def snapshot(self) -> dict:
        with self._lock:
            latency = {
                name: {
                    "count": len(samples),
                    "avg_ms": round(sum(samples) / len(samples), 1) if samples else 0.0,
                    "p50_ms": round(percentile(list(samples), 50), 1),
                    "p95_ms": round(percentile(list(samples), 95), 1),
                    "max_ms": round(max(samples), 1) if samples else 0.0,
                }
                for name, samples in self._latencies.items()
            }
            return {
                "uptime_s": round(time.time() - self._started, 1),
                "counters": dict(self._counters),
                "latency": latency,
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._latencies.clear()


metrics = MetricsRegistry()
