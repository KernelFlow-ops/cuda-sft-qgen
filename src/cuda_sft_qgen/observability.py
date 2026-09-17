"""Observability module - tracing and metrics support.

This module provides optional observability features. OpenTelemetry integration
can be added as an optional dependency.
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile of ``values`` (``p`` in 0..100)."""
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    rank = (len(xs) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo)


def trace_node(node_name: str) -> Callable[[F], F]:
    """Decorator to trace node execution time.

    Usage:
        @trace_node("my_node")
        def my_node(state: PipelineState) -> dict:
            ...

    Future: Can be extended to integrate with OpenTelemetry for distributed tracing.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                duration = time.monotonic() - start
                get_metrics().observe("node_seconds", duration, {"node": node_name})
                logger.debug(
                    "node_duration node=%s duration_seconds=%.3f",
                    node_name,
                    duration,
                )
                return result
            except Exception as e:
                duration = time.monotonic() - start
                get_metrics().increment("node_errors", labels={"node": node_name})
                logger.error(
                    "node_error node=%s duration_seconds=%.3f error=%s",
                    node_name,
                    duration,
                    str(e),
                )
                raise
        return wrapper  # type: ignore
    return decorator


class MetricsCollector:
    """Simple metrics collector for pipeline monitoring.

    Tracks counters and histograms. Can be extended to export to Prometheus.
    """

    def __init__(self) -> None:
        """Create empty counter/histogram maps guarded by an RLock."""
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Observe a histogram value."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float] | None:
        """Get histogram statistics (count, sum, avg, min, max)."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return None
        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    def snapshot(self) -> dict[str, Any]:
        """Export counters and histogram stats including p50/p95."""
        with self._lock:
            histograms = {}
            for key, values in self._histograms.items():
                if not values:
                    continue
                histograms[key] = {
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p50": _percentile(values, 50),
                    "p95": _percentile(values, 95),
                }
            return {"counters": dict(self._counters), "histograms": histograms}

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics
