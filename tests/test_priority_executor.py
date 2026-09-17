"""Tests for PriorityExecutor ordering, backpressure, and shutdown."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from cuda_sft_qgen.executor import PriorityExecutor


def test_priority_executor_basic():
    """Test basic task submission and execution."""
    executor = PriorityExecutor(max_workers=2)

    def add(a: int, b: int) -> int:
        return a + b

    try:
        fut = executor.submit(0, add, 1, 2)
        assert isinstance(fut, Future)
        assert fut.result(timeout=1.0) == 3
    finally:
        executor.shutdown(wait=True)


def test_priority_executor_ordering():
    """Test that tasks execute in priority order."""
    executor = PriorityExecutor(max_workers=1, max_queue_size=10)
    results: list[int] = []
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        started.set()
        release.wait(timeout=5)

    def record(priority: int) -> None:
        time.sleep(0.01)  # Small delay to ensure ordering
        with lock:
            results.append(priority)

    try:
        blocker = executor.submit(0, block)
        assert started.wait(timeout=1)
        futs = []
        for pri in [3, 2, 1, 0]:
            fut = executor.submit(pri, record, pri)
            futs.append(fut)
        release.set()
        blocker.result(timeout=2)

        # Wait for all to complete
        for fut in futs:
            fut.result(timeout=2.0)

        # Should execute in priority order: 0, 1, 2, 3
        assert results == [0, 1, 2, 3]
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_priority_executor_backpressure():
    """Test that bounded queue blocks when full."""
    executor = PriorityExecutor(max_workers=1, max_queue_size=2)

    started = threading.Event()
    can_finish = threading.Event()

    def blocking_task() -> None:
        started.set()
        can_finish.wait(timeout=5.0)

    try:
        # Fill the worker
        fut1 = executor.submit(0, blocking_task)
        started.wait(timeout=1.0)

        # Fill the queue (max_queue_size=2)
        fut2 = executor.submit(0, lambda: None)
        fut3 = executor.submit(0, lambda: None)

        # Next submit should block
        submitted = threading.Event()

        def try_submit() -> None:
            executor.submit(0, lambda: None)
            submitted.set()

        blocker = threading.Thread(target=try_submit, daemon=True)
        blocker.start()

        # Should not complete immediately
        time.sleep(0.1)
        assert not submitted.is_set(), "Submit should block when queue is full"

        # Unblock and verify submit completes
        can_finish.set()
        fut1.result(timeout=1.0)
        blocker.join(timeout=1.0)
        assert submitted.is_set(), "Submit should complete after space is available"
    finally:
        can_finish.set()
        executor.shutdown(wait=True)


def test_priority_executor_exception_handling():
    """Test that exceptions are properly propagated."""
    executor = PriorityExecutor(max_workers=2)

    def failing_task() -> None:
        raise ValueError("test error")

    try:
        fut = executor.submit(0, failing_task)
        with pytest.raises(ValueError, match="test error"):
            fut.result(timeout=1.0)
    finally:
        executor.shutdown(wait=True)


def test_priority_executor_queue_size():
    """Test queue_size() method."""
    executor = PriorityExecutor(max_workers=1, max_queue_size=10)

    started = threading.Event()
    can_finish = threading.Event()

    def blocking_task() -> None:
        started.set()
        can_finish.wait(timeout=5.0)

    try:
        # Block the worker
        fut1 = executor.submit(0, blocking_task)
        started.wait(timeout=1.0)

        # Submit 3 more tasks
        fut2 = executor.submit(0, lambda: None)
        fut3 = executor.submit(0, lambda: None)
        fut4 = executor.submit(0, lambda: None)
        time.sleep(0.05)  # Let dispatcher catch up

        # Queue should contain 3 pending tasks
        assert executor.queue_size() >= 3, f"Expected >=3, got {executor.queue_size()}"

        # Unblock and verify queue drains
        can_finish.set()
        for fut in [fut1, fut2, fut3, fut4]:
            fut.result(timeout=1.0)

        time.sleep(0.05)
        assert executor.queue_size() == 0, "Queue should be empty after completion"
    finally:
        can_finish.set()
        executor.shutdown(wait=True)


def test_priority_executor_shutdown():
    """Test clean shutdown behavior."""
    executor = PriorityExecutor(max_workers=2)

    counter = {"value": 0}
    lock = threading.Lock()

    def increment() -> None:
        time.sleep(0.01)
        with lock:
            counter["value"] += 1

    # Submit tasks
    futs = [executor.submit(0, increment) for _ in range(5)]

    # Shutdown and wait
    executor.shutdown(wait=True)

    # All tasks should complete
    assert counter["value"] == 5, f"Expected 5, got {counter['value']}"

    # Further submits should fail
    with pytest.raises(RuntimeError, match="shutdown"):
        executor.submit(0, lambda: None)


def test_priority_executor_priorities():
    """Test that Writer (0) has higher priority than Review (2) and Taxonomy (3)."""
    executor = PriorityExecutor(max_workers=1, max_queue_size=20)
    results: list[str] = []
    lock = threading.Lock()

    started = threading.Event()
    can_finish = threading.Event()

    def blocking_task() -> None:
        started.set()
        can_finish.wait(timeout=5.0)

    def record(name: str) -> None:
        time.sleep(0.001)
        with lock:
            results.append(name)

    try:
        # Block the worker
        executor.submit(0, blocking_task)
        started.wait(timeout=1.0)

        # Submit in mixed order: Taxonomy (3), Review (2), Writer (0)
        executor.submit(3, record, "taxonomy")
        executor.submit(2, record, "review")
        executor.submit(0, record, "writer")
        executor.submit(1, record, "salvage")

        # Unblock and wait
        time.sleep(0.05)
        can_finish.set()
        time.sleep(0.2)

        # Writer should execute before Review and Taxonomy
        # Expected order: writer (0), salvage (1), review (2), taxonomy (3)
        assert results.index("writer") < results.index("review"), \
            f"Writer should execute before Review: {results}"
        assert results.index("writer") < results.index("taxonomy"), \
            f"Writer should execute before Taxonomy: {results}"
        assert results.index("salvage") < results.index("taxonomy"), \
            f"Salvage should execute before Taxonomy: {results}"
    finally:
        can_finish.set()
        executor.shutdown(wait=True)
