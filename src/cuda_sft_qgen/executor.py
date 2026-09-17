"""Bounded priority thread pool used by generate, salvage, and review."""

from __future__ import annotations

import heapq
import logging
import threading
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _PriorityTask:
    """Heap item: lower ``priority`` runs first; ``seq`` breaks ties FIFO."""

    priority: int
    seq: int
    fn: Callable[..., Any] = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    future: Future = field(default_factory=Future, compare=False)


class PriorityExecutor:
    """Thread-safe priority executor with bounded queue and backpressure support.

    Tasks are executed in priority order (lower number = higher priority).
    When the queue reaches max_queue_size, submit() blocks until space is available.

    Priority levels (from keys.py):
    - Writer: 0
    - Salvage: 1
    - Review: 2
    - Taxonomy: 3
    """

    def __init__(self, max_workers: int, max_queue_size: int | None = None):
        """Initialize priority executor.

        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum pending tasks. None = unbounded. When full, submit() blocks.
        """
        self.max_workers = max(1, max_workers)
        if max_queue_size is not None and max_queue_size < 1:
            raise ValueError("max_queue_size must be positive or None")
        self.max_queue_size = max_queue_size
        self._queue: list[_PriorityTask] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._seq = 0
        self._shutdown = False
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"priority-exec-{i}", daemon=True)
            for i in range(self.max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(
        self,
        priority: int,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Submit a task with given priority.

        Args:
            priority: Task priority (0 = highest, 3 = lowest)
            fn: Callable to execute
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn

        Returns:
            Future representing the task result

        Raises:
            RuntimeError: If executor is shutdown
        """
        with self._cv:
            if self._shutdown:
                raise RuntimeError("PriorityExecutor is shutdown")

            # Block if queue is full
            while (
                self.max_queue_size is not None
                and len(self._queue) >= self.max_queue_size
                and not self._shutdown
            ):
                self._cv.wait()

            if self._shutdown:
                raise RuntimeError("PriorityExecutor is shutdown")

            task = _PriorityTask(
                priority=priority,
                seq=self._seq,
                fn=copy_context().run,
                args=(fn, *args),
                kwargs=kwargs,
            )
            self._seq += 1
            heapq.heappush(self._queue, task)
            self._cv.notify_all()
            return task.future

    def queue_size(self) -> int:
        """Return current number of pending tasks."""
        with self._lock:
            return len(self._queue)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for all pending tasks to complete
        """
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()

        if wait:
            for worker in self._workers:
                worker.join()

    def _worker_loop(self) -> None:
        """Only an available worker removes pending work from the bounded heap."""
        while True:
            with self._cv:
                while not self._queue and not self._shutdown:
                    self._cv.wait()

                if self._shutdown and not self._queue:
                    break

                if not self._queue:
                    continue

                task = heapq.heappop(self._queue)
                self._cv.notify_all()

            if not task.future.set_running_or_notify_cancel():
                continue
            try:
                result = task.fn(*task.args, **task.kwargs)
            except BaseException as exc:
                task.future.set_exception(exc)
            else:
                task.future.set_result(result)
