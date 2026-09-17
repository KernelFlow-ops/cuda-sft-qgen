"""Multi-key NVIDIA API pool with priority waiters, cooldowns, and backpressure.

LLM and embedding slots are independent. Writer work outranks salvage, chair,
sieve, and taxonomy. Repeated 429s shrink effective capacity; successes grow
it back toward the configured ceiling.
"""

from __future__ import annotations

import heapq
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Lower number = higher priority when waiting for an LLM slot.
PRI_WRITER = 0
PRI_SALVAGE = 1
PRI_CHAIR = 2
PRI_SIEVE = 3
PRI_TAXONOMY = 4


def parse_api_keys(
    environ: dict[str, str] | None = None,
) -> list[str]:
    """Collect up to several NVIDIA keys from env.

    Accepted forms (later duplicates are skipped):
    - NVIDIA_API_KEYS=k1,k2,k3
    - NVIDIA_API_KEY
    - NVIDIA_API_KEY_1 / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3 ...
    """
    env = environ if environ is not None else os.environ
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str | None) -> None:
        if not raw:
            return
        for part in raw.replace(";", ",").split(","):
            item = part.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            keys.append(item)

    add(env.get("NVIDIA_API_KEYS"))
    add(env.get("NVIDIA_API_KEY"))
    for i in range(1, 8):
        add(env.get(f"NVIDIA_API_KEY_{i}"))
    return keys


def mask_key(key: str) -> str:
    """Redact an API key for logs (prefix + suffix only)."""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}…{key[-4:]}"


@dataclass(order=True)
class _Waiter:
    """Priority-queue entry; cancelled waiters are skipped at pop time."""

    priority: int
    seq: int
    cancelled: bool = field(default=False, compare=False)


@dataclass
class KeySlot:
    """One API key with independent LLM/embed semaphores and cooldowns."""

    index: int
    key: str
    client: OpenAI
    llm_sem: threading.Semaphore
    embed_sem: threading.Semaphore
    llm_cooldown_until: float = 0.0
    embed_cooldown_until: float = 0.0

    def sem(self, kind: str) -> threading.Semaphore:
        """Semaphore for ``llm`` or ``embed`` in-flight calls on this key."""
        return self.llm_sem if kind == "llm" else self.embed_sem

    def kind_cooldown(self, kind: str) -> float:
        """Monotonic deadline until this kind may be acquired again."""
        return self.llm_cooldown_until if kind == "llm" else self.embed_cooldown_until

    @property
    def cooldown_until(self) -> float:
        """Latest cooldown across kinds (debug / backward compat)."""
        return max(self.llm_cooldown_until, self.embed_cooldown_until)

    @property
    def label(self) -> str:
        """Stable log label that never prints the full secret."""
        return f"key#{self.index + 1}({mask_key(self.key)})"


@dataclass
class ApiKeyPool:
    """Round-robin key pool with a priority heap of waiters per kind."""

    slots: list[KeySlot]
    llm_per_key: int
    embed_per_key: int
    embed_oversub: int = 2
    _rr: int = 0
    _seq: int = 0
    _cv: threading.Condition = field(default_factory=threading.Condition)
    inflight: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})
    peak: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})
    acquire_counts: dict[str, list[int]] = field(default_factory=dict)
    overlap_samples: int = 0
    _waiters: dict[str, list[_Waiter]] = field(default_factory=dict)
    _effective_cap: dict[str, int] = field(default_factory=dict)
    _rl_streak: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})
    _ok_streak: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})

    def __post_init__(self) -> None:
        """Initialize per-kind acquire counters, waiter heaps, and caps."""
        n = len(self.slots)
        self.acquire_counts = {
            "llm": [0] * n,
            "embed": [0] * n,
        }
        self._waiters = {"llm": [], "embed": []}
        self._effective_cap = {
            "llm": self.llm_capacity,
            "embed": self.embed_capacity,
        }
        self._rl_streak = {"llm": 0, "embed": 0}
        self._ok_streak = {"llm": 0, "embed": 0}

    @property
    def n_keys(self) -> int:
        """Number of configured API keys."""
        return len(self.slots)

    @property
    def llm_capacity(self) -> int:
        """Configured concurrent LLM calls across all keys."""
        return max(1, self.n_keys * max(1, self.llm_per_key))

    @property
    def embed_capacity(self) -> int:
        """Configured concurrent embed calls, including oversubscription."""
        over = max(1, int(self.embed_oversub))
        return max(1, self.n_keys * max(1, self.embed_per_key) * over)

    def stats(self) -> dict[str, Any]:
        """Snapshot of capacities, peaks, and per-key acquire counts."""
        return {
            "n_keys": self.n_keys,
            "llm_per_key": self.llm_per_key,
            "embed_per_key": self.embed_per_key,
            "embed_oversub": self.embed_oversub,
            "llm_capacity": self.llm_capacity,
            "embed_capacity": self.embed_capacity,
            "effective_llm_cap": self._effective_cap.get("llm", self.llm_capacity),
            "effective_embed_cap": self._effective_cap.get("embed", self.embed_capacity),
            "peak_llm": self.peak["llm"],
            "peak_embed": self.peak["embed"],
            "overlap_samples": self.overlap_samples,
            "llm_acquires_per_key": list(self.acquire_counts["llm"]),
            "embed_acquires_per_key": list(self.acquire_counts["embed"]),
        }

    def waiter_count(self, kind: str) -> int:
        """Count of non-cancelled waiters for ``llm`` or ``embed``."""
        with self._cv:
            return sum(1 for w in self._waiters.get(kind, []) if not w.cancelled)

    def _try_take_unlocked(self, kind: str, now: float) -> KeySlot | None:
        """Non-blocking acquire from the next non-cooling key (caller holds ``_cv``)."""
        n = len(self.slots)
        if n == 0:
            return None
        start = self._rr
        self._rr = (self._rr + 1) % n
        for offset in range(n):
            slot = self.slots[(start + offset) % n]
            if slot.kind_cooldown(kind) > now:
                continue
            if slot.sem(kind).acquire(blocking=False):
                self.inflight[kind] += 1
                if self.inflight[kind] > self.peak[kind]:
                    self.peak[kind] = self.inflight[kind]
                if self.inflight["llm"] > 0 and self.inflight["embed"] > 0:
                    self.overlap_samples += 1
                self.acquire_counts[kind][slot.index] += 1
                logger.debug(
                    "acquire %s %s inflight=%s/%s",
                    kind,
                    slot.label,
                    self.inflight[kind],
                    self.llm_capacity if kind == "llm" else self.embed_capacity,
                )
                return slot
        return None

    def acquire(self, kind: str, priority: int = 0) -> KeySlot:
        """Block until a slot of ``kind`` is available at this priority.

        Embedding waiters ignore ``priority`` (always FIFO). LLM waiters use a
        heap so writers run ahead of salvage/chair/taxonomy.

        Args:
            kind: ``llm`` or ``embed``.
            priority: Lower is higher priority; see ``PRI_*`` constants.

        Returns:
            The acquired :class:`KeySlot`. Caller must :meth:`release`.
        """
        if kind not in {"llm", "embed"}:
            raise ValueError(kind)
        pri = 0 if kind == "embed" else int(priority)
        waiter: _Waiter | None = None
        try:
            while True:
                now = time.monotonic()
                with self._cv:
                    if waiter is None:
                        waiter = _Waiter(priority=pri, seq=self._seq)
                        self._seq += 1
                        heapq.heappush(self._waiters[kind], waiter)
                    while self._waiters[kind] and self._waiters[kind][0].cancelled:
                        heapq.heappop(self._waiters[kind])
                    cap = max(1, int(self._effective_cap.get(kind, self.llm_capacity)))
                    head = self._waiters[kind][0] if self._waiters[kind] else None
                    if head is waiter and self.inflight[kind] < cap:
                        slot = self._try_take_unlocked(kind, now)
                        if slot is not None:
                            waiter.cancelled = True
                            heapq.heappop(self._waiters[kind])
                            return slot
                    soonest: float | None = None
                    for s in self.slots:
                        cd = s.kind_cooldown(kind)
                        if cd > now:
                            soonest = cd if soonest is None else min(soonest, cd)
                    timeout = 0.25
                    if soonest is not None:
                        timeout = min(1.0, max(0.05, soonest - now))
                    self._cv.wait(timeout=timeout)
        finally:
            if waiter is not None and not waiter.cancelled:
                waiter.cancelled = True
                with self._cv:
                    self._cv.notify_all()

    def release(self, kind: str, slot: KeySlot) -> None:
        """Return a slot and wake waiters."""
        slot.sem(kind).release()
        with self._cv:
            self.inflight[kind] = max(0, self.inflight[kind] - 1)
            self._cv.notify_all()

    def mark_cooldown(self, slot: KeySlot, seconds: float, kind: str = "llm") -> None:
        """Park a key after 429/overload; three hits in a row shrink effective cap."""
        if kind not in {"llm", "embed"}:
            kind = "llm"
        seconds = max(1.0, float(seconds))
        until = time.monotonic() + seconds
        if kind == "llm":
            slot.llm_cooldown_until = until
        else:
            slot.embed_cooldown_until = until
        self._rl_streak[kind] = self._rl_streak.get(kind, 0) + 1
        self._ok_streak[kind] = 0
        if self._rl_streak[kind] >= 3:
            floor = max(1, self.n_keys)
            self._effective_cap[kind] = min(self._effective_cap.get(kind, floor), floor)
            logger.warning(
                "%s %s cooldown %.0fs; effective_cap=%s",
                slot.label,
                kind,
                seconds,
                self._effective_cap[kind],
            )
        else:
            logger.warning("%s %s cooldown %.0fs", slot.label, kind, seconds)
        with self._cv:
            self._cv.notify_all()

    def note_success(self, kind: str) -> None:
        """Clear the rate-limit streak and slowly restore effective capacity."""
        if kind not in {"llm", "embed"}:
            return
        full = self.llm_capacity if kind == "llm" else self.embed_capacity
        self._rl_streak[kind] = 0
        self._ok_streak[kind] = self._ok_streak.get(kind, 0) + 1
        cur = int(self._effective_cap.get(kind, full))
        if cur < full and self._ok_streak[kind] >= 1:
            self._effective_cap[kind] = min(full, cur + 1)
            self._ok_streak[kind] = 0
            with self._cv:
                self._cv.notify_all()

    def remaining_capacity(self, kind: str) -> int:
        """Return number of available slots for the given kind (llm or embed).

        Returns:
            Number of slots available (capacity - inflight)
        """
        if kind not in {"llm", "embed"}:
            raise ValueError(f"kind must be 'llm' or 'embed', got {kind!r}")
        with self._cv:
            cap = self._effective_cap.get(kind, self.llm_capacity if kind == "llm" else self.embed_capacity)
            return max(0, cap - self.inflight.get(kind, 0))

    def remaining_capacity_fraction(self, kind: str) -> float:
        """Return fraction of remaining capacity (0.0 to 1.0).

        Returns:
            remaining / total_capacity
        """
        if kind not in {"llm", "embed"}:
            raise ValueError(f"kind must be 'llm' or 'embed', got {kind!r}")
        with self._cv:
            cap = self._effective_cap.get(kind, self.llm_capacity if kind == "llm" else self.embed_capacity)
            if cap <= 0:
                return 0.0
            remaining = max(0, cap - self.inflight.get(kind, 0))
            return remaining / cap


def build_key_pool(
    keys: list[str],
    *,
    base_url: str,
    llm_per_key: int,
    embed_per_key: int,
    embed_oversub: int = 2,
    timeout: float = 600.0,
) -> ApiKeyPool:
    """Construct OpenAI-compatible clients and per-key semaphores.

    Args:
        keys: Unique NVIDIA API keys.
        base_url: Chat/embeddings endpoint.
        llm_per_key: Concurrent LLM streams allowed per key.
        embed_per_key: Base concurrent embedding calls per key.
        embed_oversub: Multiplier on embed slots (embeddings are cheaper).
        timeout: Per-request HTTP timeout in seconds.

    Raises:
        RuntimeError: If ``keys`` is empty.
    """
    if not keys:
        raise RuntimeError(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3."
        )
    llm_per_key = max(1, int(llm_per_key))
    embed_per_key = max(1, int(embed_per_key))
    embed_oversub = max(1, int(embed_oversub))
    embed_slots = embed_per_key * embed_oversub
    slots: list[KeySlot] = []
    for i, key in enumerate(keys):
        client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=timeout,
            max_retries=0,
        )
        slots.append(
            KeySlot(
                index=i,
                key=key,
                client=client,
                llm_sem=threading.Semaphore(llm_per_key),
                embed_sem=threading.Semaphore(embed_slots),
            )
        )
    logger.info(
        "API key pool: %s key(s), llm %s/key, embed %s/key x%s (total llm=%s embed=%s)",
        len(slots),
        llm_per_key,
        embed_per_key,
        embed_oversub,
        len(slots) * llm_per_key,
        len(slots) * embed_slots,
    )
    return ApiKeyPool(
        slots=slots,
        llm_per_key=llm_per_key,
        embed_per_key=embed_per_key,
        embed_oversub=embed_oversub,
    )
