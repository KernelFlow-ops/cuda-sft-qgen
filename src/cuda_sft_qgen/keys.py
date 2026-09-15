from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


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
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}…{key[-4:]}"


@dataclass
class KeySlot:
    index: int
    key: str
    client: OpenAI
    llm_sem: threading.Semaphore
    embed_sem: threading.Semaphore
    cooldown_until: float = 0.0

    def sem(self, kind: str) -> threading.Semaphore:
        return self.llm_sem if kind == "llm" else self.embed_sem

    @property
    def label(self) -> str:
        return f"key#{self.index + 1}({mask_key(self.key)})"


@dataclass
class ApiKeyPool:
    slots: list[KeySlot]
    llm_per_key: int
    embed_per_key: int
    _rr: int = 0
    _cv: threading.Condition = field(default_factory=threading.Condition)
    inflight: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})
    peak: dict[str, int] = field(default_factory=lambda: {"llm": 0, "embed": 0})
    acquire_counts: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.slots)
        self.acquire_counts = {
            "llm": [0] * n,
            "embed": [0] * n,
        }

    @property
    def n_keys(self) -> int:
        return len(self.slots)

    @property
    def llm_capacity(self) -> int:
        return max(1, self.n_keys * max(1, self.llm_per_key))

    @property
    def embed_capacity(self) -> int:
        return max(1, self.n_keys * max(1, self.embed_per_key))

    def stats(self) -> dict[str, Any]:
        return {
            "n_keys": self.n_keys,
            "llm_per_key": self.llm_per_key,
            "embed_per_key": self.embed_per_key,
            "llm_capacity": self.llm_capacity,
            "embed_capacity": self.embed_capacity,
            "peak_llm": self.peak["llm"],
            "peak_embed": self.peak["embed"],
            "llm_acquires_per_key": list(self.acquire_counts["llm"]),
            "embed_acquires_per_key": list(self.acquire_counts["embed"]),
        }

    def acquire(self, kind: str) -> KeySlot:
        if kind not in {"llm", "embed"}:
            raise ValueError(kind)
        while True:
            now = time.monotonic()
            soonest: float | None = None
            with self._cv:
                start = self._rr
                self._rr = (self._rr + 1) % max(len(self.slots), 1)
                for offset in range(len(self.slots)):
                    slot = self.slots[(start + offset) % len(self.slots)]
                    if slot.cooldown_until > now:
                        soonest = (
                            slot.cooldown_until
                            if soonest is None
                            else min(soonest, slot.cooldown_until)
                        )
                        continue
                    if slot.sem(kind).acquire(blocking=False):
                        self.inflight[kind] += 1
                        if self.inflight[kind] > self.peak[kind]:
                            self.peak[kind] = self.inflight[kind]
                        self.acquire_counts[kind][slot.index] += 1
                        logger.debug(
                            "acquire %s %s inflight=%s/%s",
                            kind,
                            slot.label,
                            self.inflight[kind],
                            self.llm_capacity if kind == "llm" else self.embed_capacity,
                        )
                        return slot
                timeout = 0.25
                if soonest is not None:
                    timeout = min(1.0, max(0.05, soonest - now))
                self._cv.wait(timeout=timeout)

    def release(self, kind: str, slot: KeySlot) -> None:
        slot.sem(kind).release()
        with self._cv:
            self.inflight[kind] = max(0, self.inflight[kind] - 1)
            self._cv.notify_all()

    def mark_cooldown(self, slot: KeySlot, seconds: float) -> None:
        seconds = max(1.0, float(seconds))
        slot.cooldown_until = time.monotonic() + seconds
        logger.warning("%s cooldown %.0fs", slot.label, seconds)
        with self._cv:
            self._cv.notify_all()


def build_key_pool(
    keys: list[str],
    *,
    base_url: str,
    llm_per_key: int,
    embed_per_key: int,
    timeout: float = 600.0,
) -> ApiKeyPool:
    if not keys:
        raise RuntimeError(
            "No NVIDIA API keys. Set NVIDIA_API_KEY / NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3."
        )
    llm_per_key = max(1, int(llm_per_key))
    embed_per_key = max(1, int(embed_per_key))
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
                embed_sem=threading.Semaphore(embed_per_key),
            )
        )
    logger.info(
        "API key pool: %s key(s), llm %s/key, embed %s/key (total llm=%s embed=%s)",
        len(slots),
        llm_per_key,
        embed_per_key,
        len(slots) * llm_per_key,
        len(slots) * embed_per_key,
    )
    return ApiKeyPool(
        slots=slots,
        llm_per_key=llm_per_key,
        embed_per_key=embed_per_key,
    )
