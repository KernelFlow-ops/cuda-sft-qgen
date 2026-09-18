"""Tests that the API key pool serves writers ahead of sieve/chair waiters."""

from __future__ import annotations

import threading
import time

from cuda_sft_qgen.keys import PRI_CHAIR, PRI_SIEVE, PRI_WRITER, build_key_pool
from cuda_sft_qgen.llm import FakeLLM
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.sieve import llm_sieve


def test_writer_priority_beats_sieve():
    pool = build_key_pool(
        ["k1"],
        base_url="https://example.invalid/v1",
        llm_per_key=1,
        embed_per_key=1,
        embed_oversub=1,
    )
    held = pool.acquire("llm", priority=PRI_WRITER)
    order: list[str] = []
    started_low = threading.Event()
    started_high = threading.Event()

    def worker(name: str, pri: int, started: threading.Event) -> None:
        started.set()
        slot = pool.acquire("llm", priority=pri)
        order.append(name)
        time.sleep(0.05)
        pool.release("llm", slot)

    t_low = threading.Thread(target=worker, args=("sieve", PRI_SIEVE, started_low))
    t_high = threading.Thread(target=worker, args=("writer", PRI_WRITER, started_high))
    t_low.start()
    started_low.wait(1)
    for _ in range(80):
        if pool.waiter_count("llm") >= 1:
            break
        time.sleep(0.01)
    t_high.start()
    started_high.wait(1)
    for _ in range(80):
        if pool.waiter_count("llm") >= 2:
            break
        time.sleep(0.01)
    pool.release("llm", held)
    t_high.join(2)
    t_low.join(2)
    assert order, "waiters never acquired"
    assert order[0] == "writer"


def test_chair_reserve_takes_a_slot_while_writers_wait():
    """A reserved chair lane is granted even if a writer is also queued."""
    pool = build_key_pool(
        ["k1", "k2"],
        base_url="https://example.invalid/v1",
        llm_per_key=1,
        embed_per_key=1,
        embed_oversub=1,
        llm_chair_reserve=1,
    )
    held_a = pool.acquire("llm", priority=PRI_WRITER)
    held_b = pool.acquire("llm", priority=PRI_WRITER)
    order: list[str] = []
    started_chair = threading.Event()
    started_writer = threading.Event()

    def worker(name: str, pri: int, started: threading.Event) -> None:
        started.set()
        slot = pool.acquire("llm", priority=pri)
        order.append(name)
        time.sleep(0.05)
        pool.release("llm", slot)

    t_chair = threading.Thread(target=worker, args=("chair", PRI_CHAIR, started_chair))
    t_writer = threading.Thread(target=worker, args=("writer", PRI_WRITER, started_writer))
    t_chair.start()
    started_chair.wait(1)
    t_writer.start()
    started_writer.wait(1)
    for _ in range(80):
        if pool.waiter_count("llm") >= 2:
            break
        time.sleep(0.01)
    pool.release("llm", held_a)
    t_chair.join(2)
    assert order, "no waiter acquired after first release"
    assert order[0] == "chair"
    pool.release("llm", held_b)
    t_writer.join(2)
    assert "writer" in order


def test_llm_sieve_uses_low_priority():
    class Rec(FakeLLM):
        def __init__(self) -> None:
            self.prios: list[int] = []

        def complete(self, messages, *, priority=0, **kwargs):  # noqa: ANN001
            self.prios.append(int(priority))
            return super().complete(messages, priority=priority, **kwargs)

    llm = Rec()
    q = Question.model_validate(
        {
            "id": "q1",
            "question": "为什么 Hopper 上 TMA 对齐失败会导致静默错结果？",
            "tests_claim": "TMA alignment",
            "topic_path": ["T08", "tma"],
            "question_type": "debug",
            "difficulty": 3,
            "bloom": "analyze",
            "persona": "code_reviewer",
            "hardware": "hopper",
            "granularity": "kernel",
            "language": "zh",
        }
    )
    llm_sieve(q, llm)
    assert llm.prios
    assert llm.prios[0] == PRI_SIEVE
