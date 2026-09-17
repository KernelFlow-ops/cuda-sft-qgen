"""Tests that the API key pool serves writers ahead of sieve/chair waiters."""

from __future__ import annotations

import threading
import time

from cuda_sft_qgen.keys import PRI_SIEVE, PRI_WRITER, build_key_pool
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
