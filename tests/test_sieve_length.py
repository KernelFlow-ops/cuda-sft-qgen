from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.sieve import rule_sieve


def _q(text: str) -> Question:
    return Question(
        id="q1",
        question=text,
        tests_claim="__syncthreads deadlock",
        topic_path=["T04", "barriers", "syncthreads"],
        l3_id="syncthreads",
        question_type="debug",
        difficulty=2,
        bloom="analyze",
        persona="code_reviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
    )


def test_sieve_too_short_and_too_long():
    settings = load_settings(fake_llm=True)
    settings.question_min_chars = 40
    settings.question_max_chars = 80
    short = _q("太短的 CUDA 题")
    assert rule_sieve(short, settings) == "sieve_too_short"
    long = _q("CUDA kernel " + ("occupancy " * 40) + "为什么？")
    assert rule_sieve(long, settings) == "sieve_too_long"
    ok = _q("为什么 __syncthreads 不能写在只被一半线程走到的分支里？请说明死锁原因。")
    assert rule_sieve(ok, settings) is None


def test_question_length_from_env(monkeypatch):
    monkeypatch.setenv("QUESTION_MIN_CHARS", "55")
    monkeypatch.setenv("QUESTION_MAX_CHARS", "900")
    settings = load_settings(fake_llm=True)
    assert settings.question_min_chars == 55
    assert settings.question_max_chars == 900
    assert settings.sieve_min_chars == 55
