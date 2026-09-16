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


def test_sieve_accepts_exam_stems_without_question_mark():
    settings = load_settings(fake_llm=True)
    settings.question_min_chars = 40
    fill = _q(
        "下面是 Hopper 上使用 Tensor Core 与 TMA 的 kernel 片段。"
        "请在标记 // <<<FILL>>> 处补全两行代码，使 copy_async 的提交与等待配对。"
        "题干考查 __syncthreads 与 TMA wait 的配对。"
    )
    compute = _q(
        "A GEMM kernel computes C = A*B with fp16 C. Given M = 1024, N = 2048: "
        "Compute the total HBM traffic before fusion. Mention __syncthreads only "
        "as the CTA barrier before the store."
    )
    which = _q(
        "Examine the three host-side fragments below and indicate which ones are legal "
        "CUDA launches that still reach __syncthreads. Select the fragment(s) that are legal."
    )
    assert rule_sieve(fill, settings) is None
    assert rule_sieve(compute, settings) is None
    assert rule_sieve(which, settings) is None


def test_question_length_from_env(monkeypatch):
    monkeypatch.setenv("QUESTION_MIN_CHARS", "55")
    monkeypatch.setenv("QUESTION_MAX_CHARS", "900")
    settings = load_settings(fake_llm=True)
    assert settings.question_min_chars == 55
    assert settings.question_max_chars == 900
    assert settings.sieve_min_chars == 55
