from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.fingerprint import fingerprint


def _q(text: str, qid: str) -> Question:
    fp = fingerprint(text)
    return Question(
        id=qid,
        question=text,
        tests_claim="claim",
        topic_path=["T05", "access", "coalescing"],
        question_type="optimization",
        difficulty=2,
        bloom="apply",
        persona="ncu_perf_engineer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        template_fingerprint=fp.skeleton,
        simhash=fp.simhash,
        exact_hash=fp.exact,
    )


def test_l0_exact_duplicate():
    settings = load_settings(fake_llm=True)
    d = Deduper(settings)
    a = _q("为什么 warp 内 stride=2 会破坏 coalescing？", "q1")
    d.add_question(a)
    fp = fingerprint("为什么  warp 内 stride=2 会破坏 coalescing？")
    dec = d.decide(fp, nearest_id=None, nearest_cosine=0.2)
    assert dec.verdict == "dup"
    assert dec.reason == "L0_exact"


def test_l1_template_duplicate_even_if_cosine_is_not_that_close():
    settings = load_settings(fake_llm=True)
    d = Deduper(settings)
    a = _q("在 H100 上 cudaMalloc 16 个元素会不会破坏 coalescing？", "q1")
    d.add_question(a)
    b_text = "在 B200 上 cudaMalloc 32 个元素会不会破坏 coalescing？"
    fp = fingerprint(b_text)
    dec = d.decide(fp, nearest_id="q1", nearest_cosine=0.70)
    assert dec.verdict == "dup"
    assert dec.reason == "L1_template"


def test_semantic_dup_at_high_cosine():
    settings = load_settings(fake_llm=True)
    d = Deduper(settings)
    a = _q("解释 TMA descriptor 的 128B 对齐要求。", "q1")
    d.add_question(a)
    fp = fingerprint("请说明 Hopper 上 TMA 描述符为什么必须满足 128 字节对齐。")
    dec = d.decide(fp, nearest_id="q1", nearest_cosine=0.95)
    assert dec.verdict == "dup"
    assert dec.reason == "L2_semantic"


def test_lenient_accept_when_same_topic_but_far_embedding():
    settings = load_settings(fake_llm=True)
    d = Deduper(settings)
    a = _q("为什么 __syncthreads 不能放进 divergent 分支？", "q1")
    d.add_question(a)
    fp = fingerprint("grid_group.sync 在非 cooperative launch 下为什么未定义？")
    dec = d.decide(fp, nearest_id="q1", nearest_cosine=0.55)
    assert dec.verdict == "unique"
