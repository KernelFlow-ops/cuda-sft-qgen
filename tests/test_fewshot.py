from __future__ import annotations

from cuda_sft_qgen.models import Question, QuestionBrief
from cuda_sft_qgen.prompts.fewshot import (
    FewshotCfg,
    FewshotCorpus,
    is_quality_reject,
    select_anti,
    select_negatives,
    select_positives,
)
from cuda_sft_qgen.review.fingerprint import fingerprint


def _q(**kwargs) -> Question:
    text = kwargs.pop("question", "为什么 CUDA kernel 在 __syncthreads 发散时会死锁？请说明。")
    fp = fingerprint(text)
    base = dict(
        id="q1",
        question=text,
        tests_claim="syncthreads deadlock",
        topic_path=["T04", "barriers", "syncthreads"],
        l3_id="syncthreads",
        question_type="debug",
        difficulty=2,
        bloom="analyze",
        persona="code_reviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        template_fingerprint=fp.skeleton,
        simhash=fp.simhash,
        exact_hash=fp.exact,
        dup_verdict="unique",
        warnings=[],
        nearest_cosine=0.3,
    )
    base.update(kwargs)
    return Question.model_validate(base)


def _brief(**kwargs) -> QuestionBrief:
    data = dict(
        topic_path=["T08", "tma", "tma_desc"],
        l3_id="tma_desc",
        l3_claim="TMA descriptor alignment",
        question_type="debug",
        difficulty=3,
        bloom="analyze",
        persona="code_reviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
        kind="failure",
    )
    data.update(kwargs)
    return QuestionBrief.model_validate(data)


def test_quality_reject_filters_429():
    assert is_quality_reject("sieve_too_short")
    assert is_quality_reject("L1_template")
    assert not is_quality_reject("generate_error:429")
    assert not is_quality_reject("retry_exhausted:429")


def test_positive_slots_cover_dimensions():
    accepted = [
        _q(id="a1", l3_id="div", question_type="debug", topic_path=["T03", "warp", "div"],
           nearest_cosine=0.2, question="debug A 为什么 __syncwarp mask 不全会死锁？"),
        _q(id="a2", l3_id="bank", question_type="debug", topic_path=["T02", "smem", "bank"],
           nearest_cosine=0.25, question="debug B 共享内存 stride=2 为何 2-way conflict？"),
        _q(id="a3", l3_id="tma_desc", question_type="layout", topic_path=["T08", "tma", "tma_desc"],
           difficulty=3, bloom="analyze", question="layout 同叶 求 TMA box 的 Shape/Stride？"),
        _q(id="a4", l3_id="occ", question_type="optimization", topic_path=["T03", "occ", "occ"],
           difficulty=3, bloom="analyze", nearest_cosine=0.4,
           question="optimization 同难度 占用公式 regs=64 block=256 求驻留 CTA？"),
        _q(id="a5", l3_id="cute", question_type="layout", topic_path=["T07", "lay", "cute"],
           granularity="kernel", simhash=0,
           question="layout 跨域 composition(Shape<_16,_8>,Stride<_8,_1>) 的偏移？"),
        _q(id="a6", l3_id="fast", question_type="debug", topic_path=["T08", "tma", "fast"],
           nearest_cosine=0.69, warnings=["gray_zone"],
           question="高相似 debug 不应进槽 A 的同 type 题？"),
    ]
    # Give a5 a simhash close to a1 for slot D.
    a1 = accepted[0]
    accepted[4].simhash = int(a1.simhash) ^ 0xFF
    cfg = FewshotCfg(simhash_gray_hamming=0, simhash_hamming_max=64)
    corpus = FewshotCorpus(accepted=accepted)
    picks = select_positives(_brief(), corpus, cfg)
    ids = [p.question.id for p in picks]
    assert "a6" not in ids
    slots = {p.slot for p in picks}
    assert "A" in slots
    assert len(ids) == len(set(ids))
    assert len(picks) >= 4


def test_negatives_prefer_same_l3_and_skip_empty_429():
    rejected = [
        _q(id="r429", question="", reject_reason="generate_error:429", l3_id="tma_desc"),
        _q(id="r1", l3_id="tma_desc", question_type="debug",
           reject_reason="sieve_too_short",
           question="太短"),
        _q(id="r2", l3_id="other", question_type="layout",
           reject_reason="L1_template",
           question="另一叶被骨架判重的 CUDA 题？"),
    ]
    corpus = FewshotCorpus(rejected_quality=[q for q in rejected if q.question])
    picks = select_negatives(_brief(), corpus, FewshotCfg(negatives=2))
    assert picks[0].question.id == "r1"
    assert all(p.question.question for p in picks)


def test_anti_includes_static_and_caps():
    brief = _brief(anti_templates=["dynamic-skel"])
    corpus = FewshotCorpus(anti_skeletons=["L1-skel"])
    items = select_anti(brief, corpus, FewshotCfg(anti_max=15), ["static-a", "static-b"])
    assert items[0] == "static-a"
    assert "dynamic-skel" in items
    assert "L1-skel" in items
