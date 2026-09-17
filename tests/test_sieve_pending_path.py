"""Writer-exit sieve: hw/leaf no longer hard-fail after a finished generate."""
from __future__ import annotations

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.consistency import consistency_reason, leaf_mismatch
from cuda_sft_qgen.sieve import rule_sieve


def _q(**kwargs) -> Question:
    data = dict(
        id="q1",
        question="为什么 __syncthreads 不能写在只被一半线程走到的分支里？请说明死锁原因。",
        tests_claim="__syncthreads must be reached by every thread in the block",
        topic_path=["T04", "barriers", "syncthreads"],
        l3_id="syncthreads",
        l3_claim="__syncthreads 必须被 block 内所有线程执行到",
        question_type="debug",
        difficulty=2,
        bloom="analyze",
        persona="code_reviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
    )
    data.update(kwargs)
    return Question.model_validate(data)


def test_rule_sieve_does_not_hard_fail_hw_mismatch():
    settings = load_settings(fake_llm=True)
    settings.question_min_chars = 40
    q = _q(
        hardware="hopper",
        l3_id="sm80_cp_async_l2_persist",
        l3_claim="Ampere sm_80 L2 persisting window via cudaStreamSetAttribute",
        question=(
            "A developer enables L2 cache persistence on an Ampere (sm_80) GPU using "
            "cudaStreamSetAttribute. Which statement is correct about the window?"
        ),
    )
    assert rule_sieve(q, settings) is None
    assert consistency_reason(q) == "sieve_hw_mismatch"


def test_chinese_on_leaf_stem_is_not_off_leaf():
    q = _q(
        l3_id="tile_scheduler_mutex",
        l3_claim="StreamKScheduler 与 PersistentScheduler 互斥",
        tests_claim="同一 Kernel 不能同时实例化 StreamKScheduler 与 PersistentScheduler",
        question=(
            "下列 Hopper 内核启动配置中，哪一项会触发编译期错误，因为同一 Kernel 模板 "
            "GemmKernel 同时实例化了 StreamKScheduler 与 PersistentScheduler？"
        ),
        hardware="hopper",
    )
    assert leaf_mismatch(q) is None


def test_true_off_leaf_still_detected_by_consistency():
    q = _q(
        l3_id="memcpy_peer_ipc",
        l3_claim="跨进程 memcpy 需要 cudaIpcGetMemHandle 打开的指针",
        tests_claim="grid.sync requires a cooperative launch on Ampere",
        hardware="ampere",
        question=(
            "On an Ampere GPU, a developer wants to synchronize all thread blocks "
            "in a grid using grid.sync(). Which snippet is correct?"
        ),
    )
    assert leaf_mismatch(q) == "sieve_off_leaf"
    settings = load_settings(fake_llm=True)
    settings.question_min_chars = 40
    assert rule_sieve(q, settings) is None
