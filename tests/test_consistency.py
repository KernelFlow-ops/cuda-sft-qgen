"""Tests for hardware/leaf consistency heuristics used as chair diagnostics."""

from __future__ import annotations

from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.consistency import (
    chair_may_reject_unique,
    consistency_reason,
    hardware_mismatch,
    leaf_mismatch,
    leaf_soft_suspect,
    scene_arches,
    sm_name_conflicts,
)
from cuda_sft_qgen.sieve import rule_sieve
from cuda_sft_qgen.config import load_settings


def _q(**kwargs) -> Question:
    data = dict(
        id="q1",
        question="为什么 __syncthreads 不能写在只被一半线程走到的分支里？请说明死锁原因。",
        tests_claim="__syncthreads must be reached by every thread in the block",
        topic_path=["T04", "barriers", "syncthreads"],
        l3_id="syncthreads",
        l3_claim="__syncthreads 必须被 block 内所有线程执行到，发散会死锁",
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


def test_blackwell_sm90_is_chair_suspect_not_hard_sieve():
    q = _q(
        hardware="blackwell",
        question=(
            "在 Blackwell (SM90) 上运行一个使用 wgmma.mma.sync 的双缓冲 GEMM kernel，"
            "NCU 报告 Long Scoreboard。请指出该 stall 的最小改动。"
        ),
    )
    assert sm_name_conflicts(q.question)
    assert hardware_mismatch(q) is None
    assert consistency_reason(q) == "sieve_off_leaf"
    assert leaf_soft_suspect(q) is True


def test_ampere_label_hopper_scene_mismatch():
    q = _q(
        hardware="ampere",
        l3_id="epilogue_fusion_tma_store",
        l3_claim="Hopper 可选 TMA store epilogue 把 D tile 写回 global",
        question=(
            "On a Hopper (sm_90) GPU you want to use the optional TMA store epilogue "
            "to write a D tile back to global memory. Which snippet is legal? "
            "Compute the required alignment."
        ),
    )
    assert hardware_mismatch(q) == "sieve_hw_mismatch"


def test_matching_arch_passes():
    q = _q(
        hardware="hopper",
        question=(
            "在 Hopper (SM90) 上，__syncthreads 写在只被一半线程走到的分支里会怎样？"
            "请说明死锁原因。"
        ),
    )
    assert hardware_mismatch(q) is None
    assert consistency_reason(q) is None


def test_agnostic_may_name_an_arch():
    q = _q(
        hardware="agnostic",
        question="On an Ampere GPU, why must __syncthreads be reached by every thread?",
    )
    assert hardware_mismatch(q) is None


def test_ipc_leaf_vs_grid_sync_stem_is_off_leaf():
    q = _q(
        l3_id="memcpy_peer_ipc",
        l3_claim="跨进程 memcpy 需要 cudaIpcGetMemHandle 打开的指针；普通 peer 指针不能交给另一进程",
        tests_claim="grid.sync requires a cooperative launch on Ampere",
        hardware="ampere",
        question=(
            "On an Ampere GPU, a developer wants to synchronize all thread blocks "
            "in a grid using grid.sync(). Two launch configurations are shown. "
            "Which snippet is guaranteed to execute grid.sync() correctly?"
        ),
    )
    assert leaf_mismatch(q) == "sieve_off_leaf"


def test_leaf_tokens_in_stem_pass():
    q = _q(
        l3_id="memcpy_peer_ipc",
        l3_claim="跨进程 memcpy 需要 cudaIpcGetMemHandle 打开的指针",
        tests_claim="cudaIpcGetMemHandle is required for cross-process memcpyPeer",
        question=(
            "Which host fragment uses cudaIpcGetMemHandle correctly before "
            "cudaMemcpyPeerAsync across processes? Identify the illegal call."
        ),
        hardware="agnostic",
    )
    assert leaf_mismatch(q) is None
    assert consistency_reason(q) is None


def test_sieve_runs_consistency():
    settings = load_settings(fake_llm=True)
    settings.question_min_chars = 40
    bad = _q(
        hardware="ampere",
        question=(
            "On a Hopper (sm_90) GPU you want to use the optional TMA store epilogue "
            "to write a D tile back to global memory. Which snippet is legal?"
        ),
        l3_id="graph_update",
        l3_claim="cudaGraphExecUpdate 可更新的参数集合",
    )
    # Cheap length/topic filters still run; hw/leaf are warnings, not hard-fail.
    assert rule_sieve(bad, settings) is None
    assert hardware_mismatch(bad) == "sieve_hw_mismatch"


def test_hopper_blackwell_slash_matches_labeled_blackwell():
    q = _q(
        hardware="blackwell",
        question_type="compare",
        question=(
            "Consider a decoupled look-back prefix-sum on Hopper/Blackwell where each "
            "CTA publishes a flag. Why must the next CTA acquire the flag before reading "
            "the partial sum? Mention __syncthreads only as a contrast."
        ),
    )
    assert "hopper" in scene_arches(q.question)
    assert "blackwell" in scene_arches(q.question)
    assert hardware_mismatch(q) is None


def test_chair_unique_reject_only_for_consistency():
    assert chair_may_reject_unique("leaf_mismatch")
    assert chair_may_reject_unique("hw_mismatch")
    assert chair_may_reject_unique("sieve_hw_mismatch")
    assert not chair_may_reject_unique("too easy")
    assert not chair_may_reject_unique(None)


def test_purge_rate_limit_rejects(tmp_path):
    from cuda_sft_qgen.store.archive import Archive

    store = Archive(tmp_path)
    store.append_rejected(
        _q(id="q_rl", reject_reason="generate_error:LLM call failed after retries: Error code: 429")
    )
    store.append_rejected(_q(id="q_ok", reject_reason="sieve_too_short"))
    dropped = store.purge_rate_limit_rejects()
    assert dropped == 1
    left = store.load_rejected()
    assert [q.id for q in left] == ["q_ok"]
