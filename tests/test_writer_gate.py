"""Pre-chair writer consistency gate: local defects and rewrite-once."""

from __future__ import annotations

from cuda_sft_qgen.graph.nodes.generate_core import generate_or_salvage
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.writer_gate import local_writer_defects, writer_gate_reason
from cuda_sft_qgen.runtime import set_runtime


def _q(**kwargs) -> Question:
    data = dict(
        id="q1",
        question="为什么 Hopper 上 TMA 128B 对齐失败会导致静默错结果？请给出可观察后果。",
        tests_claim="TMA 128B alignment failure is silent corruption",
        topic_path=["T08", "tma", "tma_align_128"],
        l3_id="tma_align_128",
        l3_claim="Hopper TMA 要求 128B 对齐",
        question_type="debug",
        difficulty=3,
        bloom="analyze",
        persona="code_reviewer",
        hardware="hopper",
        granularity="kernel",
        language="zh",
    )
    data.update(kwargs)
    return Question.model_validate(data)


def test_local_gate_passes_clean_stem():
    assert local_writer_defects(_q()) is None


def test_local_gate_catches_cute_kernel_vs_numba():
    q = _q(
        l3_claim="在 @cute.kernel 里捕获普通 Python list 当闭包会失败",
        tests_claim="capturing host objects in @cute.kernel is illegal",
        question=(
            "下面 Numba 代码能编译但结果全是 0。请指出闭包捕获 numpy 的问题，并对比 CUTLASS @cute.kernel。\n"
            "```python\nfrom numba import cuda\n@cuda.jit\ndef k(out):\n    out[0]=1\n```\n为什么？"
        ),
    )
    assert local_writer_defects(q) == "writer_off_claim:cute_kernel_vs_numba"


def test_local_gate_catches_hopper_api_on_ampere_code():
    q = _q(
        hardware="ampere",
        question_type="debug",
        question=(
            "On Ampere sm_80 the kernel uses this PTX:\n"
            "```ptx\ncp.async.bulk.tensor.shared.global ...\n```\n"
            "What happens at runtime?"
        ),
    )
    assert local_writer_defects(q) == "writer_hw:hopper_api_on_ampere"


def test_local_gate_ignores_ampere_compare_without_code():
    q = _q(
        hardware="ampere",
        question_type="compare",
        question="比较 Ampere 与 Hopper 上 TMA 是否可用，不要给答案。",
    )
    assert local_writer_defects(q) is None


def test_local_gate_catches_sm_name_conflict():
    q = _q(question="On Blackwell (SM90) what is TMEM capacity? 请给出数字。")
    reason = local_writer_defects(q)
    assert reason is not None
    assert reason.startswith("writer_sm_conflict:")


def test_writer_gate_reason_skips_llm_when_local_hits():
    q = _q(
        l3_claim="@cute.kernel host capture",
        tests_claim="@cute.kernel must not capture host tensors",
        question="Numba @cuda.jit 捕获 numpy 为什么静默错误？请说明。",
    )

    class Boom:
        def complete_json(self, *a, **k):
            raise AssertionError("llm must not run after local defect")

    assert writer_gate_reason(q, Boom(), use_llm=True) == "writer_off_claim:cute_kernel_vs_numba"


def test_generate_rewrites_then_accepts(fake_runtime, sample_brief, sample_steering_brief):
    """A local defect triggers one rewrite; the second draft can pass."""
    set_runtime(fake_runtime)
    calls = {"n": 0}
    orig = fake_runtime.llm.complete

    def complete(messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                '{"question":"下面是 Numba @cuda.jit 内核，闭包捕获了主机 numpy 数组。'
                '代码能编译启动但输出全为 0 且无报错。请指出静默错误的根本原因，并说明为何 '
                'CUTLASS @cute.kernel 同类写法会失败。\\n\\n```python\\n@cuda.jit\\ndef k(o):\\n    o[0]=1\\n```",'
                '"tests_claim":"@cute.kernel host capture is illegal",'
                '"language":"zh"}'
            )
        return orig(messages, **kwargs)

    fake_runtime.llm.complete = complete
    brief = sample_brief(
        l3_id="cute_dsl_capture_host",
        l3_claim="在 @cute.kernel 里捕获普通 Python list/Tensor 主机对象当闭包",
    )
    out = generate_or_salvage(brief.model_dump(), sample_steering_brief())
    assert out.status == "ok"
    assert calls["n"] >= 2
    assert local_writer_defects(out.question) is None


def test_generate_rejects_when_rewrite_still_broken(
    fake_runtime, sample_brief, sample_steering_brief
):
    set_runtime(fake_runtime)

    def complete(messages, **kwargs):
        return (
            '{"question":"下面是 Numba @cuda.jit 内核，闭包捕获了主机 numpy 数组。'
            '代码能编译启动但输出全为 0 且无报错。请指出静默错误的根本原因，并说明为何 '
            'CUTLASS @cute.kernel 同类写法会失败。\\n\\n```python\\n@cuda.jit\\ndef k(o):\\n    o[0]=1\\n```",'
            '"tests_claim":"@cute.kernel host capture is illegal",'
            '"language":"zh"}'
        )

    fake_runtime.llm.complete = complete
    brief = sample_brief(
        l3_id="cute_dsl_capture_host",
        l3_claim="在 @cute.kernel 里捕获普通 Python list/Tensor 主机对象当闭包",
    )
    out = generate_or_salvage(brief.model_dump(), sample_steering_brief())
    assert out.status == "hard_fail"
    assert "cute_kernel_vs_numba" in (out.reason or "")
