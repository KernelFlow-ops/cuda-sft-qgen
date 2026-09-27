"""Pre-chair writer consistency gate: local defects and rewrite-once."""

from __future__ import annotations

from types import SimpleNamespace

from cuda_sft_qgen.graph.nodes.generate_core import _gate_or_rewrite, _run_writer_gate, rewrite_item
from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.review.writer_gate import local_writer_defects, writer_gate_reason
from cuda_sft_qgen.runtime import reset_runtime, set_runtime


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


def test_local_gate_routes_contradictory_cute_shapes_to_rewrite():
    q = _q(
        answer_format="single_choice",
        draft_target="A",
        internal_reference_answer="A",
        question=(
            "What are the shapes from zipped_divide and tiled_divide?\n"
            "```cpp\n"
            "constexpr auto layout = make_layout(make_shape(_12{}, Int<18>{}));\n"
            "constexpr auto tiler = make_tile(_3{}, _3{});\n"
            "constexpr auto z = zipped_divide(layout, tiler);\n"
            "constexpr auto t = tiled_divide(layout, tiler);\n"
            "```\n"
            "A. Both calls are well-formed; z has shape ((_3,_3),_4,_6) and t has shape ((_3,_3),(_4,_6)).\n"
            "B. Both calls are well-formed; z has shape ((_4,_6),(_3,_3)) and t has shape ((_4,_6),_3,_3).\n"
            "C. Both calls are well-formed; z has shape ((_3,_3),(_4,_5)) and t has shape ((_3,_3),_4,_5).\n"
            "D. Both calls are well-formed; z has shape ((_3,_2),(_4,_6)) and t has shape ((_3,_2),_4,_6)."
        ),
    )

    assert local_writer_defects(q) == "writer_cute_shape_contradiction:no_correct_option"


def test_local_gate_fails_closed_on_unsupported_cute_shape_source():
    q = _q(
        answer_format="single_choice",
        draft_target="A",
        internal_reference_answer="A",
        question=(
            "What are the shapes from zipped_divide and tiled_divide?\n"
            "```cpp\n"
            "constexpr auto layout = make_layout(make_shape(M, Int<18>{}));\n"
            "constexpr auto tiler = make_tile(_3{}, _3{});\n"
            "constexpr auto z = zipped_divide(layout, tiler);\n"
            "constexpr auto t = tiled_divide(layout, tiler);\n"
            "```\n"
            "A. Both calls are well-formed; z has shape ((_3,_3),(_4,_6)) and t has shape ((_3,_3),_4,_6).\n"
            "B. Both calls are well-formed; z has shape ((_3,_3),_4,_6) and t has shape ((_3,_3),(_4,_6)).\n"
            "C. Both calls are well-formed; z has shape ((_4,_6),(_3,_3)) and t has shape ((_4,_6),_3,_3).\n"
            "D. Both calls are well-formed; z has shape ((_3,_2),(_4,_6)) and t has shape ((_3,_2),_4,_6)."
        ),
    )

    assert local_writer_defects(q) == "writer_cute_shape_unsupported:source_layout_or_tile_not_whitelisted"


def test_local_gate_rewrites_incomplete_private_contract_before_chair():
    q = _q(draft_validation_errors=["missing_draft_target", "missing_verification_spec"])
    assert local_writer_defects(q) == "writer_contract:missing_draft_target,missing_verification_spec"


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


def test_writer_gate_llm_is_opt_in(monkeypatch):
    calls = []

    class Critic:
        def complete_json(self, *args, **kwargs):
            calls.append(1)
            return {"ok": True}

    settings = SimpleNamespace(fake_llm=False, generation={
        "writer_consistency_gate": True, "writer_gate_llm": False,
    })
    monkeypatch.delenv("WRITER_CONSISTENCY_GATE", raising=False)
    monkeypatch.delenv("WRITER_GATE_LLM", raising=False)
    assert _run_writer_gate(_q(), settings, Critic()) is None
    assert calls == []
    settings.generation["writer_gate_llm"] = True
    assert _run_writer_gate(_q(), settings, Critic()) is None
    assert calls == [1]


def test_writer_rewrite_can_be_deferred_without_second_model_call(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate_core

    token = set_runtime(fake_runtime)
    monkeypatch.setenv("WRITER_CONSISTENCY_GATE", "1")
    monkeypatch.setenv("WRITER_GATE_REWRITE", "1")
    monkeypatch.setattr(generate_core, "generate_from_brief", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Writer worker must not run repair")))
    try:
        outcome = _gate_or_rewrite(
            _q(question="On sm_90, may tcgen05 issue from this kernel? Explain the legality."),
            sample_brief(), SteeringBrief(), corpus=None, attempts=1, defer_rewrite=True,
        )
    finally:
        reset_runtime(token)
    assert outcome.status == "repair"
    assert outcome.question is not None


def test_repair_worker_accepts_rechecked_draft(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate_core

    monkeypatch.setattr(generate_core, "generate_from_brief", lambda *a, **k: _q())
    monkeypatch.setattr(generate_core, "sieve_question", lambda *a, **k: None)
    token = set_runtime(fake_runtime)
    try:
        out = rewrite_item({"brief": sample_brief().model_dump(), "question": _q().model_dump(),
                            "reason": "writer_hw", "steering": SteeringBrief().model_dump()})
    finally:
        reset_runtime(token)
    assert out.status == "ok"
    assert local_writer_defects(out.question) is None


def test_repair_worker_rejects_still_broken_draft(fake_runtime, sample_brief, monkeypatch):
    from cuda_sft_qgen.graph.nodes import generate_core

    bad = _q(question="On sm_90, may tcgen05 issue from this kernel? Explain the legality.")
    monkeypatch.setattr(generate_core, "generate_from_brief", lambda *a, **k: bad)
    monkeypatch.setattr(generate_core, "sieve_question", lambda *a, **k: None)
    token = set_runtime(fake_runtime)
    try:
        out = rewrite_item({"brief": sample_brief().model_dump(), "question": _q().model_dump(),
                            "reason": "writer_hw", "steering": SteeringBrief().model_dump()})
    finally:
        reset_runtime(token)
    assert out.status == "hard_fail"
    assert "tcgen05_on_non_sm100a" in (out.reason or "")
