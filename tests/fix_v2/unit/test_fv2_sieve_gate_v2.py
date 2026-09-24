"""Sieve and writer-gate v2 rules (task T25)."""
from types import SimpleNamespace

import pytest

from fixv2_helpers import make_question

S = SimpleNamespace(question_min_chars=40, question_max_chars=2500)


def test_sieve_leak_phrases():
    from cuda_sft_qgen.sieve import rule_sieve
    q = make_question(question="移除第二个 __syncthreads 会怎样？Your answer must mention the WAR hazard and the fix.")
    assert rule_sieve(q, S) == "sieve_leak_must_mention"


def test_sieve_off_topic_strict():
    from cuda_sft_qgen.sieve import rule_sieve
    # passes the legacy _CUDA_HINTS regex ("thread") but has no CUDA-specific token
    q = make_question(question="Which Java thread-pool size is best for a web server that receives ten requests per second?",
                      tests_claim="A fixed pool of ten Java threads serves ten requests.")
    assert rule_sieve(q, S) == "sieve_off_topic_strict"


def test_sieve_accepts_normal_item():
    from cuda_sft_qgen.sieve import rule_sieve
    assert rule_sieve(make_question(), S) is None


@pytest.mark.parametrize("over,code", [
    ({"question": "On sm_90 compile a tcgen05.mma kernel: what fails and why? Explain the arch gate.", "arch_target": "sm_90",
      "hardware": "hopper"}, "writer_hw:tcgen05_on_non_sm100a"),
    ({"question": "On an A100 call wgmma.mma_async in a 128-thread block: what happens and why?", "arch_target": "sm_80",
      "hardware": "ampere"}, "writer_hw:wgmma_on_non_hopper"),
    ({"question": "单选：下列哪个 __shfl_down_sync 调用是正确的？A. mask 全 1 B. mask 低 16 位", "answer_format": "single_choice"},
     "writer_format:choice_without_options"),
])
def test_writer_gate_rules(over, code):
    from cuda_sft_qgen.review.writer_gate import local_writer_defects
    assert local_writer_defects(make_question(**over)) == code


def test_writer_gate_claim_in_stem():
    from cuda_sft_qgen.review.writer_gate import local_writer_defects
    claim = "Warp shuffle mask must name exactly the lanes that execute the call"
    q = make_question(l3_claim=claim, question=f"Given that {claim}, which lanes are defined after __shfl_down_sync?")
    assert local_writer_defects(q) == "writer_leakage:claim_in_stem"


def test_compare_items_may_name_other_arch():
    from cuda_sft_qgen.review.writer_gate import local_writer_defects
    q = make_question(question="Compare wgmma on sm_90a with tcgen05 on sm_100a: which one keeps accumulators in TMEM?",
                      question_type="compare", arch_target="sm_90a", hardware="hopper")
    assert local_writer_defects(q) is None
