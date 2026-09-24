"""Test data builders for tests/fix_v2 (importable as `fixv2_helpers`; conftest puts this dir on sys.path)."""
from __future__ import annotations


def make_question(**over):
    """A minimal valid v2 Question (import inside tests to keep collection import-free)."""
    from cuda_sft_qgen.models import Question
    base = dict(id="q_test", question="__global__ kernel with warp shuffle: which lane gets the sum?",
                tests_claim="Warp shuffle down with offset 16 moves lane i+16 into lane i.",
                topic_path=["P01", "warp_shuffle", "warp_shuffle_mask_width"], l3_id="warp_shuffle_mask_width",
                l3_claim="shuffle mask must name participating lanes", question_type="debug", difficulty=2,
                bloom="apply", persona="code_reviewer", hardware="agnostic", granularity="kernel", language="en",
                kind="api")
    base.update(over)
    return Question.model_validate(base)


def make_brief(**over):
    from cuda_sft_qgen.models import QuestionBrief
    base = dict(topic_path=["T02", "smem", "smem_bank_stride_gcd"], l3_id="smem_bank_stride_gcd",
                l3_claim="For 4-byte words at stride S over 32 banks the conflict degree is gcd(S,32).",
                question_type="conceptual", difficulty=2, bloom="apply", persona="perf_engineer",
                hardware="agnostic", granularity="kernel", language="zh", kind="formula",
                angle="给定参数计算唯一结果", angle_id="con_plug_numbers", answer_format="multi_numeric",
                scenario="generic", arch_target="any")
    base.update(over)
    return QuestionBrief.model_validate(base)


def tiny_taxonomy(extra_leaves: list[dict] | None = None, *, with_arch: bool = True):
    """Two L1 x two L2 in-memory taxonomy used by planner/diversity tests."""
    from cuda_sft_qgen.taxonomy import Taxonomy, _leaves_from_domains
    def leaf(i, kind, types, claim, hints=(), status="verified", diff=(2, 3)):
        return {"id": i, "claim": claim, "kind": kind, "types": list(types), "difficulty": list(diff),
                "hints": list(hints), "status": status}
    domains = [
        {"id": "T02", "name": "mem", "weight": 1.6, "children": [
            {"id": "smem", "name": "smem", **({"arch": ["any"]} if with_arch else {}), "leaves": [
                leaf("smem_bank_gcd", "formula", ["conceptual", "layout"], "ways = gcd(S,32) for 4B words", hints=["S=6", "S=12"]),
                leaf("smem_pad", "failure", ["debug", "profiling"], "tile[tid][k] on 32x32 float is 32-way conflicted"),
            ]},
            {"id": "gmem", "name": "gmem", **({"arch": ["any"]} if with_arch else {}), "leaves": [
                leaf("gmem_sectors", "numeric", ["profiling", "conceptual"], "a[lane+1] touches 5 sectors = ceil(132/32)"),
            ]},
        ]},
        {"id": "T08", "name": "async", "weight": 1.3, "children": [
            {"id": "tma", "name": "tma", **({"arch": ["sm_90", "sm_90a", "sm_100a"]} if with_arch else {}), "leaves": [
                leaf("tma_encode", "feature", ["syntax", "compare"], "TMA on sm_90 needs 16B-multiple global strides"),
                leaf("tma_expect_tx", "numeric", ["counterfactual", "conceptual"], "expect_tx = full box bytes incl OOB; mbarrier", status="provisional"),
            ]},
            {"id": "async_copy", "name": "cp.async", **({"arch": ["sm_80", "sm_86", "sm_89"]} if with_arch else {}), "leaves": [
                leaf("cp_async_sizes", "api", ["syntax", "debug"], "cp.async.cg only supports 16 bytes on sm_80"),
            ]},
        ]},
    ]
    for item in extra_leaves or []:
        domains[0]["children"][0]["leaves"].append(item)
    leaves = _leaves_from_domains(domains)
    return Taxonomy(1, "tiny", leaves, domains)
