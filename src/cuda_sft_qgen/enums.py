"""Canonical string enums shared by planner, prompts, review and knowledge (v2)."""

ANSWER_FORMATS = ("single_choice", "multi_choice", "numeric", "multi_numeric", "fill_in",
                  "code_fix", "code_complete", "short_answer", "trace_output")
ARCH_TARGETS = ("any", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100a", "sm_120")
ARCH_TO_HARDWARE = {"any": "agnostic", "sm_70": "volta", "sm_75": "turing",
                    "sm_80": "ampere", "sm_86": "ampere", "sm_89": "ada",
                    "sm_90": "hopper", "sm_90a": "hopper", "sm_100a": "blackwell", "sm_120": "blackwell"}
HARDWARE_BUCKETS = ("agnostic", "volta", "turing", "ampere", "ada", "hopper", "blackwell")


def hardware_bucket(arch_target: str) -> str:
    """Return the canonical hardware family for a validated architecture."""
    try:
        return ARCH_TO_HARDWARE[arch_target]
    except KeyError as exc:
        raise ValueError(f"unknown arch_target {arch_target}") from exc
SCENARIOS = ("generic", "elementwise", "reduction", "scan", "histogram", "sort", "transpose", "stencil", "spmv",
             "gemm", "conv", "attention", "softmax", "layernorm", "moe_routing", "kv_cache", "quantization",
             "distributed_training", "inference_serving", "data_pipeline", "graph_analytics", "scientific", "deployment")
DEFECT_CODES = ("correctness", "answerability", "multiple_correct", "insufficient_premises", "claim_alignment",
                "hardware", "leakage", "metadata", "duplicate", "ub_misstated", "invented_api", "low_value")
DEFECT_CODES += ("writer_contract",)
OPERATORS = ("deepen", "re-arch", "inject-bug", "layout-eq", "profiler", "compare", "counterfactual",
             "format-shift", "scenario-transfer", "trace", "decompose-reverse")
CHECK_TO_DEFECT = {"correctness": "correctness", "answerability": "answerability",
                   "claim_alignment": "claim_alignment", "hardware_compatibility": "hardware",
                   "answer_leakage": "leakage", "metadata": "metadata", "duplication": "duplicate"}
# Chair defects that come from an incoherent brief (leaf/angle/arch/labels) rather
# than the Writer's draft: rewriting the same brief repeats them, re-planning does not.
BRIEF_LEVEL_DEFECTS = frozenset({"claim_alignment", "hardware", "metadata"})
VERIFIED_BY = ("none", "quote", "computation", "execution", "expert")
