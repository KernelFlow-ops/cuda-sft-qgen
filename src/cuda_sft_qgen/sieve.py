"""Cheap local filters that reject empty, off-topic, or non-question text.

Hardware/leaf mismatches are warnings on the writer path, not hard rejects;
the chair still decides those cases.
"""

from __future__ import annotations

import re

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.llm import LLMClient
from cuda_sft_qgen.models import Question

_CUDA_HINTS = re.compile(
    r"cuda|kernel|内核|warp|thread|block|grid|shared|global memory|occupancy|"
    r"nvcc|ptx|sass|tensor core|cutlass|cute|tma|wgmma|umma|tmem|"
    r"nsight|cublas|nccl|hopper|ampere|blackwell|smem|gmem|"
    r"__syncthreads|coalesc|bank conflict|mma|layout|stride|swizzle|"
    r"cluster|nvlink|cub::|launch_bounds|__fma|fma|ulp",
    re.IGNORECASE,
)

_QUESTIONISH = re.compile(
    r"[?？]|为什么|为何|如何|怎样|哪|什么|是否|给出|求|指出|解释|"
    r"请|补全|下列|选择|选出|填空|计算|判断|"
    r"compare|why|how|what|which|identify|select|choose|examine|"
    r"complete|fill|compute|calculate|determine|provide|indicate",
    re.IGNORECASE,
)

_BAD_OPEN = re.compile(
    r"请介绍|谈谈你对|what is cuda\??$|介绍一下cuda优化$|任意发挥|畅所欲言",
    re.IGNORECASE,
)


def rule_sieve(question: Question, settings: Settings) -> str | None:
    """Return reject reason or None if it passes."""
    text = (question.question or "").strip()
    if len(text) < settings.question_min_chars:
        return "sieve_too_short"
    if len(text) > settings.question_max_chars:
        return "sieve_too_long"
    if not (question.tests_claim or "").strip():
        return "sieve_missing_claim"
    if not _QUESTIONISH.search(text):
        return "sieve_not_a_question"
    if not _CUDA_HINTS.search(text) and not _CUDA_HINTS.search(question.tests_claim):
        return "sieve_off_topic"
    if _BAD_OPEN.search(text):
        return "sieve_too_vague"
    if text.endswith("。") and "？" not in text and "?" not in text and len(text) < 40:
        return "sieve_not_a_question"
    # hw/leaf mismatches are warnings on the Writer path, not hard rejects.
    return None


def llm_sieve(question: Question, llm: LLMClient) -> str | None:
    """Optional cheap LLM filter; parse/call failures never reject the item."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cheap filter. Label the item meaningful or garbage. "
                "Garbage = not CUDA, not a question, too vague, or already contains "
                "the full answer. Do not reject for style or difficulty. "
                'Return JSON {"label":"meaningful"|"garbage","reason":"..."}.'
            ),
        },
        {
            "role": "user",
            "content": f"QUESTION:\n{question.question}\nCLAIM:\n{question.tests_claim}",
        },
    ]
    try:
        from cuda_sft_qgen.keys import PRI_SIEVE

        obj = llm.complete_json(
            messages,
            thinking_level="none",
            temperature=0.0,
            retries=0,
            priority=PRI_SIEVE,
        )
    except Exception:
        return None
    label = str(obj.get("label", "meaningful")).lower()
    if label == "garbage":
        return f"sieve_llm:{obj.get('reason', 'garbage')}"
    return None


def sieve_question(
    question: Question, settings: Settings, llm: LLMClient | None
) -> str | None:
    """Run rule sieve, then optional LLM sieve.

    Returns:
        Reject reason string, or None if the question may proceed to review.
    """
    reason = rule_sieve(question, settings)
    if reason:
        return reason
    if settings.sieve_use_llm and llm is not None:
        return llm_sieve(question, llm)
    return None
