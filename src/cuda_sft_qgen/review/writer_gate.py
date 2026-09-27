"""Catch internally broken writer drafts before they spend a chair call.

Local checks are high-precision only. The cheap LLM gate may fail an item
only when it is certain; uncertainty must pass through to the chair.
This module does not accept questions or commit the archive.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from cuda_sft_qgen.keys import PRI_WRITER
from cuda_sft_qgen.llm import LLMClient
from cuda_sft_qgen.models import Question
from cuda_sft_qgen.review.cute_shapes import check_cute_shapes
from cuda_sft_qgen.review.consistency import sm_name_conflicts

logger = logging.getLogger(__name__)

GATE_MARK = "WRITER_GATE_V1"

_CUTE_CLAIM = re.compile(r"@cute\.kernel|cute\.kernel", re.I)
_NUMBA_STEM = re.compile(r"@cuda\.jit|\bnumba\b", re.I)
_HOPPER_API = re.compile(
    r"\b(?:tcgen05|wgmma\.mma_async|cp\.async\.bulk|tensormap)\b",
    re.I,
)
_LEAK_EXAMPLE = re.compile(
    r"(?:例如|e\.g\.|answer is|答案是)[：:\s]+\d+(?:\s*[,，]\s*\d+)+",
    re.I,
)

_GATE_PROMPT = f"""You are a CUDA exam-item consistency checker. {GATE_MARK}
Fail ONLY when a defect is certain. If unsure, return ok=true and let the chair decide.
Do not judge style, difficulty, or missing official citations.

Fail when:
- code, comments, options, and prose disagree on a number, mask, lane count, or identifier
- a single-choice / "only one legal snippet" item has zero or more than one correct option
- the stem tests a different mechanism than l3_claim (example: @cuda.jit vs @cute.kernel)
- the stem leaks the worked answer
- the unique correct option is an invented API or illegal PTX

Return JSON only, no extra keys:
{{"ok": true}}
or
{{"ok": false, "reason": "<=200 chars, name the concrete contradiction"}}
"""


def local_writer_defects(question: Question) -> str | None:
    """Return a high-precision local defect code, or None if rules do not fire."""
    stem = question.question or ""
    tests = question.tests_claim or ""
    claim = getattr(question, "l3_claim", None) or ""
    if question.draft_validation_errors:
        return "writer_contract:" + ",".join(question.draft_validation_errors[:3])
    shape_check = check_cute_shapes(question)
    if shape_check["status"] in {"contradiction", "unsupported"}:
        return f"writer_cute_shape_{shape_check['status']}:{shape_check['reason']}"
    conflicts = sm_name_conflicts(stem)
    if conflicts:
        return "writer_sm_conflict:" + ",".join(conflicts)
    fences = "\n".join(re.findall(r"```(?:\w+)?\n([\s\S]*?)```", stem))
    code = fences or stem
    if (
        _CUTE_CLAIM.search(claim + "\n" + tests)
        and _NUMBA_STEM.search(code)
        and not _CUTE_CLAIM.search(code)
    ):
        return "writer_off_claim:cute_kernel_vs_numba"
    if (
        (question.hardware or "").lower() == "ampere"
        and (question.question_type or "") != "compare"
        and "```" in stem
        and _HOPPER_API.search(stem)
    ):
        return "writer_hw:hopper_api_on_ampere"
    if len(tests) >= 80 and tests[:80] in stem:
        return "writer_leakage:tests_claim_in_stem"
    if _LEAK_EXAMPLE.search(stem):
        return "writer_leakage:worked_example_in_stem"
    arch = (getattr(question, "arch_target", "") or "any")
    qtype = question.question_type or ""
    if len(claim) >= 40 and claim[:40] in stem:
        return "writer_leakage:claim_in_stem"
    if re.search(r"tcgen05", stem, re.I) and arch != "sm_100a" and qtype != "compare":
        return "writer_hw:tcgen05_on_non_sm100a"
    if re.search(r"\bwgmma", stem, re.I) and arch != "sm_90a" and qtype != "compare":
        return "writer_hw:wgmma_on_non_hopper"
    fmt = getattr(question, "answer_format", "") or ""
    if fmt in {"single_choice", "multi_choice"}:
        option_lines = re.findall(r"(?m)^\s*(?:[A-F][\.\)、]|\([a-f]\))", stem)
        if len(option_lines) < 3:
            return "writer_format:choice_without_options"
    return None


def llm_writer_gate(question: Question, llm: LLMClient, *, max_tokens: int = 512) -> str | None:
    """Cheap LLM consistency check. None means pass; a string is a certain defect.

    Parse/call failures return None so the chair still sees the item.
    """
    payload = {
        "id": question.id,
        "question": question.question,
        "l3_id": question.l3_id,
        "l3_claim": question.l3_claim,
        "tests_claim": question.tests_claim,
        "hardware": question.hardware,
        "question_type": question.question_type,
    }
    try:
        obj: dict[str, Any] = llm.complete_json(
            [
                {"role": "system", "content": _GATE_PROMPT},
                {"role": "user", "content": str(payload)},
            ],
            thinking_level="none",
            temperature=0.0,
            retries=0,
            priority=PRI_WRITER,
            max_tokens=max(256, int(max_tokens)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("writer_gate llm skipped: %s", exc)
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("ok") is True or str(obj.get("ok")).lower() == "true":
        return None
    if obj.get("ok") is False or str(obj.get("ok")).lower() == "false":
        reason = str(obj.get("reason") or "internal_contradiction").strip()[:200]
        return "writer_inconsistent:" + reason
    return None


def writer_gate_reason(
    question: Question,
    llm: LLMClient | None,
    *,
    use_llm: bool,
    max_tokens: int = 512,
) -> str | None:
    """Local defects first; optional LLM gate only if local rules are silent."""
    local = local_writer_defects(question)
    if local:
        return local
    if use_llm and llm is not None:
        return llm_writer_gate(question, llm, max_tokens=max_tokens)
    return None
