"""Writer/evolver helpers: one-brief generation, JSON salvage, retry, and apply.

JSON parse failures become salvage or retry; they never become a silent accept.
This module does not run the batch executor or LangGraph wiring.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from cuda_sft_qgen.errors import JsonSalvageNeeded, RateLimitError
from cuda_sft_qgen.jsonutil import normalize_language, parse_question_payload
from cuda_sft_qgen.keys import PRI_SALVAGE, PRI_WRITER
from cuda_sft_qgen.llm import NvidiaLLM
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.observability import get_metrics
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.consistency import consistency_reason
from cuda_sft_qgen.review.fingerprint import fingerprint
from cuda_sft_qgen.review.writer_gate import writer_gate_reason
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.sieve import sieve_question

logger = logging.getLogger(__name__)


def _with_consistency_warning(question: Question) -> Question:
    """Attach hw/leaf mismatch as a warning; do not reject on the writer path."""
    cons = consistency_reason(question)
    if cons and cons not in question.warnings:
        question.warnings.append(cons)
    return question


@dataclass
class GenerateOutcome:
    """Result of one brief: ok question, salvage payload, retry, or hard fail."""

    status: str
    question: Question | None = None
    brief: dict[str, Any] | None = None
    raw: str | None = None
    reason: str | None = None
    attempts: int = 1


def retry_wait_sec(attempts: int) -> float:
    """Exponential backoff clipped to ``retry_max_cooldown_sec``."""
    rt = get_runtime()
    wait = rt.settings.retry_base_cooldown_sec * (2 ** min(max(attempts - 1, 0), 30))
    return min(rt.settings.retry_max_cooldown_sec, wait)


def question_from_obj(
    brief: QuestionBrief,
    obj: dict[str, Any],
    *,
    source: str,
) -> Question:
    """Build a pending :class:`Question` from parsed writer JSON and the brief."""
    question_text = str(obj.get("question") or "").strip()
    tests_claim = str(obj.get("tests_claim") or "").strip()
    language = normalize_language(obj.get("language"), fallback=brief.language)
    warnings = obj.get("warnings") if isinstance(obj.get("warnings"), list) else []
    fp = fingerprint(question_text)
    return Question(
        id="q_" + (brief.task_id or uuid.uuid4().hex),
        question=question_text,
        tests_claim=tests_claim,
        topic_path=list(brief.topic_path),
        l3_id=brief.l3_id,
        l3_claim=brief.l3_claim,
        question_type=brief.question_type,
        difficulty=int(brief.difficulty),
        bloom=brief.bloom,
        persona=brief.persona,
        hardware=brief.hardware,
        granularity=brief.granularity,
        language=language,
        template_fingerprint=fp.skeleton,
        simhash=fp.simhash,
        exact_hash=fp.exact,
        source=source,
        review_status="pending",
        warnings=[str(w) for w in warnings],
        kind=brief.kind or "",
        requested_metadata={
            key: getattr(brief, key)
            for key in ("question_type", "difficulty", "bloom", "hardware", "granularity", "language")
        },
        seed_id=brief.seed_id,
        operator=str(obj["operator"]) if obj.get("operator") else None,
        angle_id=brief.angle_id,
        claim_revision=brief.claim_revision,
        source_ids=list(brief.source_ids),
        knowledge_dependencies=[
            {"claim_id": claim["id"], "revision": claim.get("revision", 1)}
            for claim in brief.knowledge_context.get("claims", []) if claim.get("id")
        ],
    )


def generate_from_brief(
    brief: QuestionBrief,
    steering: SteeringBrief,
    *,
    corpus: FewshotCorpus | None = None,
    repair_note: str | None = None,
) -> Question:
    """Call the writer/evolver LLM and parse a question; may raise JsonSalvageNeeded."""
    rt = get_runtime()
    context = brief.knowledge_context or {}
    get_metrics().observe("knowledge_context_chars", len(json.dumps(context, ensure_ascii=False)))
    anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
    if corpus is None:
        corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
    builder = getattr(rt, "prompt_builder", None)
    if builder is None:
        builder = StructuredPrompt(
            rt.settings, corpus, cache=getattr(rt, "prompt_cache", None)
        )
    else:
        builder.corpus = corpus
    bundle = builder.build(brief, steering)
    source = "evolver" if brief.role == "evolver" and brief.seed_question else "writer"
    user = bundle.user
    if repair_note:
        user = (
            user
            + "\n\nThe previous draft was rejected before review for this defect:\n"
            + str(repair_note).strip()[:400]
            + "\nRewrite ONE new question for the same brief. Keep the leaf claim. "
            "Fix the defect. Return JSON only."
        )
    messages = [
        {"role": "system", "content": bundle.system},
        {"role": "user", "content": user},
    ]
    raw = rt.llm.complete(
        messages,
        priority=PRI_WRITER,
        prompt_cache_key=bundle.cache_key or None,
        thinking_level=rt.settings.writer_thinking_level,
    )
    obj = parse_question_payload(
        raw,
        fallback_claim=brief.l3_claim,
        min_question_chars=rt.settings.sieve_min_chars,
        keep_truncated=rt.settings.json_keep_truncated,
    )
    if obj is None or not str(obj.get("question") or "").strip():
        raise JsonSalvageNeeded(raw, brief.model_dump())
    question = question_from_obj(brief, obj, source=source)
    if len(question.question) > rt.settings.question_max_chars:
        raise JsonSalvageNeeded(
            json.dumps(
                {
                    "question": question.question,
                    "tests_claim": question.tests_claim,
                    "language": question.language,
                },
                ensure_ascii=False,
            ),
            brief.model_dump(),
        )
    return question


def _safe_generate(
    brief_data: dict,
    steering: SteeringBrief,
    *,
    attempts: int = 1,
    corpus: FewshotCorpus | None = None,
) -> GenerateOutcome:
    """Generate one question and map exceptions onto retry/salvage/hard-fail."""
    brief = QuestionBrief.model_validate(brief_data)
    rt = get_runtime()
    try:
        question = generate_from_brief(brief, steering, corpus=corpus)
    except RateLimitError as exc:
        logger.warning("generation rate-limited, will retry later: %s", exc)
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="429",
            attempts=attempts,
        )
    except JsonSalvageNeeded as exc:
        return GenerateOutcome(
            status="salvage",
            brief=exc.brief,
            raw=exc.raw,
            reason="json_parse",
            attempts=attempts,
        )
    except Exception as exc:  # noqa: BLE001
        if NvidiaLLM._is_rate_limit(exc):
            return GenerateOutcome(
                status="retry",
                brief=brief.model_dump(),
                reason="429",
                attempts=attempts,
            )
        logger.warning("generation failed: %s", exc)
        failed = Question(
            id="q_" + (brief.task_id or uuid.uuid4().hex),
            question="",
            tests_claim=brief.l3_claim,
            topic_path=list(brief.topic_path),
            l3_id=brief.l3_id,
            l3_claim=brief.l3_claim,
            question_type=brief.question_type,
            difficulty=brief.difficulty,
            bloom=brief.bloom,
            persona=brief.persona,
            hardware=brief.hardware,
            granularity=brief.granularity,
            language=brief.language,
            source=brief.role,
            reject_reason=f"generate_error:{exc}",
        )
        rt.store.append_rejected(failed)
        return GenerateOutcome(status="hard_fail", brief=brief.model_dump(), reason=str(exc), attempts=attempts)
    reason = sieve_question(question, rt.settings, rt.llm)
    if reason:
        question.reject_reason = reason
        rt.store.append_rejected(question)
        return GenerateOutcome(status="hard_fail", brief=brief.model_dump(), reason=reason, attempts=attempts)
    return _gate_or_rewrite(
        question, brief, steering, corpus=corpus, attempts=attempts
    )


def _gate_enabled(settings: Any) -> bool:
    """True when the pre-chair writer consistency gate is on (default on)."""
    import os

    from cuda_sft_qgen.config import _as_bool

    return _as_bool(
        os.getenv("WRITER_CONSISTENCY_GATE"),
        bool(settings.generation.get("writer_consistency_gate", True)),
    )


def _rewrite_enabled(settings: Any) -> bool:
    """True when a gated draft is rewritten once (default on)."""
    import os

    from cuda_sft_qgen.config import _as_bool

    return _as_bool(
        os.getenv("WRITER_GATE_REWRITE"),
        bool(settings.generation.get("writer_gate_rewrite", True)),
    )


def _run_writer_gate(question: Question, settings: Any, llm: Any) -> str | None:
    """Local then optional LLM gate. FakeLLM skips the LLM critic."""
    use_llm = _gate_enabled(settings) and not bool(getattr(settings, "fake_llm", False))
    max_tokens = int(settings.generation.get("writer_gate_max_tokens", 512) or 512)
    return writer_gate_reason(
        question, llm if use_llm else None, use_llm=use_llm, max_tokens=max_tokens
    )


def _reject_draft(question: Question, brief: QuestionBrief, reason: str, attempts: int) -> GenerateOutcome:
    """Record a pre-chair reject and stop this brief."""
    rt = get_runtime()
    question.reject_reason = reason
    rt.store.append_rejected(question)
    logger.info("writer_gate reject l3=%s reason=%s", brief.l3_id, reason[:160])
    return GenerateOutcome(
        status="hard_fail", brief=brief.model_dump(), reason=reason, attempts=attempts
    )


def _gate_or_rewrite(
    question: Question,
    brief: QuestionBrief,
    steering: SteeringBrief,
    *,
    corpus: FewshotCorpus | None,
    attempts: int,
) -> GenerateOutcome:
    """Pass, rewrite once on a certain defect, or reject without a chair call."""
    rt = get_runtime()
    if not _gate_enabled(rt.settings):
        return GenerateOutcome(
            status="ok", question=_with_consistency_warning(question), attempts=attempts
        )
    reason = _run_writer_gate(question, rt.settings, rt.llm)
    if not reason:
        return GenerateOutcome(
            status="ok", question=_with_consistency_warning(question), attempts=attempts
        )
    if not _rewrite_enabled(rt.settings):
        return _reject_draft(question, brief, reason, attempts)
    logger.info("writer_gate rewrite l3=%s reason=%s", brief.l3_id, reason[:160])
    try:
        rewritten = generate_from_brief(
            brief, steering, corpus=corpus, repair_note=reason
        )
    except (RateLimitError, JsonSalvageNeeded):
        return _reject_draft(question, brief, reason, attempts)
    except Exception as exc:  # noqa: BLE001
        if NvidiaLLM._is_rate_limit(exc):
            return GenerateOutcome(
                status="retry", brief=brief.model_dump(), reason="429", attempts=attempts
            )
        logger.warning("writer_gate rewrite failed: %s", exc)
        return _reject_draft(question, brief, reason, attempts)
    sieve_reason = sieve_question(rewritten, rt.settings, rt.llm)
    if sieve_reason:
        return _reject_draft(rewritten, brief, sieve_reason, attempts)
    again = _run_writer_gate(rewritten, rt.settings, rt.llm)
    if again:
        return _reject_draft(rewritten, brief, again, attempts)
    return GenerateOutcome(
        status="ok", question=_with_consistency_warning(rewritten), attempts=attempts
    )


def salvage_item(item: dict[str, Any]) -> GenerateOutcome:
    """Repair one broken JSON payload locally, optionally with a salvage LLM call."""
    rt = get_runtime()
    brief = QuestionBrief.model_validate(item["brief"])
    raw = str(item.get("raw") or "")
    obj = parse_question_payload(
        raw,
        fallback_claim=brief.l3_claim,
        min_question_chars=rt.settings.sieve_min_chars,
        keep_truncated=rt.settings.json_keep_truncated,
    )
    if obj and str(obj.get("question") or "").strip():
        q = question_from_obj(brief, obj, source=brief.role)
        reason = sieve_question(q, rt.settings, rt.llm)
        if reason:
            q.reject_reason = reason
            rt.store.append_rejected(q)
            return GenerateOutcome(status="hard_fail", brief=brief.model_dump(), reason=reason)
        return _gate_or_rewrite(q, brief, SteeringBrief(), corpus=None, attempts=1)

    if not rt.settings.salvage_use_llm:
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="json_parse",
            attempts=1,
        )
    # Conceptual stems are cheap to redraw; a salvage LLM call often invents tests_claim.
    if (brief.kind or "") == "conceptual" and len(raw) < max(800, rt.settings.question_min_chars * 8):
        logger.warning("salvage skip conceptual short dump raw_chars=%s", len(raw))
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="json_parse",
            attempts=1,
        )
    # Empty / non-JSON writer dumps cannot be repaired; skip a second 10k-token call.
    if '"question"' not in raw and "'question'" not in raw:
        logger.warning("salvage skip empty/non-json raw_chars=%s", len(raw))
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="json_parse",
            attempts=1,
        )

    prompt = (
        load_prompt("salvage")
        .replace("{{l3_claim}}", brief.l3_claim)
        .replace("{{raw}}", raw[:12000])
        .replace("{{min_chars}}", str(rt.settings.question_min_chars))
        .replace("{{max_chars}}", str(rt.settings.question_max_chars))
    )
    last_err: str | None = f"local parse failed raw_chars={len(raw)}"
    for _ in range(max(1, rt.settings.salvage_llm_attempts)):
        try:
            text = rt.llm.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Return the repaired JSON object now."},
                ],
                thinking_level=rt.settings.salvage_thinking_level,
                max_tokens=rt.settings.salvage_max_tokens,
                priority=PRI_SALVAGE,
            )
        except RateLimitError:
            return GenerateOutcome(
                status="retry",
                brief=brief.model_dump(),
                reason="429",
                attempts=1,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        obj = parse_question_payload(
            text,
            fallback_claim=brief.l3_claim,
            min_question_chars=rt.settings.sieve_min_chars,
            keep_truncated=rt.settings.json_keep_truncated,
        )
        if obj and str(obj.get("question") or "").strip():
            q = question_from_obj(brief, obj, source=brief.role)
            reason = sieve_question(q, rt.settings, rt.llm)
            if reason:
                q.reject_reason = reason
                rt.store.append_rejected(q)
                return GenerateOutcome(status="hard_fail", brief=brief.model_dump(), reason=reason)
            return _gate_or_rewrite(q, brief, SteeringBrief(), corpus=None, attempts=1)
        last_err = f"unparseable salvage output chars={len(text or '')}"
    logger.warning("salvage failed: %s", last_err)
    return GenerateOutcome(
        status="retry",
        brief=brief.model_dump(),
        reason="json_parse",
        attempts=1,
    )


def generate_or_salvage(
    brief_data: dict,
    steering: SteeringBrief,
    *,
    attempts: int = 1,
    corpus: FewshotCorpus | None = None,
) -> GenerateOutcome:
    """Generate, and if salvage_inline is set, immediately repair JSON failures."""
    outcome = _safe_generate(brief_data, steering, attempts=attempts, corpus=corpus)
    if (
        outcome.status == "salvage"
        and outcome.brief is not None
        and get_runtime().settings.salvage_inline
    ):
        return salvage_item({"brief": outcome.brief, "raw": outcome.raw or ""})
    return outcome


def enqueue_retry(brief: dict[str, Any], *, reason: str, attempts: int) -> None:
    """Queue a brief for later retry, or reject it when attempts are exhausted."""
    rt = get_runtime()
    if attempts >= rt.settings.retry_max_attempts:
        failed = Question(
            id="q_" + (str(brief.get("task_id") or "") or uuid.uuid4().hex),
            question="",
            tests_claim=str(brief.get("l3_claim") or ""),
            topic_path=list(brief.get("topic_path") or []),
            l3_id=str(brief.get("l3_id") or ""),
            l3_claim=str(brief.get("l3_claim") or ""),
            question_type=str(brief.get("question_type") or "conceptual"),
            difficulty=int(brief.get("difficulty") or 2),
            bloom=str(brief.get("bloom") or "apply"),
            persona=str(brief.get("persona") or ""),
            hardware=str(brief.get("hardware") or "agnostic"),
            granularity=str(brief.get("granularity") or "kernel"),
            language=str(brief.get("language") or "zh"),
            source=str(brief.get("role") or "writer"),
            reject_reason=f"retry_exhausted:{reason}",
        )
        rt.store.append_rejected(failed)
        if brief.get("task_id"):
            rt.store.complete_generation_task(str(brief["task_id"]))
        return
    rt.store.enqueue_retry(
        {
            "brief": brief,
            "reason": reason,
            "attempts": attempts,
            "next_ok_at": time.time() + retry_wait_sec(attempts),
        }
    )
    if brief.get("task_id"):
        rt.store.complete_generation_task(str(brief["task_id"]))


def apply_outcome(
    outcome: GenerateOutcome,
    *,
    pending: list[dict],
    salvage_items: list[dict],
    lock: threading.Lock,
) -> str:
    """Fold an outcome into pending/salvage/retry under ``lock``. Returns status."""
    rt = get_runtime()
    task_id = str((outcome.brief or {}).get("task_id") or "")
    if not task_id and outcome.question is not None:
        task_id = outcome.question.id.removeprefix("q_")

    def complete_task() -> None:
        if task_id:
            rt.store.complete_generation_task(task_id)

    if outcome.status == "ok" and outcome.question is not None:
        rec = outcome.question.model_dump()
        with lock:
            pending.append(rec)
            rt.store.save_pending([outcome.question])
        complete_task()
        return "ok"
    if outcome.status == "salvage" and outcome.brief is not None:
        item = {"brief": outcome.brief, "raw": outcome.raw or ""}
        with lock:
            salvage_items.append(item)
            rt.store.append_salvage_raw(
                {"reason": "json_parse", "raw": outcome.raw or "", "brief": outcome.brief}
            )
        return "salvage"
    if outcome.status == "retry" and outcome.brief is not None:
        enqueue_retry(outcome.brief, reason=outcome.reason or "429", attempts=outcome.attempts)
        complete_task()
        return "retry"
    complete_task()
    return "hard_fail"
