"""Writer/evolver helpers: one-brief generation, JSON salvage, retry, and apply.

JSON parse failures become salvage or retry; they never become a silent accept.
This module does not run the batch executor or LangGraph wiring.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from cuda_sft_qgen.errors import JsonSalvageNeeded, RateLimitError
from cuda_sft_qgen.enums import ANSWER_FORMATS, hardware_bucket
from cuda_sft_qgen.jsonutil import extract_complete_json_field, normalize_language, parse_question_payload
from cuda_sft_qgen.keys import PRI_SALVAGE, PRI_WRITER
from cuda_sft_qgen.llm import FakeLLM, NvidiaLLM, ReasoningBudgetExceeded
from cuda_sft_qgen.models import (
    PairPlan,
    Question,
    QuestionBrief,
    SeedAnalysis,
    SteeringBrief,
    VerificationEvidence,
    VerificationSpec,
)
from cuda_sft_qgen.observability import agent_role, get_metrics
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.consistency import consistency_reason
from cuda_sft_qgen.review.fingerprint import fingerprint
from cuda_sft_qgen.review.writer_gate import writer_gate_reason
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.sieve import sieve_question

logger = logging.getLogger(__name__)

# One retry after a reasoning-only reply: twice the budget (capped) and one
# thinking level less when the role thinks above ``low``.
_REASONING_RETRY_MAX_TOKENS = 32768
_THINKING_STEP_DOWN = {"high": "medium", "medium": "low"}


def _with_consistency_warning(question: Question) -> Question:
    """Attach hw/leaf mismatch as a warning; do not reject on the writer path."""
    cons = consistency_reason(question)
    if cons and cons not in question.warnings:
        question.warnings.append(cons)
    return question


@dataclass
class GenerateOutcome:
    """Result of one brief: ok question, repair work, retry, or hard fail."""

    status: str
    question: Question | None = None
    brief: dict[str, Any] | None = None
    raw: str | None = None
    reason: str | None = None
    attempts: int = 1
    writer_real_api: bool = False


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
    generation_provenance: dict[str, Any] | None = None,
) -> Question:
    """Build a pending :class:`Question` from parsed writer JSON and the brief."""
    question_text = str(obj.get("question") or "").strip()
    tests_claim = str(obj.get("tests_claim") or "").strip()
    language = normalize_language(obj.get("language"), fallback=brief.language)
    warnings = list(obj.get("warnings")) if isinstance(obj.get("warnings"), list) else []
    answer_format = str(obj.get("answer_format") or "").strip()
    if brief.answer_format and answer_format != brief.answer_format:
        warnings.append("answer_format_mismatch")
        answer_format = brief.answer_format
    elif answer_format not in ANSWER_FORMATS:
        answer_format = brief.answer_format
    draft_rationale = str(obj.get("draft_rationale") or "").strip() or None
    raw_premises = obj.get("premise_check")
    premise_check = [str(item) for item in raw_premises if str(item).strip()][:12] if isinstance(raw_premises, list) else []
    if re.search(r"[\u4e00-\u9fff]", tests_claim):
        warnings.append("tests_claim_not_english")
    raw_plan = obj.get("draft_plan")
    if isinstance(raw_plan, dict):
        draft_plan = {str(key): value for key, value in raw_plan.items()}
    elif isinstance(raw_plan, str) and raw_plan.strip():
        draft_plan = {"plan": raw_plan.strip()}
    else:
        draft_plan = None
    raw_target = obj.get("draft_target")
    if raw_target is None:
        draft_target = None
    elif isinstance(raw_target, str):
        draft_target = raw_target.strip() or None
    else:
        draft_target = json.dumps(raw_target, ensure_ascii=False, sort_keys=True)
    mixup = brief.knowledge_context.get("mixup") if isinstance(brief.knowledge_context, dict) else None
    mixup = mixup if isinstance(mixup, dict) else {}
    validation_errors: list[str] = []
    draft_plan = draft_plan or {}
    seed_analysis = draft_plan.get("seed_analysis")
    pair_plan = draft_plan.get("pair_plan")
    if not draft_plan:
        validation_errors.append("missing_draft_plan")
    elif not isinstance(seed_analysis, dict):
        validation_errors.append("missing_seed_analysis")
    else:
        try:
            analyzed = SeedAnalysis.model_validate(seed_analysis)
            expected_domain = brief.topic_path[0] if brief.topic_path else ""
            if analyzed.domain_id and expected_domain and analyzed.domain_id != expected_domain:
                if analyzed.domain_id == ".".join(brief.topic_path):
                    draft_plan["seed_analysis"] = {**seed_analysis, "domain_id": expected_domain}
                else:
                    validation_errors.append("seed_domain_mismatch")
        except Exception as exc:
            validation_errors.append(f"invalid_seed_analysis:{type(exc).__name__}")
    if not isinstance(pair_plan, dict):
        validation_errors.append("missing_pair_plan")
    else:
        try:
            checked_plan = PairPlan.model_validate(pair_plan)
            expected_mode = brief.mixup_mode or mixup.get("mode")
            if expected_mode and checked_plan.mode != expected_mode:
                validation_errors.append("mixup_plan_mode_mismatch")
            if expected_mode and set(checked_plan.parent_ids) != set(brief.parent_ids or mixup.get("parent_ids") or []):
                validation_errors.append("mixup_plan_parent_mismatch")
        except Exception as exc:
            validation_errors.append(f"invalid_pair_plan:{type(exc).__name__}")
    if not draft_target:
        validation_errors.append("missing_draft_target")
    if not draft_rationale:
        validation_errors.append("missing_draft_rationale")
    if not premise_check:
        validation_errors.append("missing_premise_check")
    raw_spec = obj.get("verification_spec")
    verification_spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
    if not verification_spec:
        validation_errors.append("missing_verification_spec")
    else:
        try:
            VerificationSpec.model_validate(verification_spec)
        except Exception as exc:
            validation_errors.append(f"invalid_verification_spec:{type(exc).__name__}")
    # Evidence is produced only by the independent verifier.  In particular a
    # Writer claiming its program already passed compilation cannot authorize
    # export, even if the rest of its JSON contract is perfectly valid.
    verification_evidence = {"status": "not_run"}
    mixup = brief.knowledge_context.get("mixup") if isinstance(brief.knowledge_context, dict) else None
    mixup = mixup if isinstance(mixup, dict) else {}
    provenance = dict(generation_provenance or {})
    provenance.setdefault("writer_real_api", False)
    if provenance.get("structured_cot") is False:
        # Explicit baseline/legacy Writer mode does not request private fields.
        # Malformed fields that *are* supplied remain visible for review.
        validation_errors = [e for e in validation_errors if not e.startswith("missing_")]
    provenance.setdefault("task_id", brief.task_id)
    provenance.setdefault("role", brief.role)
    provenance.setdefault("seed_id", brief.seed_id)
    provenance.setdefault("seed_domain_id", brief.topic_path[0] if brief.topic_path else "")
    provenance.setdefault("claim_revision", brief.claim_revision)
    provenance.setdefault("source_ids", list(brief.source_ids))
    provenance["mixup_context"] = mixup
    provenance["draft_validation_errors"] = list(validation_errors)
    brief_snapshot = {
        "task_id": brief.task_id,
        "topic_path": list(brief.topic_path),
        "l3_id": brief.l3_id,
        "l3_claim": brief.l3_claim,
        "question_type": brief.question_type,
        "difficulty": brief.difficulty,
        "bloom": brief.bloom,
        "hardware": brief.hardware,
        "granularity": brief.granularity,
        "language": brief.language,
        "answer_format": brief.answer_format,
        "scenario": brief.scenario,
        "arch_target": brief.arch_target,
        "operator": brief.operator,
        "exploration": brief.exploration,
        "kind": brief.kind,
        "role": brief.role,
        "seed_id": brief.seed_id,
        "angle_id": brief.angle_id,
        "claim_revision": brief.claim_revision,
        "source_ids": list(brief.source_ids),
        "mixup_mode": brief.mixup_mode,
        "parent_ids": list(brief.parent_ids),
        "parent_difficulties": list(brief.parent_difficulties),
        "difficulty_target": brief.difficulty_target,
        "mixup_context": mixup,
    }
    fp = fingerprint(question_text)
    dependencies: list[dict[str, Any]] = []
    for claim in (brief.knowledge_context.get("claims", []) if isinstance(brief.knowledge_context, dict) else []):
        if isinstance(claim, dict) and claim.get("id"):
            dependencies.append({"claim_id": claim["id"], "revision": claim.get("revision", 1)})
    for parent in mixup.get("parents", []) if isinstance(mixup.get("parents"), list) else []:
        if isinstance(parent, dict) and parent.get("l3_id"):
            dependencies.append({"claim_id": parent["l3_id"], "revision": parent.get("claim_revision", 1)})
    seen_dependencies = set()
    dependencies = [d for d in dependencies if not ((d["claim_id"], d["revision"]) in seen_dependencies or seen_dependencies.add((d["claim_id"], d["revision"])))]
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
        hardware=hardware_bucket(brief.arch_target) if brief.arch_target != "any" else brief.hardware,
        granularity=brief.granularity,
        language=language,
        answer_format=answer_format,
        scenario=brief.scenario,
        arch_target=brief.arch_target,
        draft_rationale=draft_rationale,
        premise_check=premise_check,
        exploration=brief.exploration,
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
        knowledge_dependencies=dependencies,
        draft_plan=draft_plan,
        draft_target=draft_target,
        verification_spec=verification_spec,
        verification_evidence=verification_evidence,
        draft_validation_errors=validation_errors,
        generation_provenance=provenance,
        mixup_context=mixup,
        brief_snapshot=brief_snapshot,
        mixup_mode=brief.mixup_mode or mixup.get("mode"),
        parent_ids=list(brief.parent_ids or mixup.get("parent_ids") or []),
        parent_difficulties=list(brief.parent_difficulties or mixup.get("parent_difficulties") or []),
        difficulty_target=brief.difficulty_target or mixup.get("difficulty_target"),
        difficulty_raw_score=(
            brief.difficulty_raw_score
            if brief.difficulty_raw_score is not None
            else mixup.get("difficulty_raw_score")
        ),
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
    if isinstance(rt.llm, FakeLLM):
        raise RuntimeError("Question generation requires a real LLM API client")
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
    bundle = builder.build(brief, steering, corpus=corpus)
    source = "evolver" if brief.role == "evolver" and brief.seed_question else "writer"
    generator = (rt.evolver_llm or rt.llm) if source == "evolver" else rt.llm
    api = rt.settings.agent_apis[source]
    user = bundle.user
    if repair_note:
        user = (
            user
            + "\n\nA previous draft failed a quality check for this defect:\n"
            + str(repair_note).strip()[:400]
            + "\nRewrite ONE new question for the same brief. Keep the leaf claim. "
            "Fix the defect. Return JSON only."
        )
    messages = [
        {"role": "system", "content": bundle.system},
        {"role": "user", "content": user},
    ]
    level = rt.settings.evolver_thinking_level if source == "evolver" else rt.settings.writer_thinking_level
    with agent_role(source):
        try:
            raw = generator.complete(
                messages,
                priority=PRI_WRITER,
                prompt_cache_key=bundle.cache_key or None,
                thinking_level=level,
            )
        except ReasoningBudgetExceeded:
            # Reasoning ate the whole budget: retry once with more room. A second
            # failure propagates and _safe_generate queues the brief for retry.
            budget = min(2 * int(rt.settings.max_tokens), _REASONING_RETRY_MAX_TOKENS)
            retry_level = _THINKING_STEP_DOWN.get(level, level)
            logger.warning("writer_reasoning_budget_exceeded role=%s retry_max_tokens=%s thinking=%s->%s",
                           source, budget, level, retry_level)
            raw = generator.complete(
                messages,
                priority=PRI_WRITER,
                prompt_cache_key=bundle.cache_key or None,
                thinking_level=retry_level,
                max_tokens=budget,
            )
    obj = parse_question_payload(
        raw,
        fallback_claim=brief.l3_claim,
        min_question_chars=rt.settings.sieve_min_chars,
        keep_truncated=rt.settings.json_keep_truncated,
    )
    if obj is None or not str(obj.get("question") or "").strip():
        raise JsonSalvageNeeded(raw, brief.model_dump())
    question = question_from_obj(
        brief, obj, source=source,
        generation_provenance={
            **(getattr(bundle, "generation_provenance", None) or {}),
            "writer_real_api": not rt.settings.fake_llm and isinstance(generator, NvidiaLLM),
            "writer_provider": api.provider,
            "writer_model": api.model,
        },
    )
    if len(question.question) > rt.settings.question_max_chars:
        raise JsonSalvageNeeded(json.dumps(obj, ensure_ascii=False), brief.model_dump())
    return question


def _safe_generate(
    brief_data: dict,
    steering: SteeringBrief,
    *,
    attempts: int = 1,
    corpus: FewshotCorpus | None = None,
    defer_rewrite: bool = False,
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
            writer_real_api=not rt.settings.fake_llm and isinstance(
                (rt.evolver_llm or rt.llm) if brief.role == "evolver" and brief.seed_question else rt.llm,
                NvidiaLLM,
            ),
        )
    except ReasoningBudgetExceeded:
        # Already retried with a larger budget; like a 429 this is capacity, not quality.
        logger.warning("generation reasoning budget exhausted twice, will retry later l3=%s", brief.l3_id)
        return GenerateOutcome(
            status="retry",
            brief=brief.model_dump(),
            reason="reasoning_budget",
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
        question, brief, steering, corpus=corpus, attempts=attempts,
        defer_rewrite=defer_rewrite,
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
    use_llm = (_gate_enabled(settings)
               and bool(settings.generation.get("writer_gate_llm", False))
               and not bool(getattr(settings, "fake_llm", False)))
    max_tokens = int(settings.generation.get("writer_gate_max_tokens", 512) or 512)
    with agent_role("writer_gate"):
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
    defer_rewrite: bool = False,
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
    if defer_rewrite:
        return GenerateOutcome(
            status="repair", question=question, brief=brief.model_dump(),
            reason=reason, attempts=attempts,
        )
    return _rewrite_draft(question, brief, steering, reason, corpus=corpus, attempts=attempts)


def _rewrite_draft(
    question: Question,
    brief: QuestionBrief,
    steering: SteeringBrief,
    reason: str,
    *,
    corpus: FewshotCorpus | None,
    attempts: int,
) -> GenerateOutcome:
    """Use one real Writer call to repair a gated draft, then recheck it."""
    rt = get_runtime()
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


def rewrite_item(item: dict[str, Any]) -> GenerateOutcome:
    """Run a deferred pre-chair rewrite on the repair worker."""
    brief = QuestionBrief.model_validate(item["brief"])
    question = Question.model_validate(item["question"])
    steering = SteeringBrief.model_validate(item.get("steering") or {})
    return _rewrite_draft(
        question, brief, steering, str(item.get("reason") or "writer_gate"),
        corpus=None, attempts=int(item.get("attempts") or 1),
    )


def review_repair_item(item: dict[str, Any]) -> GenerateOutcome:
    """Regenerate one rejected item from its original brief and review feedback."""
    rt = get_runtime()
    brief = QuestionBrief.model_validate(item["brief"])
    steering = SteeringBrief.model_validate(item.get("steering") or {})
    attempts = int(item.get("attempts") or 1)
    try:
        question = generate_from_brief(
            brief, steering, repair_note=str(item.get("reason") or "quality review")
        )
    except (RateLimitError, JsonSalvageNeeded):
        return GenerateOutcome(status="retry", brief=brief.model_dump(),
                               reason="review_repair_api", attempts=attempts)
    except Exception as exc:  # noqa: BLE001
        logger.warning("review repair deferred error=%s", type(exc).__name__)
        return GenerateOutcome(status="retry", brief=brief.model_dump(),
                               reason=f"review_repair_error:{type(exc).__name__}", attempts=attempts)
    reason = sieve_question(question, rt.settings, rt.llm)
    if reason:
        return _reject_draft(question, brief, reason, attempts)
    outcome = _gate_or_rewrite(
        question, brief, steering, corpus=None, attempts=attempts
    )
    if outcome.status == "ok" and outcome.question is not None:
        outcome.question.generation_provenance["review_repair_of"] = str(item["source_id"])
        outcome.question.generation_provenance["review_repair_attempt"] = attempts
    return outcome


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
        q = question_from_obj(
            brief, obj, source=brief.role,
            generation_provenance={
                "structured_cot": bool(rt.settings.generation.get("prompt_cot", True)),
                "salvaged": True, "writer_real_api": item.get("writer_real_api") is True,
                "writer_provider": rt.settings.agent_apis["evolver" if brief.role == "evolver" else "writer"].provider,
                "writer_model": rt.settings.agent_apis["evolver" if brief.role == "evolver" else "writer"].model,
            },
        )
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
    original_question = extract_complete_json_field(raw, "question")
    if not isinstance(original_question, str) or not original_question.strip():
        return GenerateOutcome(status="retry", brief=brief.model_dump(),
                               reason="salvage_question_unverifiable", attempts=1)
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
            with agent_role("salvage"):
                text = (rt.salvage_llm or rt.llm).complete(
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
            if str(obj["question"]).strip() != original_question.strip():
                last_err = "salvage_changed_question"
                continue
            preserved_keys = ("tests_claim", "language", "answer_format", "draft_target",
                              "draft_rationale", "premise_check", "verification_spec", "draft_plan")
            if any(extract_complete_json_field(raw, key) != obj.get(key)
                   for key in preserved_keys if key in obj):
                last_err = "salvage_changed_field"
                continue
            q = question_from_obj(
                brief, obj, source=brief.role,
                generation_provenance={
                    "structured_cot": bool(rt.settings.generation.get("prompt_cot", True)),
                    "salvaged": True, "writer_real_api": item.get("writer_real_api") is True,
                    "writer_provider": rt.settings.agent_apis["evolver" if brief.role == "evolver" else "writer"].provider,
                    "writer_model": rt.settings.agent_apis["evolver" if brief.role == "evolver" else "writer"].model,
                },
            )
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
    defer_rewrite: bool = False,
) -> GenerateOutcome:
    """Generate, and if salvage_inline is set, immediately repair JSON failures."""
    outcome = _safe_generate(
        brief_data, steering, attempts=attempts, corpus=corpus,
        defer_rewrite=defer_rewrite,
    )
    if (
        outcome.status == "salvage"
        and outcome.brief is not None
        and get_runtime().settings.salvage_inline
    ):
        return salvage_item({"brief": outcome.brief, "raw": outcome.raw or "",
                             "writer_real_api": outcome.writer_real_api})
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
    repair_items: list[dict] | None = None,
    steering: SteeringBrief | None = None,
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
        item = {"brief": outcome.brief, "raw": outcome.raw or "",
                "writer_real_api": outcome.writer_real_api}
        with lock:
            salvage_items.append(item)
            rt.store.append_salvage_raw(
                {"reason": "json_parse", "raw": outcome.raw or "", "brief": outcome.brief}
            )
        return "salvage"
    if outcome.status == "repair" and outcome.brief is not None and outcome.question is not None:
        item = {
            "brief": outcome.brief,
            "question": outcome.question.model_dump(),
            "reason": outcome.reason or "writer_gate",
            "attempts": outcome.attempts,
            "steering": (steering or SteeringBrief()).model_dump(),
        }
        if repair_items is None:
            raise ValueError("deferred rewrite requires a repair queue")
        with lock:
            repair_items.append(item)
        return "repair"
    if outcome.status == "retry" and outcome.brief is not None:
        enqueue_retry(outcome.brief, reason=outcome.reason or "429", attempts=outcome.attempts)
        complete_task()
        return "retry"
    complete_task()
    return "hard_fail"
