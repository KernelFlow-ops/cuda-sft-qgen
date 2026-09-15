from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from cuda_sft_qgen.errors import JsonSalvageNeeded, RateLimitError
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.jsonutil import parse_question_payload
from cuda_sft_qgen.llm import NvidiaLLM
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.review.fingerprint import fingerprint
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.sieve import sieve_question

logger = logging.getLogger(__name__)


@dataclass
class GenerateOutcome:
    status: str
    question: Question | None = None
    brief: dict[str, Any] | None = None
    raw: str | None = None
    reason: str | None = None
    attempts: int = 1


def retry_wait_sec(attempts: int) -> float:
    rt = get_runtime()
    wait = rt.settings.retry_base_cooldown_sec * (2 ** min(max(attempts - 1, 0), 4))
    return min(rt.settings.retry_max_cooldown_sec, wait)


def question_from_obj(
    brief: QuestionBrief,
    obj: dict[str, Any],
    *,
    source: str,
) -> Question:
    question_text = str(obj.get("question") or "").strip()
    tests_claim = str(obj.get("tests_claim") or brief.l3_claim).strip()
    language = str(obj.get("language") or brief.language).strip() or brief.language
    warnings = obj.get("warnings") if isinstance(obj.get("warnings"), list) else []
    fp = fingerprint(question_text)
    return Question(
        id="q_" + uuid.uuid4().hex,
        question=question_text,
        tests_claim=tests_claim,
        topic_path=list(brief.topic_path),
        l3_id=brief.l3_id,
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
    )


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _fewshot_block(brief: QuestionBrief) -> str:
    rt = get_runtime()
    accepted = rt.store.load_accepted()
    if not accepted:
        return "(none yet)"
    picked: list = []
    seen_l3: set[str] = {brief.l3_id}
    seen_type: set[str] = {brief.question_type}
    for q in reversed(accepted):
        if q.l3_id in seen_l3 and q.question_type in seen_type:
            continue
        picked.append(q)
        seen_l3.add(q.l3_id)
        seen_type.add(q.question_type)
        if len(picked) >= 2:
            break
    if not picked:
        picked = accepted[-2:]
    lines = []
    for q in picked:
        lines.append(f"- ({q.question_type}/{q.l3_id}) {q.question[:400]}")
    return "\n".join(lines)


def _anti_block(brief: QuestionBrief) -> str:
    items = list(brief.anti_templates)
    rt = get_runtime()
    k = int(rt.settings.generation.get("fewshot_negatives", 8))
    for skel in rt.store.recent_rejected_skeletons(k):
        if skel not in items:
            items.append(skel)
    if not items:
        return "(none)"
    return "\n".join(f"- {s[:240]}" for s in items[:k])


def _persona(brief: QuestionBrief) -> tuple[str, str]:
    meta = get_runtime().settings.personas.get(brief.persona) or {}
    return str(meta.get("name") or brief.persona), str(meta.get("voice") or "")


def generate_from_brief(brief: QuestionBrief, steering: SteeringBrief) -> Question:
    rt = get_runtime()
    persona_name, persona_voice = _persona(brief)
    mapping = {
        "persona_name": persona_name,
        "persona_voice": persona_voice,
        "l3_claim": brief.l3_claim,
        "topic_path": ".".join(brief.topic_path),
        "question_type": brief.question_type,
        "difficulty": str(brief.difficulty),
        "bloom": brief.bloom,
        "hardware": brief.hardware,
        "granularity": brief.granularity,
        "language": brief.language,
        "must_include": brief.must_include,
        "must_avoid": brief.must_avoid,
        "steering": steering.to_prompt_block(),
        "anti_templates": _anti_block(brief),
        "fewshot": _fewshot_block(brief),
        "seed_question": brief.seed_question or "",
        "angle": brief.angle or "",
        "instance_hint": brief.instance_hint or brief.must_include or brief.l3_claim,
        "must_avoid_stems": "\n".join(f"- {s}" for s in brief.must_avoid_stems) or "(none)",
        "angle_guide": brief.angle_guide or "",
        "min_chars": str(rt.settings.question_min_chars),
        "max_chars": str(rt.settings.question_max_chars),
    }
    if brief.role == "evolver" and brief.seed_question:
        system = _render(load_prompt("evolver"), mapping)
        user = (
            f"CLAIM: {brief.l3_claim}\nTYPE: {brief.question_type}\n"
            f"HARDWARE: {brief.hardware}\nSEED: {brief.seed_question}"
        )
        source = "evolver"
    else:
        system = _render(load_prompt("writer"), mapping)
        user = (
            f"CLAIM: {brief.l3_claim}\nTYPE: {brief.question_type}\n"
            f"HARDWARE: {brief.hardware}\nBLOOM: {brief.bloom}\n"
            f"Write the JSON now."
        )
        source = "writer"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    raw = rt.llm.complete(messages)
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
) -> GenerateOutcome:
    brief = QuestionBrief.model_validate(brief_data)
    rt = get_runtime()
    try:
        question = generate_from_brief(brief, steering)
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
            id="q_" + uuid.uuid4().hex,
            question="",
            tests_claim=brief.l3_claim,
            topic_path=list(brief.topic_path),
            l3_id=brief.l3_id,
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
        return GenerateOutcome(status="hard_fail", reason=str(exc), attempts=attempts)
    reason = sieve_question(question, rt.settings, rt.llm)
    if reason:
        question.reject_reason = reason
        rt.store.append_rejected(question)
        return GenerateOutcome(status="hard_fail", reason=reason, attempts=attempts)
    return GenerateOutcome(status="ok", question=question, attempts=attempts)


def enqueue_retry(brief: dict[str, Any], *, reason: str, attempts: int) -> None:
    rt = get_runtime()
    if attempts >= rt.settings.retry_max_attempts:
        failed = Question(
            id="q_" + uuid.uuid4().hex,
            question="",
            tests_claim=str(brief.get("l3_claim") or ""),
            topic_path=list(brief.get("topic_path") or []),
            l3_id=str(brief.get("l3_id") or ""),
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
        return
    rt.store.enqueue_retry(
        {
            "brief": brief,
            "reason": reason,
            "attempts": attempts,
            "next_ok_at": time.time() + retry_wait_sec(attempts),
        }
    )


def apply_outcome(
    outcome: GenerateOutcome,
    *,
    pending: list[dict],
    salvage_items: list[dict],
    lock: threading.Lock,
) -> str:
    rt = get_runtime()
    if outcome.status == "ok" and outcome.question is not None:
        rec = outcome.question.to_jsonl()
        with lock:
            pending.append(rec)
            rt.store.save_pending([Question.model_validate(x) for x in pending])
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
        return "retry"
    return "hard_fail"


def generate_batch_node(state: PipelineState) -> dict:
    rt = get_runtime()
    briefs = state.get("batch_briefs") or []
    steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
    pending = list(state.get("pending") or [])
    if not briefs:
        return {"status": "generated_0", "pending": pending, "salvage_items": []}
    workers = max(1, rt.settings.writer_concurrency)
    logger.info(
        "generate_batch briefs=%s workers=%s keys=%s llm_per_key=%s",
        len(briefs),
        workers,
        len(rt.settings.nvidia_api_keys),
        rt.settings.llm_per_key_concurrency,
    )
    salvage_items: list[dict] = []
    counts = {"ok": 0, "salvage": 0, "retry": 0, "hard_fail": 0}
    lock = threading.Lock()

    def _one(brief_data: dict) -> None:
        outcome = _safe_generate(brief_data, steering, attempts=1)
        kind = apply_outcome(
            outcome, pending=pending, salvage_items=salvage_items, lock=lock
        )
        with lock:
            counts[kind] = counts.get(kind, 0) + 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, b) for b in briefs]
        for fut in as_completed(futs):
            fut.result()
    return {
        "pending": pending,
        "batch_briefs": [],
        "salvage_items": salvage_items,
        "rejected_count": int(state.get("rejected_count") or 0) + counts["hard_fail"],
        "status": f"generated_{counts['ok']}",
        "log": [
            f"generate ok={counts['ok']} salvage={counts['salvage']} "
            f"retry={counts['retry']} fail={counts['hard_fail']} pending={len(pending)}"
        ],
    }
