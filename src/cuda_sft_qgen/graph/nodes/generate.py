from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from cuda_sft_qgen.errors import JsonSalvageNeeded, RateLimitError
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.jsonutil import normalize_language, parse_question_payload
from cuda_sft_qgen.keys import PRI_SALVAGE, PRI_WRITER
from cuda_sft_qgen.llm import NvidiaLLM
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
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
    language = normalize_language(obj.get("language"), fallback=brief.language)
    warnings = obj.get("warnings") if isinstance(obj.get("warnings"), list) else []
    fp = fingerprint(question_text)
    return Question(
        id="q_" + uuid.uuid4().hex,
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
    )


def generate_from_brief(
    brief: QuestionBrief,
    steering: SteeringBrief,
    *,
    corpus: FewshotCorpus | None = None,
) -> Question:
    rt = get_runtime()
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
    messages = [
        {"role": "system", "content": bundle.system},
        {"role": "user", "content": bundle.user},
    ]
    raw = rt.llm.complete(
        messages,
        priority=PRI_WRITER,
        prompt_cache_key=bundle.cache_key or None,
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
            id="q_" + uuid.uuid4().hex,
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
        return GenerateOutcome(status="hard_fail", reason=str(exc), attempts=attempts)
    reason = sieve_question(question, rt.settings, rt.llm)
    if reason:
        question.reject_reason = reason
        rt.store.append_rejected(question)
        return GenerateOutcome(status="hard_fail", reason=reason, attempts=attempts)
    return GenerateOutcome(status="ok", question=question, attempts=attempts)


def salvage_item(item: dict[str, Any]) -> GenerateOutcome:
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
            return GenerateOutcome(status="hard_fail", reason=reason)
        return GenerateOutcome(status="ok", question=q)

    if not rt.settings.salvage_use_llm:
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
                return GenerateOutcome(status="hard_fail", reason=reason)
            return GenerateOutcome(status="ok", question=q)
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
    outcome = _safe_generate(brief_data, steering, attempts=attempts, corpus=corpus)
    if outcome.status == "salvage" and outcome.brief is not None:
        return salvage_item({"brief": outcome.brief, "raw": outcome.raw or ""})
    return outcome


def enqueue_retry(brief: dict[str, Any], *, reason: str, attempts: int) -> None:
    rt = get_runtime()
    if attempts >= rt.settings.retry_max_attempts:
        failed = Question(
            id="q_" + uuid.uuid4().hex,
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
    anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
    corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)

    def _one(brief_data: dict) -> None:
        outcome = generate_or_salvage(brief_data, steering, attempts=1, corpus=corpus)
        kind = apply_outcome(
            outcome, pending=pending, salvage_items=salvage_items, lock=lock
        )
        if kind == "ok" and outcome.question is not None:
            prefetch = getattr(rt, "prefetch_embed", None)
            if callable(prefetch):
                prefetch(outcome.question.question)
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
