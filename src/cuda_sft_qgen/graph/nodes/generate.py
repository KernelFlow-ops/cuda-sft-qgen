"""Writer/evolver node: parallel generation, local sieve, and JSON salvage hooks."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from cuda_sft_qgen.errors import GeneratorError, JsonSalvageNeeded, RateLimitError
from cuda_sft_qgen.executor import PriorityExecutor
from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.jsonutil import normalize_language, parse_question_payload
from cuda_sft_qgen.keys import PRI_CHAIR, PRI_SALVAGE, PRI_TAXONOMY, PRI_WRITER
from cuda_sft_qgen.llm import NvidiaLLM
from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.observability import get_metrics, trace_node
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.prompts.fewshot import FewshotCorpus
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.fingerprint import fingerprint
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.review.consistency import consistency_reason
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
    messages = [
        {"role": "system", "content": bundle.system},
        {"role": "user", "content": bundle.user},
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
    return GenerateOutcome(
        status="ok",
        question=_with_consistency_warning(question),
        attempts=attempts,
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
        return GenerateOutcome(status="ok", question=_with_consistency_warning(q))

    if not rt.settings.salvage_use_llm:
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
            return GenerateOutcome(status="ok", question=_with_consistency_warning(q))
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


@trace_node("generate_batch")
def generate_batch_node(state: PipelineState) -> dict:
    """Run the current brief batch in a priority executor; collect salvage items."""
    try:
        rt = get_runtime()
        briefs = state.get("batch_briefs") or []
        steering = SteeringBrief.model_validate(state.get("steering_brief") or {})
        pending = list(state.get("pending") or [])
        if not briefs:
            return {"status": "generated_0", "pending": pending, "salvage_items": []}

        workers = max(1, min(len(briefs), rt.settings.writer_concurrency))
        queue_size = int(rt.settings.generation.get("priority_queue_size", 200))
        backpressure_threshold = float(rt.settings.generation.get("backpressure_threshold", 0.20))
        backpressure_sleep = float(rt.settings.generation.get("backpressure_sleep", 0.5))
        stagger_sec = float(rt.settings.generation.get("submit_stagger_sec", 0.8))
        if rt.settings.fake_llm:
            stagger_sec = 0.0

        logger.info(
            "generate_batch briefs=%s workers=%s keys=%s llm_per_key=%s queue_size=%s stagger=%.2fs",
            len(briefs),
            workers,
            len(rt.settings.nvidia_api_keys),
            rt.settings.llm_per_key_concurrency,
            queue_size,
            stagger_sec,
        )

        salvage_items: list[dict] = []
        counts = {"ok": 0, "salvage": 0, "retry": 0, "hard_fail": 0}
        lock = threading.Lock()
        anti_k = int(rt.settings.generation.get("anti_templates_max", 15))
        corpus = FewshotCorpus.from_store(rt.store, anti_k=anti_k)
        batch_t0 = time.monotonic()

        # Use PriorityExecutor instead of ThreadPoolExecutor
        executor = PriorityExecutor(max_workers=workers, max_queue_size=queue_size)

        # Optional tqdm progress bar
        pbar = None
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(briefs), desc="Generating", unit="brief")
        except ImportError:
            pass

        def _one(brief_data: dict) -> None:
            t0 = time.monotonic()
            rt.store.save_generation_task(brief_data)
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
            logger.info(
                "generate_brief status=%s duration_s=%.2f l3=%s",
                kind,
                time.monotonic() - t0,
                (brief_data or {}).get("l3_id") or "",
            )
            if pbar is not None:
                pbar.update(1)

        try:
            futs = []
            for i, brief in enumerate(briefs):
                # Backpressure control: check remaining capacity before submitting
                if rt.pool is not None:
                    capacity_frac = rt.pool.remaining_capacity_fraction("llm")
                    if capacity_frac < backpressure_threshold:
                        logger.debug(
                            "Backpressure triggered: llm capacity=%.1f%% < %.1f%%, sleeping %.1fs",
                            capacity_frac * 100,
                            backpressure_threshold * 100,
                            backpressure_sleep,
                        )
                        time.sleep(backpressure_sleep)
                if i and stagger_sec > 0:
                    time.sleep(stagger_sec)

                # Submit with Writer priority (0)
                fut = executor.submit(PRI_WRITER, _one, brief)
                futs.append(fut)

            # Wait for all tasks to complete
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Task failed: %s", exc)
        finally:
            executor.shutdown(wait=True)
            if pbar is not None:
                pbar.close()
            rt.store.save_pending([Question.model_validate(x) for x in pending])

        elapsed = time.monotonic() - batch_t0
        rate = (counts["ok"] / elapsed * 60.0) if elapsed > 0 else 0.0
        logger.info(
            "generate_batch done ok=%s salvage=%s retry=%s fail=%s pending=%s "
            "duration_s=%.1f ok_per_min=%.2f",
            counts["ok"],
            counts["salvage"],
            counts["retry"],
            counts["hard_fail"],
            len(pending),
            elapsed,
            rate,
        )
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
    except Exception as exc:
        logger.error("Critical error in generate_batch_node: %s", exc)
        raise GeneratorError(
            f"Generation batch failed: {exc}",
            context={"brief_count": len(state.get("batch_briefs") or [])},
            recoverable=True,
        ) from exc
