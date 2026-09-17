"""NVIDIA chat-completions client, FakeLLM, and JSON-completion helper.

Streaming is required so usage and reasoning tokens can be recorded. Empty
``content`` with non-empty ``reasoning_content`` is recovered as the reply.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterable

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.errors import RateLimitError
from cuda_sft_qgen.jsonutil import parse_json_object, strip_think
from cuda_sft_qgen.keys import PRI_WRITER, ApiKeyPool
from cuda_sft_qgen.observability import get_metrics

logger = logging.getLogger(__name__)


def thinking_kwargs(level: str) -> dict[str, Any]:
    """Map ``none|low|medium|high`` onto Nemotron ``chat_template_kwargs``."""
    level = (level or "medium").lower()
    if level == "none":
        return {"enable_thinking": False}
    if level in {"low", "low_effort"}:
        # Nemotron 3 Ultra: short reasoning (chat_template_kwargs.low_effort).
        return {
            "enable_thinking": True,
            "low_effort": True,
            "force_nonempty_content": True,
        }
    if level == "medium":
        return {
            "enable_thinking": True,
            "medium_effort": True,
            "force_nonempty_content": True,
        }
    if level == "high":
        return {"enable_thinking": True, "force_nonempty_content": True}
    raise ValueError(f"unknown thinking level: {level}")


class LLMClient:
    """Minimal chat client: complete text or a parsed JSON object."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        priority: int = PRI_WRITER,
        prompt_cache_key: str | None = None,
    ) -> str:
        """Return assistant text for ``messages``."""
        raise NotImplementedError

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        retries: int = 1,
        priority: int = PRI_WRITER,
        prompt_cache_key: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call :meth:`complete` and parse a JSON object, with a repair retry."""
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            text = self.complete(
                messages,
                thinking_level=thinking_level,
                temperature=temperature,
                priority=priority,
                prompt_cache_key=prompt_cache_key,
                max_tokens=max_tokens,
            )
            try:
                return parse_json_object(text)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("JSON parse failed (attempt %s): %s", attempt + 1, exc)
                messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON. "
                            "Reply with a single JSON object only, no markdown."
                        ),
                    }
                ]
        raise ValueError(f"failed to parse JSON: {last_err}")


class NvidiaLLM(LLMClient):
    """Streaming NVIDIA Integrate API client backed by :class:`ApiKeyPool`."""

    def __init__(self, settings: Settings, pool: ApiKeyPool) -> None:
        """Bind runtime settings and the shared API key pool."""
        self.settings = settings
        self.pool = pool

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        priority: int = PRI_WRITER,
        prompt_cache_key: str | None = None,
    ) -> str:
        """Stream a completion; records duration, tokens, and failures."""
        level = thinking_level or self.settings.thinking_level
        extra: dict[str, Any] = {"chat_template_kwargs": thinking_kwargs(level)}
        payload = {
            "model": self.settings.nvidia_model,
            "messages": messages,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": extra,
        }
        t0 = time.monotonic()
        metrics = get_metrics()
        labels = {"priority": str(priority), "model": self.settings.nvidia_model}
        metrics.increment("llm_requests", labels=labels)
        try:
            text = self._stream_with_retry(payload, priority=priority)
        except Exception:
            metrics.increment("llm_failures", labels=labels)
            raise
        finally:
            metrics.observe("llm_seconds", time.monotonic() - t0, labels)
        logger.info(
            "llm_complete duration_s=%.2f out_chars=%s thinking=%s max_tokens=%s priority=%s",
            time.monotonic() - t0,
            len(text or ""),
            level,
            payload["max_tokens"],
            priority,
        )
        return text

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        """True for 429 / overloaded / temporarily unavailable errors."""
        text = str(exc)
        lowered = text.lower()
        return (
            "429" in text
            or "too many requests" in lowered
            or "rate limit" in lowered
            or "overloaded" in lowered
            or "temporarily unavailable" in lowered
        )

    def _stream_with_retry(
        self,
        payload: dict[str, Any],
        retries: int | None = None,
        priority: int = PRI_WRITER,
    ) -> str:
        """Acquire a key, stream, and rotate on 429 until per-key budget is spent."""
        per_key = max(1, int(self.settings.llm_rate_limit_retries_per_call))
        max_rate = per_key * max(1, self.pool.n_keys)
        hard_retries = retries if retries is not None else max(max_rate, 6)
        rate_hits = [0] * max(1, self.pool.n_keys)
        last: Exception | None = None
        for attempt in range(hard_retries):
            if all(h >= per_key for h in rate_hits) and sum(rate_hits) > 0:
                raise RateLimitError(str(last) if last else "rate limited")
            slot = self.pool.acquire("llm", priority=priority)
            try:
                text = self._stream_once(slot.client, payload)
                self.pool.note_success("llm")
                return text
            except Exception as exc:  # noqa: BLE001
                last = exc
                if self._is_rate_limit(exc):
                    rate_hits[slot.index] = rate_hits[slot.index] + 1
                    wait = min(
                        self.settings.retry_max_cooldown_sec,
                        self.settings.retry_base_cooldown_sec
                        * (2 ** min(rate_hits[slot.index] - 1, 3)),
                    )
                    self.pool.mark_cooldown(slot, wait, kind="llm")
                    logger.warning(
                        "LLM %s rate-limited (attempt %s, key_hits=%s/%s): %s",
                        slot.label,
                        attempt + 1,
                        rate_hits[slot.index],
                        per_key,
                        exc,
                    )
                    # Do not hop onto the remaining keys in the same millisecond.
                    time.sleep(min(2.0, max(0.4, wait * 0.1)))
                    continue
                wait = min(60.0, float(2 ** attempt))
                logger.warning(
                    "LLM %s failed (attempt %s/%s): %s; sleep %.0fs",
                    slot.label,
                    attempt + 1,
                    hard_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            finally:
                self.pool.release("llm", slot)
        if last and self._is_rate_limit(last):
            raise RateLimitError(str(last))
        raise RuntimeError(f"LLM call failed after retries: {last}")

    def _stream_once(self, client: Any, payload: dict[str, Any]) -> str:
        """Consume one streamed completion; recover content from reasoning if empty."""
        stream = client.chat.completions.create(**payload)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = None
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                labels = {"model": str(payload["model"])}
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = getattr(usage, field, None)
                    if isinstance(value, int):
                        get_metrics().increment("llm_" + field, value, labels)
                get_metrics().increment("llm_usage_reports", labels=labels)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = choice.delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if delta.content:
                content_parts.append(delta.content)
        content = strip_think("".join(content_parts))
        reasoning = "".join(reasoning_parts)
        if self.settings.log_reasoning and reasoning:
            logger.info("reasoning: %s", reasoning[:2000])
        if not content and reasoning:
            recovered = strip_think(reasoning)
            if recovered:
                logger.warning(
                    "empty content after stream; recovered %s chars from reasoning (finish=%s)",
                    len(recovered),
                    finish_reason,
                )
                return recovered
            logger.warning(
                "empty content after stream; reasoning tokens were present finish=%s",
                finish_reason,
            )
        elif finish_reason and finish_reason not in {"stop", "None", None}:
            logger.info("llm_finish reason=%s content_chars=%s", finish_reason, len(content))
        return content


class FakeLLM(LLMClient):
    """Deterministic stand-in for tests and --fake runs."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        priority: int = PRI_WRITER,
        prompt_cache_key: str | None = None,
    ) -> str:
        """Return a deterministic JSON fixture matching the prompt role (writer/chair/taxonomy)."""
        user = ""
        system = ""
        for m in messages:
            if m.get("role") == "system" and not system:
                system = m.get("content") or ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content") or ""
                break
        if "QUALITY_REVIEW_V1" in system:
            payload = json.loads(user)
            # Offline fixtures exercise the contract, not CUDA factual correctness.
            return json.dumps({
                "id": payload["id"], "action": "accept",
                "reason": "Offline fixture contract accepted; not a factual assessment.",
                "checks": {key: "pass" for key in (
                    "correctness", "answerability", "claim_alignment", "hardware_compatibility",
                    "answer_leakage", "metadata", "duplication",
                )},
                "evidence": ["Offline fixture response; no factual verification performed."],
                "internal_reference_answer": "Offline fixture reference answer.",
                "assessed_metadata": payload["requested_metadata"],
            }, ensure_ascii=False)
        if "suggested_claim" in user:
            return json.dumps({"suggestions": []}, ensure_ascii=False)
        if "health_notes" in user or "WorkOrder" in user or '"refine_ids"' in user:
            return json.dumps(
                {
                    "health_notes": [],
                    "misplaced": [],
                    "unanswerable": [],
                    "kind_mismatch": [],
                    "expansion_priority": [],
                    "search_queries": [],
                    "refine_ids": [],
                },
                ensure_ascii=False,
            )
        if "Taxonomy" in (messages[0].get("content") if messages else "") or "new_leaves" in user:
            return json.dumps({"new_leaves": []}, ensure_ascii=False)
        blob = f"{system}{user}"
        if "suggestions" in blob and "new_leaves" not in user:
            return json.dumps({"suggestions": []}, ensure_ascii=False)
        if "keep_both" in user or "retire_id" in user:
            return json.dumps({"decisions": []}, ensure_ascii=False)
        if (
            '"decisions"' in user
            or "SteeringBrief" in user
            or "审题" in user
            or "Diversity report" in user
            or "Gray items" in user
        ):
            return json.dumps(
                {
                    "decisions": [],
                    "steering_brief": {
                        "underrepresented": [],
                        "overrepresented": [],
                        "forbidden_templates": [],
                        "preferred_types": ["debug", "layout"],
                        "difficulty_nudge": "hold",
                        "style_notes": "keep questions answerable",
                    },
                },
                ensure_ascii=False,
            )
        if "garbage" in user.lower() and "meaningful" in user.lower():
            return json.dumps({"label": "meaningful"})
        nonce = uuid.uuid4().hex[:8]
        claim = _extract_between(user, "CLAIM:", "\n") or "CUDA occupancy vs register pressure"
        qtype = _extract_between(user, "TYPE:", "\n") or "conceptual"
        hw = _extract_between(user, "HARDWARE:", "\n") or "hopper"
        question = (
            f"在 {hw} 架构上，针对「{claim}」，从 {qtype} 角度给出一道可验证的问题"
            f"（实例标记 {nonce}）：如果违反该约束，会出现什么可观察的错误或性能现象？"
            "请写明判断依据，不要直接给出完整答案。"
        )
        return json.dumps(
            {
                "question": question,
                "tests_claim": claim,
                "language": "zh",
            },
            ensure_ascii=False,
        )


def _extract_between(text: str, start: str, end: str) -> str:
    """Slice ``text`` between ``start`` and ``end`` for FakeLLM brief parsing."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        return text[i + len(start) :].strip()
    return text[i + len(start) : j].strip()


def build_llm(settings: Settings, pool: ApiKeyPool | None = None) -> LLMClient:
    """Return FakeLLM when ``fake_llm`` is set; otherwise NvidiaLLM.

    Raises:
        RuntimeError: If a real client is requested without an API key pool.
    """
    if settings.fake_llm:
        return FakeLLM()
    if pool is None:
        raise RuntimeError("NvidiaLLM requires an ApiKeyPool")
    return NvidiaLLM(settings, pool)


def iter_message_dicts(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Materialize a message iterable (used by tests and prompt dumps)."""
    return list(messages)
