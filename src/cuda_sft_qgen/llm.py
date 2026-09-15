from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Iterable

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.errors import RateLimitError
from cuda_sft_qgen.jsonutil import parse_json_object, strip_think
from cuda_sft_qgen.keys import ApiKeyPool

logger = logging.getLogger(__name__)


def thinking_kwargs(level: str) -> dict[str, Any]:
    level = (level or "medium").lower()
    if level == "none":
        return {"enable_thinking": False}
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
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        retries: int = 1,
    ) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            text = self.complete(
                messages,
                thinking_level=thinking_level,
                temperature=temperature,
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
    def __init__(self, settings: Settings, pool: ApiKeyPool) -> None:
        self.settings = settings
        self.pool = pool

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        thinking_level: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        level = thinking_level or self.settings.thinking_level
        extra = {"chat_template_kwargs": thinking_kwargs(level)}
        payload = {
            "model": self.settings.nvidia_model,
            "messages": messages,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
            "extra_body": extra,
        }
        return self._stream_with_retry(payload)

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        text = str(exc)
        lowered = text.lower()
        return (
            "429" in text
            or "too many requests" in lowered
            or "rate limit" in lowered
            or "overloaded" in lowered
            or "temporarily unavailable" in lowered
        )

    def _stream_with_retry(self, payload: dict[str, Any], retries: int | None = None) -> str:
        per_key = max(1, int(self.settings.llm_rate_limit_retries_per_call))
        max_rate = per_key * max(1, self.pool.n_keys)
        hard_retries = retries if retries is not None else max(max_rate, 6)
        rate_hits = [0] * max(1, self.pool.n_keys)
        last: Exception | None = None
        for attempt in range(hard_retries):
            if all(h >= per_key for h in rate_hits) and sum(rate_hits) > 0:
                raise RateLimitError(str(last) if last else "rate limited")
            slot = self.pool.acquire("llm")
            try:
                return self._stream_once(slot.client, payload)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if self._is_rate_limit(exc):
                    rate_hits[slot.index] = rate_hits[slot.index] + 1
                    wait = min(
                        self.settings.retry_max_cooldown_sec,
                        self.settings.retry_base_cooldown_sec
                        * (2 ** min(rate_hits[slot.index] - 1, 3)),
                    )
                    self.pool.mark_cooldown(slot, wait)
                    logger.warning(
                        "LLM %s rate-limited (attempt %s, key_hits=%s/%s): %s",
                        slot.label,
                        attempt + 1,
                        rate_hits[slot.index],
                        per_key,
                        exc,
                    )
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
        stream = client.chat.completions.create(**payload)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            if delta.content:
                content_parts.append(delta.content)
        content = strip_think("".join(content_parts))
        if self.settings.log_reasoning and reasoning_parts:
            logger.info("reasoning: %s", "".join(reasoning_parts)[:2000])
        if not content and reasoning_parts:
            logger.warning("empty content after stream; reasoning tokens were present")
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
    ) -> str:
        user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user = m.get("content") or ""
                break
        if "Taxonomy" in (messages[0].get("content") if messages else "") or "new_leaves" in user:
            return json.dumps({"new_leaves": []}, ensure_ascii=False)
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
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    if j < 0:
        return text[i + len(start) :].strip()
    return text[i + len(start) : j].strip()


def build_llm(settings: Settings, pool: ApiKeyPool | None = None) -> LLMClient:
    if settings.fake_llm:
        return FakeLLM()
    if pool is None:
        raise RuntimeError("NvidiaLLM requires an ApiKeyPool")
    return NvidiaLLM(settings, pool)


def iter_message_dicts(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return list(messages)
