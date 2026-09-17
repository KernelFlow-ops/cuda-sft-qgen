"""Robust JSON extraction from LLM output (fences, think traces, truncated objects).

Writers often wrap JSON in markdown, leak ``<think>`` traces, or emit raw
newlines inside strings. This module prefers the last valid object and can
fall back to field scanning when ``json.loads`` cannot accept the payload.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FENCE = re.compile(r"```([a-zA-Z0-9+-]*)\s*(.*?)\s*```", re.DOTALL)
_CODE_FENCE = re.compile(r"```[\w+-]*\n.*?```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_KEYS = ("question", "tests_claim", "language")
_LANG_TOKEN = re.compile(r"^[\"']?(en|zh|cn)(?:glish|inese)?", re.IGNORECASE)


def normalize_language(value: Any, fallback: str = "zh") -> str:
    """Map model output to zh/en; tolerate leaked extra JSON after the token."""
    text = str(value or "").strip()
    if text:
        m = _LANG_TOKEN.match(text.strip("\"'"))
        if m:
            token = m.group(1).lower()
            return "en" if token == "en" else "zh"
    fb = str(fallback or "zh").strip().strip("\"'").lower()
    return "en" if fb.startswith("en") else "zh"


def strip_think(text: str) -> str:
    """Drop Nemotron thinking traces that leak into content."""
    if not text:
        return ""
    lower = text.lower()
    marker = "</think>"
    idx = lower.rfind(marker)
    if idx >= 0:
        after = text[idx + len(marker) :].strip()
        if after:
            return after
    return _THINK.sub("", text).strip()


def _escape_raw_controls_in_strings(text: str) -> str:
    """Turn raw newlines/tabs inside JSON strings into escaped sequences."""
    out: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 32:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _extract_code_fences(text: str) -> tuple[str, list[str]]:
    """Replace inner code fences with placeholders so JSON parse can succeed."""
    blocks: list[str] = []

    def repl(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"__CODE_{len(blocks) - 1}__"

    return _CODE_FENCE.sub(repl, text), blocks


def _restore_code_fences(obj: dict[str, Any], blocks: list[str]) -> dict[str, Any]:
    """Put extracted fenced code back into string fields after JSON parse."""
    out: dict[str, Any] = {}
    for key, value in obj.items():
        if isinstance(value, str):
            for i, block in enumerate(blocks):
                value = value.replace(f"__CODE_{i}__", block)
            out[key] = value
        else:
            out[key] = value
    return out


def _loads_obj(raw: str) -> dict[str, Any] | None:
    """Parse a dict (or first dict in a list); return None on decode failure."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj = json.loads(_escape_raw_controls_in_strings(raw))
        except json.JSONDecodeError:
            return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return None


def _scan_objects(raw: str) -> dict[str, Any] | None:
    """Walk the text for ``{...}`` objects and return the last successful decode."""
    last: dict[str, Any] | None = None
    decoder = json.JSONDecoder()
    for src in (raw, _escape_raw_controls_in_strings(raw)):
        i = 0
        while i < len(src):
            j = src.find("{", i)
            if j < 0:
                break
            try:
                obj, end = decoder.raw_decode(src, j)
            except json.JSONDecodeError:
                i = j + 1
                continue
            if isinstance(obj, dict):
                last = obj
            i = end
        if last is not None:
            return last
    return None


def _unescape_json_string(value: str) -> str:
    """Undo a minimal set of JSON string escapes used by the field scanner."""
    return (
        value.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def _field_start(text: str, key: str) -> int | None:
    """Index of the value start for a JSON object key, or None if missing."""
    m = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
    if not m:
        return None
    return m.end()


def _next_key_index(text: str, start: int, keys: tuple[str, ...]) -> int | None:
    """Earliest sibling-key boundary after ``start`` (comma or newline form)."""
    found: list[int] = []
    for key in keys:
        m = re.search(rf'"\s*,\s*"{re.escape(key)}"\s*:', text[start:])
        if m:
            found.append(start + m.start())
        m = re.search(rf'\n\s*"{re.escape(key)}"\s*:', text[start:])
        if m:
            found.append(start + m.start())
    if not found:
        return None
    return min(found)


def _read_jsonish_value(text: str, start: int, end: int | None) -> str:
    """Slice a possibly unquoted/truncated string value between key boundaries."""
    chunk = text[start : end if end is not None else len(text)].strip()
    if chunk.startswith('"'):
        chunk = chunk[1:]
    chunk = chunk.rstrip()
    if chunk.endswith(","):
        chunk = chunk[:-1].rstrip()
    if chunk.endswith('"'):
        chunk = chunk[:-1]
    chunk = chunk.rstrip()
    if chunk.endswith(","):
        chunk = chunk[:-1].rstrip()
    return _unescape_json_string(chunk)


def _read_short_token(text: str, start: int) -> str | None:
    """Read a short unquoted or quoted token such as language ``zh`` / ``en``."""
    m = re.match(r'\s*"([A-Za-z_-]{1,16})"', text[start:])
    if m:
        return m.group(1)
    m = re.match(r"\s*'([A-Za-z_-]{1,16})'", text[start:])
    if m:
        return m.group(1)
    m = re.match(r"\s*([A-Za-z_-]{1,16})", text[start:])
    if m:
        return m.group(1)
    return None


def _scan_fields(text: str) -> dict[str, Any] | None:
    """Pull schema fields without treating inner quotes as JSON string ends."""
    out: dict[str, Any] = {}
    for key in _KEYS:
        start = _field_start(text, key)
        if start is None:
            continue
        if key == "language":
            token = _read_short_token(text, start)
            if token:
                out[key] = token
                continue
        others = tuple(k for k in _KEYS if k != key)
        end = _next_key_index(text, start, others)
        if end is None:
            close = re.search(r'"\s*}', text[start:])
            end = start + close.start() if close else None
        out[key] = _read_jsonish_value(text, start, end)
    if out.get("question"):
        return out
    return None


def _loose_extract(text: str) -> dict[str, Any] | None:
    """Last-resort regex extraction of question/tests_claim/language fields."""
    scanned = _scan_fields(text)
    if scanned:
        return scanned
    keys = _KEYS
    out: dict[str, Any] = {}
    for key in keys:
        m = re.search(rf'"{key}"\s*:\s*"', text)
        if not m:
            continue
        rest = text[m.end() :]
        ends: list[int] = []
        for other in keys:
            if other == key:
                continue
            dm = re.search(rf'"\s*,\s*"{other}"\s*:', rest)
            if dm:
                ends.append(dm.start())
        dm = re.search(r'"\s*}', rest)
        if dm:
            ends.append(dm.start())
        if ends:
            value = rest[: min(ends)]
        else:
            value = rest.rstrip().rstrip("}").rstrip().rstrip('"')
        out[key] = _unescape_json_string(value.rstrip())
    if out.get("question"):
        return out
    return None


def _json_fence_body(raw: str) -> str | None:
    """Return the body of a ``` / ```json fence that looks like an object.

    Inner ```cuda / ```cpp fences belong to the question string and must not
    replace the outer JSON payload.
    """
    for match in _FENCE.finditer(raw or ""):
        lang = (match.group(1) or "").strip().lower()
        body = (match.group(2) or "").strip()
        if lang in {"", "json"} and body.startswith("{"):
            return body
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output (prefers the last valid object)."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    raw = strip_think(text)
    candidates = [raw]
    fenced = _json_fence_body(raw)
    if fenced and fenced not in candidates:
        candidates.append(fenced)
    last_obj: dict[str, Any] | None = None
    for cand in candidates:
        placeholders, blocks = _extract_code_fences(cand)
        obj = (
            _loads_obj(placeholders)
            or _scan_objects(placeholders)
            or _loads_obj(cand)
            or _scan_objects(cand)
        )
        if obj is None:
            continue
        obj = _restore_code_fences(obj, blocks)
        if str(obj.get("question") or "").strip():
            return obj
        last_obj = obj
    if last_obj is not None:
        return last_obj
    raise ValueError(f"could not parse JSON object from: {text[:400]}")


def parse_question_payload(
    text: str,
    *,
    fallback_claim: str | None = None,
    min_question_chars: int = 24,
    keep_truncated: bool = True,
) -> dict[str, Any] | None:
    """Parse a writer payload, preferring the longest recovered question text.

    Args:
        text: Raw model output.
        fallback_claim: Brief claim used when ``tests_claim`` was truncated.
        min_question_chars: Minimum length before a truncated payload is kept.
        keep_truncated: If True, accept a question that lost its claim field.

    Returns:
        Normalized dict with ``question``, ``tests_claim``, and ``language``,
        or None if no question text could be recovered.
    """
    if not text or not text.strip():
        return None
    raw = strip_think(text)
    placeholders, blocks = _extract_code_fences(raw)
    obj = None
    try:
        obj = parse_json_object(text)
    except ValueError:
        obj = None
    scanned = (
        _scan_fields(placeholders)
        or _loose_extract(placeholders)
        or _scan_fields(raw)
        or _loose_extract(raw)
    )
    parsed_q = str((obj or {}).get("question") or "").strip()
    scanned_q = str((scanned or {}).get("question") or "").strip()
    if scanned_q and (not parsed_q or len(scanned_q) > len(parsed_q)):
        obj = scanned
    if obj is None:
        return None
    obj = _restore_code_fences(obj, blocks)
    question = str(obj.get("question") or "").strip()
    if not question:
        return None
    obj["question"] = question
    claim = str(obj.get("tests_claim") or "").strip()
    if not claim and fallback_claim and keep_truncated and len(question) >= min_question_chars:
        obj["tests_claim"] = fallback_claim
        obj.setdefault("warnings", [])
        if isinstance(obj["warnings"], list):
            obj["warnings"].append("json_truncated")
    obj["language"] = normalize_language(obj.get("language"), fallback="zh")
    return obj
