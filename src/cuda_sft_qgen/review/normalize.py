"""Text normalization used by exact-hash dedup and embedding cache keys."""

from __future__ import annotations

import hashlib
import re
import unicodedata


_WS = re.compile(r"\s+")
_SPACE_PUNCT = re.compile(r"\s+([?!,.;:，。？！])")


def normalize_text(text: str) -> str:
    """NFKC, collapse whitespace, lowercase, and glue punctuation to tokens."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    text = _WS.sub(" ", text).strip().lower()
    text = _SPACE_PUNCT.sub(r"\1", text)
    return text


def exact_hash(text: str) -> str:
    """SHA-256 hex digest of :func:`normalize_text` (L0 exact-duplicate key)."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
