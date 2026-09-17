"""In-process prompt prefix cache and embedding vector cache (optional npz)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cuda_sft_qgen.review.normalize import exact_hash

logger = logging.getLogger(__name__)


class PromptCache:
    """Stable system-prefix assembly. Hits are local; API TTFT needs byte-identical prefix."""

    def __init__(self) -> None:
        """Create empty static/semi-static maps and hit counters."""
        self._static: dict[str, str] = {}
        self._semi: dict[tuple[str, ...], str] = {}
        self._lock = threading.Lock()
        self.static_hits = 0
        self.semi_hits = 0
        self.static_misses = 0
        self.semi_misses = 0

    def get_static(self, role: str, builder: Callable[[], str]) -> str:
        """Return a role-level system prefix, building it once per process."""
        with self._lock:
            cached = self._static.get(role)
            if cached is not None:
                self.static_hits += 1
                return cached
            self.static_misses += 1
        text = builder()
        with self._lock:
            self._static[role] = text
        return text

    def get_semi(self, key: tuple[str, ...], builder: Callable[[], str]) -> str:
        """Return a semi-static prefix keyed by taxonomy/persona tuple."""
        with self._lock:
            cached = self._semi.get(key)
            if cached is not None:
                self.semi_hits += 1
                return cached
            self.semi_misses += 1
        text = builder()
        with self._lock:
            self._semi[key] = text
        return text

    @staticmethod
    def compose(*parts: str) -> str:
        """Join nonempty prompt parts with blank lines (byte-stable for cache)."""
        return "\n\n".join(p.strip() for p in parts if p and p.strip())

    def stats(self) -> dict[str, int]:
        """Hit/miss counters for static and semi-static prefixes."""
        with self._lock:
            return {
                "static_hits": self.static_hits,
                "static_misses": self.static_misses,
                "semi_hits": self.semi_hits,
                "semi_misses": self.semi_misses,
            }


class EmbeddingCache:
    """In-process + optional npz cache keyed by embedding_model + exact_hash."""

    def __init__(self, model: str, path: Path | None = None) -> None:
        """Load an optional npz cache scoped to ``model``."""
        self.model = model
        self.path = path
        self._mem: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._dirty = 0
        if path is not None and path.exists():
            self._load(path)

    def _key(self, digest: str) -> str:
        """Namespace a text digest by embedding model so caches do not mix."""
        return f"{self.model}:{digest}"

    def hash_text(self, text: str) -> str:
        """SHA-256 of NFKC-normalized text; same key as exact-hash dedup."""
        return exact_hash(text)

    def get(self, digest: str) -> np.ndarray | None:
        """Return a copy of a cached vector, or None on miss."""
        key = self._key(digest)
        with self._lock:
            vec = self._mem.get(key)
            if vec is None:
                self.misses += 1
                return None
            self.hits += 1
            return vec.copy()

    def get_text(self, text: str) -> np.ndarray | None:
        """Look up by raw text rather than a precomputed digest."""
        return self.get(self.hash_text(text))

    def put(self, digest: str, vec: np.ndarray) -> None:
        """Store a float32 row vector; marks the cache dirty if the key is new."""
        key = self._key(digest)
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        with self._lock:
            if key not in self._mem:
                self._dirty += 1
            self._mem[key] = arr

    def put_text(self, text: str, vec: np.ndarray) -> None:
        """Store a vector under the exact-hash of ``text``."""
        self.put(self.hash_text(text), vec)

    def put_many(self, digests: Sequence[str], mat: np.ndarray) -> None:
        """Bulk-insert rows aligned with ``digests``."""
        rows = np.asarray(mat, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        for digest, row in zip(digests, rows):
            self.put(digest, row)

    def lookup_texts(self, texts: Sequence[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Return per-text hits and the indexes that still need an API call."""
        out: list[np.ndarray | None] = []
        misses: list[int] = []
        for i, text in enumerate(texts):
            vec = self.get_text(text)
            out.append(vec)
            if vec is None:
                misses.append(i)
        return out, misses

    def warm(self, digest: str, vec: np.ndarray) -> None:
        """Seed the cache from an already-indexed accepted question."""
        self.put(digest, vec)

    def stats(self) -> dict[str, int]:
        """Size and hit/miss counters."""
        with self._lock:
            return {
                "size": len(self._mem),
                "hits": self.hits,
                "misses": self.misses,
            }

    def save(self, path: Path | None = None) -> None:
        """Persist the in-memory map to compressed npz (no-op if empty/unpathed)."""
        dest = path or self.path
        if dest is None:
            return
        with self._lock:
            if not self._mem:
                return
            keys = np.array(list(self._mem.keys()), dtype=object)
            mat = np.stack(list(self._mem.values()), axis=0)
            dirty = self._dirty
            self._dirty = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, keys=keys, mat=mat, model=np.array(self.model))
        logger.debug("embedding cache saved %s entries dirty=%s -> %s", len(keys), dirty, dest)

    def _load(self, path: Path) -> None:
        """Load npz entries that belong to this embedding model; skip corrupt files."""
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding cache load failed: %s", exc)
            return
        model = str(data["model"]) if "model" in data.files else ""
        keys = [str(x) for x in data["keys"].tolist()]
        mat = np.asarray(data["mat"], dtype=np.float32)
        if mat.ndim == 1:
            if not keys:
                return
            mat = mat.reshape(1, -1)
        with self._lock:
            for key, row in zip(keys, mat):
                if model and not key.startswith(f"{self.model}:"):
                    continue
                if model and model != self.model and not key.startswith(f"{self.model}:"):
                    continue
                self._mem[key] = np.asarray(row, dtype=np.float32).reshape(-1)
        logger.info("embedding cache loaded %s vectors from %s", len(self._mem), path)
