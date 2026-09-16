from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

import numpy as np

from cuda_sft_qgen.cache import EmbeddingCache
from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.keys import ApiKeyPool
from cuda_sft_qgen.review.normalize import exact_hash

logger = logging.getLogger(__name__)


def _split_even(items: list[str], n_parts: int) -> list[list[str]]:
    n_parts = max(1, min(len(items), n_parts))
    k, m = divmod(len(items), n_parts)
    batches: list[list[str]] = []
    i = 0
    for b in range(n_parts):
        size = k + (1 if b < m else 0)
        if size <= 0:
            continue
        batches.append(items[i : i + size])
        i += size
    return batches


def _chunks(items: list[str], size: int) -> list[list[str]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


class Embedder:
    dim: int | None = None
    cache: EmbeddingCache | None = None
    uncached_calls: int = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim or 1024), dtype=np.float32)
        cache = self.cache
        if cache is None:
            return self._embed_uncached(items)
        hits, miss_idx = cache.lookup_texts(items)
        if not miss_idx:
            stacked = np.stack([h for h in hits if h is not None], axis=0)
            self.dim = stacked.shape[1]
            return stacked
        miss_texts = [items[i] for i in miss_idx]
        mat = self._embed_uncached(miss_texts)
        digests = [exact_hash(t) for t in miss_texts]
        cache.put_many(digests, mat)
        out_dim = mat.shape[1]
        self.dim = out_dim
        rows: list[np.ndarray] = []
        miss_iter = 0
        for i, hit in enumerate(hits):
            if hit is None:
                rows.append(np.asarray(mat[miss_iter], dtype=np.float32))
                miss_iter += 1
            else:
                rows.append(hit)
        return np.stack(rows, axis=0)

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class NvidiaEmbedder(Embedder):
    def __init__(self, settings: Settings, pool: ApiKeyPool) -> None:
        self.settings = settings
        self.pool = pool
        self.dim = None
        self.uncached_calls = 0

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim or 1024), dtype=np.float32)
        chunk_size = max(1, int(self.settings.embed_chunk_size))
        batches = _chunks(items, chunk_size)
        workers = min(len(batches), max(1, self.pool.embed_capacity))
        rows: list[np.ndarray | None] = [None] * len(batches)
        self.uncached_calls += len(batches)

        def _one(idx: int, batch: list[str]) -> None:
            rows[idx] = self._embed_batch_with_retry(batch)

        if workers <= 1 or len(batches) == 1:
            for i, batch in enumerate(batches):
                _one(i, batch)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_one, i, batch) for i, batch in enumerate(batches)]
                for fut in as_completed(futs):
                    fut.result()
        mat = np.concatenate([r for r in rows if r is not None], axis=0)
        self.dim = mat.shape[1]
        return mat

    def _embed_batch_with_retry(self, texts: list[str], retries: int = 8) -> np.ndarray:
        last: Exception | None = None
        for attempt in range(retries):
            slot = self.pool.acquire("embed")
            try:
                return self._embed_once(slot.client, texts)
            except Exception as exc:  # noqa: BLE001
                last = exc
                text = str(exc).lower()
                if "429" in str(exc) or "too many requests" in text or "overloaded" in text:
                    wait = min(60.0, 8.0 * (2 ** min(attempt, 3)))
                    self.pool.mark_cooldown(slot, wait, kind="embed")
                    logger.warning(
                        "embed %s rate-limited (attempt %s/%s)",
                        slot.label,
                        attempt + 1,
                        retries,
                    )
                    continue
                wait = min(30.0, float(2 ** attempt))
                logger.warning("embed failed (attempt %s): %s", attempt + 1, exc)
                time.sleep(wait)
            finally:
                self.pool.release("embed", slot)
        raise RuntimeError(f"embed failed: {last}")

    def _embed_once(self, client: Any, texts: list[str]) -> np.ndarray:
        kwargs = {
            "model": self.settings.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        try:
            resp = client.embeddings.create(
                **kwargs,
                extra_body={"input_type": "passage"},
            )
        except Exception:
            resp = client.embeddings.create(**kwargs)
        vectors = [item.embedding for item in resp.data]
        mat = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self.dim = mat.shape[1]
        self.pool.note_success("embed")
        return mat


class HuggingFaceEmbedder(Embedder):
    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'Install the HF extra: pip install -e ".[hf]"'
            ) from exc
        self._model = SentenceTransformer(settings.embedding_model)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        self.uncached_calls += 1
        if not texts:
            return np.zeros((0, self.dim or 1024), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(vecs, dtype=np.float32))


class FakeEmbedder(Embedder):
    """Hash-seeded vectors so identical strings have cosine 1."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.uncached_calls = 0

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        self.uncached_calls += 1
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (
                2**32
            )
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=(self.dim,)).astype(np.float32)
            rows.append(vec)
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack(rows, axis=0))


def build_embedder(settings: Settings, pool: ApiKeyPool | None = None) -> Embedder:
    if settings.fake_llm or settings.embedding_provider == "fake":
        return FakeEmbedder()
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbedder(settings)
    if pool is None:
        raise RuntimeError("NvidiaEmbedder requires an ApiKeyPool")
    return NvidiaEmbedder(settings, pool)
