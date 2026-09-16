from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from cuda_sft_qgen.cache import EmbeddingCache, PromptCache
from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.embeddings import Embedder, build_embedder
from cuda_sft_qgen.keys import ApiKeyPool, build_key_pool
from cuda_sft_qgen.llm import LLMClient, build_llm
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.index import VectorIndex
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    store: Archive
    llm: LLMClient
    embedder: Embedder
    index: VectorIndex
    deduper: Deduper
    taxonomy: Taxonomy
    pool: ApiKeyPool | None = None
    prompt_cache: PromptCache | None = None
    embedding_cache: EmbeddingCache | None = None
    prompt_builder: StructuredPrompt | None = None
    _embed_pool: ThreadPoolExecutor | None = None
    _prefetch_futs: list[Future] = field(default_factory=list)
    _prefetch_lock: threading.Lock = field(default_factory=threading.Lock)

    def prefetch_embed(self, text: str) -> None:
        if not text or self._embed_pool is None:
            return

        def _run() -> None:
            try:
                self.embedder.embed([text])
            except Exception as exc:  # noqa: BLE001
                logger.warning("embed prefetch failed: %s", exc)

        fut = self._embed_pool.submit(_run)
        with self._prefetch_lock:
            self._prefetch_futs.append(fut)

    def drain_prefetch(self) -> None:
        with self._prefetch_lock:
            futs = list(self._prefetch_futs)
            self._prefetch_futs.clear()
        for fut in futs:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("embed prefetch join failed: %s", exc)

    def close(self) -> None:
        self.drain_prefetch()
        pool = self._embed_pool
        self._embed_pool = None
        if pool is not None:
            pool.shutdown(wait=False)
        cache = self.embedding_cache
        if cache is not None:
            try:
                cache.save()
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding cache save failed: %s", exc)


_RUNTIME: ContextVar[Runtime | None] = ContextVar("cuda_sft_qgen_runtime", default=None)
_RUNTIME_FALLBACK: Runtime | None = None


def get_runtime() -> Runtime:
    rt = _RUNTIME.get() or _RUNTIME_FALLBACK
    if rt is None:
        raise RuntimeError("runtime is not initialized")
    return rt


def set_runtime(runtime: Runtime) -> Token:
    global _RUNTIME_FALLBACK
    _RUNTIME_FALLBACK = runtime
    return _RUNTIME.set(runtime)


def reset_runtime(token: Token) -> None:
    global _RUNTIME_FALLBACK
    _RUNTIME.reset(token)
    _RUNTIME_FALLBACK = _RUNTIME.get()


def build_runtime(settings: Settings) -> Runtime:
    store = Archive(settings.data_dir)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    pool: ApiKeyPool | None = None
    if not settings.fake_llm and settings.nvidia_api_keys:
        pool = build_key_pool(
            settings.nvidia_api_keys,
            base_url=settings.nvidia_base_url,
            llm_per_key=settings.llm_per_key_concurrency,
            embed_per_key=settings.embed_per_key_concurrency,
            embed_oversub=settings.embed_oversub,
        )
    llm = build_llm(settings, pool)
    embedder = build_embedder(settings, pool)
    index_path = settings.data_dir / "embeddings.npz"
    index = VectorIndex.load(index_path)
    prompt_cache = PromptCache()
    embedding_cache: EmbeddingCache | None = None
    if settings.embedding_cache_enabled:
        embedding_cache = EmbeddingCache(
            settings.embedding_model,
            settings.data_dir / "embed_cache.npz",
        )
        embedder.cache = embedding_cache
        accepted = store.load_accepted()
        for q in accepted:
            if not q.exact_hash:
                continue
            vec = index.vector_for(q.id)
            if vec is not None:
                embedding_cache.warm(q.exact_hash, vec)
    prompt_builder = StructuredPrompt(settings, cache=prompt_cache)
    deduper = Deduper(settings)
    for q in store.load_accepted():
        deduper.add_question(q)
    workers = pool.embed_capacity if pool is not None else max(1, settings.embed_chunk_size)
    embed_pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="embed-pre")
    return Runtime(
        settings=settings,
        store=store,
        llm=llm,
        embedder=embedder,
        index=index,
        deduper=deduper,
        taxonomy=taxonomy,
        pool=pool,
        prompt_cache=prompt_cache,
        embedding_cache=embedding_cache,
        prompt_builder=prompt_builder,
        _embed_pool=embed_pool,
    )
