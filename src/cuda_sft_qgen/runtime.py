"""Process-wide runtime: LLM, embedder, archive, taxonomy, and knowledge store.

Graph nodes read this via a :class:`contextvars.ContextVar` so worker threads
inherit the same objects. ``build_runtime`` wires caches, the vector index,
and the knowledge overlay.
"""

from __future__ import annotations

import logging
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace

import numpy as np

from cuda_sft_qgen.cache import EmbeddingCache, PromptCache
from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.embeddings import Embedder, build_embedder
from cuda_sft_qgen.keys import ApiKeyPool, build_key_pool
from cuda_sft_qgen.llm import LLMClient, build_llm
from cuda_sft_qgen.prompts.structured_writer import StructuredPrompt
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.index import VectorIndex
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.store.knowledge import KnowledgeStore
from cuda_sft_qgen.taxonomy import Taxonomy, load_taxonomy

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Shared services for one generate/review run (not process-global)."""

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
    review_llm: LLMClient | None = None
    knowledge: KnowledgeStore | None = None
    knowledge_worker: object | None = None
    _knowledge_refreshed_at: float = 0.0
    _verification_lock: threading.Lock = field(default_factory=threading.Lock)
    _verification_calls: int = 0

    def reserve_verification_call(self) -> bool:
        """Atomically enforce the run-wide solver/judge call ceiling.

        The normal key pool bounds concurrency, not cumulative spend. This
        separate counter includes retried reviews and is shared by all Chair
        workers; exhausting it defers candidates instead of bypassing gates.
        """
        limit = int(self.settings.review.get("verification_run_max_calls", 600))
        with self._verification_lock:
            if self._verification_calls >= max(0, limit):
                return False
            self._verification_calls += 1
            return True

    def reconcile_index(self) -> None:
        """Rebuild the cosine index and deduper from accepted questions."""
        accepted = self.store.load_accepted()
        ids = [q.id for q in accepted]
        if self.index.ids == ids:
            return
        rebuilt = VectorIndex()
        missing = [q for q in accepted if self.index.vector_for(q.id) is None]
        recovered = {}
        if missing:
            vecs = self.embedder.embed([q.question for q in missing])
            recovered = dict(zip((q.id for q in missing), vecs))
        for q in accepted:
            vec = self.index.vector_for(q.id)
            if vec is None:
                vec = recovered[q.id]
            rebuilt.add([q.id], np.asarray(vec).reshape(1, -1))
        self.index = rebuilt
        self.index.save(self.settings.data_dir / "embeddings.npz")
        self.deduper = Deduper(self.settings)
        for q in accepted:
            self.deduper.add_question(q)

    def refresh_knowledge(self, *, force: bool = False) -> None:
        """Overlay latest claims onto taxonomy and invalidate stale questions.

        Skips if called more than once per second unless ``force`` is True.
        Also catches audits that committed between a chair check and archive.
        """
        if self.knowledge is None:
            return
        if not force and time.monotonic() - self._knowledge_refreshed_at < 1:
            return
        self._knowledge_refreshed_at = time.monotonic()
        self.taxonomy = self.knowledge.overlay_taxonomy(self.taxonomy)
        cursor = int(self.store._get_meta("knowledge_cursor", 0))
        for event in self.knowledge.impact_events(cursor):
            if (event.get("new_revision") != event.get("old_revision")
                    or event.get("reason", "").startswith("audit:contradicted:")):
                self.store.invalidate_claim(
                    event["claim_id"], event.get("old_revision"), event.get("reason", "knowledge_changed"),
                    event["id"], question_ids=event.get("question_ids") or None,
                )
            cursor = int(event["id"])
        # Reconcile accepted dependencies as well as events: an audit may commit
        # between a reviewer's revision check and its question transaction.
        current_claims = {claim["id"]: claim for claim in self.knowledge.list_claims()}
        for q in self.store.load_accepted():
            dependencies = [{"claim_id": q.l3_id, "revision": getattr(q, "claim_revision", None)}]
            dependencies.extend(getattr(q, "knowledge_dependencies", []))
            for dependency in dependencies:
                claim = current_claims.get(dependency["claim_id"])
                revision = dependency.get("revision")
                if claim and (claim["status"] in {"quarantined", "retired"} or
                              (revision is not None and int(revision) != claim["revision"])):
                    self.store.invalidate_claim(dependency["claim_id"], revision,
                                                "stale_dependency", question_ids=[q.id])
                    break
        with self.store._lock, self.store._db:
            self.store._set_meta("knowledge_cursor", cursor)

    def prefetch_embed(self, text: str) -> None:
        """Queue an embedding call so review can hit cache after generate."""
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
        """Wait for outstanding prefetch futures (best-effort)."""
        with self._prefetch_lock:
            futs = list(self._prefetch_futs)
            self._prefetch_futs.clear()
        for fut in futs:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("embed prefetch join failed: %s", exc)

    def close(self) -> None:
        """Stop the knowledge worker, persist caches/metrics, close SQLite."""
        worker = self.knowledge_worker
        if worker is not None:
            worker.stop()
            self.knowledge_worker = None
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
        from cuda_sft_qgen.observability import get_metrics
        (self.settings.data_dir / "metrics.json").write_text(
            json.dumps(get_metrics().snapshot(), indent=2), encoding="utf-8")
        try:
            self.store.close()
        except Exception as exc:
            logger.warning("archive close failed: %s", exc)


_RUNTIME: ContextVar[Runtime | None] = ContextVar("cuda_sft_qgen_runtime", default=None)


def get_runtime() -> Runtime:
    """Return the runtime bound to the current context.

    Raises:
        RuntimeError: If :func:`set_runtime` has not been called.
    """
    rt = _RUNTIME.get()
    if rt is None:
        raise RuntimeError("runtime is not initialized")
    return rt


def set_runtime(runtime: Runtime) -> Token:
    """Bind ``runtime`` to this context; return a token for :func:`reset_runtime`."""
    return _RUNTIME.set(runtime)


def reset_runtime(token: Token) -> None:
    """Restore the previous runtime binding from ``token``."""
    _RUNTIME.reset(token)


def build_runtime(settings: Settings) -> Runtime:
    """Construct archive, knowledge store, LLM/embed clients, index, and caches.

    Does not start the knowledge worker; :func:`cuda_sft_qgen.graph.app.run_pipeline`
    does that after the data-dir lock is held.
    """
    store = Archive(settings.data_dir)
    taxonomy = load_taxonomy(settings.taxonomy_path)
    knowledge = KnowledgeStore(settings.data_dir / "knowledge.sqlite3")
    knowledge.import_taxonomy(taxonomy)
    taxonomy = knowledge.overlay_taxonomy(taxonomy)
    pool: ApiKeyPool | None = None
    if not settings.fake_llm and settings.nvidia_api_keys:
        pool = build_key_pool(
            settings.nvidia_api_keys,
            base_url=settings.nvidia_base_url,
            timeout=float(settings.generation.get("request_timeout_seconds", 90)),
            llm_per_key=settings.llm_per_key_concurrency,
            embed_per_key=settings.embed_per_key_concurrency,
            embed_oversub=settings.embed_oversub,
            llm_chair_reserve=int(settings.review.get("chair_slot_reserve", 1) or 0),
        )
    llm = build_llm(settings, pool)
    review_model = getattr(settings, "review_model", settings.nvidia_model)
    review_llm = (build_llm(replace(settings, nvidia_model=review_model), pool)
                  if review_model != settings.nvidia_model else None)
    embedder = build_embedder(settings, pool)
    index_path = settings.data_dir / "embeddings.npz"
    try:
        index = VectorIndex.load(index_path)
    except (ValueError, OSError, EOFError):
        logger.warning("Vector index is unreadable; it will be rebuilt from the archive")
        index = VectorIndex()
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
        review_llm=review_llm,
        knowledge=knowledge,
    )
