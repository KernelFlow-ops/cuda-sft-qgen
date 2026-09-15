from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.embeddings import Embedder, build_embedder
from cuda_sft_qgen.keys import ApiKeyPool, build_key_pool
from cuda_sft_qgen.llm import LLMClient, build_llm
from cuda_sft_qgen.review.dedup import Deduper
from cuda_sft_qgen.review.index import VectorIndex
from cuda_sft_qgen.store.archive import Archive
from cuda_sft_qgen.taxonomy import Taxonomy, load_taxonomy


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
        )
    llm = build_llm(settings, pool)
    embedder = build_embedder(settings, pool)
    index_path = settings.data_dir / "embeddings.npz"
    index = VectorIndex.load(index_path)
    deduper = Deduper(settings)
    for q in store.load_accepted():
        deduper.add_question(q)
    return Runtime(
        settings=settings,
        store=store,
        llm=llm,
        embedder=embedder,
        index=index,
        deduper=deduper,
        taxonomy=taxonomy,
        pool=pool,
    )
