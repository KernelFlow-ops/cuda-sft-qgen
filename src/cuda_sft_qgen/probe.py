"""Live concurrency probes against the configured API key pools.

Used by ``cuda-sft-qgen probe-concurrency`` to confirm that LLM and embedding
calls overlap across keys before a long generate run.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cuda_sft_qgen.config import load_settings
from cuda_sft_qgen.graph.app import pipeline_lock, run_pipeline
from cuda_sft_qgen.runtime import build_runtime, reset_runtime, set_runtime

logger = logging.getLogger(__name__)


def _run_parallel(n: int, fn) -> tuple[float, list]:
    """Run ``fn(i)`` for ``i in range(n)`` concurrently; return (seconds, results)."""
    t0 = time.monotonic()
    out: list = [None] * n
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(fn, i): i for i in range(n)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
    return time.monotonic() - t0, out


def probe_concurrency(data_dir: Path | None = None) -> dict:
    """Issue parallel LLM and embedding calls at pool capacity.

    Raises:
        RuntimeError: If no real API keys are configured (fake mode is refused).
    """
    settings = load_settings(data_dir=data_dir)
    missing = settings.missing_api_keys + settings.missing_api_fields
    if settings.fake_llm or missing:
        raise RuntimeError(f"Need real API configuration: {', '.join(missing)}")
    with pipeline_lock(settings.data_dir):
        return _probe_concurrency_locked(settings)


def _probe_concurrency_locked(settings) -> dict:
    rt = build_runtime(settings)
    token = set_runtime(rt)
    try:
        assert rt.pool is not None
        embed_pool = rt.embed_key_pool or rt.pool
        n_llm = rt.pool.llm_capacity
        n_emb = embed_pool.embed_capacity
        logger.info(
            "probe llm_keys=%s embed_keys=%s llm_capacity=%s embed_capacity=%s",
            rt.pool.n_keys,
            embed_pool.n_keys,
            n_llm,
            n_emb,
        )

        def llm_job(i: int) -> str:
            return rt.llm.complete(
                [{"role": "user", "content": f"Reply with only the integer {i}."}],
                thinking_level=settings.thinking_level,
                temperature=0.0,
                max_tokens=256,
            )

        def emb_job(i: int) -> int:
            vec = rt.embedder.embed([f"CUDA coalescing test sentence {i}."])
            return int(vec.shape[-1])

        llm_s = dict(rt.pool.stats())
        llm_dt, llm_out = _run_parallel(n_llm, llm_job)
        llm_peak = rt.pool.peak["llm"]
        llm_keys = list(rt.pool.acquire_counts["llm"])

        emb_dt, emb_out = _run_parallel(n_emb, emb_job)
        emb_peak = embed_pool.peak["embed"]
        emb_keys = list(embed_pool.acquire_counts["embed"])

        mix_n = min(n_llm, 4) + min(n_emb, 4)

        def mixed_job(i: int):
            if i % 2 == 0:
                return llm_job(i)
            return emb_job(i)

        mix_dt, _ = _run_parallel(max(2, mix_n), mixed_job)

        report = {
            "keys": rt.pool.n_keys,
            "embed_keys": embed_pool.n_keys,
            "llm_per_key": rt.pool.llm_per_key,
            "embed_per_key": embed_pool.embed_per_key,
            "llm_capacity": n_llm,
            "embed_capacity": n_emb,
            "llm_parallel_sec": round(llm_dt, 2),
            "embed_parallel_sec": round(emb_dt, 2),
            "peak_llm_inflight": llm_peak,
            "peak_embed_inflight": emb_peak,
            "llm_acquires_per_key": llm_keys,
            "embed_acquires_per_key": emb_keys,
            "llm_ok": sum(1 for x in llm_out if x),
            "embed_ok": sum(1 for x in emb_out if x),
            "llm_overlap": llm_peak >= 2,
            "embed_overlap": emb_peak >= 2,
            "mixed_parallel_sec": round(mix_dt, 2),
            "overlap_samples": rt.pool.overlap_samples if embed_pool is rt.pool else None,
            "llm_embed_overlap": rt.pool.overlap_samples > 0 if embed_pool is rt.pool else None,
            "used_multiple_keys_llm": sum(1 for c in llm_keys if c > 0) >= min(2, rt.pool.n_keys),
            "stats_before_llm": llm_s,
            "prompt_cache": None if rt.prompt_cache is None else rt.prompt_cache.stats(),
            "embedding_cache": None if rt.embedding_cache is None else rt.embedding_cache.stats(),
        }
        return report
    finally:
        reset_runtime(token)
        closer = getattr(rt, "close", None)
        if callable(closer):
            closer()


def probe_pipeline(tmp: Path, *, target: int = 6, batch: int = 6) -> dict:
    """Run a tiny generate pipeline and report accepted counts plus pool overlap."""
    settings = load_settings(target=target, review_batch_size=batch, data_dir=tmp)
    with pipeline_lock(tmp):
        rt = build_runtime(settings)
        try:
            result = run_pipeline(settings, runtime=rt, lock_held=True)
            stats = None if rt.pool is None else rt.pool.stats()
            embed_stats = None if rt.embed_key_pool is None else rt.embed_key_pool.stats()
            return {
                "accepted": result.get("accepted_count"),
                "rejected": result.get("rejected_count"),
                "pending": len(result.get("pending") or []),
                "done": result.get("done"),
                "status": result.get("status"),
                "pool": stats,
                "embed_pool": embed_stats,
                "generate_overlap": bool(stats and stats["peak_llm"] >= 2),
                "review_embed_overlap": bool(embed_stats and embed_stats["peak_embed"] >= 2),
            }
        finally:
            rt.close()
