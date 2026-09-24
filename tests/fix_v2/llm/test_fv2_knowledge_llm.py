"""Real-LLM knowledge paths (tasks T33, T35, T37, T34). Budget ≈ 10 requests (+2 if mirrored docs exist)."""
import json

import numpy as np
import pytest

pytestmark = [pytest.mark.llm, pytest.mark.slow]

FORMULA_CLAIMS = [
    ("T03", "occupancy_math", "occupancy_math_fv2_regs",
     "On sm_80 with 36 registers per thread, registers per warp round up to 1280, so 65536 // 1280 = 51 warps fit."),
    ("T25", "smem_bank", "smem_bank_fv2_gcd",
     "A warp reading 4-byte words at stride 12 over 32 banks is gcd(12,32) = 4-way conflicted."),
    ("T25", "roofline", "roofline_fv2_saxpy",
     "FP32 SAXPY moves 12 bytes per 2 FLOPs, so at 3.0e12 B/s it attains 3.0e12 * 2 / 12 = 5.0e11 FLOP/s."),
]


def _worker(rt, tmp_path):
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    ks = KnowledgeStore(tmp_path / "kv.sqlite3")
    return ks, KnowledgeWorker(ks, rt.llm, embedder=rt.embedder, settings=rt.settings, taxonomy=rt.taxonomy)


def test_compute_verify_real(real_runtime, tmp_path, eval_out):
    ks, w = _worker(real_runtime, tmp_path)
    for l1, l2, cid, claim in FORMULA_CLAIMS:
        assert ks.add_claim({"l3_id": cid, "claim": claim, "domain_id": l1, "l2_id": l2, "kind": "formula"})
        w._compute_verify({"claim_id": cid, "revision": 1})
    status = {c[2]: ks.get_claim(c[2])["status"] for c in FORMULA_CLAIMS}
    (eval_out / "compute_verify.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    assert sum(v == "verified" for v in status.values()) >= 2, status


def test_relation_candidates_real_embedding(real_runtime, tmp_path):
    ks, _ = _worker(real_runtime, tmp_path)
    for cid, claim, l2 in (("tma_fv2_root", "TMA global address must be 16-byte aligned on sm_90", "tma"),
                           ("tma_fv2_near", "TMA box inner dimension in bytes must be a multiple of 16", "tma"),
                           ("nccl_fv2_far", "NCCL busbw equals algbw times 2(n-1)/n for all-reduce", "nccl")):
        ks.add_claim({"l3_id": cid, "claim": claim, "domain_id": "T08", "l2_id": l2, "kind": "feature"})
    cands = ks.relation_candidates("tma_fv2_root", limit=2, embedder=real_runtime.embedder)
    assert cands[0]["id"] == "tma_fv2_near"


def test_refine_legacy_real(real_runtime, tmp_path, eval_out):
    """T37: one real refine call on 10 title-like legacy leaves of the real taxonomy (dry run, no apply)."""
    ks, w = _worker(real_runtime, tmp_path)
    ks.import_taxonomy(real_runtime.taxonomy)
    result = w.refine_legacy_batch("T08.tma", limit=10, apply=False)
    (eval_out / "refine_legacy.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    assert result["selected"] >= 3
    assert result["valid_suggestions"] >= max(1, int(0.7 * result["selected"])), result


def test_audit_with_local_docs_real(real_runtime, tmp_path, eval_out):
    docs = real_runtime.settings.root / str(real_runtime.settings.knowledge_cfg.get("docs_dir") or "data/docs")
    if not docs.is_dir() or not any(docs.glob("*.meta.json")):
        pytest.fail(f"no mirrored docs in {docs}: run `python scripts/mirror_docs.py` first (task T34 step 6)")
    ks, w = _worker(real_runtime, tmp_path)
    ks.add_claim({"l3_id": "tma_fv2_audit", "claim": "cuTensorMapEncodeTiled requires each boxDim entry to be at most 256",
                  "domain_id": "T08", "l2_id": "tma", "kind": "feature"})
    w._audit({"claim_id": "tma_fv2_audit", "revision": 1})
    with ks.connect() as db:
        audits = db.execute("SELECT verdict, reason FROM audits WHERE claim_id='tma_fv2_audit'").fetchall()
    (eval_out / "audit_local_docs.json").write_text(json.dumps([list(a) for a in audits]), encoding="utf-8")
    assert audits, "audit produced no record"
