"""Knowledge store v2 (tasks T33, T34, T35; invariant I11)."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from fixv2_helpers import tiny_taxonomy


def _store(tmp_path):
    from cuda_sft_qgen.store.knowledge import KnowledgeStore
    return KnowledgeStore(tmp_path / "k.sqlite3")


def _add(ks, cid, claim, kind="formula", l2="la", status="provisional"):
    assert ks.add_claim({"l3_id": cid, "claim": claim, "domain_id": "T07", "l2_id": l2, "kind": kind}, status=status)


def test_migration_idempotent(tmp_path):
    _store(tmp_path); _store(tmp_path)
    db = sqlite3.connect(tmp_path / "k.sqlite3")
    cols = lambda t: {r[1] for r in db.execute(f"PRAGMA table_info({t})")}
    assert {"verified_by", "audit_due_at"} <= cols("claims") and {"kind", "toolkit_version"} <= cols("evidence")


def test_computation_verifies_formula(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_off", "(4,8):(8,1) at index 13 has offset = 11")
    assert ks.apply_computation_check("la_off", 1, expression="(13 % 4)*8 + (13 // 4)*1", variables={}, expected="11", trace="t")
    c = ks.get_claim("la_off")
    assert c["status"] == "verified" and c["verified_by"] == "computation"


def test_computation_variables_and_mismatch(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_regs", "regs per warp = regs*32 = 1152 for 36 regs")
    assert ks.apply_computation_check("la_regs", 1, expression="regs*32", variables={"regs": 36}, expected="1152", trace="t")
    assert ks.get_claim("la_regs")["status"] == "verified"
    _add(ks, "la_bad", "offset = 12 for index 13 in (4,8):(8,1)")
    assert ks.apply_computation_check("la_bad", 1, expression="(13 % 4)*8 + 13 // 4", variables={}, expected="12", trace="t")
    assert ks.get_claim("la_bad")["status"] == "provisional"


def test_computation_rejects_non_formula_and_bad_names(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_api", "cp.async.cg supports only 16 bytes on sm_80", kind="api")
    with pytest.raises(ValueError):
        ks.apply_computation_check("la_api", 1, expression="1", variables={}, expected="1", trace="t")
    _add(ks, "la_x", "x = 2 for the given input")
    with pytest.raises(ValueError):
        ks.apply_computation_check("la_x", 1, expression="__import__('os')", variables={}, expected="2", trace="t")


def test_expert_and_execution(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_e", "ldmatrix x4 gives 4 regs per thread on sm_80", kind="feature")
    assert ks.apply_expert_verification("la_e", 1, reviewer="alice", note="checked PTX ISA")
    assert ks.get_claim("la_e")["verified_by"] == "expert"
    with pytest.raises(ValueError):
        ks.apply_expert_verification("la_e", 1, reviewer="", note="x")
    _add(ks, "la_r", "racecheck reports shared memory hazards only", kind="failure")
    assert ks.apply_execution_check("la_r", 1, log="0 errors", passed=False)
    assert ks.get_claim("la_r")["status"] == "provisional"


def test_whitespace_normalized_quote(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_q", "The TMA global address must be aligned to 16 bytes on sm_90", kind="feature")
    ok = ks.apply_audit("la_q", 1, verdict="supported", url="https://docs.nvidia.com/cuda/x.html",
                        quote="must be\n  aligned to   16 bytes", document="The global address must be aligned to 16 bytes.")
    assert ok and ks.get_claim("la_q")["verified_by"] == "quote"


def test_import_conflicts_recorded(tmp_path):
    from cuda_sft_qgen.taxonomy import Taxonomy, _leaves_from_domains
    doms = [{"id": "T02", "name": "m", "children": [
        {"id": "a", "name": "a", "leaves": [{"id": "dup_leaf", "claim": "claim one about banks and strides", "kind": "api"}]},
        {"id": "b", "name": "b", "leaves": [{"id": "dup_leaf", "claim": "claim two about sectors and loads", "kind": "api"}]}]}]
    ks = _store(tmp_path)
    ks.import_taxonomy(Taxonomy(1, "t", _leaves_from_domains(doms), doms))
    conflicts = ks.get_metadata("import_conflicts")
    assert len(conflicts) == 1 and conflicts[0]["l3_id"] == "dup_leaf"


def test_overlay_contains_all(tmp_path):
    ks = _store(tmp_path)
    tax = tiny_taxonomy()
    ks.import_taxonomy(tax)
    ks.add_claim({"l3_id": "smem_new", "claim": "new smem claim about swizzle = xor", "domain_id": "T02", "l2_id": "smem", "kind": "formula"})
    out = ks.overlay_taxonomy(tax)
    assert {lf.l3_id for lf in out.leaves} == {lf.l3_id for lf in tax.leaves} | {"smem_new"}


def test_relation_candidates_fallback_tokenizer(tmp_path):
    ks = _store(tmp_path)
    _add(ks, "la_root", "TMA 全局地址必须 16 字节对齐", kind="feature")
    _add(ks, "la_near", "TMA 的 box 内维字节数必须是 16 的倍数", kind="feature")
    _add(ks, "la_far", "NCCL busbw equals algbw times 2(n-1)/n", kind="formula", l2="nccl")
    cands = ks.relation_candidates("la_root", limit=2, embedder=None)
    assert cands[0]["id"] == "la_near"            # CJK-aware tokenizer (no model involved)


def test_retrieve_documents_prefers_doc_index(tmp_path):
    """Pure retrieval step of the audit: local DocIndex hits, network fetcher never called, no model call."""
    from cuda_sft_qgen.knowledge_worker import KnowledgeWorker
    docs = tmp_path / "docs"; docs.mkdir()
    (docs / "tma.txt").write_text("The global address must be aligned to 16 bytes. " * 30, encoding="utf-8")
    (docs / "tma.meta.json").write_text(json.dumps({"url": "https://docs.nvidia.com/cuda/tma.html", "title": "t",
                                                     "toolkit_version": "12.x", "sha256": "x"}), encoding="utf-8")
    ks = _store(tmp_path)
    _add(ks, "la_doc", "The TMA global address must be aligned to 16 bytes on sm_90", kind="feature")

    def no_fetch(url):
        raise AssertionError("network fetch must not be used when the doc index has hits")
    settings = SimpleNamespace(root=tmp_path, knowledge_cfg={"docs_dir": str(docs)})
    w = KnowledgeWorker(ks, None, settings=settings, official_fetcher=no_fetch)
    docs_found = w.retrieve_documents(ks.get_claim("la_doc"))
    assert docs_found and docs_found[0][0] == "https://docs.nvidia.com/cuda/tma.html"
    assert "aligned to 16 bytes" in docs_found[0][1]
