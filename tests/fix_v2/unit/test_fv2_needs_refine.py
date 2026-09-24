"""needs_refine heuristic (task T37) on real taxonomy text patterns. Pure; no model call."""
from types import SimpleNamespace


def _leaf(claim, hints=("h",)):
    return SimpleNamespace(claim=claim, hints=tuple(hints))


def test_title_like_and_hintless():
    from cuda_sft_qgen.knowledge_worker import needs_refine
    assert needs_refine(_leaf("TMA descriptor 的 box size、stride、OOB fill 与 128B 对齐要求"))
    assert needs_refine(_leaf("cp.async.cg supports only 16-byte copies on sm_80", hints=()))
    assert not needs_refine(_leaf("cp.async.cg 在 sm_80 上只支持 16 字节拷贝，8 字节会导致 ptxas 报错"))
    assert not needs_refine(_leaf("A 32-way conflict occurs when lanes read tile[tid][k]; padding to 33 columns is conflict-free"))


def test_real_taxonomy_majority_flagged(settings):
    from cuda_sft_qgen.knowledge_worker import needs_refine
    from cuda_sft_qgen.taxonomy import load_taxonomy
    leaves = load_taxonomy(settings.taxonomy_path).leaves
    share = sum(needs_refine(lf) for lf in leaves) / len(leaves)
    assert share > 0.5          # 98.6% have no hints at T00 time
