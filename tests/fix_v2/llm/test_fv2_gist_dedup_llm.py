"""Cross-language gist dedup with the REAL embedder (task T30). Budget: embedding calls only."""
import pytest

from fixv2_helpers import make_question

pytestmark = pytest.mark.llm


def test_translation_marked_dup(real_runtime):
    from cuda_sft_qgen.graph.nodes.review_score import review_score_node
    rt = real_runtime
    claim = "Warp stride-S access over 32 banks is gcd(S,32)-way conflicted."
    acc = make_question(id="en1", language="en", tests_claim=claim,
                        question="A warp reads __shared__ float s[tid*S] with 32 banks. What is the conflict degree for S=6?")
    new = make_question(id="zh1", language="zh", tests_claim=claim,
                        question="一个 warp 读取 __shared__ float s[tid*S]，共 32 个 bank。S=6 时冲突路数是多少？")
    rt.store.append_accepted(acc)
    rt.index.add([acc.id], rt.embedder.embed([acc.question]))
    rt.gist_index.add([acc.id], rt.embedder.embed([acc.tests_claim]))
    rt.deduper.add_question(acc)
    out = review_score_node({"pending": [new.model_dump()]})
    item = out["last_diversity"]["scored"][0]
    assert item["decision"]["verdict"] == "dup" and item["decision"]["reason"] == "L2_gist_crosslingual"
    assert any(n["id"] == "en1" for n in item["neighbors"])


def test_same_language_not_gist_dup(real_runtime):
    from cuda_sft_qgen.graph.nodes.review_score import review_score_node
    rt = real_runtime
    claim = "cp.async.cg supports only a 16-byte copy size on sm_80."
    a = make_question(id="en_a", language="en", tests_claim=claim, question="On sm_80 which cp.async variants accept an 8-byte size? List them.")
    b = make_question(id="en_b", language="en", tests_claim=claim, question="Write the PTX for a legal 16-byte cp.async.cg copy on sm_80 and explain why.")
    rt.store.append_accepted(a)
    rt.gist_index.add([a.id], rt.embedder.embed([a.tests_claim]))
    out = review_score_node({"pending": [b.model_dump()]})
    assert out["last_diversity"]["scored"][0]["decision"]["reason"] != "L2_gist_crosslingual"
