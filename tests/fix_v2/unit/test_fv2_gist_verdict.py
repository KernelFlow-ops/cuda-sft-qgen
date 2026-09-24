"""gist_verdict pure rules (task T30). Real-embedding behaviour: llm/test_fv2_gist_dedup_llm.py."""


def test_gist_verdict_rules():
    from cuda_sft_qgen.models import DupDecision
    from cuda_sft_qgen.review.dedup import gist_verdict
    uniq = DupDecision(verdict="unique")
    assert gist_verdict(uniq, None, dup=0.92, gray=0.86, nearest_id=None) == uniq
    d = gist_verdict(uniq, 0.95, dup=0.92, gray=0.86, nearest_id="a")
    assert (d.verdict, d.reason, d.nearest_id) == ("dup", "L2_gist_crosslingual", "a")
    g = gist_verdict(uniq, 0.9, dup=0.92, gray=0.86, nearest_id="a")
    assert (g.verdict, g.reason) == ("gray", "gist_gray")
    dup = DupDecision(verdict="dup", reason="L0_exact")
    assert gist_verdict(dup, 0.99, dup=0.92, gray=0.86, nearest_id="a").reason == "L0_exact"
    gray = DupDecision(verdict="gray", reason="gray_zone")
    assert gist_verdict(gray, 0.9, dup=0.92, gray=0.86, nearest_id="a").reason == "gray_zone"


def test_runtime_has_gist_index_field():
    import dataclasses
    from cuda_sft_qgen.runtime import Runtime
    assert "gist_index" in {f.name for f in dataclasses.fields(Runtime)}
