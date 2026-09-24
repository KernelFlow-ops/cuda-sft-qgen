"""MMR curation (task T23; 01_ARCHITECTURE §7.2; invariant I10)."""
from types import SimpleNamespace

import numpy as np

from fixv2_helpers import make_question

SETTINGS = SimpleNamespace(review={"rip_enabled": True, "rip_keep_per_bucket": 3, "rip_min": 0.34, "mmr_lambda": 0.5})


def _items(rips, scores=None, same_bucket=True):
    scored, results = [], {}
    for i, rip in enumerate(rips):
        q = make_question(id=f"q{i}", topic_path=["T02", "smem", f"l{i}"] if same_bucket else [f"T0{i}", "x", "y"])
        scored.append({"question": q.model_dump(), "vector_row": i})
        vs = (scores or {}).get(i)
        results[q.id] = {"action": "accept", "review": {"verification_evidence": {"rip_score": rip},
                                                          "value_scores": vs}}
    return scored, results


def test_keeps_three_prefers_harder_on_ties():
    from cuda_sft_qgen.graph.nodes.review_chair import _curate_select
    scored, results = _items([1.0, 0.9, 0.6, 0.5])
    not_sel, below = _curate_select(scored, results, SETTINGS, None)
    assert below == set() and not_sel == {"q0"}


def test_rip_below_min_is_pending_not_rejected():
    from cuda_sft_qgen.graph.nodes.review_chair import _curate_select
    scored, results = _items([0.2, 0.9])
    not_sel, below = _curate_select(scored, results, SETTINGS, None)
    assert below == {"q0"} and not_sel == set()


def test_mmr_penalizes_near_duplicates():
    from cuda_sft_qgen.graph.nodes.review_chair import _curate_select
    scored, results = _items([0.8, 0.8, 0.8, 0.8], scores={i: {"pedagogical_value": 5, "discriminativeness": 5, "realism": 5} for i in range(4)})
    v = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    not_sel, _ = _curate_select(scored, results, SETTINGS, v)
    assert not_sel in ({"q0"}, {"q1"})           # one of the two identical vectors is dropped


def test_different_buckets_untouched():
    from cuda_sft_qgen.graph.nodes.review_chair import _curate_select
    scored, results = _items([1.0, 0.9, 0.6, 0.5], same_bucket=False)
    assert _curate_select(scored, results, SETTINGS, None) == (set(), set())


def test_disabled_and_alias():
    from cuda_sft_qgen.graph.nodes.review_chair import _curate_select, _rip_select
    scored, results = _items([1.0, 0.9, 0.6, 0.5])
    off = SimpleNamespace(review={**SETTINGS.review, "rip_enabled": False})
    assert _curate_select(scored, results, off, None) == (set(), set())
    assert _rip_select(scored, results, SETTINGS) == {"q0"}
