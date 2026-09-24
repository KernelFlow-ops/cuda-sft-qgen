"""Weighted random facet choice with cooldown (task T35)."""
import random
from collections import Counter
from types import SimpleNamespace


def _manager():
    from cuda_sft_qgen.taxonomy_manager import TaxonomyManager
    m = TaxonomyManager.__new__(TaxonomyManager)
    m.facet_ids = ["formula", "numeric", "failure", "feature", "mechanism", "api", "layout", "profiling"]
    return m


def test_cooldown_and_boost():
    m = _manager()
    health = SimpleNamespace(kind_distribution={"formula": 50, "numeric": 50, "failure": 50, "feature": 0,
                                                "mechanism": 50, "api": 50}, total_leaves=250, leaves_with_high_reject_rate=[])
    rng = random.Random(0)
    counts = Counter(m.choose_facet(health, "layout", rng=rng) for _ in range(4000))
    assert counts["feature"] > counts["api"]            # under-represented kind boosted
    assert counts["layout"] < counts["profiling"]       # cooldown on last facet
    assert len(counts) == len(m.facet_ids)              # no facet starved


def test_deterministic_with_seed():
    m = _manager()
    h = SimpleNamespace(kind_distribution={}, total_leaves=0, leaves_with_high_reject_rate=[])
    a = [m.choose_facet(h, None, rng=random.Random(9)) for _ in range(5)]
    b = [m.choose_facet(h, None, rng=random.Random(9)) for _ in range(5)]
    assert a == b
