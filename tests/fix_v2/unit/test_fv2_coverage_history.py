"""Cross-batch coverage memory (task T27)."""
import random
from collections import Counter

import yaml

from fixv2_helpers import make_question, tiny_taxonomy


def _archive(tmp_path, questions):
    from cuda_sft_qgen.store.archive import Archive
    store = Archive(tmp_path)
    for q in questions:
        store.append_accepted(q)
    return store


def test_counts(tmp_path):
    qs = [make_question(id=f"q{i}", l3_id="smem_bank_gcd", angle_id="con_plug_numbers", answer_format="multi_numeric",
                        scenario="reduction" if i % 2 else "generic", arch_target="sm_80", hardware="ampere") for i in range(4)]
    cov = _archive(tmp_path, qs).coverage_history()["smem_bank_gcd"]
    assert cov.count == 4 and cov.angles["con_plug_numbers"] == 4 and cov.formats["multi_numeric"] == 4
    assert cov.scenarios == Counter({"reduction": 2, "generic": 2}) and cov.archs["sm_80"] == 4


def test_briefs_carry_used_lists_and_avoid_repeats(tmp_path, repo_root):
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.sampling import load_angles, plan_briefs
    load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))
    qs = [make_question(id=f"q{i}", l3_id="smem_bank_gcd", topic_path=["T02", "smem", "smem_bank_gcd"],
                        angle_id="con_plug_numbers", answer_format="multi_numeric", kind="formula") for i in range(6)]
    cov = _archive(tmp_path, qs).coverage_history()
    kw = dict(n=3000, taxonomy=tiny_taxonomy(), quotas=load("quotas.yaml"), personas=load("personas.yaml")["personas"],
              accepted=[], pending=[], steering=SteeringBrief(), evolver_fraction=0.0, seed_pool=[],
              angles=load_angles(repo_root / "configs" / "angles.yaml"), compat=load("compat_matrix.yaml"),
              scope=load("scope.yaml"), joint=True)
    with_cov = [b for b in plan_briefs(rng=random.Random(5), coverage=cov, **kw) if b.l3_id == "smem_bank_gcd" and b.question_type == "conceptual"]
    no_cov = [b for b in plan_briefs(rng=random.Random(5), coverage=None, **kw) if b.l3_id == "smem_bank_gcd" and b.question_type == "conceptual"]
    assert with_cov and all("con_plug_numbers" in b.used_angles for b in with_cov)
    rate = lambda bs: sum(b.angle_id == "con_plug_numbers" for b in bs) / max(len(bs), 1)
    assert rate(with_cov) < rate(no_cov)
