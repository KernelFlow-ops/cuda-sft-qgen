"""v1-format questions stay readable and plannable (task T40; invariant I12)."""
import json
import random

import yaml

from fixv2_helpers import tiny_taxonomy


def test_legacy_rows(tmp_path, repo_root):
    from cuda_sft_qgen.models import Question, SteeringBrief
    from cuda_sft_qgen.planner.sampling import load_angles, plan_briefs
    from cuda_sft_qgen.store.archive import Archive
    rows = [json.loads(l) for l in (repo_root / "tests" / "fix_v2" / "fixtures" / "legacy_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    store = Archive(tmp_path)
    for r in rows:
        store.append_accepted(Question.model_validate(r))
    acc = store.load_accepted()
    assert len(acc) == len(rows) and all(q.answer_format == "" and q.arch_target == "any" for q in acc)
    cov = store.coverage_history()
    assert sum(c.count for c in cov.values()) == len(rows)
    load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))
    briefs = plan_briefs(n=50, taxonomy=tiny_taxonomy(), quotas=load("quotas.yaml"), personas=load("personas.yaml")["personas"],
                         accepted=acc, pending=[], steering=SteeringBrief(), evolver_fraction=0.3, seed_pool=acc,
                         rng=random.Random(4), angles=load_angles(repo_root / "configs" / "angles.yaml"),
                         compat=load("compat_matrix.yaml"), scope=load("scope.yaml"), coverage=cov, joint=True)
    assert len(briefs) == 50
