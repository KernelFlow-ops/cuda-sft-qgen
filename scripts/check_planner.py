"""Sample N briefs from the REAL taxonomy/configs and check invariant I3 (fix/01_ARCHITECTURE.md §6, §14).

python scripts/check_planner.py --n 2000 [--seed 7]
Prints per-axis distributions and `violations=<k>`; exit 1 when k > 0.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from cuda_sft_qgen.config import load_settings
    from cuda_sft_qgen.enums import ARCH_TO_HARDWARE
    from cuda_sft_qgen.models import SteeringBrief
    from cuda_sft_qgen.planner.joint import allowed_arch_targets, l2_meta
    from cuda_sft_qgen.planner.sampling import load_angles, plan_briefs
    from cuda_sft_qgen.taxonomy import load_taxonomy
    s = load_settings()
    tax = load_taxonomy(s.taxonomy_path)
    briefs = plan_briefs(n=a.n, taxonomy=tax, quotas=s.quotas, personas=s.personas, accepted=[], pending=[],
                         steering=SteeringBrief(), evolver_fraction=0.3, seed_pool=[], rng=random.Random(a.seed),
                         angles=load_angles(s.angles_path), compat=s.compat, scope=s.scope, joint=True)
    compat, meta = s.compat, l2_meta(tax)
    leaves = {lf.l3_id: lf for lf in tax.leaves}
    bad = []
    for b in briefs:
        leaf = leaves[b.l3_id]
        allowed = allowed_arch_targets(leaf, meta.get((leaf.domain_id, leaf.l2_id), {}).get("arch") or ["any"], compat["arch_claim_rules"])
        checks = {
            "type": b.question_type in compat["kind_to_types"][b.kind],
            "format": b.answer_format in compat["type_to_formats"][b.question_type],
            "bloom": b.bloom in compat["difficulty_to_bloom"][str(b.difficulty)],
            "arch": b.arch_target in allowed,
            "hardware": b.hardware == ARCH_TO_HARDWARE[b.arch_target],
            "instance": (not b.instance_hint) or b.instance_hint in leaf.hints,
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            bad.append({"l3_id": b.l3_id, "failed": failed})
    dist = {axis: dict(Counter(getattr(b, axis) for b in briefs).most_common())
            for axis in ("question_type", "answer_format", "difficulty", "bloom", "arch_target", "scenario", "language", "granularity")}
    dist["l1"] = dict(Counter(b.topic_path[0] for b in briefs).most_common())
    dist["exploration_share"] = sum(b.exploration for b in briefs) / len(briefs)
    print(json.dumps(dist, ensure_ascii=False, indent=1))
    print(json.dumps(bad[:20], ensure_ascii=False))
    print(f"violations={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
