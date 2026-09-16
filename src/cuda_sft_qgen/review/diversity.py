from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from cuda_sft_qgen.models import DiversityReport, Question, SteeringBrief
from cuda_sft_qgen.taxonomy import Taxonomy


def _hist(items: Iterable[str]) -> dict[str, int]:
    return dict(Counter(items))


def _kl(current: Mapping[str, int], target: Mapping[str, float]) -> float:
    total = sum(current.values())
    if total <= 0:
        return 0.0
    kl = 0.0
    keys = set(current) | set(target)
    for key in keys:
        p = current.get(key, 0) / total
        q = float(target.get(key, 0.0))
        q = max(q, 1e-8)
        if p <= 0:
            continue
        kl += p * math.log(p / q)
    return kl


def _entropy_ratio(current: Mapping[str, int], n_bins: int) -> float:
    total = sum(current.values())
    if total <= 0 or n_bins <= 1:
        return 1.0
    ent = 0.0
    for c in current.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    max_ent = math.log(n_bins)
    return ent / max_ent if max_ent else 1.0


def residual_weights(
    current: Mapping[str, int],
    target_fracs: Mapping[str, float],
    *,
    floor: float = 0.02,
    overrep_factor: float = 2.0,
) -> dict[str, float]:
    total = max(sum(current.values()), 1)
    weights: dict[str, float] = {}
    for key, frac in target_fracs.items():
        frac = float(frac)
        count = current.get(key, 0)
        cur = count / total
        w = max(frac - cur, floor)
        if count > overrep_factor * frac * max(sum(current.values()), 20):
            w = 0.0
        weights[key] = w
    return weights


def build_diversity_report(
    accepted: Sequence[Question],
    pending: Sequence[Question],
    taxonomy: Taxonomy,
    quotas: dict[str, Any],
    *,
    mean_nn_cosine: float | None = None,
    leaf_min: int = 3,
    kl_alert: float = 0.15,
    entropy_alert: float = 0.85,
    overrep_factor: float = 2.0,
) -> DiversityReport:
    pool = list(accepted) + list(pending)
    l1 = _hist(q.topic_path[0] for q in pool if q.topic_path)
    types = _hist(q.question_type for q in pool)
    kinds = _hist(q.kind or "mechanism" for q in pool)
    diffs = _hist(str(q.difficulty) for q in pool)
    hw = _hist(q.hardware for q in pool)
    gran = _hist(q.granularity for q in pool)

    l2_keys = taxonomy.l2_keys()
    l2_counts = Counter(
        f"{q.topic_path[0]}.{q.topic_path[1]}"
        for q in pool
        if len(q.topic_path) >= 2
    )
    covered = sum(1 for k in l2_keys if l2_counts.get(k, 0) >= 1)
    coverage = covered / max(len(l2_keys), 1)

    # Uniform-ish L1 target inside [min, max] band.
    l1_ids = taxonomy.l1_ids()
    uniform = 1.0 / max(len(l1_ids), 1)
    l1_target = {d: uniform for d in l1_ids}
    type_target = {str(k): float(v) for k, v in (quotas.get("question_types") or {}).items()}
    diff_target = {str(k): float(v) for k, v in (quotas.get("difficulty") or {}).items()}

    kl_l1 = _kl(l1, l1_target)
    kl_type = _kl(types, type_target) if type_target else 0.0
    kl_diff = _kl(diffs, diff_target) if diff_target else 0.0
    ent = _entropy_ratio(l1, max(len(l1_ids), 1))

    total = max(len(pool), 1)
    under: list[str] = []
    over: list[str] = []
    n_l1 = max(len(l1_ids), 1)
    l1_min = float(quotas.get("l1_share", {}).get("min", 0.5 / n_l1))
    l1_max = float(quotas.get("l1_share", {}).get("max", 2.5 / n_l1))
    for did in l1_ids:
        share = l1.get(did, 0) / total
        if l1.get(did, 0) == 0 or share < l1_min * 0.5:
            under.append(did)
        if share > l1_max:
            over.append(did)
    for key, count in l2_counts.items():
        target_each = max(total / max(len(l2_keys), 1), leaf_min)
        if count > overrep_factor * target_each:
            over.append(key)

    notes: list[str] = []
    if kl_l1 > kl_alert:
        notes.append(f"KL(L1)={kl_l1:.3f} > {kl_alert}")
    if kl_type > kl_alert:
        notes.append(f"KL(type)={kl_type:.3f} > {kl_alert}")
    if ent < entropy_alert:
        notes.append(f"L1 entropy ratio={ent:.3f} < {entropy_alert}")
    if coverage < 0.8:
        notes.append(f"L2 coverage={coverage:.2f}")
    if mean_nn_cosine is not None and mean_nn_cosine > 0.80:
        notes.append(f"mean NN cosine={mean_nn_cosine:.3f} (question phrasing collapsing)")

    return DiversityReport(
        accepted_count=len(accepted),
        pending_count=len(pending),
        l1_hist=l1,
        type_hist=types,
        difficulty_hist=diffs,
        hardware_hist=hw,
        granularity_hist=gran,
        leaf_coverage=coverage,
        kind_hist=kinds,
        kl_l1=kl_l1,
        kl_type=kl_type,
        kl_difficulty=kl_diff,
        entropy_ratio_l1=ent,
        mean_nn_cosine=mean_nn_cosine,
        underrepresented=under,
        overrepresented=over,
        notes=notes,
    )


def steering_from_report(
    report: DiversityReport,
    quotas: dict[str, Any],
    forbidden_templates: Sequence[str] | None = None,
    *,
    anti_cap: int = 15,
) -> SteeringBrief:
    type_target = quotas.get("question_types") or {}
    total_types = sum(report.type_hist.values()) or 1
    preferred = []
    for name, frac in type_target.items():
        cur = report.type_hist.get(name, 0) / total_types
        if cur < float(frac) * 0.6:
            preferred.append(str(name))
    nudge: str = "hold"
    dhist = report.difficulty_hist
    hard = dhist.get("3", 0) + dhist.get("4", 0)
    easy = dhist.get("1", 0)
    if hard < easy:
        nudge = "up"
    elif easy == 0 and hard > (dhist.get("2", 0)):
        nudge = "down"
    notes = "; ".join(report.notes) if report.notes else "keep generating answerable, non-template questions"
    return SteeringBrief(
        underrepresented=report.underrepresented[:12],
        overrepresented=report.overrepresented[:12],
        forbidden_templates=list(forbidden_templates or [])[: max(1, int(anti_cap))],
        preferred_types=preferred[:6],
        difficulty_nudge=nudge,  # type: ignore[arg-type]
        style_notes=notes,
    )
