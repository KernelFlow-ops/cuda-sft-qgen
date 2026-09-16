from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.review.diversity import residual_weights
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy

BLOOMS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
GRANULARITY = ["api", "kernel", "pipeline", "system"]
HARDWARE = ["agnostic", "ampere", "hopper", "blackwell"]


def _as_angle_card(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        ask = str(item.get("ask") or item.get("angle") or "")
        card = dict(item)
        card["ask"] = ask
        card["id"] = str(item.get("id") or "")
        return card
    return {"id": "", "ask": str(item), "kinds": []}


def load_angles(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        from cuda_sft_qgen.config import find_project_root

        path = find_project_root() / "configs" / "angles.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    table = raw.get("angles") or {}
    return {str(k): [_as_angle_card(x) for x in (v or [])] for k, v in table.items()}


def _instance_hint(leaf: Leaf, rng: random.Random) -> str:
    if leaf.hints:
        return rng.choice(list(leaf.hints))
    # Do not split on `/` — it appears in fractions (3/5) and API paths.
    parts = re.split(r"[、，,]| vs |\s+and\s+", leaf.claim)
    parts = [p.strip(" .") for p in parts if len(p.strip()) >= 6]
    return rng.choice(parts) if parts else leaf.claim[:80]


def _stems_for_leaf(accepted: Sequence[Question], l3_id: str, k: int = 3) -> list[str]:
    out: list[str] = []
    for q in reversed(list(accepted)):
        if q.l3_id == l3_id and (q.question or "").strip():
            out.append(q.question.strip()[:80])
        if len(out) >= k:
            break
    return out


def _pick_angle(
    rng: random.Random,
    qtype: str,
    table: dict[str, list[dict[str, Any]]],
    *,
    avoid: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    fallback = {"id": "", "ask": "机制因果：给出可证伪的判断", "kinds": []}
    options = list(table.get(qtype) or [fallback])
    if kind:
        matched = [c for c in options if kind in (c.get("kinds") or [])]
        if matched:
            options = matched
        else:
            # Do not attach an unrelated ask (e.g. missing __device__ on a formula leaf).
            options = [fallback]
    if avoid:
        alt = [c for c in options if c.get("ask") != avoid]
        if alt:
            options = alt
    return rng.choice(options)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(k, max(float(v), 0.0)) for k, v in weights.items()]
    total = sum(w for _, w in items)
    if total <= 0:
        return rng.choice([k for k, _ in items])
    pick = rng.random() * total
    acc = 0.0
    for key, w in items:
        acc += w
        if pick <= acc:
            return key
    return items[-1][0]


def _count_axis(questions: Sequence[Question], attr: str) -> dict[str, int]:
    if attr == "l1":
        return dict(Counter(q.topic_path[0] for q in questions if q.topic_path))
    if attr == "l2":
        return dict(
            Counter(
                ".".join(q.topic_path[:2])
                for q in questions
                if len(q.topic_path) >= 2
            )
        )
    return dict(Counter(str(getattr(q, attr)) for q in questions))


def plan_briefs(
    *,
    n: int,
    taxonomy: Taxonomy,
    quotas: dict[str, Any],
    personas: dict[str, Any],
    accepted: Sequence[Question],
    pending: Sequence[Question],
    steering: SteeringBrief,
    evolver_fraction: float,
    seed_pool: Sequence[Question],
    rng: random.Random | None = None,
    angles: dict[str, list[Any]] | None = None,
    anti_cap: int = 15,
) -> list[QuestionBrief]:
    rng = rng or random.Random()
    raw_angles = angles if angles is not None else load_angles()
    angle_table: dict[str, list[dict[str, Any]]] = {
        k: [_as_angle_card(x) for x in (v or [])] for k, v in raw_angles.items()
    }
    pool = list(accepted) + list(pending)
    l1_counts = _count_axis(pool, "l1")
    type_counts = _count_axis(pool, "question_type")
    diff_counts = _count_axis(pool, "difficulty")
    hw_counts = _count_axis(pool, "hardware")
    gran_counts = _count_axis(pool, "granularity")
    lang_counts = _count_axis(pool, "language")
    l2_counts = _count_axis(pool, "l2")

    l1_ids = taxonomy.l1_ids()
    uniform = {d: 1.0 / max(len(l1_ids), 1) for d in l1_ids}
    # Mix quota band with uniform so empty domains get boosted.
    l1_weights = residual_weights(l1_counts, uniform, overrep_factor=2.0)
    for key in steering.underrepresented:
        l1 = key.split(".")[0]
        if l1 in l1_weights:
            l1_weights[l1] = max(l1_weights[l1], 0.05) * 3.0
    for key in steering.overrepresented:
        l1 = key.split(".")[0]
        if l1 in l1_weights:
            l1_weights[l1] *= 0.05
        # Hard-block an over-represented L2 by skipping those leaves later.

    blocked_l2 = {k for k in steering.overrepresented if k.count(".") >= 1}

    type_w = residual_weights(
        type_counts,
        {str(k): float(v) for k, v in (quotas.get("question_types") or {}).items()},
    )
    for t in steering.preferred_types:
        if t in type_w:
            type_w[t] = max(type_w[t], 0.05) * 2.5
    diff_w = residual_weights(
        diff_counts,
        {str(k): float(v) for k, v in (quotas.get("difficulty") or {}).items()},
    )
    if steering.difficulty_nudge == "up":
        for k in list(diff_w):
            if int(k) >= 3:
                diff_w[k] *= 2.0
            if int(k) == 1:
                diff_w[k] *= 0.4
    elif steering.difficulty_nudge == "down":
        for k in list(diff_w):
            if int(k) == 1:
                diff_w[k] *= 1.5
    hw_w = residual_weights(
        hw_counts,
        {str(k): float(v) for k, v in (quotas.get("hardware") or {}).items()},
    )
    gran_w = residual_weights(
        gran_counts,
        {str(k): float(v) for k, v in (quotas.get("granularity") or {}).items()},
    )
    lang_w = residual_weights(
        lang_counts,
        {str(k): float(v) for k, v in (quotas.get("language") or {}).items()},
    )
    bloom_w = residual_weights(
        _count_axis(pool, "bloom"),
        {str(k): float(v) for k, v in (quotas.get("bloom") or {}).items()}
        or {b: 1 / len(BLOOMS) for b in BLOOMS},
    )

    persona_ids = list(personas.keys()) or ["cuda_interviewer"]
    leaves_by_l1: dict[str, list[Leaf]] = {}
    for leaf in taxonomy.leaves:
        l2_key = f"{leaf.domain_id}.{leaf.l2_id}"
        if l2_key in blocked_l2:
            continue
        leaves_by_l1.setdefault(leaf.domain_id, []).append(leaf)
    usable_l1 = [k for k, v in leaves_by_l1.items() if v]
    if not usable_l1:
        leaves_by_l1 = {}
        for leaf in taxonomy.leaves:
            leaves_by_l1.setdefault(leaf.domain_id, []).append(leaf)
        usable_l1 = list(leaves_by_l1)

    l1_weights = {k: v for k, v in l1_weights.items() if k in usable_l1}
    if not l1_weights:
        l1_weights = {k: 1.0 for k in usable_l1}

    n_evolver = 0
    if seed_pool:
        n_evolver = min(int(round(n * evolver_fraction)), n, len(seed_pool))

    briefs: list[QuestionBrief] = []
    anti = list(steering.forbidden_templates[: max(1, int(anti_cap))])
    # style_notes already go into the steering prompt block; stuffing them
    # into MUST_AVOID inverts the chair's request.
    avoid = "do not write definition-only questions"
    used_l3: set[str] = set()
    used_pair: set[tuple[str, str]] = set()
    all_leaves = [lf for v in leaves_by_l1.values() for lf in v]

    for i in range(n):
        l1 = _weighted_choice(rng, l1_weights)
        candidates = leaves_by_l1[l1]
        unused = [lf for lf in candidates if lf.l3_id not in used_l3]
        if unused:
            leaf = rng.choice(unused)
        elif len(used_l3) < len(all_leaves):
            pool_unused = [lf for lf in all_leaves if lf.l3_id not in used_l3]
            leaf = rng.choice(pool_unused or candidates)
        else:
            leaf = rng.choice(candidates)
        # Prefer leaf-native types if they have residual mass.
        native = [t for t in leaf.types if t in type_w]
        if native and rng.random() < 0.7:
            tw = {t: type_w.get(t, 0.05) for t in native}
            qtype = _weighted_choice(rng, tw)
        else:
            qtype = _weighted_choice(rng, type_w)
        native_d = [str(d) for d in leaf.difficulty if str(d) in diff_w]
        if native_d and rng.random() < 0.7:
            difficulty = int(_weighted_choice(rng, {d: diff_w.get(d, 0.05) for d in native_d}))
        else:
            difficulty = int(_weighted_choice(rng, diff_w))
        bloom = _weighted_choice(rng, bloom_w) if bloom_w else rng.choice(BLOOMS)
        # Definition-only remember+conceptual is allowed but rare.
        if bloom == "remember" and qtype == "conceptual" and rng.random() < 0.7:
            bloom = "apply"
        hardware = _weighted_choice(rng, hw_w) if hw_w else "agnostic"
        granularity = _weighted_choice(rng, gran_w) if gran_w else "kernel"
        language = _weighted_choice(rng, lang_w) if lang_w else "zh"
        persona = rng.choice(persona_ids)
        role: str = "writer"
        seed_q = None
        seed_id = None
        if i < n_evolver:
            seed = rng.choice(list(seed_pool))
            role = "evolver"
            seed_q = seed.question
            seed_id = seed.id
        hint = _instance_hint(leaf, rng)
        card = _pick_angle(rng, qtype, angle_table, kind=leaf.kind or None)
        angle = str(card.get("ask") or "")
        if (leaf.l3_id, angle) in used_pair:
            card = _pick_angle(
                rng, qtype, angle_table, avoid=angle, kind=leaf.kind or None
            )
            angle = str(card.get("ask") or angle)
        used_l3.add(leaf.l3_id)
        used_pair.add((leaf.l3_id, angle))
        guide_parts: list[str] = []
        if card.get("must_have"):
            guide_parts.append(
                "Must have:\n" + "\n".join(f"- {x}" for x in card["must_have"])
            )
        if card.get("must_not"):
            guide_parts.append(
                "Must not:\n" + "\n".join(f"- {x}" for x in card["must_not"])
            )
        others = [
            c for c in (angle_table.get(qtype) or []) if c.get("ask") != angle
        ]
        if others:
            guide_parts.append(
                "Other angles (contrast only, do not mix in):\n"
                + "\n".join(f"- {c.get('ask')}" for c in others)
            )
        briefs.append(
            QuestionBrief(
                topic_path=list(leaf.path),
                l3_id=leaf.l3_id,
                l3_claim=leaf.claim,
                question_type=qtype,
                difficulty=difficulty,
                bloom=bloom,
                persona=persona,
                hardware=hardware,
                granularity=granularity,
                language=language,
                must_include=hint,
                must_avoid=avoid,
                anti_templates=anti,
                angle=angle,
                instance_hint=hint,
                must_avoid_stems=_stems_for_leaf(accepted, leaf.l3_id),
                angle_guide="\n".join(guide_parts),
                angle_id=str(card.get("id") or ""),
                kind=leaf.kind or "mechanism",
                role=role,  # type: ignore[arg-type]
                seed_question=seed_q,
                seed_id=seed_id,
            )
        )
        # Light recency penalty so a batch does not collapse onto one L1.
        l1_weights[l1] *= 0.7
        l2_key = f"{leaf.domain_id}.{leaf.l2_id}"
        l2_counts[l2_key] = l2_counts.get(l2_key, 0) + 1
    rng.shuffle(briefs)
    return briefs
