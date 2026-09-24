"""Quota-weighted brief sampling with hardware/leaf compatibility and steering.

Hardware labels stay within architectures compatible with the leaf claim;
agnostic is always allowed. Residual quota weights and steering briefs
drive L1/type/difficulty mix. This module does not write questions or
review them.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import yaml

from cuda_sft_qgen.models import Question, QuestionBrief, SteeringBrief
from cuda_sft_qgen.planner.pair_mining import (
    PairCandidate,
    env_mixup_enabled,
    mine_pairs,
    mixup_payload,
)
from cuda_sft_qgen.review.diversity import residual_weights
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy

BLOOMS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
GRANULARITY = ["api", "kernel", "pipeline", "system"]
HARDWARE = ["agnostic", "ampere", "hopper", "blackwell"]

_HW_CLAIM = (
    ("ampere", re.compile(r"sm_?80|\bampere\b|\ba100\b|ga100", re.I)),
    ("hopper", re.compile(r"sm_?90|\bhopper\b|\bh100\b|gh100", re.I)),
    ("blackwell", re.compile(r"sm_?10[0-9]|\bblackwell\b|\bb200\b|gb200|tcgen05|\bumma\b", re.I)),
)
_HOPPER_FEAT = re.compile(
    r"\b(?:tma|wgmma|tmem|mbarrier|cp\.async\.bulk|tensormap)\b",
    re.I,
)


def leaf_hardware_allowed(leaf: Leaf) -> frozenset[str]:
    """Architectures this leaf may be labeled with.

    Inferred from claim / l3_id so Planner never assigns Hopper to an Ampere-only
    persist-window leaf (that pair used to be written then sieve-killed).
    Agnostic is always allowed. Unknown leaves keep the full quota set.
    """
    blob = f"{leaf.claim} {leaf.l3_id} {leaf.l2_id}"
    found: set[str] = set()
    for name, rx in _HW_CLAIM:
        if rx.search(blob):
            found.add(name)
    lower = blob.lower()
    if _HOPPER_FEAT.search(blob) and "ampere" not in lower and not re.search(
        r"sm_?80", blob, re.I
    ):
        found.update({"hopper", "blackwell"})
    if not found:
        return frozenset(HARDWARE)
    return frozenset(found | {"agnostic"})


def _as_angle_card(item: Any) -> dict[str, Any]:
    """Normalize an angles.yaml entry to ``{id, ask, kinds, ...}``."""
    if isinstance(item, dict):
        ask = str(item.get("ask") or item.get("angle") or "")
        card = dict(item)
        card["ask"] = ask
        card["id"] = str(item.get("id") or "")
        return card
    return {"id": "", "ask": str(item), "kinds": []}


def load_angles(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load per-kind question-angle cards from ``configs/angles.yaml``."""
    if path is None:
        from cuda_sft_qgen.config import find_project_root

        path = find_project_root() / "configs" / "angles.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    table = raw.get("angles") or {}
    return {str(k): [_as_angle_card(x) for x in (v or [])] for k, v in table.items()}


def _instance_hint(leaf: Leaf, rng: random.Random) -> str:
    """Pick a concrete fragment of the claim so writers do not stay generic."""
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


def _load_mixup_config() -> dict[str, Any]:
    """Load the planner-level mixup feature flag without importing runtime config."""
    try:
        from cuda_sft_qgen.config import find_project_root

        path = find_project_root() / "configs" / "default.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = raw.get("mixup") or {}
        return dict(value) if isinstance(value, dict) else {}
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return {}


def _resolve_mixup_config(
    explicit: dict[str, Any] | None,
    adaptive_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve explicit, runtime, file, and environment mixup settings."""
    if explicit is not None:
        config = dict(explicit)
    else:
        config = _load_mixup_config()
        nested = (adaptive_config or {}).get("mixup")
        if isinstance(nested, dict):
            config.update(nested)
    config["enabled"] = env_mixup_enabled(bool(config.get("enabled", False)))
    return config


def _mixup_specs(
    seed_pool: Sequence[Question],
    *,
    n: int,
    config: dict[str, Any],
    embedder: Any | None = None,
    active_claims: dict[str, Leaf] | None = None,
    parent_reuse_counts: dict[str, int] | None = None,
    knowledge: Any | None = None,
) -> list[tuple[PairCandidate, str]]:
    """Build bounded mode/pair assignments for the next writer batch."""
    if not config.get("enabled", False) or not seed_pool:
        return []
    modes = [str(mode).strip().lower() for mode in (config.get("modes") or ["decomposed", "hybrid"])]
    modes = [mode for mode in modes if mode in {"decomposed", "hybrid"}]
    if not modes:
        return []
    fraction = min(1.0, max(0.0, float(config.get("fraction", 0.30))))
    wanted = min(int(round(max(0, n) * fraction)), int(config.get("max_batch_items", n)))
    if wanted <= 0:
        return []
    pairs = mine_pairs(
        seed_pool,
        # Production planning uses the configured embedding service.  The
        # hashed fallback remains deterministic for offline tests only.
        embedder=embedder,
        min_similarity=float(config.get("pair_similarity_min", 0.80)),
        max_similarity=float(config.get("pair_similarity_max", 0.995)),
        top_k=max(1, int(config.get("pair_top_k", 8))),
        max_pairs=max(1, int(config.get("max_pairs", wanted))),
        max_pairs_per_item=max(1, int(config.get("max_pairs_per_item", 1))),
        parent_reuse_counts=parent_reuse_counts,
        max_parent_reuse=max(1, int(config.get("max_parent_reuse", 3))),
        min_difficulty_gap=max(1, int(config.get("min_difficulty_gap", 1))),
        same_l2=bool(config.get("same_l2", True)),
        related_claims=({q.l3_id: set(knowledge.related_claim_ids(q.l3_id)) for q in seed_pool}
                        if knowledge is not None and not config.get("same_l2", True) else None),
        same_language=bool(config.get("same_language", True)),
        same_granularity=bool(config.get("same_granularity", True)),
        active_claims=active_claims,
        lsh_tables=max(1, int(config.get("lsh_tables", 6))),
        lsh_bits=max(2, int(config.get("lsh_bits", 12))),
        lsh_max_bucket_size=max(2, int(config.get("lsh_max_bucket_size", 64))),
        lsh_max_candidates=max(4, int(config.get("lsh_max_candidates", 128))),
        lsh_seed=int(config.get("lsh_seed", 17)),
    )
    specs: list[tuple[PairCandidate, str]] = []
    reuse = dict(parent_reuse_counts or {})
    cap = max(1, int(config.get("max_parent_reuse", 3)))
    for pair in pairs:
        for mode in modes:
            if len(specs) >= wanted:
                return specs
            parent_ids = [pair.low.id, pair.high.id]
            if any(reuse.get(qid, 0) >= cap for qid in parent_ids):
                continue
            specs.append((pair, mode))
            for qid in parent_ids:
                reuse[qid] = reuse.get(qid, 0) + 1
    return specs


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
    adaptive_config: dict[str, Any] | None = None,
    rejected: Sequence[Question] | None = None,
    knowledge: Any | None = None,
    mixup_config: dict[str, Any] | None = None,
    embedder: Any | None = None,
) -> list[QuestionBrief]:
    """Sample ``n`` writer/evolver briefs from residual quotas and steering.

    Hardware labels are restricted to architectures compatible with the leaf
    claim. When adaptive sampling is enabled, L1 weights are boosted by leaf
    health. Knowledge context is attached when a store is provided.
    """
    rng = rng or random.Random()
    resolved_mixup = _resolve_mixup_config(mixup_config, adaptive_config)
    active_claims = {
        leaf.l3_id: leaf
        for leaf in taxonomy.leaves
        if not leaf.retired and str(getattr(leaf, "status", "") or "").lower()
        not in {"retired", "quarantined", "provisional"}
    }
    parent_reuse_counts: Counter[str] = Counter()
    for existing in list(accepted) + list(pending):
        for parent_id in getattr(existing, "parent_ids", ()) or ():
            parent_reuse_counts[str(parent_id)] += 1
    mixup_specs = _mixup_specs(
        seed_pool,
        n=n,
        config=resolved_mixup,
        embedder=embedder,
        active_claims=active_claims,
        parent_reuse_counts=dict(parent_reuse_counts),
        knowledge=knowledge,
    )
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
        if leaf.retired or str(getattr(leaf, "status", "") or "") in {"quarantined", "provisional"}:
            continue
        l2_key = f"{leaf.domain_id}.{leaf.l2_id}"
        if l2_key in blocked_l2:
            continue
        leaves_by_l1.setdefault(leaf.domain_id, []).append(leaf)
    usable_l1 = [k for k, v in leaves_by_l1.items() if v]
    if not usable_l1:
        leaves_by_l1 = {}
        for leaf in taxonomy.leaves:
            if leaf.retired or str(getattr(leaf, "status", "") or "") in {"quarantined", "provisional"}:
                continue
            leaves_by_l1.setdefault(leaf.domain_id, []).append(leaf)
        usable_l1 = list(leaves_by_l1)

    l1_weights = {k: v for k, v in l1_weights.items() if k in usable_l1}
    if not l1_weights:
        l1_weights = {k: 1.0 for k in usable_l1}

    sampler = None
    leaf_stats = {}
    # Apply adaptive sampling if enabled, including taxonomy leaves with no samples.
    if adaptive_config and adaptive_config.get("enabled", False):
        from cuda_sft_qgen.planner.adaptive import AdaptiveSampler, compute_leaf_stats

        rejected_seq = list(rejected) if rejected else []
        leaf_stats = compute_leaf_stats(
            accepted=accepted,
            rejected=rejected_seq,
            target_per_leaf=adaptive_config.get("target_per_leaf", 3),
            taxonomy=taxonomy,
        )

        sampler = AdaptiveSampler(
            epsilon=adaptive_config.get("epsilon", 0.15),
            accept_weight=adaptive_config.get("accept_weight", 0.6),
            coverage_weight=adaptive_config.get("coverage_weight", 0.4),
            min_samples=int(adaptive_config.get("min_samples_threshold", 5) or 5),
        )

        l1_weights = sampler.adjust_weights(l1_weights, leaf_stats, rng=rng)

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
    if not all_leaves:
        raise ValueError("No active taxonomy claims are available for planning")

    def choose_leaf(candidates: list[Leaf]) -> Leaf:
        if sampler is None or rng.random() < sampler.epsilon:
            return rng.choice(candidates)
        by_id = {leaf.l3_id: leaf for leaf in candidates}
        weights = {leaf_id: 0.5 + sampler.compute_health_score(leaf_stats[leaf_id])
                   for leaf_id in by_id}
        return by_id[_weighted_choice(rng, weights)]

    leaf_by_id = {leaf.l3_id: leaf for leaf in all_leaves}
    for i in range(n):
        mixup_spec = mixup_specs[i] if i < len(mixup_specs) else None
        l1 = _weighted_choice(rng, l1_weights)
        candidates = leaves_by_l1[l1]
        unused = [lf for lf in candidates if lf.l3_id not in used_l3]
        if unused:
            leaf = choose_leaf(unused)
        elif len(used_l3) < len(all_leaves):
            pool_unused = [lf for lf in all_leaves if lf.l3_id not in used_l3]
            leaf = choose_leaf(pool_unused or candidates)
        else:
            leaf = choose_leaf(candidates)
        if mixup_spec is not None:
            pair, _mode = mixup_spec
            # The generated question remains anchored to the lower-difficulty
            # parent's leaf; the second parent is supplied as a compatible
            # constraint in the evolver seed/context below.
            anchored = leaf_by_id.get(str(getattr(pair.high, "l3_id", "")))
            if anchored is None:
                anchored = leaf_by_id.get(str(getattr(pair.low, "l3_id", "")))
            if anchored is not None:
                leaf = anchored
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
        if mixup_spec is not None:
            # Use a deterministic target as a request; independent review may
            # revise the assessed difficulty later.
            difficulty = int(mixup_payload(*mixup_spec)["difficulty_target"])
        bloom = _weighted_choice(rng, bloom_w) if bloom_w else rng.choice(BLOOMS)
        # Definition-only remember+conceptual is allowed but rare.
        if bloom == "remember" and qtype == "conceptual" and rng.random() < 0.7:
            bloom = "apply"
        allowed_hw = leaf_hardware_allowed(leaf)
        hw_filtered = {k: v for k, v in (hw_w or {}).items() if k in allowed_hw}
        arch_specific = [name for name in ("ampere", "hopper", "blackwell") if name in allowed_hw]
        # Arch-locked claims still allow agnostic for compare items; otherwise
        # prefer a named architecture so Ampere stems do not grow TMA.
        if arch_specific and qtype != "compare":
            named = {k: v for k, v in hw_filtered.items() if k != "agnostic"}
            if named:
                hw_filtered = named
        if hw_filtered:
            hardware = _weighted_choice(rng, hw_filtered)
        elif "agnostic" in allowed_hw:
            hardware = "agnostic"
        else:
            hardware = next(iter(allowed_hw))
        granularity = _weighted_choice(rng, gran_w) if gran_w else "kernel"
        language = _weighted_choice(rng, lang_w) if lang_w else "zh"
        if mixup_spec is not None:
            # Pair compatibility is wasted if the subsequent quota draw assigns
            # a different language/architecture to its child. Carry the parents'
            # common environment into the synthesis request.
            parent = mixup_spec[0].high
            hardware = parent.hardware
            granularity = parent.granularity
            language = parent.language
        persona = rng.choice(persona_ids)
        role: str = "writer"
        seed_q = None
        seed_id = None
        mixup_context: dict[str, Any] | None = None
        if mixup_spec is not None:
            pair, mode = mixup_spec
            mixup_context = mixup_payload(pair, mode)
            role = "evolver"
            # Pair identity is retained for seed tracing; mode and parent
            # metadata live in the structured fields below.
            seed_id = mixup_context["pair_id"]
            seed_q = (
                f"PARENT A (lower difficulty):\n{getattr(pair.low, 'question', '')}\n\n"
                f"PARENT B (higher difficulty):\n{getattr(pair.high, 'question', '')}"
            ).strip()
        if mixup_spec is None and i < n_evolver:
            seed_candidates = [q for q in seed_pool if q.l3_id == leaf.l3_id]
            if not seed_candidates and knowledge is not None:
                related = set(knowledge.related_claim_ids(leaf.l3_id))
                seed_candidates = [q for q in seed_pool if q.l3_id in related]
            if seed_candidates:
                seed = rng.choice(seed_candidates)
                role = "evolver"
                seed_q = seed.question
                seed_id = seed.id
        hint = _instance_hint(leaf, rng)
        if mixup_context is not None:
            parent_claims = [
                str(getattr(mixup_context_item, "tests_claim", "") or "").strip()
                for mixup_context_item in (mixup_spec[0].low, mixup_spec[0].high)
            ]
            hint = (
                f"MIXUP_MODE={mixup_context['mode']}; target difficulty="
                f"{mixup_context['difficulty_target']}; raw target="
                f"{mixup_context['difficulty_raw_score']}. "
                f"{mixup_context['difficulty_policy']} Parent claims: "
                + " | ".join(claim for claim in parent_claims if claim)
                + f". Concrete instance: {hint}"
            )
        card = _pick_angle(rng, qtype, angle_table, kind=leaf.kind or None)
        angle = str(card.get("ask") or "")
        if (leaf.l3_id, angle) in used_pair:
            card = _pick_angle(
                rng, qtype, angle_table, avoid=angle, kind=leaf.kind or None
            )
            angle = str(card.get("ask") or angle)
        used_l3.add(leaf.l3_id)
        used_pair.add((leaf.l3_id, angle))

        # Breadth vs depth signal
        leaf_questions = [q for q in accepted if q.l3_id == leaf.l3_id]
        n_leaf = len(leaf_questions)
        if n_leaf < 3:
            breadth_signal = "BREADTH"
            breadth_guidance = f"This leaf has only {n_leaf} questions; explore different facets of the claim"
        elif n_leaf < 8:
            breadth_signal = "DEPTH"
            breadth_guidance = f"This leaf has {n_leaf} questions; go deeper with harder constraints on one facet"
        else:
            breadth_signal = "DEPTH"
            breadth_guidance = f"This leaf is well-covered ({n_leaf} questions); challenge with edge cases"

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
        ][:3]
        if others:
            guide_parts.append(
                "Other angles (contrast only, do not mix in):\n"
                + "\n".join(f"- {c.get('ask')}" for c in others)
            )
        context = (
            knowledge.context_for_generation(
                leaf.l3_id, revision=getattr(leaf, "revision", 1), max_claims=6
            )
            if knowledge is not None and hasattr(knowledge, "context_for_generation")
            else (
                knowledge.context_for_claim(leaf.l3_id, revision=getattr(leaf, "revision", 1))
                if knowledge is not None
                else {}
            )
        )
        if not isinstance(context, dict):
            context = {}
        if mixup_context is not None:
            context = dict(context)
            context["mixup"] = mixup_context
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
                breadth_or_depth=breadth_signal,
                claim_revision=getattr(leaf, "revision", 1),
                source_ids=list(getattr(leaf, "sources", ())),
                knowledge_context=context,
                mixup_mode=(mixup_context or {}).get("mode") if mixup_context else None,
                parent_ids=list((mixup_context or {}).get("parent_ids") or []),
                parent_difficulties=list((mixup_context or {}).get("parent_difficulties") or []),
                difficulty_target=(mixup_context or {}).get("difficulty_target") if mixup_context else None,
                difficulty_raw_score=(
                    float((mixup_context or {}).get("difficulty_raw_score"))
                    if mixup_context and (mixup_context.get("difficulty_raw_score")) is not None
                    else None
                ),
            )
        )
        # Light recency penalty so a batch does not collapse onto one L1.
        l1_weights[leaf.domain_id] = l1_weights.get(leaf.domain_id, 1.0) * 0.7
        l2_key = f"{leaf.domain_id}.{leaf.l2_id}"
        l2_counts[l2_key] = l2_counts.get(l2_key, 0) + 1
    rng.shuffle(briefs)
    return briefs
