"""Joint, compatibility-constrained sampling of brief axes (v2). Does not call LLMs or read files."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from cuda_sft_qgen.enums import ANSWER_FORMATS, ARCH_TARGETS, ARCH_TO_HARDWARE, OPERATORS
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy

logger = logging.getLogger(__name__)

FALLBACK_CARD = {"id": "fallback", "ask": "机制因果：给出可证伪的判断", "kinds": [], "formats": []}


@dataclass
class LeafCoverage:
    count: int = 0
    angles: Counter = field(default_factory=Counter)
    formats: Counter = field(default_factory=Counter)
    scenarios: Counter = field(default_factory=Counter)
    archs: Counter = field(default_factory=Counter)
    operators: Counter = field(default_factory=Counter)


@dataclass
class AxisWeights:
    type: dict[str, float]
    difficulty: dict[str, float]
    bloom: dict[str, float]
    granularity: dict[str, float]
    language: dict[str, float]
    arch: dict[str, float]
    format: dict[str, float]


@dataclass(frozen=True)
class BriefAxes:
    question_type: str
    card: dict
    answer_format: str
    difficulty: int
    bloom: str
    arch_target: str
    hardware: str
    granularity: str
    language: str
    persona: str
    scenario: str
    operator: str | None


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    """Choose a key with nonnegative weights in stable key order."""
    if not weights:
        raise ValueError("empty choice")
    items = [(key, max(0.0, float(weights[key]))) for key in sorted(weights)]
    total = sum(weight for _, weight in items)
    if total <= 0:
        return rng.choice([key for key, _ in items])
    pick = rng.random() * total
    cumulative = 0.0
    for key, weight in items:
        cumulative += weight
        if pick <= cumulative:
            return key
    return items[-1][0]


def l2_meta(taxonomy: Taxonomy) -> dict[tuple[str, str], dict[str, Any]]:
    """Index raw L2 dictionaries by domain and L2 id."""
    return {(str(domain["id"]), str(child["id"])): child
            for domain in taxonomy.domains for child in domain.get("children", [])}


def domain_weight(taxonomy: Taxonomy, scope: dict, domain_id: str) -> float:
    """Read taxonomy domain weight, then scope prior, then one."""
    domain = next((item for item in taxonomy.domains if item.get("id") == domain_id), None)
    if domain and domain.get("weight") is not None:
        return float(domain["weight"])
    scoped = next((item for item in (scope.get("domains") or []) if item.get("id") == domain_id), None)
    return float(scoped.get("weight", 1.0)) if scoped else 1.0


def allowed_arch_targets(leaf: Leaf, l2_arch: Sequence[str], rules: Sequence[dict]) -> list[str]:
    """Constrain architecture labels by L2 metadata and the first claim rule."""
    valid = set(ARCH_TARGETS)
    selected = set(l2_arch) & valid
    base = valid if "any" in selected else selected
    if not base:
        base = {"any"}
    blob = f"{leaf.claim} {' '.join(leaf.hints)} {leaf.l3_id}"
    for rule in rules:
        if re.search(str(rule["match"]), blob, re.I):
            allowed = set(rule.get("allow") or []) & valid
            base = (base & allowed) or allowed
            break
    return [arch for arch in ARCH_TARGETS if arch in base]


def violates(combo: dict[str, Any], forbidden: Sequence[dict]) -> bool:
    """True if every constrained field in any forbidden rule matches."""
    return violation_count(combo, forbidden) > 0


def violation_count(combo: dict[str, Any], forbidden: Sequence[dict]) -> int:
    """Number of forbidden rules whose constrained fields all match ``combo``."""
    return sum(all(combo.get(key) == value for key, value in rule.items() if key != "note")
               for rule in forbidden)


def _normalized(weights: dict[str, float]) -> dict[str, float]:
    """Per-level probabilities with the same clipping/uniform rule as weighted_choice."""
    clipped = {key: max(0.0, float(value)) for key, value in weights.items()}
    total = sum(clipped.values())
    return {key: (value / total if total > 0 else 1.0 / len(clipped)) for key, value in clipped.items()}


def leaf_weight(count: int, per_leaf: dict[str, int]) -> float:
    """Prefer underused leaves while enforcing the hard maximum."""
    if count < per_leaf["min"]:
        return 2.0
    if count < per_leaf["soft_max"]:
        return 1.0
    if count < per_leaf["hard_max"]:
        return 0.3
    return 0.0


def l1_weights(taxonomy: Taxonomy, scope: dict, l1_counts: dict[str, int], total: int) -> dict[str, float]:
    """Weight L1 domains by the gap between scope prior and current share."""
    priors = {domain_id: domain_weight(taxonomy, scope, domain_id) for domain_id in taxonomy.l1_ids()}
    norm = sum(priors.values()) or 1.0
    return {domain_id: max(prior / norm - l1_counts.get(domain_id, 0) / total if total else prior / norm,
                           0.2 * prior / norm)
            for domain_id, prior in priors.items()}


def sample_axes(leaf: Leaf, *, rng: random.Random, weights: AxisWeights, compat: dict,
                angles: dict[str, list[dict]], personas: dict[str, dict], l2_arch: Sequence[str],
                coverage: LeafCoverage, role: str, generic_share: float, generic_cap: float) -> BriefAxes:
    """Sample a valid set of axes; forbidden tuples fall back to an exact draw over valid ones."""
    kind = leaf.kind or "mechanism"
    kind_types = list(compat["kind_to_types"].get(kind) or compat["kind_to_types"]["mechanism"])
    type_candidates = [question_type for question_type in leaf.types if question_type in kind_types] or kind_types
    arch_candidates = allowed_arch_targets(leaf, l2_arch, compat.get("arch_claim_rules") or [])
    forbidden = compat.get("forbidden_combinations") or []
    difficulties = list(leaf.difficulty or (2,))
    gran_options = compat["domain_to_granularity"].get(leaf.domain_id) or compat["domain_to_granularity"]["default"]

    def cards_for(qtype: str) -> list[dict]:
        cards = list(angles.get(qtype) or [])
        matched = [card for card in cards if kind in (card.get("kinds") or [])]
        return matched or cards or [FALLBACK_CARD]

    def formats_for(qtype: str, card: dict) -> list[str]:
        all_formats = list(compat["type_to_formats"][qtype])
        return [fmt for fmt in all_formats if fmt in (card.get("formats") or [])] or all_formats

    def type_weights(types: Sequence[str]) -> dict[str, float]:
        return {item: weights.type.get(item, 0.05) for item in types}

    def card_weights(cards: list[dict]) -> dict[str, float]:
        return {str(i): 1 / (1 + coverage.angles[card.get("id", "")]) for i, card in enumerate(cards)}

    def format_weights(formats: list[str]) -> dict[str, float]:
        return {fmt: weights.format.get(fmt, 0.05) / (1 + coverage.formats[fmt]) for fmt in formats}

    def difficulty_weights() -> dict[str, float]:
        return {str(d): weights.difficulty.get(str(d), 0.05) for d in difficulties}

    def bloom_weights(difficulty: int) -> dict[str, float]:
        return {name: weights.bloom.get(name, 0.05) for name in compat["difficulty_to_bloom"][str(difficulty)]}

    def granularity_weights() -> dict[str, float]:
        return {name: weights.granularity.get(name, 0.05) for name in gran_options}

    def pick_arch() -> str:
        return weighted_choice(rng, {name: weights.arch.get(name, 0.05) / (1 + coverage.archs[name])
                                     for name in arch_candidates})

    def pick_language() -> str:
        return weighted_choice(rng, weights.language or {"zh": 0.6, "en": 0.4})

    def pick_persona() -> str:
        persona_options = [name for name, meta in personas.items() if leaf.domain_id in (meta.get("domains") or [])]
        return rng.choice(persona_options or list(personas) or ["cuda_interviewer"])

    def pick_scenario() -> str:
        scen_options = list(compat["domain_to_scenarios"].get(leaf.domain_id)
                            or compat["domain_to_scenarios"]["default"])
        if generic_share >= generic_cap:
            scen_options = [name for name in scen_options if name != "generic"] or scen_options
        return weighted_choice(rng, {name: 1 / (1 + coverage.scenarios[name]) for name in scen_options})

    def pick_operator() -> str | None:
        if role != "evolver":
            return None
        ops = list(compat["kind_to_operators"][kind])
        return weighted_choice(rng, {name: 1 / (1 + coverage.operators[name]) for name in ops})

    def one() -> BriefAxes:
        # Draw order is part of the seeded behaviour; keep arch between bloom and granularity.
        qtype = weighted_choice(rng, type_weights(type_candidates))
        cards = cards_for(qtype)
        card = cards[int(weighted_choice(rng, card_weights(cards)))]
        answer_format = weighted_choice(rng, format_weights(formats_for(qtype, card)))
        difficulty = int(weighted_choice(rng, difficulty_weights()))
        bloom = weighted_choice(rng, bloom_weights(difficulty))
        arch = pick_arch()
        granularity = weighted_choice(rng, granularity_weights())
        language, persona, scenario, operator = pick_language(), pick_persona(), pick_scenario(), pick_operator()
        return BriefAxes(qtype, card, answer_format, difficulty, bloom, arch, ARCH_TO_HARDWARE[arch],
                         granularity, language, persona, scenario, operator)

    def combo(qtype: str, answer_format: str, difficulty: int, bloom: str, granularity: str) -> dict[str, Any]:
        return {"kind": kind, "question_type": qtype, "answer_format": answer_format,
                "difficulty": difficulty, "bloom": bloom, "granularity": granularity}

    def fewest_violations(types: Sequence[str]) -> tuple[int, list[tuple], dict[str, float]]:
        """Combinations with the fewest forbidden-rule hits, weighted as one() would draw them."""
        best, options, probs = None, [], {}
        type_p, diff_p, gran_p = (_normalized(type_weights(types)), _normalized(difficulty_weights()),
                                  _normalized(granularity_weights()))
        for qtype in types:
            cards = cards_for(qtype)
            card_p = _normalized(card_weights(cards))
            for index, card in enumerate(cards):
                for answer_format, fmt_p in _normalized(format_weights(formats_for(qtype, card))).items():
                    for diff_key, d_p in diff_p.items():
                        for bloom, b_p in _normalized(bloom_weights(int(diff_key))).items():
                            for granularity, g_p in gran_p.items():
                                bad = violation_count(combo(qtype, answer_format, int(diff_key), bloom, granularity),
                                                      forbidden)
                                if best is not None and bad > best:
                                    continue
                                if best is None or bad < best:
                                    best, options, probs = bad, [], {}
                                probs[str(len(options))] = (type_p[qtype] * card_p[str(index)] * fmt_p
                                                            * d_p * b_p * g_p)
                                options.append((qtype, card, answer_format, int(diff_key), bloom, granularity))
        return (best if best is not None else len(forbidden) + 1), options, probs

    def draw(options: list[tuple], probs: dict[str, float]) -> BriefAxes:
        qtype, card, answer_format, difficulty, bloom, granularity = options[int(weighted_choice(rng, probs))]
        arch = pick_arch()
        return BriefAxes(qtype, card, answer_format, difficulty, bloom, arch, ARCH_TO_HARDWARE[arch],
                         granularity, pick_language(), pick_persona(), pick_scenario(), pick_operator())

    for _ in range(5):
        axes = one()
        if not violates(combo(axes.question_type, axes.answer_format, axes.difficulty, axes.bloom,
                              axes.granularity), forbidden):
            return axes
    # Rejection sampling failed: enumerate the combinations exactly, first within
    # the leaf's own types, then within every type its kind allows.
    scopes = [type_candidates] + ([kind_types] if list(kind_types) != list(type_candidates) else [])
    results = []
    for types in scopes:
        bad, options, probs = fewest_violations(types)
        if bad == 0:
            return draw(options, probs)
        results.append((bad, options, probs))
    bad, options, probs = min(results, key=lambda item: item[0])
    logger.warning("planner_forbidden_fallback leaf=%s kind=%s types=%s difficulty=%s violations=%s",
                   leaf.l3_id, kind, list(type_candidates), difficulties, bad)
    return draw(options, probs)
