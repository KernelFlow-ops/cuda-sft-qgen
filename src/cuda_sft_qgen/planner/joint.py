"""Joint, compatibility-constrained sampling of brief axes (v2). Does not call LLMs or read files."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

from cuda_sft_qgen.enums import ANSWER_FORMATS, ARCH_TARGETS, ARCH_TO_HARDWARE, OPERATORS
from cuda_sft_qgen.taxonomy import Leaf, Taxonomy

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
    return any(all(combo.get(key) == value for key, value in rule.items() if key != "note")
               for rule in forbidden)


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
    """Sample a valid set of axes with bounded retries for forbidden tuples."""
    kind = leaf.kind or "mechanism"
    kind_types = list(compat["kind_to_types"].get(kind) or compat["kind_to_types"]["mechanism"])
    type_candidates = [question_type for question_type in leaf.types if question_type in kind_types] or kind_types
    arch_candidates = allowed_arch_targets(leaf, l2_arch, compat.get("arch_claim_rules") or [])

    def one() -> BriefAxes:
        qtype = weighted_choice(rng, {item: weights.type.get(item, 0.05) for item in type_candidates})
        cards = list(angles.get(qtype) or [])
        matched = [card for card in cards if kind in (card.get("kinds") or [])]
        cards = matched or cards or [FALLBACK_CARD]
        index = weighted_choice(rng, {str(i): 1 / (1 + coverage.angles[card.get("id", "")])
                                      for i, card in enumerate(cards)})
        card = cards[int(index)]
        all_formats = list(compat["type_to_formats"][qtype])
        format_candidates = [fmt for fmt in all_formats if fmt in (card.get("formats") or [])] or all_formats
        answer_format = weighted_choice(rng, {fmt: weights.format.get(fmt, 0.05) / (1 + coverage.formats[fmt])
                                              for fmt in format_candidates})
        difficulties = list(leaf.difficulty or (2,))
        difficulty = int(weighted_choice(rng, {str(d): weights.difficulty.get(str(d), 0.05) for d in difficulties}))
        bloom_options = compat["difficulty_to_bloom"][str(difficulty)]
        bloom = weighted_choice(rng, {name: weights.bloom.get(name, 0.05) for name in bloom_options})
        arch = weighted_choice(rng, {name: weights.arch.get(name, 0.05) / (1 + coverage.archs[name])
                                     for name in arch_candidates})
        gran_options = compat["domain_to_granularity"].get(leaf.domain_id) or compat["domain_to_granularity"]["default"]
        granularity = weighted_choice(rng, {name: weights.granularity.get(name, 0.05) for name in gran_options})
        language = weighted_choice(rng, weights.language or {"zh": 0.6, "en": 0.4})
        persona_options = [name for name, meta in personas.items() if leaf.domain_id in (meta.get("domains") or [])]
        persona = rng.choice(persona_options or list(personas) or ["cuda_interviewer"])
        scen_options = list(compat["domain_to_scenarios"].get(leaf.domain_id)
                            or compat["domain_to_scenarios"]["default"])
        if generic_share >= generic_cap:
            scen_options = [name for name in scen_options if name != "generic"] or scen_options
        scenario = weighted_choice(rng, {name: 1 / (1 + coverage.scenarios[name]) for name in scen_options})
        operator = None
        if role == "evolver":
            ops = list(compat["kind_to_operators"][kind])
            operator = weighted_choice(rng, {name: 1 / (1 + coverage.operators[name]) for name in ops})
        return BriefAxes(qtype, card, answer_format, difficulty, bloom, arch, ARCH_TO_HARDWARE[arch],
                         granularity, language, persona, scenario, operator)

    for _ in range(5):
        axes = one()
        combo = {"kind": kind, "question_type": axes.question_type, "answer_format": axes.answer_format,
                 "difficulty": axes.difficulty, "bloom": axes.bloom, "granularity": axes.granularity}
        if not violates(combo, compat.get("forbidden_combinations") or []):
            return axes
    qtype = type_candidates[0]
    cards = list(angles.get(qtype) or [FALLBACK_CARD])
    card = cards[0]
    answer_format = compat["type_to_formats"][qtype][0]
    difficulty = min(leaf.difficulty or (2,))
    bloom = compat["difficulty_to_bloom"][str(difficulty)][0]
    return BriefAxes(qtype, card, answer_format, difficulty, bloom, axes.arch_target, axes.hardware,
                     axes.granularity, axes.language, axes.persona, axes.scenario, axes.operator)
