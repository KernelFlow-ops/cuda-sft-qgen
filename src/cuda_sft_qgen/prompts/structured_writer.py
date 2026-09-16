from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cuda_sft_qgen.cache import PromptCache
from cuda_sft_qgen.models import QuestionBrief, SteeringBrief
from cuda_sft_qgen.prompts import load_prompt, load_prompt_yaml
from cuda_sft_qgen.prompts.fewshot import (
    FewshotCfg,
    FewshotCorpus,
    format_anti,
    format_negatives,
    format_positives,
    select_anti,
    select_negatives,
    select_positives,
)
from cuda_sft_qgen.review.fingerprint import fingerprint

KIND_CONSTRAINTS = {
    "formula": (
        "- KIND=formula: you MAY pose a closed-form or plug-in-numbers question; "
        "the numeric/symbolic answer must be unique. Do not ask to 'derive a performance model'."
    ),
    "feature": (
        "- KIND=feature: you MAY introduce the feature ONLY through a hard constraint "
        "(arch, version, alignment, shape, scope). No marketing copy."
    ),
    "mechanism": (
        "- KIND=mechanism: ask why A causes B in the hardware/execution model; no glossary entry."
    ),
    "api": "- KIND=api: legal vs illegal call/qualifier/launch; prefer 2–3 snippets over an essay.",
    "failure": "- KIND=failure: name the path and the observable consequence (deadlock, silent wrong, UB).",
    "numeric": "- KIND=numeric: plug in the given numbers and demand a unique result.",
}

COT_SCENE = {
    "formula": "A formula scene MUST include substitutable numbers or symbols and a unique output.",
    "feature": "A feature scene MUST name an architecture/version gate and one hard constraint.",
    "numeric": "A numeric scene MUST include the inputs needed to compute a unique value.",
    "failure": "A failure scene MUST include a concrete path (code, mask, or launch) and an observable.",
}


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _bullet(items: list[str] | None) -> str:
    if not items:
        return ""
    return "\n".join(f"- {x}" for x in items)


def _card_text(table: dict[str, Any], key: str) -> str:
    val = table.get(key) or table.get(str(key)) or ""
    return str(val).strip()


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str
    static_prefix: str = ""
    cache_key: str = ""


class StructuredPrompt:
    def __init__(
        self,
        settings: Any,
        corpus: FewshotCorpus | None = None,
        cache: PromptCache | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache if cache is not None else PromptCache()
        gen = getattr(settings, "generation", {}) or {}
        if callable(gen):
            gen = settings.generation
        gray = int(getattr(settings, "simhash_gray_hamming", 8))
        self.cfg = FewshotCfg.from_generation(gen, gray_hamming=gray)
        self.corpus = corpus or FewshotCorpus()
        self.quality = load_prompt_yaml("quality_cards")
        anti_raw = load_prompt_yaml("anti_patterns")
        self.static_anti = list(anti_raw.get("patterns") or [])
        self._constraints = load_prompt("constraints")
        self._quality = load_prompt("quality")
        self._context = load_prompt("context")
        self._cot = load_prompt("cot")

    def build(self, brief: QuestionBrief, steering: SteeringBrief) -> PromptBundle:
        gen = self.settings.generation
        use_cot = bool(gen.get("prompt_cot", True))
        anti_cap = int(gen.get("anti_templates_max", 15))
        kind = brief.kind or "mechanism"
        personas = getattr(self.settings, "personas", {}) or {}
        meta = personas.get(brief.persona) or {}
        persona_name = str(meta.get("name") or brief.persona)
        persona_voice = str(meta.get("voice") or "")

        seed_hash = None
        if brief.role == "evolver" and brief.seed_question:
            seed_hash = fingerprint(brief.seed_question).simhash
        pos = select_positives(brief, self.corpus, self.cfg, seed_simhash=seed_hash)
        used = {p.question.id for p in pos}
        neg = select_negatives(brief, self.corpus, self.cfg, used_ids=used)
        anti = select_anti(brief, self.corpus, self.cfg, self.static_anti)

        angle_card = ""
        if brief.angle:
            angle_card = self._angle_card_block(brief)

        mapping = {
            "persona_name": persona_name,
            "persona_voice": persona_voice,
            "min_chars": str(self.settings.question_min_chars),
            "max_chars": str(self.settings.question_max_chars),
            "must_include": brief.must_include or brief.instance_hint or "",
            "must_avoid": brief.must_avoid or "do not write definition-only questions",
            "must_avoid_stems": "\n".join(f"- {s}" for s in brief.must_avoid_stems) or "(none)",
            "kind_constraint": KIND_CONSTRAINTS.get(kind, KIND_CONSTRAINTS["mechanism"]),
            "topic_path": ".".join(brief.topic_path),
            "question_type": brief.question_type,
            "difficulty": str(brief.difficulty),
            "bloom": brief.bloom,
            "granularity": brief.granularity,
            "language": brief.language,
            "hardware": brief.hardware,
            "kind": kind,
            "angle": brief.angle or "",
            "angle_card": angle_card,
            "angle_guide": brief.angle_guide or "",
            "bloom_card": _card_text(self.quality.get("bloom") or {}, brief.bloom),
            "diff_card": _card_text(self.quality.get("difficulty") or {}, str(brief.difficulty)),
            "kind_card": _card_text(self.quality.get("kind") or {}, kind),
            "fewshot_pos": format_positives(pos, self.cfg),
            "fewshot_neg": format_negatives(neg, self.cfg),
            "steering": steering.to_prompt_block(anti_cap=anti_cap),
            "anti_templates": format_anti(anti),
            "cot_scene": COT_SCENE.get(
                kind,
                "Design a concrete, checkable scene (numbers, launch, layout, code, or counters).",
            ),
            "cot_evolver": (
                "Evolver extra: name exactly one operator deepen|re-arch|inject-bug|layout-eq|profiler "
                "and say how it changes the seed's training structure (not a synonym rewrite)."
                if brief.role == "evolver"
                else ""
            ),
            "l3_claim": brief.l3_claim,
            "instance_hint": brief.instance_hint or brief.must_include or brief.l3_claim,
            "seed_question": brief.seed_question or "",
        }

        role_key = "evolver" if brief.role == "evolver" and brief.seed_question else "writer"
        static_prefix = self.cache.get_static(
            f"{role_key}:{mapping['min_chars']}:{mapping['max_chars']}",
            lambda: self._static_prefix(role_key, mapping["min_chars"], mapping["max_chars"]),
        )
        semi_key = (
            role_key,
            brief.persona,
            brief.question_type,
            kind,
            brief.bloom,
            str(brief.difficulty),
        )
        semi = self.cache.get_semi(
            semi_key,
            lambda: self._semi_block(
                role_key,
                persona_name,
                persona_voice,
                kind,
                mapping["bloom_card"],
                mapping["diff_card"],
                mapping["kind_card"],
                mapping["kind_constraint"],
            ),
        )
        dynamic_parts = [
            _render(self._constraints, mapping).strip(),
            _render(self._quality, mapping).strip(),
        ]
        if use_cot:
            dynamic_parts.append(_render(self._cot, mapping).strip())
        dynamic_parts.append(_render(self._context, mapping).strip())
        dynamic_parts.append(
            f'Reminder: len in [{mapping["min_chars"]}, {mapping["max_chars"]}]; '
            f'follow angle "{brief.angle}"; JSON only.'
        )
        system = PromptCache.compose(static_prefix, semi, *dynamic_parts)

        user_lines = [
            f"CLAIM: {brief.l3_claim}",
            f"INSTANCE: {mapping['instance_hint']}",
            f"ANGLE: {brief.angle}",
            f"KIND: {kind}",
            f"TYPE: {brief.question_type}",
            f"DIFFICULTY: {brief.difficulty}",
            f"BLOOM: {brief.bloom}",
            f"HARDWARE: {brief.hardware}",
            f"GRANULARITY: {brief.granularity}",
            f"LANGUAGE: {brief.language}",
            f"MUST_INCLUDE: {mapping['must_include']}",
            f"MUST_AVOID: {mapping['must_avoid']}",
        ]
        if brief.role == "evolver" and brief.seed_question:
            user_lines.append(f"SEED: {brief.seed_question}")
            user_lines.append("Write the JSON now. Change the training structure.")
        else:
            user_lines.append("Write the JSON now.")
        user = "\n".join(user_lines)
        return PromptBundle(
            system=system,
            user=user,
            static_prefix=static_prefix,
            cache_key=f"{role_key}:{brief.persona}:{brief.question_type}:{kind}",
        )

    def _static_prefix(self, role_key: str, min_chars: str, max_chars: str) -> str:
        if role_key == "evolver":
            role = (
                "You are an Evol-Instruct coach for CUDA exam questions.\n"
                "Take the seed question and produce a NEW question that is clearly different "
                "in training structure, not a synonym rewrite."
            )
            schema = (
                "Return JSON only (one object). CUDA code may live in markdown fences inside the question string.\n\n"
                "{\n"
                '  "question": "...",\n'
                '  "tests_claim": "...",\n'
                '  "language": "zh" or "en",\n'
                '  "operator": "deepen|re-arch|inject-bug|layout-eq|profiler"\n'
                "}"
            )
        else:
            role = (
                "You are a specialist CUDA examination author. You write ONE high-value "
                "training question, not an article and not an answer."
            )
            schema = (
                "Return JSON only (one object). CUDA code may live in markdown fences inside the question string.\n\n"
                "{\n"
                '  "question": "the question text",\n'
                '  "tests_claim": "one-sentence claim this item actually tests",\n'
                '  "language": "zh" or "en"\n'
                "}"
            )
        generic = (
            f"Length (hard): the question string MUST have Python len() in [{min_chars}, {max_chars}] "
            "(code fences count). Too short is a slogan; too long is a blog.\n"
            "Do NOT write a glossary, a blog intro, or hide the full worked answer in the question.\n"
            "Open-ended \"how would you optimize\" with no numbers, counters, or code is forbidden."
        )
        return PromptCache.compose(role, generic, schema)

    def _semi_block(
        self,
        role_key: str,
        persona_name: str,
        persona_voice: str,
        kind: str,
        bloom_card: str,
        diff_card: str,
        kind_card: str,
        kind_constraint: str,
    ) -> str:
        parts = []
        if role_key != "evolver":
            parts.append(f"Persona you must inhabit:\n{persona_name}\n{persona_voice}")
        if kind_constraint:
            parts.append(kind_constraint)
        cards = []
        if bloom_card:
            cards.append(bloom_card)
        if diff_card:
            cards.append(diff_card)
        if kind_card:
            cards.append(kind_card)
        if cards:
            parts.append("\n".join(cards))
        return PromptCache.compose(*parts)

    def _angle_card_block(self, brief: QuestionBrief) -> str:
        extra = []
        # Planner may have stuffed must_have into angle_guide; keep brief fields if present.
        must_have = getattr(brief, "angle_must_have", None)
        must_not = getattr(brief, "angle_must_not", None)
        if isinstance(must_have, list) and must_have:
            extra.append("Must have:\n" + _bullet(must_have))
        if isinstance(must_not, list) and must_not:
            extra.append("Must not:\n" + _bullet(must_not))
        return "\n".join(extra)
