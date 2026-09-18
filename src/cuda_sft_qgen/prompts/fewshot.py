"""Select and format few-shot positives, quality negatives, and anti-templates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from cuda_sft_qgen.models import Question, QuestionBrief

_SEED_PATH = Path(__file__).parent / "fewshot_seed.jsonl"
from cuda_sft_qgen.review.fingerprint import hamming64, speech_act

QUALITY_REJECT_PREFIXES = (
    "sieve_",
    "L0_",
    "L1_",
    "L2_",
    "gray_chair",
)

TYPE_TO_ACT = {
    "debug": "debug",
    "layout": "layout-eq",
    "compare": "compare",
    "counterfactual": "what-if",
    "profiling": "profile",
    "code_complete": "complete",
    "syntax": "ask",
    "optimization": "how",
    "conceptual": "why",
}


def load_seed_questions() -> list[Question]:
    """Frozen golden stems shipped with the package; used on cold start."""
    if not _SEED_PATH.is_file():
        return []
    out: list[Question] = []
    for line in _SEED_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            rec.setdefault("tests_claim", rec.get("l3_claim") or "seed")
            rec.setdefault("dup_verdict", "unique")
            rec.setdefault("warnings", [])
            out.append(Question.model_validate(rec))
        except Exception:
            continue
    return out


def _merge_seed_questions(accepted: list[Question]) -> list[Question]:
    """Prepend seed items whose question text is not already in the archive."""
    seen = {q.question.strip() for q in accepted if (q.question or "").strip()}
    seeds = [q for q in load_seed_questions() if q.question.strip() not in seen]
    return seeds + accepted


def is_quality_reject(reason: str | None) -> bool:
    """True for sieve/dedup rejects; false for generate errors and 429 exhaust."""
    text = reason or ""
    if not text or text.startswith("generate_error") or text.startswith("retry_exhausted"):
        return False
    return any(text.startswith(p) for p in QUALITY_REJECT_PREFIXES)


def _l1(q: Question) -> str:
    return q.topic_path[0] if q.topic_path else ""


def _snippet(text: str, limit: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= limit:
        return raw
    fence = raw.find("```")
    if 0 <= fence < limit:
        # Keep the stem plus a truncated first fence.
        return raw[:limit].rstrip() + "…"
    return raw[:limit].rstrip() + "…"


def _unique_ok(q: Question) -> bool:
    if q.warnings:
        return False
    if q.dup_verdict and q.dup_verdict != "unique":
        return False
    return bool((q.question or "").strip())


@dataclass
class FewshotCfg:
    """Caps and similarity bands for few-shot selection."""
    positives: int = 5
    negatives: int = 2
    anti_max: int = 15
    snippet_chars: int = 500
    max_nn_cosine: float = 0.50
    nn_cosine_fallback: float = 0.80
    simhash_hamming_max: int = 20
    simhash_gray_hamming: int = 8

    @classmethod
    def from_generation(cls, gen: dict[str, Any], *, gray_hamming: int = 8) -> "FewshotCfg":
        """Build a config from the ``generation`` YAML mapping."""
        return cls(
            positives=int(gen.get("fewshot_positives", 5)),
            negatives=int(gen.get("fewshot_negatives", 2)),
            anti_max=int(gen.get("anti_templates_max", 15)),
            snippet_chars=int(gen.get("fewshot_snippet_chars", 500)),
            max_nn_cosine=float(gen.get("fewshot_max_nn_cosine", 0.50)),
            nn_cosine_fallback=float(gen.get("fewshot_nn_cosine_fallback", 0.80)),
            simhash_hamming_max=int(gen.get("fewshot_simhash_hamming_max", 20)),
            simhash_gray_hamming=int(gray_hamming),
        )


@dataclass
class FewshotPick:
    """One example plus the slot it occupies (same-leaf, same-L1, ...)."""
    question: Question
    slot: str
    note: str = ""


@dataclass
class FewshotCorpus:
    """Accepted/rejected pools used to pick positives, negatives, and anti-stems."""
    accepted: list[Question] = field(default_factory=list)
    rejected_quality: list[Question] = field(default_factory=list)
    anti_skeletons: list[str] = field(default_factory=list)

    @classmethod
    def from_store(cls, store: Any, *, anti_k: int = 15) -> "FewshotCorpus":
        """Build a corpus from the archive's accepted, quality-rejected, and anti stems."""
        accepted = _merge_seed_questions(list(store.load_accepted()))
        rejected = []
        if hasattr(store, "iter_quality_rejected"):
            rejected = list(store.iter_quality_rejected(limit=200))
        else:
            for q in reversed(store.load_rejected()):
                if (q.question or "").strip() and is_quality_reject(q.reject_reason):
                    rejected.append(q)
                if len(rejected) >= 200:
                    break
        skeletons = []
        if hasattr(store, "recent_rejected_skeletons"):
            skeletons = list(store.recent_rejected_skeletons(anti_k))
        return cls(accepted=accepted, rejected_quality=rejected, anti_skeletons=skeletons)


def _cosine_rank(q: Question, prefer: float, fallback: float) -> tuple[int, float, int]:
    """Lower tuple sorts better: preferred band first, then lower cosine, then recency proxy."""
    cos = q.nearest_cosine
    if cos is None:
        band = 0
        key = -1.0
    elif cos < prefer:
        band = 0
        key = float(cos)
    elif cos < fallback:
        band = 1
        key = float(cos)
    else:
        band = 2
        key = float(cos)
    return (band, key, 0)


def _take(
    cands: Iterable[Question],
    n: int,
    used: set[str],
    slot: str,
    note: str,
) -> list[FewshotPick]:
    out: list[FewshotPick] = []
    for q in cands:
        if not q.id or q.id in used:
            continue
        if not (q.question or "").strip():
            continue
        used.add(q.id)
        out.append(FewshotPick(question=q, slot=slot, note=note))
        if len(out) >= n:
            break
    return out


def select_positives(
    brief: QuestionBrief,
    corpus: FewshotCorpus,
    cfg: FewshotCfg,
    *,
    seed_simhash: int | None = None,
) -> list[FewshotPick]:
    """Pick accepted examples near the brief's leaf/L1 without near-duplicates of the seed."""
    accepted = [q for q in corpus.accepted if _unique_ok(q)]
    same_lang = [q for q in accepted if (q.language or "") == brief.language]
    if same_lang:
        accepted = same_lang
    used: set[str] = set()
    picks: list[FewshotPick] = []
    brief_l1 = brief.topic_path[0] if brief.topic_path else ""
    kind = brief.kind or ""

    def recency_first(items: Sequence[Question]) -> list[Question]:
        return list(reversed(items))

    same_type = [
        q
        for q in recency_first(accepted)
        if q.question_type == brief.question_type and q.l3_id != brief.l3_id
    ]
    if kind:
        same_type.sort(
            key=lambda q: (
                0 if (q.kind or "mechanism") == kind else 1,
                *_cosine_rank(q, cfg.max_nn_cosine, cfg.nn_cosine_fallback),
            )
        )
    else:
        same_type.sort(key=lambda q: _cosine_rank(q, cfg.max_nn_cosine, cfg.nn_cosine_fallback))
    picks.extend(_take(same_type, 2, used, "A", "same question_type"))

    same_l3 = [
        q
        for q in recency_first(accepted)
        if q.l3_id == brief.l3_id and q.question_type != brief.question_type
    ]
    picks.extend(_take(same_l3, 1, used, "B", "same l3, different type"))

    same_diff_bloom = [
        q
        for q in recency_first(accepted)
        if int(q.difficulty) == int(brief.difficulty) and q.bloom == brief.bloom
    ]
    same_diff_bloom.sort(key=lambda q: _cosine_rank(q, cfg.max_nn_cosine, cfg.nn_cosine_fallback))
    picks.extend(_take(same_diff_bloom, 1, used, "C", "same difficulty+bloom"))

    anchor_hash = seed_simhash
    if anchor_hash is None and picks:
        anchor_hash = int(picks[0].question.simhash)
    want_act = TYPE_TO_ACT.get(brief.question_type, "ask")
    cross: list[Question] = []
    if anchor_hash is not None:
        scored: list[tuple[int, Question]] = []
        for q in accepted:
            if q.id in used:
                continue
            if _l1(q) == brief_l1 or not _l1(q):
                continue
            ham = hamming64(int(q.simhash), int(anchor_hash))
            if cfg.simhash_gray_hamming < ham <= cfg.simhash_hamming_max:
                scored.append((ham, q))
        scored.sort(key=lambda t: t[0])
        cross = [q for _, q in scored]
    if not cross:
        for q in recency_first(accepted):
            if q.id in used:
                continue
            if _l1(q) == brief_l1:
                continue
            skel = q.template_fingerprint or ""
            act = speech_act(q.question or "")
            if skel.startswith(f"[{want_act}]") or act == want_act:
                cross.append(q)
            elif q.granularity == brief.granularity:
                cross.append(q)
    picks.extend(_take(cross, 1, used, "D", "cross-L1 similar structure"))

    if len(picks) < cfg.positives:
        fill = recency_first(accepted)
        picks.extend(_take(fill, cfg.positives - len(picks), used, "F", "recency fill"))
    return picks[: cfg.positives]


def select_negatives(
    brief: QuestionBrief,
    corpus: FewshotCorpus,
    cfg: FewshotCfg,
    used_ids: set[str] | None = None,
) -> list[FewshotPick]:
    """Pick quality-rejected examples as negatives, preferring the same leaf/type."""
    used = set(used_ids or ())
    quality = [
        q
        for q in corpus.rejected_quality
        if (q.question or "").strip() and is_quality_reject(q.reject_reason)
    ]
    same_l3 = [q for q in quality if q.l3_id == brief.l3_id]
    same_type = [q for q in quality if q.question_type == brief.question_type]
    ordered = same_l3 + [q for q in same_type if q.id not in {x.id for x in same_l3}] + quality
    return _take(ordered, cfg.negatives, used, "N", "quality reject")


def select_anti(
    brief: QuestionBrief,
    corpus: FewshotCorpus,
    cfg: FewshotCfg,
    static_patterns: Sequence[str],
) -> list[str]:
    """Merge static anti-patterns, brief stems, and recent rejected skeletons."""
    out: list[str] = []
    seen: set[str] = set()
    for item in list(static_patterns) + list(brief.anti_templates) + list(corpus.anti_skeletons):
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= cfg.anti_max:
            break
    return out


def format_positives(picks: Sequence[FewshotPick], cfg: FewshotCfg) -> str:
    """Render positive few-shots as a numbered prompt block."""
    if not picks:
        return "(none yet)"
    lines = []
    for p in picks:
        q = p.question
        nn = q.nearest_cosine
        nn_s = "null" if nn is None else f"{nn:.2f}"
        lines.append(
            f"[POS slot={p.slot} type={q.question_type} l3={q.l3_id} "
            f"diff={q.difficulty} bloom={q.bloom} nn={nn_s}]"
        )
        lines.append(_snippet(q.question, cfg.snippet_chars))
        if p.note:
            lines.append(f"Why this pattern: {p.note}.")
        lines.append("")
    return "\n".join(lines).strip()


def format_negatives(picks: Sequence[FewshotPick], cfg: FewshotCfg) -> str:
    """Render quality-reject few-shots as a numbered prompt block."""
    if not picks:
        return "(none yet)"
    lines = []
    for p in picks:
        q = p.question
        reason = q.reject_reason or "unknown"
        lines.append(f"[NEG l3={q.l3_id} type={q.question_type} reason={reason}]")
        lines.append(_snippet(q.question, cfg.snippet_chars))
        lines.append(f"Do not repeat this failure mode ({reason}).")
        lines.append("")
    return "\n".join(lines).strip()


def format_anti(items: Sequence[str]) -> str:
    """Render forbidden stems as a bullet list."""
    if not items:
        return "(none)"
    return "\n".join(f"- {s[:240]}" for s in items)
