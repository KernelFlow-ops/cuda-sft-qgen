"""Select and format few-shot positives, quality negatives, and anti-templates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from cuda_sft_qgen.models import Question, QuestionBrief

_SEED_PATH = Path(__file__).parent / "fewshot_seed.jsonl"
_NEG_SEED_PATH = Path(__file__).parent / "fewshot_negative.jsonl"
from cuda_sft_qgen.review.fingerprint import hamming64, speech_act

QUALITY_REJECT_PREFIXES = (
    "sieve_",
    "L0_",
    "L1_",
    "L2_",
    "gray_chair",
    "chair:",
    "writer_",
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


def load_negative_seeds() -> list[Question]:
    """Load annotated quality negatives shipped with the package."""
    if not _NEG_SEED_PATH.is_file():
        return []
    out: list[Question] = []
    for line in _NEG_SEED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            rec["reject_reason"] = f"chair:{str(rec.get('defect', 'seed')).split('+')[0]}:seed"
            rec["review_details"] = {"evidence": [rec.get("why", "")]}
            rec.setdefault("dup_verdict", "unique")
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
    snippet_chars_code: int = 1200
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
            snippet_chars_code=int(gen.get("fewshot_snippet_chars_code", 1200)),
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
        return cls(accepted=accepted, rejected_quality=load_negative_seeds() + rejected, anti_skeletons=skeletons)


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
    """Fill same-leaf, kind, format, then domain slots without seed copies."""
    accepted = [q for q in corpus.accepted if _unique_ok(q)]
    same_lang = [q for q in accepted if (q.language or "") == brief.language]
    if same_lang:
        accepted = same_lang
    if seed_simhash is not None:
        accepted = [q for q in accepted if hamming64(q.simhash, seed_simhash) > cfg.simhash_gray_hamming]
    used: set[str] = set()
    picks: list[FewshotPick] = []
    brief_l1 = brief.topic_path[0] if brief.topic_path else ""

    def score(q: Question) -> float:
        return sum(q.value_scores.values()) / len(q.value_scores) if q.value_scores else 3.0

    def take(candidates: Iterable[Question], cap: int, slot: str, note: str) -> None:
        room = min(cap, cfg.positives - len(picks))
        if room > 0:
            picks.extend(_take(candidates, room, used, slot, note))

    take((q for q in reversed(accepted) if q.l3_id == brief.l3_id), 1, "S",
         "already exists on this leaf — do NOT paraphrase")
    kind = brief.kind or "mechanism"
    same_kind = [q for q in accepted if q.l3_id != brief.l3_id and (q.kind or "mechanism") == kind]
    same_kind.sort(key=lambda q: (-score(q), _cosine_rank(q, cfg.max_nn_cosine, cfg.nn_cosine_fallback)))
    take(same_kind, 2, "K", "same kind, different leaf: imitate rigor, not content")
    same_format = [q for q in accepted if q.l3_id != brief.l3_id and q.answer_format
                   and q.answer_format == brief.answer_format]
    same_format.sort(key=lambda q: -score(q))
    take(same_format, 1, "F", "same answer format: imitate option/blank structure")
    take((q for q in reversed(accepted) if _l1(q) == brief_l1), cfg.positives, "D", "same domain")
    return picks


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
    same_bucket = [
        q for q in quality
        if q.l3_id == brief.l3_id
        and q.question_type == brief.question_type
        and (q.kind or "mechanism") == (brief.kind or "mechanism")
        and (q.hardware in {brief.hardware, "agnostic"} or brief.hardware == "agnostic")
    ]
    bucket_ids = {x.id for x in same_bucket}
    same_l3 = [q for q in quality if q.l3_id == brief.l3_id and q.id not in bucket_ids]
    l3_ids = bucket_ids | {x.id for x in same_l3}
    same_type = [q for q in quality if q.question_type == brief.question_type and q.id not in l3_ids]
    ordered = same_bucket + same_l3 + same_type + quality
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
            f"[POS slot={p.slot} type={q.question_type} kind={q.kind or 'mechanism'} "
            f"hardware={q.hardware} l3={q.l3_id} diff={q.difficulty} "
            f"bloom={q.bloom} format={q.answer_format or '-'} nn={nn_s}]"
        )
        lines.append(_snippet(q.question, cfg.snippet_chars_code if "```" in q.question else cfg.snippet_chars))
        if p.note:
            lines.append(f"Why this is good: {p.note}.")
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
        lines.append(_snippet(q.question, cfg.snippet_chars_code if "```" in q.question else cfg.snippet_chars))
        evidence = (q.review_details or {}).get("evidence") or []
        if isinstance(evidence, list) and evidence:
            lines.append("Reviewer finding: " + "; ".join(str(item)[:160] for item in evidence[:2]))
        lines.append(f"Do not repeat this failure mode ({reason}).")
        lines.append("")
    return "\n".join(lines).strip()


def format_anti(items: Sequence[str]) -> str:
    """Render forbidden stems as a bullet list."""
    if not items:
        return "(none)"
    return "\n".join(f"- {s[:240]}" for s in items)
