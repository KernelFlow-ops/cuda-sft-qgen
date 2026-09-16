from __future__ import annotations

from dataclasses import dataclass

from cuda_sft_qgen.config import Settings
from cuda_sft_qgen.models import DupDecision, Question
from cuda_sft_qgen.review.fingerprint import Fingerprint, hamming64, token_jaccard


@dataclass
class IndexedRecord:
    id: str
    exact: str
    skeleton: str
    simhash: int


class Deduper:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.by_exact: dict[str, str] = {}
        self.records: list[IndexedRecord] = []

    def add_question(self, q: Question) -> None:
        self.by_exact[q.exact_hash] = q.id
        self.records.append(
            IndexedRecord(
                id=q.id,
                exact=q.exact_hash,
                skeleton=q.template_fingerprint,
                simhash=int(q.simhash),
            )
        )

    def decide(
        self,
        fp: Fingerprint,
        nearest_id: str | None,
        nearest_cosine: float | None,
        *,
        skip_id: str | None = None,
    ) -> DupDecision:
        # L0 exact
        hit = self.by_exact.get(fp.exact)
        if hit and hit != skip_id:
            return DupDecision(
                verdict="dup",
                reason="L0_exact",
                nearest_id=hit,
                nearest_cosine=1.0,
            )

        best_j = -1.0
        best_h = 64
        best_tid: str | None = None
        for rec in self.records:
            if rec.id == skip_id:
                continue
            jac = token_jaccard(fp.skeleton, rec.skeleton)
            ham = hamming64(fp.simhash, rec.simhash)
            if jac > best_j or (jac == best_j and ham < best_h):
                best_j, best_h, best_tid = jac, ham, rec.id

        template_dup = (
            best_tid is not None
            and (
                best_h <= self.settings.simhash_hamming_threshold
                or best_j >= self.settings.template_jaccard_threshold
            )
        )
        template_gray = (
            best_tid is not None
            and not template_dup
            and (
                (
                    self.settings.simhash_hamming_threshold
                    < best_h
                    <= self.settings.simhash_gray_hamming
                    and best_j >= self.settings.template_gray_jaccard
                )
            )
        )

        cosine = nearest_cosine if nearest_cosine is not None else -1.0
        semantic_dup = cosine >= self.settings.semantic_dup_threshold
        semantic_gray = (
            self.settings.semantic_gray_threshold <= cosine < self.settings.semantic_dup_threshold
        )

        if semantic_dup:
            return DupDecision(
                verdict="dup",
                reason="L2_semantic",
                nearest_id=nearest_id,
                nearest_cosine=nearest_cosine,
                template_jaccard=None if best_j < 0 else best_j,
                hamming=None if best_tid is None else best_h,
            )
        if template_dup:
            return DupDecision(
                verdict="dup",
                reason="L1_template",
                nearest_id=best_tid,
                nearest_cosine=nearest_cosine,
                template_jaccard=best_j,
                hamming=best_h,
            )
        if semantic_gray and template_gray:
            # Both in gray zone: escalate to chair, don't auto-reject
            return DupDecision(
                verdict="gray",
                reason="semantic_and_template_gray",
                nearest_id=nearest_id or best_tid,
                nearest_cosine=nearest_cosine,
                template_jaccard=best_j,
                hamming=best_h,
            )
        if semantic_gray or template_gray:
            return DupDecision(
                verdict="gray",
                reason="gray_zone",
                nearest_id=nearest_id or best_tid,
                nearest_cosine=nearest_cosine,
                template_jaccard=None if best_j < 0 else best_j,
                hamming=None if best_tid is None else best_h,
            )
        return DupDecision(
            verdict="unique",
            nearest_id=nearest_id,
            nearest_cosine=nearest_cosine,
            template_jaccard=None if best_j < 0 else best_j,
            hamming=None if best_tid is None else best_h,
        )
