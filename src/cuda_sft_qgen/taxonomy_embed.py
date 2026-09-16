from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cuda_sft_qgen.embeddings import l2_normalize
from cuda_sft_qgen.review.index import VectorIndex
from cuda_sft_qgen.taxonomy import Leaf

logger = logging.getLogger(__name__)


def leaf_embed_text(kind: str, l2_id: str, claim: str) -> str:
    return f"{kind or 'mechanism'} | {l2_id} | {claim.strip()}"


class TaxonomyEmbedIndex:
    def __init__(
        self,
        path: Path,
        embedder: Any,
        cosine_dup: float = 0.92,
        cosine_gray: float = 0.85,
    ) -> None:
        self.path = Path(path)
        self.embedder = embedder
        self.cosine_dup = float(cosine_dup)
        self.cosine_gray = float(cosine_gray)
        self._index = VectorIndex()
        self._loaded = False

    def load(self) -> None:
        try:
            if self.path.exists():
                self._index = VectorIndex.load(self.path)
            else:
                self._index = VectorIndex()
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy embed load failed: %s", exc)
            self._index = VectorIndex()
        self._loaded = True

    def save(self) -> None:
        try:
            self._index.save(self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy embed save failed: %s", exc)

    def ensure_loaded(self) -> bool:
        if not self._loaded:
            self.load()
        return bool(self._index.ids)

    def rebuild(self, leaves: Sequence[Leaf], *, save: bool = True) -> int:
        active = [lf for lf in leaves if not getattr(lf, "retired", False)]
        if self.embedder is None or not active:
            self._index = VectorIndex()
            self._loaded = True
            if save:
                self.save()
            return 0
        texts = [leaf_embed_text(lf.kind, lf.l2_id, lf.claim) for lf in active]
        try:
            mat = self.embedder.embed(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy embed rebuild failed: %s", exc)
            return 0
        self._index = VectorIndex()
        self._index.add([lf.l3_id for lf in active], mat)
        self._loaded = True
        if save:
            self.save()
        return len(active)

    def _embed_texts(self, texts: list[str]) -> np.ndarray | None:
        if self.embedder is None:
            return None
        try:
            mat = self.embedder.embed(texts)
            return l2_normalize(np.asarray(mat, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            logger.warning("taxonomy embed failed: %s", exc)
            return None

    def filter_candidates(
        self, candidates: list[dict]
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return (keep, dropped_dups, gray_pairs).

        keep: unique + gray (max cosine < dup). Gray is not dropped.
        dropped_dups: cosine >= dup vs existing OR vs earlier keep.
        gray_pairs: gray <= cosine < dup, include {candidate, nearest_id, cosine}.
        If index empty/embedder None: keep all, empty dropped/gray.
        """
        if not candidates:
            return [], [], []
        if self.embedder is None:
            return list(candidates), [], []
        texts = [
            leaf_embed_text(
                item.get("kind"),
                item.get("l2_id") or "",
                str(item.get("claim") or ""),
            )
            for item in candidates
        ]
        mat = self._embed_texts(texts)
        if mat is None or mat.shape[0] != len(candidates):
            return list(candidates), [], []

        has_index = self.ensure_loaded()
        if not has_index:
            logger.warning("taxonomy_embed_missing")
        idx = scores = None
        if has_index:
            idx, scores = self._index.search(mat, top_k=1)
        keep: list[dict] = []
        dropped: list[dict] = []
        gray_pairs: list[dict] = []
        keep_vecs: list[np.ndarray] = []
        keep_ids: list[str] = []

        for i, cand in enumerate(candidates):
            best_id: str | None = None
            best_cos = -1.0
            if idx is not None and scores is not None and idx.size:
                j = int(idx[i, 0])
                if j >= 0:
                    best_cos = float(scores[i, 0])
                    best_id = self._index.ids[j]
            if keep_vecs:
                sims = mat[i] @ np.stack(keep_vecs, axis=0).T
                k = int(np.argmax(sims))
                s = float(sims[k])
                if s > best_cos:
                    best_cos = s
                    best_id = keep_ids[k]
            if best_cos >= self.cosine_dup:
                dropped.append(cand)
                continue
            if best_cos >= self.cosine_gray:
                gray_pairs.append(
                    {
                        "candidate": cand,
                        "nearest_id": best_id,
                        "cosine": best_cos,
                    }
                )
            keep.append(cand)
            keep_vecs.append(mat[i])
            keep_ids.append(str(cand.get("l3_id") or cand.get("id") or ""))
        return keep, dropped, gray_pairs

    def add_leaves(self, items: list[dict]) -> None:
        if not items or self.embedder is None:
            return
        if not self._loaded:
            self.load()
        existing = set(self._index.ids)
        fresh: list[dict] = []
        for item in items:
            if item.get("retired"):
                continue
            l3 = str(item.get("l3_id") or item.get("id") or "")
            if not l3 or l3 in existing:
                continue
            fresh.append(item)
            existing.add(l3)
        if not fresh:
            return
        texts = [
            leaf_embed_text(
                item.get("kind"),
                item.get("l2_id") or "",
                str(item.get("claim") or ""),
            )
            for item in fresh
        ]
        mat = self._embed_texts(texts)
        if mat is None or mat.shape[0] != len(fresh):
            return
        self._index.add([str(item.get("l3_id") or item.get("id")) for item in fresh], mat)
