"""In-memory L2-normalized cosine index persisted as compressed npz."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np

from cuda_sft_qgen.embeddings import l2_normalize


class VectorIndex:
    """In-memory cosine index with numpy. Fine up to ~1e5 rows."""

    def __init__(self, dim: int | None = None) -> None:
        """Create an empty index; ``dim`` is filled on the first :meth:`add`."""
        self.ids: list[str] = []
        self._mat: np.ndarray | None = None
        self.dim = dim

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Append L2-normalized rows; ids and vectors must have the same length."""
        if len(ids) == 0:
            return
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if vecs.shape[0] != len(ids):
            raise ValueError("ids/vectors length mismatch")
        if self._mat is None:
            self._mat = vecs
            self.dim = vecs.shape[1]
            self.ids.extend(ids)
            return
        if vecs.shape[1] != self._mat.shape[1]:
            raise ValueError("embedding dimension mismatch")
        self._mat = np.concatenate([self._mat, vecs], axis=0)
        self.ids.extend(ids)

    def vector_for(self, qid: str) -> np.ndarray | None:
        """Return the stored vector for ``qid``, or None if missing."""
        if self._mat is None:
            return None
        try:
            i = self.ids.index(qid)
        except ValueError:
            return None
        return np.asarray(self._mat[i], dtype=np.float32)

    def search(
        self, vectors: np.ndarray, top_k: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) each of shape (n, k). Missing slots are -1 / -inf."""
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        n = vecs.shape[0]
        k = top_k
        if self._mat is None or len(self.ids) == 0:
            idx = np.full((n, k), -1, dtype=np.int64)
            scores = np.full((n, k), -np.inf, dtype=np.float32)
            return idx, scores
        k = min(k, len(self.ids))
        sims = vecs @ self._mat.T
        # argpartition then sort the top-k slice
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(part_scores, order, axis=1)
        if k < top_k:
            pad_i = np.full((n, top_k - k), -1, dtype=np.int64)
            pad_s = np.full((n, top_k - k), -np.inf, dtype=np.float32)
            idx = np.concatenate([idx, pad_i], axis=1)
            scores = np.concatenate([scores, pad_s], axis=1)
        return idx, scores

    def save(self, path: Path) -> None:
        """Atomically write ids + matrix via temp file + os.replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(
                    stream, ids=np.array(self.ids, dtype=object),
                    mat=self._mat if self._mat is not None else np.zeros((0, self.dim or 0)),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Load an index; missing files yield an empty index."""
        idx = cls()
        if not path.exists():
            return idx
        data = np.load(path, allow_pickle=True)
        ids = [str(x) for x in data["ids"].tolist()]
        mat = np.asarray(data["mat"], dtype=np.float32)
        if mat.size == 0:
            return idx
        idx.ids = ids
        idx._mat = mat
        idx.dim = mat.shape[1]
        return idx
