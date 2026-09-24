"""Embedding pair mining and CUDA-oriented difficulty signals.

The planner uses this module as a small, dependency-free pair-mining backend.
When an embedding service is available it is used directly; otherwise a stable
hashed token embedding keeps dry-runs and unit tests deterministic.  Candidate
pairs are restricted to compatible taxonomy/hardware neighborhoods before the
top-k cosine search is materialized, which keeps the default work bounded.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|[\u4e00-\u9fff]{2,}")
_CONSTRAINT = re.compile(
    r"\b(?:must|required|requires|constraint|if|when|unless|only|exactly|at least|at most)\b"
    r"|必须|要求|约束|如果|当|除非|仅|恰好|至少|至多",
    re.I,
)
_DEPENDENCY = re.compile(
    r"\b(?:stream\w*|event\w*|barrier|mbarrier|fence|sync\w*|synchron\w*|cooperative|graph|pipeline|wait\w*|ordering)\b"
    r"|依赖|同步|顺序|屏障|流水线|等待",
    re.I,
)
_CONTROL = re.compile(
    r"\b(?:if|else|for|while|switch|branch|diverg|predicate|loop|recursion|case)\b"
    r"|条件|分支|循环|递归|发散",
    re.I,
)
_ARCH = re.compile(
    r"\b(?:sm_?\d+|ampere|hopper|blackwell|a100|h100|b200|cuda\s*\d+(?:\.\d+)?)\b",
    re.I,
)
_CODE = re.compile(r"```|__global__|__device__|<<<|>>>|\b(?:nvcc|ptx|sass)\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class DifficultyFeatures:
    """Explainable complexity features for a CUDA question."""

    constraint_count: int
    dependency_count: int
    control_flow_count: int
    architecture_count: int
    code_signal: int
    numeric_signal: int
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_count": self.constraint_count,
            "dependency_count": self.dependency_count,
            "control_flow_count": self.control_flow_count,
            "architecture_count": self.architecture_count,
            "code_signal": self.code_signal,
            "numeric_signal": self.numeric_signal,
            "score": round(float(self.score), 4),
        }


@dataclass(frozen=True)
class PairCandidate:
    """A compatible, semantically similar pair with different difficulty."""

    first: Any
    second: Any
    similarity: float
    low: Any
    high: Any
    low_features: DifficultyFeatures
    high_features: DifficultyFeatures

    @property
    def pair_id(self) -> str:
        ids = sorted((str(getattr(self.first, "id", "")), str(getattr(self.second, "id", ""))))
        return "pair:" + ":".join(ids)


def _text(item: Any) -> str:
    if isinstance(item, str):
        return item
    return " ".join(
        str(getattr(item, key, "") or "")
        for key in ("question", "tests_claim", "l3_claim")
    ).strip()


def cuda_difficulty_features(item: Any) -> DifficultyFeatures:
    """Extract deterministic complexity signals from a CUDA question."""
    text = _text(item)
    constraints = len(_CONSTRAINT.findall(text))
    dependencies = len(_DEPENDENCY.findall(text))
    controls = len(_CONTROL.findall(text))
    architectures = len(_ARCH.findall(text))
    code_signal = 1 if _CODE.search(text) else 0
    numeric_signal = min(3, len(_NUMBER.findall(text)) // 2)
    # The score is deliberately interpretable and bounded.  It is a prior for
    # pairing only; the Chair remains the source of the final assessed label.
    raw = (
        1.0
        + min(constraints, 6) * 0.18
        + min(dependencies, 5) * 0.24
        + min(controls, 5) * 0.18
        + min(architectures, 3) * 0.25
        + code_signal * 0.28
        + numeric_signal * 0.12
    )
    return DifficultyFeatures(
        constraint_count=constraints,
        dependency_count=dependencies,
        control_flow_count=controls,
        architecture_count=architectures,
        code_signal=code_signal,
        numeric_signal=numeric_signal,
        score=max(1.0, min(4.0, raw)),
    )


def _hashed_embeddings(texts: Sequence[str], dim: int = 256) -> np.ndarray:
    """Stable hashed-token vectors used only when no service is supplied."""
    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _TOKEN.findall(text.lower()) or [text.lower()]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            matrix[row, index] += sign
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _embed(items: Sequence[Any], embedder: Any | Callable[[Sequence[str]], Any] | None) -> np.ndarray:
    texts = [_text(item) for item in items]
    if embedder is None:
        return _hashed_embeddings(texts)
    if hasattr(embedder, "embed"):
        vectors = embedder.embed(texts)
    else:
        vectors = embedder(texts)
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(items):
        raise ValueError("pair embedder must return one vector per item")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _topic_compatible(first: Any, second: Any, same_l2: bool,
                      related_claims: Mapping[str, set[str]] | None = None) -> bool:
    """Only related knowledge may cross an L2 boundary, even when widened."""
    left, right = str(getattr(first, "l3_id", "")), str(getattr(second, "l3_id", ""))
    if left and left == right:
        return True
    a, b = list(getattr(first, "topic_path", ()) or ()), list(getattr(second, "topic_path", ()) or ())
    if len(a) >= 2 and len(b) >= 2 and a[:2] == b[:2]:
        return True
    return not same_l2 and right in (related_claims or {}).get(left, set())


def _version_compatible(first: Any, second: Any) -> bool:
    """Do not combine claims whose explicit verification environments differ."""
    a = getattr(first, "verification_spec", {}) or {}
    b = getattr(second, "verification_spec", {}) or {}
    return all(not a.get(k) or not b.get(k) or a[k] == b[k]
               for k in ("toolkit_version", "architecture"))


def _hardware_compatible(first: Any, second: Any) -> bool:
    left = str(getattr(first, "hardware", "agnostic") or "agnostic").lower()
    right = str(getattr(second, "hardware", "agnostic") or "agnostic").lower()
    return left == right or "agnostic" in {left, right}


def _field_compatible(first: Any, second: Any, field: str) -> bool:
    left = str(getattr(first, field, "") or "").strip().lower()
    right = str(getattr(second, field, "") or "").strip().lower()
    return not left or not right or left == right


def _active_claim(
    item: Any,
    active_claims: Mapping[str, Any] | None,
) -> bool:
    """Reject retired/quarantined/stale parents when taxonomy data is supplied."""
    if active_claims is None:
        return True
    claim_id = str(getattr(item, "l3_id", "") or "")
    active = active_claims.get(claim_id)
    if active is None:
        return False
    if bool(getattr(active, "retired", False)):
        return False
    status = str(getattr(active, "status", "") or "").lower()
    if status in {"retired", "quarantined", "provisional"}:
        return False
    item_revision = getattr(item, "claim_revision", None)
    active_revision = getattr(active, "revision", None)
    return item_revision is None or active_revision is None or int(item_revision) == int(active_revision)


def _lsh_candidate_indices(
    vectors: np.ndarray,
    *,
    tables: int,
    bits: int,
    max_bucket_size: int,
    max_candidates: int,
    seed: int,
) -> list[set[int]]:
    """Build bounded approximate-neighbor candidates with random-hyperplane LSH.

    This deliberately never materializes an ``N x N`` score matrix. Each row
    collects at most ``max_candidates`` ids from exact LSH buckets across a
    fixed number of tables. A small bounded index window supplies candidates
    for sparse buckets; exact cosine ranking happens only inside this set.
    """
    count, dim = vectors.shape
    out = [set() for _ in range(count)]
    rng = np.random.default_rng(seed)
    for _ in range(max(1, int(tables))):
        planes = rng.standard_normal((max(1, int(bits)), dim), dtype=np.float32)
        signatures = np.packbits(vectors @ planes.T >= 0, axis=1)
        buckets: dict[bytes, list[int]] = {}
        for index, signature in enumerate(signatures):
            buckets.setdefault(signature.tobytes(), []).append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for position, index in enumerate(members):
                if len(out[index]) >= max_candidates:
                    continue
                # Rotate per query so a large bucket does not always expose the
                # same prefix, while retaining a strict per-bucket bound.
                limit = min(len(members) - 1, max(1, int(max_bucket_size)))
                for offset in range(1, limit + 1):
                    neighbor = members[(position + offset) % len(members)]
                    if neighbor != index:
                        out[index].add(neighbor)
                    if len(out[index]) >= max_candidates:
                        break
    # Sparse-bucket fallback remains bounded and deterministic. It is an
    # approximate local window, not a dense exact-neighbor pass.
    window = min(max(1, max_candidates), max(0, count - 1))
    for index in range(count):
        offset = 1
        while len(out[index]) < min(window, max_candidates) and offset < count:
            for neighbor in (index - offset, index + offset):
                if 0 <= neighbor < count and neighbor != index:
                    out[index].add(neighbor)
                    if len(out[index]) >= min(window, max_candidates):
                        break
            offset += 1
    return out


def mine_pairs(
    items: Sequence[Any],
    *,
    embedder: Any | Callable[[Sequence[str]], Any] | None = None,
    min_similarity: float = 0.80,
    max_similarity: float = 0.995,
    top_k: int = 8,
    max_pairs: int = 32,
    max_pairs_per_item: int = 1,
    parent_reuse_counts: Mapping[str, int] | None = None,
    max_parent_reuse: int = 3,
    min_difficulty_gap: int = 1,
    same_l2: bool = True,
    related_claims: Mapping[str, set[str]] | None = None,
    same_language: bool = True,
    same_granularity: bool = True,
    active_claims: Mapping[str, Any] | None = None,
    lsh_tables: int = 6,
    lsh_bits: int = 12,
    lsh_max_bucket_size: int = 64,
    lsh_max_candidates: int = 128,
    lsh_seed: int = 17,
) -> list[PairCandidate]:
    """Mine bounded approximate semantic pairs with different difficulty.

    Random-hyperplane LSH provides candidate recall; cosine similarity ranks
    only the bounded candidate set. Compatibility filters run before embedding.
    """
    reuse = parent_reuse_counts or {}
    candidates = [
        item for item in items
        if _active_claim(item, active_claims)
        and int(reuse.get(str(getattr(item, "id", "")), 0)) < max(1, int(max_parent_reuse))
    ]
    if len(candidates) < 2 or max_pairs <= 0:
        return []
    vectors = _embed(candidates, embedder)
    approximate = _lsh_candidate_indices(
        vectors,
        tables=lsh_tables,
        bits=lsh_bits,
        max_bucket_size=lsh_max_bucket_size,
        max_candidates=lsh_max_candidates,
        seed=lsh_seed,
    )
    used: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    pairs: list[PairCandidate] = []
    feature_cache = [cuda_difficulty_features(item) for item in candidates]
    for i in range(len(candidates)):
        if used.get(i, 0) >= max_pairs_per_item:
            continue
        candidate_ids = np.fromiter(approximate[i], dtype=np.int64)
        if candidate_ids.size == 0:
            continue
        candidate_scores = vectors[candidate_ids] @ vectors[i]
        ranked = np.argsort(-candidate_scores)[: max(1, int(top_k))]
        for rank in ranked:
            j = int(candidate_ids[int(rank)])
            key = (min(i, j), max(i, j))
            if key in seen or used.get(j, 0) >= max_pairs_per_item:
                continue
            similarity = float(vectors[i] @ vectors[j])
            if similarity < min_similarity or similarity > max_similarity:
                continue
            first, second = candidates[i], candidates[j]
            first_diff = int(getattr(first, "difficulty", 0) or 0)
            second_diff = int(getattr(second, "difficulty", 0) or 0)
            if abs(first_diff - second_diff) < max(1, int(min_difficulty_gap)):
                continue
            if not _topic_compatible(first, second, same_l2, related_claims) or not _hardware_compatible(first, second) or not _version_compatible(first, second):
                continue
            if same_language and not _field_compatible(first, second, "language"):
                continue
            if same_granularity and not _field_compatible(first, second, "granularity"):
                continue
            if first_diff <= second_diff:
                low_i, high_i = i, j
            else:
                low_i, high_i = j, i
            seen.add(key)
            used[i] = used.get(i, 0) + 1
            used[j] = used.get(j, 0) + 1
            pairs.append(PairCandidate(
                first=first,
                second=second,
                similarity=similarity,
                low=candidates[low_i],
                high=candidates[high_i],
                low_features=feature_cache[low_i],
                high_features=feature_cache[high_i],
            ))
            break
        if len(pairs) >= int(max_pairs):
            break
    return pairs


def mixup_payload(pair: PairCandidate, mode: str) -> dict[str, Any]:
    """Return serializable generation provenance for one pair/mode."""
    normalized = str(mode).strip().lower()
    if normalized not in {"decomposed", "hybrid"}:
        raise ValueError("mixup mode must be decomposed or hybrid")
    low_diff = int(getattr(pair.low, "difficulty", 1) or 1)
    high_diff = int(getattr(pair.high, "difficulty", 4) or 4)
    if normalized == "decomposed":
        target = low_diff + max(1, (high_diff - low_diff) // 2)
        target = min(high_diff, target)
        raw_score = (pair.low_features.score + pair.high_features.score) / 2.0
        policy = (
            f"Keep difficulty between parent levels {low_diff} and {high_diff}; "
            "remove one dependency from the harder parent while preserving both parents' core claims."
        )
    else:
        target = min(4, high_diff + 1)
        raw_score = pair.high_features.score + 0.35
        policy = (
            f"Exceed the harder parent's raw complexity while targeting level {target}. "
            + (
                "The label scale is saturated at level 4: add an orthogonal CUDA constraint, "
                "but do not claim or emit a nonexistent level above 4."
                if high_diff >= 4
                else "Add an orthogonal CUDA constraint that requires both parent claims."
            )
        )
    parent_records = []
    for role, parent, features in (
        ("low", pair.low, pair.low_features),
        ("high", pair.high, pair.high_features),
    ):
        parent_records.append({
            "role": role,
            "id": str(getattr(parent, "id", "")),
            "l3_id": str(getattr(parent, "l3_id", "") or ""),
            "claim": str(
                getattr(parent, "l3_claim", "")
                or getattr(parent, "tests_claim", "")
                or ""
            ),
            "claim_revision": getattr(parent, "claim_revision", None),
            "source_ids": list(getattr(parent, "source_ids", ()) or ()),
            "difficulty": int(getattr(parent, "difficulty", 0) or 0),
            "language": str(getattr(parent, "language", "") or ""),
            "granularity": str(getattr(parent, "granularity", "") or ""),
            "hardware": str(getattr(parent, "hardware", "") or ""),
            "difficulty_features": features.as_dict(),
        })
    return {
        "mode": normalized,
        "pair_id": pair.pair_id,
        "parent_ids": [str(getattr(pair.low, "id", "")), str(getattr(pair.high, "id", ""))],
        "parent_difficulties": [low_diff, high_diff],
        "similarity": round(float(pair.similarity), 5),
        "difficulty_target": target,
        "difficulty_raw_score": round(float(raw_score), 4),
        "difficulty_features": {
            "low": pair.low_features.as_dict(),
            "high": pair.high_features.as_dict(),
        },
        "parents": parent_records,
        "difficulty_policy": policy,
        "monotonicity_rule": "decomposed <= original <= hybrid",
        "ann_method": "random_hyperplane_lsh_bounded_candidates",
    }


def monotonicity_ok(decomposed: float, original: float, hybrid: float) -> bool:
    """Check the curriculum order used by MathMixup-style data."""
    return float(decomposed) <= float(original) <= float(hybrid)


def encoded_seed_id(pair: PairCandidate, mode: str) -> str:
    """Encode pair provenance in the existing Question.seed_id field."""
    payload = mixup_payload(pair, mode)
    return f"mixup:{payload['mode']}:{payload['parent_ids'][0]}:{payload['parent_ids'][1]}"


def env_mixup_enabled(default: bool = False) -> bool:
    value = os.getenv("CUDA_MIXUP_ENABLED")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
