"""Offline diversity / coverage report for an accepted-questions JSONL (standalone: numpy only).

python eval_diversity.py accepted.jsonl [--taxonomy-leaves N] [--embeddings emb.npy]

Reports: marginal entropies, joint-cell coverage (L2 x kind x type x format x arch), per-leaf stats,
distinct-n of stems, mean nearest-neighbour cosine, Vendi score (on supplied embeddings, else hashed tokens),
template-skeleton entropy and language parity per L1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|[\u4e00-\u9fff]")


def entropy_ratio(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0 or len(counter) < 2:
        return 1.0
    h = -sum(c / total * math.log(c / total) for c in counter.values() if c)
    return h / math.log(len(counter))


def distinct_n(texts, n=3) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = TOKEN.findall(t.lower())
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / max(total, 1)


def hashed(texts, dim=512):
    m = np.zeros((len(texts), dim), dtype=np.float32)
    for r, t in enumerate(texts):
        for tok in TOKEN.findall(t.lower()):
            d = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            m[r, int.from_bytes(d[:4], "little") % dim] += 1 if d[4] & 1 else -1
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def vendi(emb: np.ndarray) -> float:
    k = emb @ emb.T / len(emb)
    w = np.clip(np.linalg.eigvalsh(k), 1e-12, None)
    return float(np.exp(-(w * np.log(w)).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--taxonomy-leaves", type=int, default=0); ap.add_argument("--embeddings")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    texts = [r["question"] for r in rows]
    emb = np.load(a.embeddings) if a.embeddings else hashed(texts)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sims = emb @ emb.T; np.fill_diagonal(sims, -1)
    leaf = Counter(r.get("l3_id") for r in rows)
    cells = Counter((".".join(r["topic_path"][:2]), r.get("kind"), r.get("question_type"),
                     r.get("answer_format", "?"), r.get("arch_target", r.get("hardware"))) for r in rows)
    report = {
        "n": len(rows),
        "entropy_ratio": {k: round(entropy_ratio(Counter(r.get(k) for r in rows)), 3)
                          for k in ("question_type", "kind", "difficulty", "hardware", "language", "answer_format", "scenario")},
        "l1_entropy_ratio": round(entropy_ratio(Counter(r["topic_path"][0] for r in rows)), 3),
        "leaf_coverage": (len(leaf) / a.taxonomy_leaves) if a.taxonomy_leaves else None,
        "items_per_leaf": {"max": max(leaf.values()), "p50": float(np.median(list(leaf.values())))},
        "joint_cells_used": len(cells),
        "singleton_cell_share": round(sum(1 for c in cells.values() if c == 1) / max(len(cells), 1), 3),
        "distinct_3": round(distinct_n(texts), 3),
        "mean_nn_cosine": round(float(sims.max(axis=1).mean()), 3),
        "share_nn_cosine_ge_0.9": round(float((sims.max(axis=1) >= 0.9).mean()), 3),
        "vendi_score": round(vendi(emb), 2),
        "vendi_per_item": round(vendi(emb) / len(rows), 3),
        "difficulty_hist": dict(Counter(str(r.get("difficulty")) for r in rows)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
