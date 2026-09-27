"""Metrics, comparison and expert-audit sampling for fix/03_TEST_CHAIN.md §3/§5.

  summarize <data_dir> [--wall-seconds S]        -> JSON summary of one run (uses the Archive)
  review <chair_calibration.json>                -> FAR/FRR/pending/defect-recall/stability from the llm test output
  run <baseline_dir> <v2_dir>                    -> table: metric, baseline, v2, threshold, pass
  sample-audit <baseline_export.jsonl> <v2_export.jsonl> --n 120 --out audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

THRESHOLDS = {  # metric: (direction, value); direction 'max' means v2 <= value, 'min' means v2 >= value
    "hard_share": ("min", 0.45), "contract_complete": ("min", 0.95), "mean_nn_cosine": ("max", 0.80),
    "max_cell_share": ("max", 0.03), "generic_share": ("max", 0.35), "answer_format_entropy": ("min", 0.85),
}


def summarize(data_dir: Path, wall: float | None) -> dict:
    from cuda_sft_qgen.store.archive import Archive
    from cuda_sft_qgen.store.jsonl import read_jsonl
    store = Archive(data_dir)
    acc, rej, pen = store.load_accepted(), store.load_rejected(), store.load_pending()
    raw_accepted = read_jsonl(data_dir / "accepted.jsonl")
    has_v2_labels = any("answer_format" in row for row in raw_accepted)
    allq = acc + rej + pen
    codes = Counter()
    for q in rej:
        r = q.reject_reason or ""
        if r.startswith("chair:"):
            codes.update(r.split(":")[1].split(","))
        else:
            codes[r.split(":")[0]] += 1
    div_path = data_dir / "diversity.json"
    div = json.loads(div_path.read_text()) if div_path.exists() else {}
    cells = Counter((".".join(q.topic_path[:2]), q.kind, q.question_type,
                     getattr(q, "answer_format", ""), getattr(q, "arch_target", "any")) for q in acc)
    return {
        "accepted": len(acc), "rejected": len(rej), "pending": len(pen),
        "accept_rate": len(acc) / max(len(acc) + len(rej), 1),
        "hard_share": sum(q.difficulty >= 3 for q in acc) / max(len(acc), 1),
        "contract_complete": (sum(not q.draft_validation_errors for q in allq) / max(len(allq), 1)
                              if has_v2_labels else None),
        "generic_share": (sum(q.scenario == "generic" for q in acc) / max(len(acc), 1)
                          if acc and has_v2_labels else None),
        "max_cell_share": max(cells.values()) / len(acc) if acc else 0.0,
        "mean_nn_cosine": div.get("mean_nn_cosine"),
        "answer_format_entropy": (div.get("entropy_ratio") or {}).get("answer_format"),
        "reject_reasons": dict(codes.most_common()), "wall_seconds": wall,
    }


def review(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    first = [r for r in rows if r["repeat"] == 0]
    rej = [r for r in first if r["expected"] == "reject"]; acc = [r for r in first if r["expected"] == "accept"]
    by = defaultdict(set)
    for r in rows:
        by[r["case"]].add(r["action"])
    hit = [r for r in rej if r["action"] == "reject"]
    return {"FAR": sum(r["action"] == "accept" for r in rej) / max(len(rej), 1),
            "FRR": sum(r["action"] == "reject" for r in acc) / max(len(acc), 1),
            "pending": sum(r["action"] == "pending" for r in acc) / max(len(acc), 1),
            "defect_recall_raw": sum(bool(r["codes"]) for r in hit) / max(len(hit), 1),
            "stability": sum(len(v) == 1 for v in by.values()) / max(len(by), 1)}


def compare(base_dir: Path, v2_dir: Path) -> list[dict]:
    b = json.loads((base_dir / "summary.json").read_text()); v = json.loads((v2_dir / "summary.json").read_text())
    rows = []
    for metric, (direction, thr) in THRESHOLDS.items():
        val = v.get(metric)
        ok = None if val is None else (val <= thr if direction == "max" else val >= thr)
        rows.append({"metric": metric, "baseline": b.get(metric), "v2": val, "threshold": f"{direction} {thr}", "pass": ok})
    ar_b, ar_v = b.get("accept_rate"), v.get("accept_rate")
    rows.append({"metric": "accept_rate", "baseline": ar_b, "v2": ar_v, "threshold": "v2 >= baseline - 0.05",
                 "pass": None if ar_b is None or ar_v is None else ar_v >= ar_b - 0.05})
    return rows


def sample_audit(base: Path, v2: Path, n: int, out: Path, seed: int = 11) -> None:
    rng = random.Random(seed)
    def load(p, tag):
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [(tag, r) for r in rows]
    pool = load(base, "A") + load(v2, "B")
    strata = defaultdict(list)
    for tag, r in pool:
        strata[(tag, (r.get("topic_path") or ["?"])[0], r.get("difficulty"))].append((tag, r))
    picked = []
    for tag in ("A", "B"):
        keys = [k for k in strata if k[0] == tag]
        per = max(1, n // max(len(keys), 1))
        chosen = []
        for k in keys:
            chosen += rng.sample(strata[k], min(per, len(strata[k])))
        rng.shuffle(chosen); picked += chosen[:n]
    rng.shuffle(picked)
    key_rows = []
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "question", "correctness", "unique_answer", "leakage", "metadata_ok",
                    "pedagogical", "discriminative", "realism", "defect_codes", "note"])
        for i, (tag, r) in enumerate(picked):
            aid = f"x{i:04d}"
            w.writerow([aid, r["question"], "", "", "", "", "", "", "", "", ""])
            key_rows.append({"audit_id": aid, "source": tag, "id": r.get("id")})
    out.with_suffix(".key.json").write_text(json.dumps(key_rows, indent=1), encoding="utf-8")   # keep away from reviewers


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize"); s.add_argument("data_dir"); s.add_argument("--wall-seconds", type=float, default=None)
    r = sub.add_parser("review"); r.add_argument("path")
    c = sub.add_parser("run"); c.add_argument("baseline_dir"); c.add_argument("v2_dir")
    a = sub.add_parser("sample-audit"); a.add_argument("baseline"); a.add_argument("v2"); a.add_argument("--n", type=int, default=120); a.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "summarize":
        print(json.dumps(summarize(Path(args.data_dir), args.wall_seconds), ensure_ascii=False, indent=1))
    elif args.cmd == "review":
        print(json.dumps(review(Path(args.path)), indent=1))
    elif args.cmd == "run":
        rows = compare(Path(args.baseline_dir), Path(args.v2_dir))
        print("| metric | baseline | v2 | threshold | pass |\n|---|---|---|---|---|")
        for row in rows:
            print(f"| {row['metric']} | {row['baseline']} | {row['v2']} | {row['threshold']} | {row['pass']} |")
    else:
        sample_audit(Path(args.baseline), Path(args.v2), args.n, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
