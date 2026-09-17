#!/usr/bin/env python3
"""Live dashboard for a cuda-sft-qgen generate run.

Reads data-dir artifacts + the pipeline log and writes agent_status.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


AGENTS = (
    "bootstrap",
    "architect",
    "planner",
    "writer",
    "salvage",
    "retry_drain",
    "review_score",
    "review_chair",
    "archive",
    "taxonomy_manager",
)

NODE_PATTERNS = {
    "bootstrap": re.compile(r"\bbootstrap\b"),
    "architect": re.compile(r"\barchitect\b"),
    "planner": re.compile(r"\bplanner\b"),
    "writer": re.compile(r"\bgenerate_batch\b|\bgeneration\b"),
    "salvage": re.compile(r"\bsalvage\b"),
    "retry_drain": re.compile(r"\bretry_drain\b"),
    "review_score": re.compile(r"\breview_score\b"),
    "review_chair": re.compile(r"\bchair\b"),
    "archive": re.compile(r"\barchive\b"),
    "taxonomy_manager": re.compile(r"\btaxonomy_(?:manager|health)\b"),
}

ERROR_RE = re.compile(r"\b(ERROR|CRITICAL|Traceback|GraphRecursionError)\b")
WARN_RE = re.compile(r"\bWARNING\b")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _tail(path: Path, n: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    lines = data.splitlines()
    return lines[-n:]


def _reject_reasons(path: Path, limit: int = 12) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            reason = rec.get("reject_reason") or "unknown"
            reason = str(reason).split(":")[0][:80]
            counts[reason] += 1
    return dict(counts.most_common(limit))


def snapshot(data_dir: Path, log_path: Path, pid: int | None, target: int) -> dict:
    """Collect counts, log tail, and inferred current agent from a data dir."""
    run = _read_json(data_dir / "run_state.json")
    stats = _read_json(data_dir / "stats.json")
    health = _read_json(data_dir / "taxonomy_health.json")
    accepted = _count_lines(data_dir / "accepted.jsonl")
    rejected = _count_lines(data_dir / "rejected.jsonl")
    pending = _count_lines(data_dir / "pending.jsonl")
    retry = _count_lines(data_dir / "retry_queue.jsonl")
    salvage = _count_lines(data_dir / "salvage_raw.jsonl")

    tail = _tail(log_path, 200)
    last_seen: dict[str, str] = {name: "idle" for name in AGENTS}
    last_line: dict[str, str] = {name: "" for name in AGENTS}
    errors: list[str] = []
    warnings: list[str] = []
    for line in tail:
        if ERROR_RE.search(line):
            errors.append(line[-300:])
        if WARN_RE.search(line):
            warnings.append(line[-300:])
        for name, pat in NODE_PATTERNS.items():
            if pat.search(line):
                last_seen[name] = "active"
                last_line[name] = line[-240:]

    # Most recently mentioned node in the tail is current.
    current = "unknown"
    for line in reversed(tail):
        for name, pat in NODE_PATTERNS.items():
            if pat.search(line):
                current = name
                break
        if current != "unknown":
            break

    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False

    accepted_n = int(run.get("accepted_count") or accepted or 0)
    iteration = int(run.get("iteration") or 0)
    review_rounds = int(run.get("review_rounds") or 0)
    rate = (accepted_n / (accepted_n + rejected)) if (accepted_n + rejected) else 0.0
    now = time.time()
    last_data = max(
        [
            t
            for t in (
                _mtime(data_dir / "accepted.jsonl"),
                _mtime(data_dir / "pending.jsonl"),
                _mtime(data_dir / "run_state.json"),
                _mtime(log_path),
            )
            if t
        ]
        or [now]
    )
    stall_sec = now - last_data

    agents = []
    for name in AGENTS:
        agents.append(
            {
                "name": name,
                "status": last_seen[name] if alive else ("done" if accepted_n >= target else last_seen[name]),
                "last_log": last_line[name],
            }
        )

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "alive": alive,
        "pid": pid,
        "current_agent": current,
        "target": target,
        "accepted": accepted_n,
        "rejected": rejected,
        "pending": pending,
        "retry_queue": retry,
        "salvage_raw": salvage,
        "iteration": iteration,
        "review_rounds": review_rounds,
        "accept_rate": round(rate, 4),
        "progress_pct": round(100.0 * accepted_n / target, 2) if target else 0.0,
        "stall_sec": round(stall_sec, 1),
        "reject_reasons": _reject_reasons(data_dir / "rejected.jsonl"),
        "taxonomy_health": {
            "overall_health": health.get("overall_health") or health.get("health_score"),
            "coverage": health.get("overall_coverage") or (health.get("coverage") if isinstance(health, dict) else None),
        },
        "stats_question_type": (stats.get("question_type") or {}) if stats else {},
        "agents": agents,
        "recent_errors": errors[-8:],
        "recent_warnings": warnings[-8:],
        "log_tail": tail[-12:],
    }


def render(snap: dict) -> str:
    """Format a snapshot as a fixed-width terminal dashboard."""
    lines = [
        "=" * 78,
        f"cuda-sft-qgen  {snap['ts']}  pid={snap['pid']} alive={snap['alive']}",
        f"progress  {snap['accepted']}/{snap['target']} ({snap['progress_pct']}%)  "
        f"rejected={snap['rejected']} pending={snap['pending']} retry={snap['retry_queue']} "
        f"salvage={snap['salvage_raw']}",
        f"iteration={snap['iteration']} review_rounds={snap['review_rounds']} "
        f"accept_rate={snap['accept_rate']} stall={snap['stall_sec']}s current={snap['current_agent']}",
        "-" * 78,
    ]
    for a in snap["agents"]:
        marker = "*" if a["name"] == snap["current_agent"] else " "
        lines.append(f"{marker} {a['name']:<18} {a['status']:<8} {a['last_log'][:90]}")
    if snap["reject_reasons"]:
        lines.append("-" * 78)
        lines.append("reject_reasons: " + ", ".join(f"{k}={v}" for k, v in snap["reject_reasons"].items()))
    if snap["recent_errors"]:
        lines.append("-" * 78)
        lines.append("errors:")
        lines.extend("  " + e for e in snap["recent_errors"][-4:])
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    """Poll a running generate job and print/write agent_status.json."""
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--pid", type=int, default=None)
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--interval", type=float, default=20)
    p.add_argument("--once", action="store_true")
    p.add_argument("--status-out", default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    log_path = Path(args.log)
    status_out = Path(args.status_out) if args.status_out else data_dir / "agent_status.json"

    while True:
        snap = snapshot(data_dir, log_path, args.pid, args.target)
        status_out.parent.mkdir(parents=True, exist_ok=True)
        status_out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        print(render(snap), flush=True)
        if args.once:
            return 0
        if not snap["alive"] and snap["accepted"] >= args.target:
            print("DONE target reached", flush=True)
            return 0
        if not snap["alive"] and snap["pid"]:
            print("PIPELINE_EXITED", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
