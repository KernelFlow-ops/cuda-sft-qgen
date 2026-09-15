from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.store.jsonl import append_jsonl, read_jsonl, write_jsonl


class Archive:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_path = data_dir / "accepted.jsonl"
        self.rejected_path = data_dir / "rejected.jsonl"
        self.pending_path = data_dir / "pending.jsonl"
        self.stats_path = data_dir / "stats.json"
        self.steering_path = data_dir / "steering.json"
        self.run_path = data_dir / "run_state.json"
        self.retry_path = data_dir / "retry_queue.jsonl"
        self.salvage_raw_path = data_dir / "salvage_raw.jsonl"
        self._lock = threading.Lock()

    def load_questions(self, path: Path) -> list[Question]:
        out: list[Question] = []
        for rec in read_jsonl(path):
            try:
                out.append(Question.model_validate(rec))
            except Exception:
                continue
        return out

    def load_accepted(self) -> list[Question]:
        return self.load_questions(self.accepted_path)

    def load_rejected(self) -> list[Question]:
        return self.load_questions(self.rejected_path)

    def load_pending(self) -> list[Question]:
        return self.load_questions(self.pending_path)

    def append_accepted(self, question: Question) -> None:
        question.review_status = "accepted"
        with self._lock:
            append_jsonl(self.accepted_path, question.to_jsonl())

    def append_rejected(self, question: Question) -> None:
        question.review_status = "rejected"
        with self._lock:
            append_jsonl(self.rejected_path, question.to_jsonl())

    def save_pending(self, questions: list[Question]) -> None:
        with self._lock:
            write_jsonl(self.pending_path, [q.to_jsonl() for q in questions])

    def save_steering(self, brief: SteeringBrief) -> None:
        with self._lock:
            self.steering_path.write_text(
                json.dumps(brief.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_steering(self) -> SteeringBrief:
        if not self.steering_path.exists():
            return SteeringBrief()
        try:
            return SteeringBrief.model_validate(
                json.loads(self.steering_path.read_text(encoding="utf-8"))
            )
        except Exception:
            return SteeringBrief()

    def save_run_state(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.run_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def load_run_state(self) -> dict[str, Any]:
        if not self.run_path.exists():
            return {}
        try:
            return json.loads(self.run_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def sample_accepted(
        self,
        *,
        topic_prefix: list[str] | None = None,
        k: int = 3,
        rng: random.Random | None = None,
    ) -> list[Question]:
        accepted = self.load_accepted()
        if topic_prefix:
            matched = [
                q
                for q in accepted
                if q.topic_path[: len(topic_prefix)] == topic_prefix
            ]
            pool = matched or accepted
        else:
            pool = accepted
        if not pool:
            return []
        rng = rng or random.Random()
        k = min(k, len(pool))
        return rng.sample(pool, k)

    def recent_rejected_skeletons(self, k: int = 8) -> list[str]:
        rejected = self.load_rejected()
        out: list[str] = []
        for q in reversed(rejected):
            if q.reject_reason and q.reject_reason.startswith("L1"):
                if q.template_fingerprint:
                    out.append(q.template_fingerprint)
            if len(out) >= k:
                break
        return out

    def append_salvage_raw(self, record: dict[str, Any]) -> None:
        with self._lock:
            append_jsonl(self.salvage_raw_path, record)

    def load_retry_queue(self) -> list[dict[str, Any]]:
        return list(read_jsonl(self.retry_path))

    def save_retry_queue(self, items: list[dict[str, Any]]) -> None:
        with self._lock:
            write_jsonl(self.retry_path, items)

    def enqueue_retry(self, item: dict[str, Any]) -> None:
        with self._lock:
            items = list(read_jsonl(self.retry_path))
            items.append(item)
            write_jsonl(self.retry_path, items)

    def due_retry_items(self, *, limit: int, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        with self._lock:
            items = list(read_jsonl(self.retry_path))
            take: list[dict[str, Any]] = []
            rest: list[dict[str, Any]] = []
            for item in items:
                if len(take) < limit and float(item.get("next_ok_at") or 0) <= now:
                    take.append(item)
                else:
                    rest.append(item)
            write_jsonl(self.retry_path, rest)
        return take

    def retry_queue_size(self) -> int:
        return sum(1 for _ in read_jsonl(self.retry_path))

    def due_retry_count(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        n = 0
        for x in read_jsonl(self.retry_path):
            if float(x.get("next_ok_at") or 0) <= now:
                n += 1
        return n

    def compute_stats(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        accepted = self.load_accepted()
        rejected = self.load_rejected()
        pending = self.load_pending()
        stats: dict[str, Any] = {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "pending": len(pending),
            "retry_queue": self.retry_queue_size(),
            "l1": dict(Counter(q.topic_path[0] for q in accepted if q.topic_path)),
            "l2": dict(
                Counter(
                    ".".join(q.topic_path[:2])
                    for q in accepted
                    if len(q.topic_path) >= 2
                )
            ),
            "question_type": dict(Counter(q.question_type for q in accepted)),
            "difficulty": dict(Counter(str(q.difficulty) for q in accepted)),
            "hardware": dict(Counter(q.hardware for q in accepted)),
            "granularity": dict(Counter(q.granularity for q in accepted)),
            "source": dict(Counter(q.source for q in accepted)),
            "reject_reason": dict(
                Counter(q.reject_reason or "unknown" for q in rejected)
            ),
        }
        if extra:
            stats.update(extra)
        with self._lock:
            self.stats_path.write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return stats
