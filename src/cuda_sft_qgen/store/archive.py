"""SQLite-backed question archive with JSONL compatibility projections.

SQLite is authoritative. JSONL sidecars are rebuilt after each transactional
commit so older tools can still read accepted/rejected/pending files.
"""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from cuda_sft_qgen.models import Question, SteeringBrief
from cuda_sft_qgen.store.jsonl import append_jsonl, read_jsonl, write_jsonl


def is_rate_limit_reject_reason(reason: str | None) -> bool:
    """True when a reject reason is a 429 / overload rather than a quality fail."""
    text = reason or ""
    lowered = text.lower()
    return (
        "429" in text
        or "too many requests" in lowered
        or "rate limit" in lowered
        or "temporarily overloaded" in lowered
    )


class Archive:
    """Durable store for questions, retries, generation tasks, and steering.

    All mutating methods take ``_lock``. Review decisions commit in one SQLite
    transaction; JSONL export may be rebuilt afterward without losing the commit.
    """

    def __init__(self, data_dir: Path) -> None:
        """Open or create ``data_dir/questions.sqlite3`` and migrate legacy JSONL."""
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
        self._lock = threading.RLock()
        self._accepted: list[Question] | None = None
        self._rejected: list[Question] | None = None
        self._pending: list[Question] | None = None
        self.database_path = data_dir / "questions.sqlite3"
        self._db = sqlite3.connect(self.database_path, timeout=30, check_same_thread=False)
        self.database_path.chmod(0o600)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS questions_status ON questions(status);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS retry_tasks (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, due_at REAL NOT NULL,
                lease_until REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS corrections (
                event_id TEXT NOT NULL, question_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(event_id, question_id)
            );
            CREATE TABLE IF NOT EXISTS export_members (
                export_id TEXT NOT NULL, question_id TEXT NOT NULL,
                PRIMARY KEY(export_id, question_id)
            );
            CREATE TABLE IF NOT EXISTS generation_tasks (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self._migrate_legacy()

    def _migrate_legacy(self) -> None:
        """One-shot import of pre-SQLite JSONL into the database."""
        with self._lock, self._db:
            if self._db.execute("SELECT 1 FROM metadata WHERE key='migrated'").fetchone():
                return
            for path in (self.accepted_path, self.rejected_path, self.pending_path,
                         self.retry_path, self.run_path, self.steering_path):
                backup = path.with_name(path.name + ".legacy")
                if path.exists() and not backup.exists():
                    shutil.copy2(path, backup)
            # A terminal record wins over a stale pending record after an old crash.
            for status, path in (("accepted", self.accepted_path),
                                 ("rejected", self.rejected_path),
                                 ("pending", self.pending_path)):
                for rec in read_jsonl(path):
                    q = Question.model_validate(rec)
                    q.review_status = status
                    self._db.execute(
                        "INSERT OR IGNORE INTO questions VALUES (?,?,?,?)",
                        (q.id, status, q.model_dump_json(), time.time()),
                    )
            for key, path in (("run", self.run_path), ("steering", self.steering_path)):
                if path.exists():
                    self._set_meta(key, json.loads(path.read_text(encoding="utf-8")))
            for item in read_jsonl(self.retry_path):
                self._put_retry(item)
            self._set_meta("migrated", True)

    def _set_meta(self, key: str, value: Any) -> None:
        self._db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                         (key, json.dumps(value, ensure_ascii=False)))

    def _get_meta(self, key: str, default: Any) -> Any:
        with self._lock:
            row = self._db.execute("SELECT payload FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _load_status(self, status: str) -> list[Question]:
        with self._lock:
            rows = self._db.execute(
                "SELECT payload FROM questions WHERE status=? ORDER BY rowid", (status,)
            ).fetchall()
        return [Question.model_validate_json(row[0]) for row in rows]

    def _put_question(self, question: Question, status: str, *, force: bool = False) -> None:
        existing = self._db.execute("SELECT status FROM questions WHERE id=?", (question.id,)).fetchone()
        if existing and existing[0] != "pending" and not force:
            return
        question.review_status = status
        self._db.execute("INSERT INTO questions VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE "
                         "SET status=excluded.status,payload=excluded.payload,updated_at=excluded.updated_at",
                         (question.id, status, question.model_dump_json(), time.time()))

    def _export_status(self, status: str) -> None:
        path = {"accepted": self.accepted_path, "rejected": self.rejected_path,
                "pending": self.pending_path}[status]
        write_jsonl(path, [q.to_jsonl() for q in self._load_status(status)])

    def export_compatibility(self) -> None:
        """Rewrite JSONL/JSON sidecars from the current SQLite state."""
        with self._lock:
            for status in ("accepted", "rejected", "pending"):
                self._export_status(status)
            self._export_retries()
            self.run_path.write_text(json.dumps(self.load_run_state(), indent=2), encoding="utf-8")
            self.steering_path.write_text(self.load_steering().model_dump_json(indent=2), encoding="utf-8")

    def commit_review(self, decisions: list[dict], steering: SteeringBrief | dict,
                      run_state: dict[str, Any]) -> dict[str, Any]:
        """Atomically apply chair accept/reject/pending decisions and persist steering."""
        with self._lock:
            with self._db:
                for item in decisions:
                    q = Question.model_validate(item["question"])
                    action = item["action"]
                    if action not in {"accept", "reject", "pending"}:
                        raise ValueError(f"Invalid review action: {action}")
                    q.reject_reason = item.get("reason") if action != "accept" else None
                    self._put_question(q, {"accept": "accepted", "reject": "rejected",
                                           "pending": "pending"}[action])
                counts = dict(self._db.execute("SELECT status,count(*) FROM questions GROUP BY status"))
                state = {**run_state, "accepted_count": counts.get("accepted", 0)}
                self._set_meta("run", state)
                self._set_meta("steering", steering.model_dump() if isinstance(steering, SteeringBrief) else steering)
            # Projections may be rebuilt after a failure; the transaction above is authoritative.
            self.export_compatibility()
            return {"accepted_count": counts.get("accepted", 0),
                    "rejected_count": counts.get("rejected", 0),
                    "pending": [q.model_dump() for q in self.load_pending()]}

    def invalidate_claim(self, claim_id: str, revision: int | str | None = None,
                         reason: str = "knowledge_changed", event_id: Any = None,
                         question_ids: list[str] | None = None) -> int:
        """Move accepted questions on a stale claim back to pending for re-review."""
        event_id = str(event_id or f"{claim_id}:{revision}:{reason}")
        changed = 0
        with self._lock:
            with self._db:
                for q in self.load_accepted():
                    if question_ids is not None:
                        match = q.id in question_ids
                    else:
                        match = q.l3_id == claim_id and (revision is None or
                            str(getattr(q, "claim_revision", None)) == str(revision))
                    if not match:
                        continue
                    correction = {"event_id": event_id, "question_id": q.id,
                                  "claim_id": claim_id, "revision": revision, "reason": reason}
                    self._db.execute("INSERT OR IGNORE INTO corrections VALUES (?,?,?)",
                                     (event_id, q.id, json.dumps(correction)))
                    q.warnings = list(dict.fromkeys([*q.warnings, "knowledge_reaudit_required"]))
                    if hasattr(q, "review_details"):
                        q.review_details = {"action": "pending", "reason": reason}
                        q.internal_reference_answer = None
                        q.verification_evidence = {}
                    for name, value in (("review_attempts", 0), ("review_next_at", 0.0),
                                        ("review_exhausted", False)):
                        if hasattr(q, name):
                            setattr(q, name, value)
                    self._put_question(q, "pending", force=True)
                    changed += 1
            if changed:
                self.export_compatibility()
                write_jsonl(self.data_dir / "corrections.jsonl", self.corrections())
        return changed

    def corrections(self) -> list[dict[str, Any]]:
        """Knowledge-invalidation events applied to archived questions."""
        with self._lock:
            rows = self._db.execute("SELECT payload FROM corrections ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def _exportable_questions(self) -> tuple[list[Question], int]:
        """Revalidate every accepted row through the same gate for all exports."""
        from cuda_sft_qgen.models import QualityReview

        rows: list[Question] = []
        skipped = 0
        for q in self.load_accepted():
            details = dict(getattr(q, "review_details", {}) or {})
            if details.get("reviewer_model") in {None, "", "fake"}:
                skipped += 1
                continue
            # Re-validate the persisted review instead of trusting its action flag.
            # This protects exports from legacy rows and hand-edited JSONL imports.
            try:
                review_fields = set(QualityReview.model_fields)
                review = QualityReview.model_validate({
                    key: value for key, value in details.items() if key in review_fields
                })
                if review.id != q.id or review.action != "accept":
                    raise ValueError("review identity/action mismatch")
                if not details.get("reviewer_independent", True):
                    raise ValueError("review is not independently produced")
                evidence = q.verification_evidence or details.get("verification_evidence") or {}
                if q.draft_validation_errors:
                    raise ValueError("invalid generation contract")
                if (q.verification_spec or q.draft_plan or q.generation_provenance.get("structured_cot")) and evidence.get("status") != "pass":
                    raise ValueError("structured question requires independent verification")
                if evidence and evidence.get("status") not in {"pass", "disabled"}:
                    raise ValueError("verification not passed")
                rows.append(q)
            except Exception:
                skipped += 1
        return rows, skipped

    def export_training(self, path: Path) -> dict[str, Any]:
        """Write accepted, independently reviewed questions for SFT (no private answers)."""
        from cuda_sft_qgen.models import QualityReview

        path = Path(path)
        if path.resolve() in {self.accepted_path.resolve(), self.rejected_path.resolve(), self.pending_path.resolve(), self.database_path.resolve()}:
            raise ValueError("Training export must not replace the archive's compatibility files")
        rows, skipped = self._exportable_questions()
        export_id = uuid.uuid4().hex
        write_jsonl(path, [q.to_jsonl() for q in rows])
        with self._lock, self._db:
            self._db.executemany("INSERT INTO export_members VALUES (?,?)", [(export_id, q.id) for q in rows])
        manifest = {"export_id": export_id, "questions": len(rows), "skipped_unverified": skipped,
                    "path": str(path),
                    "created_at": time.time(), "corrections": self.corrections()}
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        self.export_curriculum_manifest(path.with_suffix(path.suffix + ".curriculum.json"), member_ids={q.id for q in rows})
        return manifest

    def export_curriculum_manifest(self, path: Path, *, member_ids: set[str] | None = None) -> dict[str, Any]:
        """Write a mode/difficulty curriculum index beside an SFT export.

        ``mixup_mode`` is intentionally orthogonal to assessed difficulty.  A
        training consumer can therefore follow decomposed → original → hybrid
        while still balancing CUDA difficulty, hardware, and taxonomy coverage.
        """
        rows, skipped = self._exportable_questions()
        if member_ids is not None:
            rows = [q for q in rows if q.id in member_ids]
        stages = {"decomposed": 0, "original": 0, "hybrid": 0, "unmarked": 0}
        buckets: dict[str, int] = {}
        members: list[dict[str, Any]] = []
        # Union parent edges before assigning a deterministic split. Otherwise
        # a hybrid can leak either training parent into validation/test.
        import hashlib
        roots: dict[str, str] = {}
        def root(qid: str) -> str:
            roots.setdefault(qid, qid)
            if roots[qid] != qid:
                roots[qid] = root(roots[qid])
            return roots[qid]
        for q in rows:
            for parent in q.parent_ids:
                left, right = root(q.id), root(parent)
                roots[max(left, right)] = min(left, right)
        for question in rows:
            mode = str(getattr(question, "mixup_mode", None) or "original")
            if mode not in stages:
                mode = "unmarked"
            stages[mode] += 1
            difficulty = int(getattr(question, "difficulty", 0) or 0)
            key = "|".join((mode, str(difficulty), ".".join(question.topic_path[:2]), question.kind, question.hardware, question.language))
            buckets[key] = buckets.get(key, 0) + 1
            family = root(question.id)
            fold = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
            split = "train" if fold < 80 else "validation" if fold < 90 else "test"
            members.append({"id": question.id, "stage": mode, "difficulty": difficulty, "bucket": key,
                            "family_id": family, "split": split, "parent_ids": question.parent_ids,
                            "requested_difficulty": question.requested_metadata.get("difficulty"),
                            "difficulty_raw_score": question.difficulty_raw_score})
        # Produce a reproducible stratified audit queue; generating this queue
        # is not a claim that a human has reviewed its members. Small buckets
        # retain at least one sample so rare CUDA subjects are represented.
        audit_ids = []
        for bucket in sorted(buckets):
            group = sorted((m["id"] for m in members if m["bucket"] == bucket),
                           key=lambda qid: hashlib.sha256(qid.encode()).hexdigest())
            audit_ids.extend(group[:max(1, (len(group) + 9) // 10)])
        manifest = {
            "created_at": time.time(),
            "questions": len(rows),
            "stage_order": ["decomposed", "original", "hybrid"],
            "skipped_unverified": skipped,
            "split_policy": "parent-family hash 80/10/10; split before further synthesis; never use held-out seeds",
            "large_family_warning": max(Counter(m["family_id"] for m in members).values(), default=0) > max(1, len(rows) // 2),
            "stages": stages,
            "buckets": buckets,
            "members": members,
            "human_audit": {"status": "not_performed", "fraction": 0.10, "question_ids": audit_ids},
            "path": str(path),
        }
        path = Path(path)
        if path.resolve() in {p.resolve() for p in (self.accepted_path, self.rejected_path, self.pending_path, self.database_path)}:
            raise ValueError("Curriculum manifest must not replace the archive")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        path.chmod(0o600)
        return manifest

    def requeue_reviews(self, *, include_exhausted: bool = False) -> int:
        """Reset pending review backoff so chair can retry deferred items."""
        count = 0
        with self._lock:
            with self._db:
                for question in self.load_pending():
                    if question.review_exhausted and not include_exhausted:
                        continue
                    question.review_next_at = 0.0
                    if include_exhausted:
                        question.review_attempts = 0
                        question.review_exhausted = False
                    self._put_question(question, "pending")
                    count += 1
            self._export_status("pending")
        return count

    def close(self) -> None:
        """Flush compatibility files and close the SQLite connection."""
        with self._lock:
            self._db.close()

    def save_generation_task(self, brief: dict[str, Any]) -> str:
        """Persist an in-flight brief so resume can finish it after a crash."""
        task_id = str(brief.get("task_id") or uuid.uuid4().hex)
        brief["task_id"] = task_id
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO generation_tasks VALUES (?,?,?,?)",
                             (task_id, json.dumps(brief, ensure_ascii=False), "pending", time.time()))
        return task_id

    def complete_generation_task(self, task_id: str) -> None:
        """Mark a generation task done (ok, rejected, or retried)."""
        with self._lock, self._db:
            self._db.execute("UPDATE generation_tasks SET status='complete',updated_at=? WHERE id=?",
                             (time.time(), task_id))

    def unfinished_generation_tasks(self) -> list[dict]:
        """Briefs that were in-flight when the process last stopped."""
        with self._lock:
            rows = self._db.execute("SELECT payload FROM generation_tasks WHERE status='pending' ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def load_questions(self, path: Path) -> list[Question]:
        """Parse JSONL into Question models, skipping invalid rows."""
        out: list[Question] = []
        for rec in read_jsonl(path):
            try:
                out.append(Question.model_validate(rec))
            except Exception:
                continue
        return out

    def load_accepted(self) -> list[Question]:
        """Load accepted questions from SQLite (no in-memory cache)."""
        return self._load_status("accepted")

    def load_rejected(self) -> list[Question]:
        """Load rejected questions from SQLite (no in-memory cache)."""
        return self._load_status("rejected")

    def load_pending(self) -> list[Question]:
        """Pending / deferred-review questions."""
        return self._load_status("pending")

    def append_accepted(self, question: Question) -> None:
        """Insert or update an accepted question and refresh JSONL."""
        question.review_status = "accepted"
        with self._lock:
            with self._db:
                self._put_question(question, "accepted")
            self._export_status("accepted")

    def append_rejected(self, question: Question) -> None:
        """Insert or update a rejected question and refresh JSONL."""
        question.review_status = "rejected"
        with self._lock:
            with self._db:
                self._put_question(question, "rejected")
            self._export_status("rejected")

    def save_pending(self, questions: list[Question]) -> None:
        """Upsert pending questions (used after generate and deferred reviews)."""
        with self._lock:
            with self._db:
                for question in questions:
                    # Generation snapshots may be older than a concurrent reviewer.
                    self._db.execute("INSERT OR IGNORE INTO questions VALUES (?,?,?,?)",
                                     (question.id, "pending", question.model_dump_json(), time.time()))
            self._export_status("pending")

    def save_steering(self, brief: SteeringBrief) -> None:
        """Persist the latest chair-to-planner steering brief."""
        with self._lock:
            with self._db:
                self._set_meta("steering", brief.model_dump())
            self.steering_path.write_text(
                json.dumps(brief.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_steering(self) -> SteeringBrief:
        """Load steering; empty brief if none has been saved yet."""
        return SteeringBrief.model_validate(self._get_meta("steering", {}))

    def save_run_state(self, payload: dict[str, Any]) -> None:
        """Checkpoint iteration/review_rounds/accepted_count for resume."""
        with self._lock:
            with self._db:
                self._set_meta("run", payload)
            self.run_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def load_run_state(self) -> dict[str, Any]:
        """Resume counters; ``{}`` when this is a fresh data dir."""
        return self._get_meta("run", {})

    def sample_accepted(
        self,
        *,
        topic_prefix: list[str] | None = None,
        k: int = 3,
        rng: random.Random | None = None,
    ) -> list[Question]:
        """Random sample of accepted questions, optionally filtered by topic prefix."""
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
        """Most recent template skeletons from L1/L2/vague rejects (anti-templates)."""
        rejected = self.load_rejected()
        out: list[str] = []
        for q in reversed(rejected):
            reason = q.reject_reason or ""
            if reason.startswith("L1") or reason.startswith("L2") or reason.startswith(
                "sieve_too_vague"
            ):
                if q.template_fingerprint:
                    out.append(q.template_fingerprint)
            if len(out) >= k:
                break
        return out

    def iter_quality_rejected(self, limit: int = 200) -> list[Question]:
        """Newest quality rejects with nonempty stems, for few-shot negatives."""
        from cuda_sft_qgen.prompts.fewshot import is_quality_reject

        out: list[Question] = []
        for q in reversed(self.load_rejected()):
            if not (q.question or "").strip():
                continue
            if not is_quality_reject(q.reject_reason):
                continue
            out.append(q)
            if len(out) >= limit:
                break
        return out

    def append_salvage_raw(self, record: dict[str, Any]) -> None:
        """Append an unparsed writer dump for later debugging."""
        with self._lock:
            append_jsonl(self.salvage_raw_path, record)

    def load_retry_queue(self) -> list[dict[str, Any]]:
        """Unleased retry tasks, oldest first."""
        with self._lock:
            rows = self._db.execute("SELECT payload FROM retry_tasks WHERE lease_until<=? ORDER BY rowid",
                                    (time.time(),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _put_retry(self, item: dict[str, Any]) -> None:
        item = dict(item)
        task_id = str(item.setdefault("task_id", uuid.uuid4().hex))
        self._db.execute("INSERT OR REPLACE INTO retry_tasks VALUES (?,?,?,0)",
                         (task_id, json.dumps(item, ensure_ascii=False), float(item.get("next_ok_at") or 0)))

    def _export_retries(self) -> None:
        rows = self._db.execute("SELECT payload FROM retry_tasks ORDER BY rowid").fetchall()
        write_jsonl(self.retry_path, [json.loads(row[0]) for row in rows])

    def save_retry_queue(self, items: list[dict[str, Any]]) -> None:
        """Replace the retry queue with ``items`` and rewrite the JSONL sidecar."""
        with self._lock:
            with self._db:
                self._db.execute("DELETE FROM retry_tasks")
                for item in items:
                    self._put_retry(item)
            self._export_retries()

    def enqueue_retry(self, item: dict[str, Any]) -> None:
        """Add a retry-queue item with a due timestamp."""
        with self._lock:
            with self._db:
                self._put_retry(item)
            self._export_retries()

    def due_retry_items(self, *, limit: int, now: float | None = None) -> list[dict[str, Any]]:
        """Lease up to ``limit`` retry items whose ``due_at`` has passed."""
        now = time.time() if now is None else now
        with self._lock, self._db:
            rows = self._db.execute(
                "SELECT id,payload FROM retry_tasks WHERE due_at<=? AND lease_until<=? ORDER BY rowid LIMIT ?",
                (now, now, max(0, limit)),
            ).fetchall()
            self._db.executemany("UPDATE retry_tasks SET lease_until=? WHERE id=?",
                                 [(now + 900, row[0]) for row in rows])
        return [json.loads(row[1]) for row in rows]

    def complete_retry(self, task_id: str) -> None:
        """Drop a retry task after it has been drained or exhausted."""
        with self._lock:
            with self._db:
                self._db.execute("DELETE FROM retry_tasks WHERE id=?", (task_id,))
            self._export_retries()

    def retry_queue_size(self) -> int:
        """Total retry tasks, including those not yet due."""
        return len(self.load_retry_queue())

    def due_retry_count(self, now: float | None = None) -> int:
        """Number of retry tasks whose due time has passed and are unleased."""
        now = time.time() if now is None else now
        with self._lock:
            return self._db.execute("SELECT count(*) FROM retry_tasks WHERE due_at<=? AND lease_until<=?",
                                    (now, now)).fetchone()[0]

    def purge_rate_limit_rejects(self) -> int:
        """Drop 429 rejects from the rejected table so they are not few-shot negatives."""
        """Drop rejected rows that are rate-limit noise, not question quality."""
        with self._lock:
            ids = [q.id for q in self.load_rejected() if is_rate_limit_reject_reason(q.reject_reason)]
            dropped = len(ids)
            if dropped:
                with self._db:
                    self._db.executemany("DELETE FROM questions WHERE id=?", [(qid,) for qid in ids])
                self._export_status("rejected")
        return dropped

    def leaf_stats(self, target_per_leaf: int = 3) -> dict[str, dict[str, Any]]:
        """Per-leaf accepted/rejected counts used by adaptive sampling and stats."""
        """
        Compute per-leaf health statistics for adaptive planning.

        Args:
            target_per_leaf: Target quota per leaf (default 3)

        Returns:
            Dictionary mapping l3_id to health metrics
        """
        from collections import defaultdict

        accepted = self.load_accepted()
        rejected = self.load_rejected()

        leaf_accepted: dict[str, int] = defaultdict(int)
        leaf_rejected: dict[str, int] = defaultdict(int)
        leaf_difficulties: dict[str, list[int]] = defaultdict(list)

        for q in accepted:
            if q.l3_id:
                leaf_accepted[q.l3_id] += 1
                if q.difficulty:
                    leaf_difficulties[q.l3_id].append(q.difficulty)

        for q in rejected:
            if q.l3_id:
                leaf_rejected[q.l3_id] += 1

        all_l3_ids = set(leaf_accepted.keys()) | set(leaf_rejected.keys())

        stats = {}
        for l3_id in all_l3_ids:
            acc = leaf_accepted[l3_id]
            rej = leaf_rejected[l3_id]
            diffs = leaf_difficulties[l3_id]
            avg_diff = sum(diffs) / len(diffs) if diffs else 2.0
            total = acc + rej

            stats[l3_id] = {
                "l3_id": l3_id,
                "accepted": acc,
                "rejected": rej,
                "total_generated": total,
                "accept_rate": acc / total if total > 0 else 0.5,
                "avg_difficulty": avg_diff,
                "coverage": min(acc / target_per_leaf, 1.0) if target_per_leaf > 0 else 1.0,
                "target_quota": target_per_leaf,
            }

        return stats

    def compute_stats(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Write ``stats.json`` histograms for the current accepted/pending/rejected sets."""
        accepted = self.load_accepted()
        rejected = self.load_rejected()
        pending = self.load_pending()
        stats: dict[str, Any] = {
            "accepted": len(accepted),
            "accepted_count": len(accepted),
            "rejected": len(rejected),
            "rejected_count": len(rejected),
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
