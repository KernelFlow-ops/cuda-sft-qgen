"""Transactional knowledge revisions, evidence, graph edges and durable work queue."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import ast
from fractions import Fraction
import hashlib
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib.parse import urlparse

from cuda_sft_qgen.taxonomy import Taxonomy, _leaves_from_domains, dump_taxonomy

logger = logging.getLogger(__name__)

STATUSES = {"legacy_unreviewed", "provisional", "verified", "quarantined", "retired"}
RELATIONS = {
    "requires", "applies_to", "supported_by", "equivalent_to", "contradicts", "supersedes",
    "contains", "derives", "measures", "uses_feature", "implements", "precedes", "related",
}
ENTITY_KINDS = {"arch", "metric", "api", "feature", "operator", "document", "domain", "category"}
ENTITY_TARGET_KINDS = {"applies_to", "supported_by", "measures", "uses_feature", "implements", "contains"}
MECHANICAL_KINDS = {"contains", "precedes"}
OFFICIAL_HOSTS = {"docs.nvidia.com", "developer.nvidia.com", "nvidia.github.io"}


def official_url(url: str) -> bool:
    """True for HTTPS NVIDIA docs hosts with no userinfo or non-443 port."""
    try:
        parsed = urlparse(url)
        return parsed.scheme == "https" and parsed.hostname in OFFICIAL_HOSTS and not parsed.username and parsed.port in (None, 443)
    except ValueError:
        return False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _norm_ws(value: str) -> str:
    return " ".join((value or "").split())


def _ensure_column(db: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Add a schema column once, including on legacy SQLite files."""
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _safe_eval(expr: str) -> Fraction:
    """Evaluate bounded arithmetic into an exact fraction without eval()."""
    if not re.fullmatch(r"[0-9.+\-*/%() ]+", expr or ""):
        raise ValueError("unsafe expression")

    def value(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = value(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ValueError("division by zero")
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return Fraction(left // right)
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right.numerator) <= 16:
                return left ** right.numerator
        raise ValueError("unsafe expression")

    try:
        return value(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ValueError("unsafe expression") from exc


class KnowledgeStore:
    """SQLite knowledge graph: versioned claims, evidence, relations, and jobs.

    Verified claims are never overwritten by pack ingest. Audits may quarantine
    or revise a claim; impact events tell the question archive what to requeue.
    """

    def __init__(self, path: Path | str, toolkit_version: str = ""):
        """Open or create ``knowledge.sqlite3`` and ensure the schema exists."""
        self.path = Path(path)
        self.toolkit_version = toolkit_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, fingerprint TEXT NOT NULL, updated_at REAL NOT NULL,
                    audited_at REAL DEFAULT 0);
                CREATE INDEX IF NOT EXISTS claims_fingerprint ON claims(fingerprint);
                CREATE TABLE IF NOT EXISTS revisions (
                    claim_id TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY(claim_id,revision));
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY, claim_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    url TEXT NOT NULL, quote TEXT NOT NULL, content_hash TEXT NOT NULL,
                    fetched_at REAL NOT NULL, verdict TEXT NOT NULL,
                    UNIQUE(claim_id,revision,url,content_hash,quote));
                CREATE TABLE IF NOT EXISTS documents (
                    content_hash TEXT PRIMARY KEY, url TEXT NOT NULL,
                    content TEXT NOT NULL, fetched_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audits (
                    id INTEGER PRIMARY KEY, claim_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    verdict TEXT NOT NULL, reason TEXT NOT NULL, evidence_id INTEGER, audited_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
                    kind TEXT NOT NULL, status TEXT NOT NULL, evidence_id INTEGER,
                    source_revision INTEGER NOT NULL, reason TEXT NOT NULL,
                    UNIQUE(source_id,target_id,kind,source_revision));
                CREATE TABLE IF NOT EXISTS question_claims (
                    question_id TEXT NOT NULL, claim_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    PRIMARY KEY(question_id,claim_id,revision));
                CREATE TABLE IF NOT EXISTS impact_events (
                    id INTEGER PRIMARY KEY, claim_id TEXT NOT NULL, old_revision INTEGER NOT NULL,
                    new_revision INTEGER NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL, leased_at REAL, error TEXT, created_at REAL NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS jobs_active ON jobs(dedupe_key)
                    WHERE status IN ('queued','running');
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    label TEXT NOT NULL, attrs TEXT NOT NULL);
            """)
            _ensure_column(db, "claims", "verified_by", "TEXT NOT NULL DEFAULT 'none'")
            _ensure_column(db, "evidence", "kind", "TEXT NOT NULL DEFAULT 'doc_quote'")
            _ensure_column(db, "evidence", "toolkit_version", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(db, "claims", "audit_due_at", "REAL NOT NULL DEFAULT 0")

    @contextmanager
    def connect(self):
        """Context manager yielding a WAL SQLite connection with FK on."""
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _fingerprint(claim: str) -> str:
        return hashlib.sha256(" ".join(claim.lower().split()).encode()).hexdigest()

    def import_taxonomy(self, taxonomy: Taxonomy) -> int:
        """Insert missing taxonomy leaves as ``legacy_unreviewed`` claims."""
        added = 0
        conflicts: list[dict[str, Any]] = []
        with self.connect() as db:
            existing = {
                row[0]: (row[1], row[2])
                for row in db.execute(
                    "SELECT id,json_extract(payload,'$.domain_id'),json_extract(payload,'$.l2_id') FROM claims"
                )
            }
            for leaf in taxonomy.leaves:
                incoming = (leaf.domain_id, leaf.l2_id)
                prior = existing.get(leaf.l3_id)
                if prior is not None:
                    if prior != incoming:
                        conflicts.append({"l3_id": leaf.l3_id, "existing": list(prior), "incoming": list(incoming)})
                    continue
                payload = dict(vars(leaf))
                payload["id"] = leaf.l3_id
                payload["status"] = "retired" if leaf.retired else leaf.status
                now = time.time()
                inserted = db.execute(
                    "INSERT OR IGNORE INTO claims(id,revision,status,payload,fingerprint,updated_at) VALUES(?,?,?,?,?,?)",
                    (leaf.l3_id, leaf.revision, payload["status"], _json(payload), self._fingerprint(leaf.claim), now),
                ).rowcount
                if inserted:
                    db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)", (leaf.l3_id, leaf.revision, _json(payload), payload["status"], "taxonomy_import", now))
                    added += 1
                existing[leaf.l3_id] = incoming
        if conflicts:
            logger.error("import_conflicts n=%d first=%s", len(conflicts), conflicts[0])
        self.set_metadata("import_conflicts", conflicts)
        return added

    def add_claim(self, payload: dict[str, Any], *, status: str = "provisional", reason: str = "expansion") -> bool:
        """Insert a new claim if the id is unused. Returns False on conflict."""
        if status not in STATUSES or status == "verified":
            raise ValueError("New claims must be reviewed through evidence before verification")
        payload = deepcopy(payload)
        claim_id = str(payload.get("l3_id") or payload.get("id") or "")
        claim = str(payload.get("claim") or "").strip()
        if not claim_id or not claim or not payload.get("domain_id") or not payload.get("l2_id"):
            raise ValueError("claim id, text, domain_id and l2_id are required")
        fingerprint = self._fingerprint(claim)
        payload.update(id=claim_id, l3_id=claim_id, revision=1, status=status)
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM claims WHERE id=? OR fingerprint=?", (claim_id, fingerprint)).fetchone():
                return False
            db.execute("INSERT INTO claims(id,revision,status,payload,fingerprint,updated_at) VALUES(?,?,?,?,?,?)", (claim_id, 1, status, _json(payload), fingerprint, now))
            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)", (claim_id, 1, _json(payload), status, reason, now))
        return True

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        """Current payload for ``claim_id``, or None."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
        if row is None:
            return None
        result = json.loads(row["payload"])
        result.update(id=row["id"], revision=row["revision"], status=row["status"], audited_at=row["audited_at"],
                      verified_by=row["verified_by"], audit_due_at=row["audit_due_at"])
        return result

    def list_claims(self, status: str | None = None, *, limit: int = 10000) -> list[dict]:
        """List claims, optionally filtered by status."""
        with self.connect() as db:
            rows = db.execute("SELECT * FROM claims" + (" WHERE status=?" if status else "") + " ORDER BY id LIMIT ?", (status, limit) if status else (limit,)).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload.update(id=row["id"], revision=row["revision"], status=row["status"], audited_at=row["audited_at"],
                           verified_by=row["verified_by"], audit_due_at=row["audit_due_at"])
            result.append(payload)
        return result

    def revise(self, claim_id: str, changes: dict[str, Any], *, reason: str, expected_revision: int | None = None) -> int:
        """Optimistic-concurrency revision; emits an impact event. Returns new revision."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None:
                raise KeyError(claim_id)
            if expected_revision is not None and expected_revision != row["revision"]:
                raise ValueError("stale knowledge revision")
            payload = json.loads(row["payload"])
            payload.update({key: value for key, value in changes.items() if key in {"claim", "hints", "types", "difficulty", "applicability", "sources", "status"}})
            status = changes.get("status", "provisional")
            if status not in STATUSES or status == "verified":
                raise ValueError("Revisions require a new evidence audit before verification")
            revision = row["revision"] + 1
            payload.update(revision=revision, status=status)
            now = time.time()
            db.execute("UPDATE claims SET revision=?,status=?,payload=?,fingerprint=?,updated_at=?,audited_at=0,verified_by='none' WHERE id=?", (revision, status, _json(payload), self._fingerprint(payload["claim"]), now, claim_id))
            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)", (claim_id, revision, _json(payload), status, reason, now))
            db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)", (claim_id, row["revision"], revision, reason, now))
            db.execute("UPDATE relations SET status='stale' WHERE source_id=? OR target_id=?", (claim_id, claim_id))
        return revision

    def apply_audit(self, claim_id: str, revision: int, *, verdict: str, url: str = "", quote: str = "", document: str = "", reason: str = "", applicability: dict | None = None) -> bool:
        """Record an official-source audit; may verify, quarantine, or revise the claim."""
        if verdict not in {"supported", "contradicted", "insufficient"}:
            raise ValueError("invalid audit verdict")
        grounded = bool(official_url(url) and quote.strip() and _norm_ws(quote) in _norm_ws(document))
        if verdict != "insufficient" and not grounded:
            raise ValueError("audit requires a verbatim quote in retrieved official documentation")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None or row["revision"] != revision:
                return False
            now = time.time()
            evidence_id = None
            if grounded:
                digest = hashlib.sha256(document.encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)", (digest, url, document, now))
                db.execute("INSERT OR IGNORE INTO evidence(claim_id,revision,url,quote,content_hash,fetched_at,verdict,kind,toolkit_version) VALUES(?,?,?,?,?,?,?,?,?)", (claim_id, revision, url, quote.strip(), digest, now, verdict, "doc_quote", self.toolkit_version))
                evidence_id = db.execute("SELECT id FROM evidence WHERE claim_id=? AND revision=? AND url=? AND content_hash=? AND quote=?", (claim_id, revision, url, digest, quote.strip())).fetchone()[0]
            status = {"supported": "verified", "contradicted": "quarantined", "insufficient": row["status"]}[verdict]
            payload = json.loads(row["payload"])
            payload["status"] = status
            if verdict == "supported" and applicability:
                payload["applicability"] = applicability
            if evidence_id:
                payload["sources"] = list(dict.fromkeys([*(payload.get("sources") or []), url]))
                if verdict == "supported":
                    db.execute("INSERT OR REPLACE INTO relations(source_id,target_id,kind,status,evidence_id,source_revision,reason) VALUES(?,?,?,?,?,?,?)", (claim_id, f"evidence:{evidence_id}", "supported_by", "verified", evidence_id, revision, reason))
                    for key, values in (applicability or {}).items():
                        values = values if isinstance(values, list) else [values]
                        for value in values:
                            if isinstance(value, (str, int, float)):
                                db.execute("INSERT OR REPLACE INTO relations(source_id,target_id,kind,status,evidence_id,source_revision,reason) VALUES(?,?,?,?,?,?,?)", (claim_id, f"{key}:{value}", "applies_to", "verified", evidence_id, revision, reason))
            verified_by = "quote" if verdict == "supported" else ("none" if verdict == "contradicted" else row["verified_by"])
            db.execute("UPDATE claims SET status=?,payload=?,audited_at=?,updated_at=?,verified_by=? WHERE id=?", (status, _json(payload), now, now, verified_by, claim_id))
            db.execute("INSERT INTO audits(claim_id,revision,verdict,reason,evidence_id,audited_at) VALUES(?,?,?,?,?,?)", (claim_id, revision, verdict, reason, evidence_id, now))
            if status != row["status"]:
                db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)", (claim_id, revision, revision, f"audit:{verdict}:{reason}", now))
                if status == "quarantined":
                    db.execute("UPDATE relations SET status='stale' WHERE source_id=? OR target_id=?", (claim_id, claim_id))
        return True

    def _apply_local_check(self, claim_id: str, revision: int, *, passed: bool, method: str,
                           kind: str, url: str, quote: str, digest_source: str, reason: str) -> bool:
        """Persist a non-URL verification result and its state transition."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None or row["revision"] != revision or row["status"] in {"retired", "quarantined"}:
                return False
            now = time.time()
            evidence_id = None
            if passed:
                digest = hashlib.sha256(digest_source.encode()).hexdigest()
                db.execute(
                    "INSERT OR IGNORE INTO evidence(claim_id,revision,url,quote,content_hash,fetched_at,verdict,kind,toolkit_version) VALUES(?,?,?,?,?,?,?,?,?)",
                    (claim_id, revision, url, quote[:500], digest, now, "supported", kind, self.toolkit_version),
                )
                evidence_id = db.execute(
                    "SELECT id FROM evidence WHERE claim_id=? AND revision=? AND url=? AND content_hash=? AND quote=?",
                    (claim_id, revision, url, digest, quote[:500]),
                ).fetchone()[0]
                payload = json.loads(row["payload"])
                payload["status"] = "verified"
                db.execute(
                    "UPDATE claims SET status='verified',payload=?,verified_by=?,audited_at=?,updated_at=?,audit_due_at=0 WHERE id=?",
                    (_json(payload), method, now, now, claim_id),
                )
            db.execute("INSERT INTO audits(claim_id,revision,verdict,reason,evidence_id,audited_at) VALUES(?,?,?,?,?,?)",
                       (claim_id, revision, "supported" if passed else "insufficient", reason, evidence_id, now))
            if not passed:
                db.execute("UPDATE claims SET audited_at=?,updated_at=? WHERE id=?", (now, now, claim_id))
            if passed and row["status"] != "verified":
                db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)",
                           (claim_id, revision, revision, f"audit:supported:{method}", now))
        return True

    def record_insufficient_audit(self, claim_id: str, revision: int, *, reason: str) -> bool:
        """Record an inconclusive verification attempt without changing claim status."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision,status FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None or row["revision"] != revision or row["status"] in {"retired", "quarantined"}:
                return False
            now = time.time()
            db.execute("UPDATE claims SET audited_at=?,updated_at=? WHERE id=?", (now, now, claim_id))
            db.execute("INSERT INTO audits(claim_id,revision,verdict,reason,evidence_id,audited_at) VALUES(?,?,?,?,NULL,?)",
                       (claim_id, revision, "insufficient", str(reason)[:1000], now))
        return True

    def apply_computation_check(self, claim_id: str, revision: int, *, expression: str,
                                variables: dict[str, float] | None, expected: str, trace: str) -> bool:
        """Verify formula and numeric claims by exact arithmetic recomputation."""
        claim = self.get_claim(claim_id)
        if claim is None or claim["revision"] != revision or claim["status"] in {"retired", "quarantined"}:
            return False
        if claim.get("kind") not in {"formula", "numeric"}:
            raise ValueError("computation path only for formula/numeric")
        expr = expression
        for name, value in (variables or {}).items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,31}", name):
                raise ValueError("unsafe variable name")
            expr = re.sub(rf"\b{re.escape(name)}\b", repr(value), expr)
        got, want = _safe_eval(expr), _safe_eval(expected)
        return self._apply_local_check(
            claim_id, revision, passed=got == want, method="computation", kind="calc_trace",
            url="calc://local", quote=trace, digest_source=expr,
            reason="computation_matches" if got == want else "computation_mismatch",
        )

    def apply_expert_verification(self, claim_id: str, revision: int, *, reviewer: str, note: str) -> bool:
        """Record an accountable human review as expert evidence."""
        if not reviewer.strip():
            raise ValueError("reviewer is required")
        return self._apply_local_check(
            claim_id, revision, passed=True, method="expert", kind="expert_note",
            url=f"expert://{reviewer.strip()}", quote=note, digest_source=note,
            reason="expert_verified",
        )

    def apply_execution_check(self, claim_id: str, revision: int, *, log: str, passed: bool) -> bool:
        """Record a caller-provided execution result without running code."""
        return self._apply_local_check(
            claim_id, revision, passed=passed, method="execution", kind="exec_log",
            url="exec://local", quote=log, digest_source=log,
            reason="execution_pass" if passed else "execution_failed",
        )

    def add_evidence(self, claim_id: str, revision: int, *, url: str, quote: str, document: str, verdict: str = "relation") -> int:
        """Store an official-source quote that appears verbatim in ``document``.

        Raises:
            ValueError: If the URL is not official, the quote is missing from
                the document, or ``revision`` is stale.
        """
        if not official_url(url) or not quote.strip() or _norm_ws(quote) not in _norm_ws(document):
            raise ValueError("evidence requires a retrieved official verbatim quote")
        digest = hashlib.sha256(document.encode()).hexdigest()
        now = time.time()
        with self.connect() as db:
            row = db.execute("SELECT revision FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None or row[0] != revision:
                raise ValueError("stale evidence revision")
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)", (digest, url, document, now))
            db.execute("INSERT OR IGNORE INTO evidence(claim_id,revision,url,quote,content_hash,fetched_at,verdict,kind,toolkit_version) VALUES(?,?,?,?,?,?,?,?,?)", (claim_id, revision, url, quote.strip(), digest, now, verdict, "doc_quote", self.toolkit_version))
            return db.execute("SELECT id FROM evidence WHERE claim_id=? AND revision=? AND url=? AND content_hash=? AND quote=?", (claim_id, revision, url, digest, quote.strip())).fetchone()[0]

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """JSON metadata value for ``key``, or ``default`` if missing."""
        with self.connect() as db:
            row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_metadata(self, key: str, value: Any) -> None:
        """Persist a JSON-serializable metadata value."""
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, _json(value)))

    def upsert_entity(self, entity_id: str, kind: str, label: str, attrs: dict | None = None) -> str:
        """Insert or update a graph entity (arch, API, metric, ...)."""
        if kind not in ENTITY_KINDS:
            raise ValueError("invalid entity kind")
        entity_id = str(entity_id)
        with self.connect() as db:
            db.execute(
                "INSERT INTO entities(id,kind,label,attrs) VALUES(?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, label=excluded.label, attrs=excluded.attrs",
                (entity_id, kind, label, _json(attrs or {})),
            )
        return entity_id

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return an entity mapping, or None if it does not exist."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "kind": row["kind"], "label": row["label"], "attrs": json.loads(row["attrs"])}

    def add_relation(self, source_id: str, target_id: str, kind: str, *, evidence_id: int | None = None, status: str = "provisional", reason: str = "") -> int:
        """Insert a typed edge; mechanical kinds skip evidence requirements."""
        if kind not in RELATIONS or status not in {"provisional", "verified"}:
            raise ValueError("invalid relation kind or status")
        if kind == "related" and status == "verified":
            raise ValueError("related edges cannot be verified")
        with self.connect() as db:
            source = db.execute("SELECT revision FROM claims WHERE id=?", (source_id,)).fetchone()
            source_entity = None if source else db.execute("SELECT id FROM entities WHERE id=?", (source_id,)).fetchone()
            if source is None and source_entity is None:
                raise KeyError(source_id)
            target_claim = db.execute("SELECT 1 FROM claims WHERE id=?", (target_id,)).fetchone()
            evidence_target = str(target_id).startswith("evidence:")
            if not target_claim and kind not in ENTITY_TARGET_KINDS and not evidence_target:
                raise KeyError(target_id)
            source_revision = source[0] if source else 0
            if evidence_id is not None:
                evidence = db.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
                if evidence is None or evidence["claim_id"] != source_id or evidence["revision"] != source_revision:
                    raise ValueError("relation evidence must match source revision")
            if status == "verified" and kind not in MECHANICAL_KINDS and (evidence_id is None or not reason.strip()):
                raise ValueError("verified relations require evidence and audit rationale")
            if status == "verified" and kind in MECHANICAL_KINDS and not reason.strip():
                raise ValueError("mechanical relations still need a reason")
            cursor = db.execute("INSERT OR REPLACE INTO relations(source_id,target_id,kind,status,evidence_id,source_revision,reason) VALUES(?,?,?,?,?,?,?)", (source_id, target_id, kind, status, evidence_id, source_revision, reason))
            return cursor.lastrowid

    def subgraph(self, claim_id: str, *, hops: int = 1, kinds: list[str] | None = None,
                 status: str | None = "verified", max_nodes: int = 12) -> dict[str, Any]:
        """Bounded neighborhood around ``claim_id`` for chair/generation context."""
        hops = max(1, min(int(hops), 4))
        max_nodes = max(1, min(int(max_nodes), 200 if hops > 1 else 50))
        root = self.get_claim(claim_id)
        if root is None:
            return {"claims": [], "relations": [], "evidence": [], "entities": []}
        nodes = {claim_id: root}
        entities: dict[str, dict] = {}
        edges: list[dict] = []
        seen_edge = set()
        frontier = {claim_id}
        kind_filter = set(kinds) if kinds else None
        with self.connect() as db:
            for _ in range(hops):
                if not frontier or len(nodes) >= max_nodes:
                    break
                placeholders = ",".join("?" * len(frontier))
                sql = f"SELECT * FROM relations WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))"
                params: list[Any] = [*frontier, *frontier]
                if status:
                    sql += " AND status=?"
                    params.append(status)
                sql += " ORDER BY id LIMIT ?"
                params.append(max_nodes * 4)
                batch = [dict(r) for r in db.execute(sql, params)]
                nxt: set[str] = set()
                for edge in batch:
                    if kind_filter and edge["kind"] not in kind_filter:
                        continue
                    if edge["id"] in seen_edge:
                        continue
                    seen_edge.add(edge["id"])
                    edges.append(edge)
                    other = edge["target_id"] if edge["source_id"] in frontier else edge["source_id"]
                    if other in nodes or other in entities:
                        continue
                    candidate = self.get_claim(other)
                    if candidate and candidate["status"] not in {"quarantined", "retired"} and len(nodes) < max_nodes:
                        nodes[other] = candidate
                        nxt.add(other)
                        continue
                    entity = self.get_entity(other)
                    if entity:
                        entities[other] = entity
                frontier = nxt
            evidence = [dict(r) for r in db.execute("SELECT * FROM evidence WHERE claim_id=? AND revision=? ORDER BY id DESC LIMIT ?", (claim_id, root["revision"], max_nodes))]
        return {"claims": list(nodes.values()), "relations": edges, "evidence": evidence, "entities": list(entities.values())}

    def context_for_claim(self, claim_id: str, revision: int | None = None) -> dict:
        """Chair payload: claim, evidence, neighbors, and related question ids."""
        graph = self.subgraph(claim_id)
        if revision is not None and graph["claims"] and graph["claims"][0]["revision"] != int(revision):
            with self.connect() as db:
                row = db.execute("SELECT payload FROM revisions WHERE claim_id=? AND revision=?", (claim_id, int(revision))).fetchone()
                evidence = [dict(r) for r in db.execute("SELECT * FROM evidence WHERE claim_id=? AND revision=?", (claim_id, int(revision)))]
            return {"claims": [json.loads(row[0])] if row else [], "relations": [], "evidence": evidence, "entities": []}
        return graph

    def context_for_generation(self, claim_id: str, revision: int | None = None, *, max_claims: int = 6) -> dict:
        """Compact related-claim context attached to a writer brief."""
        """Writer/chair context: root + requires/derives prerequisites, hard-capped."""
        max_claims = max(1, min(int(max_claims), 6))
        graph = self.subgraph(claim_id, hops=2, kinds=["requires", "derives"], max_nodes=max_claims + 4)
        if revision is not None and graph["claims"] and graph["claims"][0].get("revision") != int(revision):
            return self.context_for_claim(claim_id, revision=revision)
        claims = []
        for claim in graph["claims"]:
            item = dict(claim)
            text = str(item.get("claim") or "")
            if len(text) > 220:
                item["claim"] = text[:217] + "..."
            claims.append(item)
            if len(claims) >= max_claims:
                break
        ids = {c["id"] for c in claims}
        relations = [
            edge for edge in graph.get("relations") or []
            if edge.get("kind") in {"requires", "derives"}
            and edge.get("source_id") in ids
            and edge.get("target_id") in ids
        ]
        evidence = list(graph.get("evidence") or [])[:3]
        entities = list(graph.get("entities") or [])[:6]
        return {"claims": claims, "relations": relations, "evidence": evidence, "entities": entities}

    def related_claim_ids(self, claim_id: str) -> set[str]:
        """Claim ids in the 1-hop subgraph excluding ``claim_id`` itself."""
        graph = self.subgraph(claim_id)
        return {claim["id"] for claim in graph["claims"] if claim["id"] != claim_id}

    def neighbors(self, node_id: str, *, direction: str = "both", status: str | None = "verified") -> list[dict]:
        """Incident relations; ``direction`` is ``in``, ``out``, or ``both``."""
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be in|out|both")
        clauses = []
        params: list[Any] = []
        if direction in {"out", "both"}:
            clauses.append("source_id=?")
            params.append(node_id)
        if direction in {"in", "both"}:
            clauses.append("target_id=?")
            params.append(node_id)
        sql = "SELECT * FROM relations WHERE (" + " OR ".join(clauses) + ")"
        if status:
            sql += " AND status=?"
            params.append(status)
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql + " ORDER BY id", params)]

    def paths(self, src: str, dst: str, *, max_hops: int = 4) -> list[list[str]]:
        """Simple paths from ``src`` to ``dst`` (at most 8, hop-capped)."""
        max_hops = max(1, min(int(max_hops), 6))
        if src == dst:
            return [[src]]
        graph = self.subgraph(src, hops=max_hops, max_nodes=200)
        adj: dict[str, list[str]] = {}
        for edge in graph["relations"]:
            adj.setdefault(edge["source_id"], []).append(edge["target_id"])
        found: list[list[str]] = []
        queue = [[src]]
        while queue and len(found) < 8:
            path = queue.pop(0)
            if len(path) > max_hops + 1:
                continue
            for nxt in adj.get(path[-1], []):
                if nxt in path:
                    continue
                cand = path + [nxt]
                if nxt == dst:
                    found.append(cand)
                else:
                    queue.append(cand)
        return found

    def export_graph(self, *, domain_id: str | None = None, max_nodes: int = 2000,
                     include_related: bool = False) -> dict[str, Any]:
        """Serialize claims, entities, and relations for visualization."""
        claims = self.list_claims()
        if domain_id:
            claims = [c for c in claims if c.get("domain_id") == domain_id]
        claim_ids = {c["id"] for c in claims}
        with self.connect() as db:
            edges = [dict(r) for r in db.execute("SELECT * FROM relations ORDER BY id")]
            entities = [
                {"id": r["id"], "kind": r["kind"], "label": r["label"], "attrs": json.loads(r["attrs"])}
                for r in db.execute("SELECT * FROM entities")
            ]
        if not include_related:
            edges = [e for e in edges if e["kind"] != "related"]
        allowed = claim_ids | {e["id"] for e in entities}
        edges = [
            e for e in edges
            if e["source_id"] in allowed and (e["target_id"] in allowed or str(e["target_id"]).startswith("evidence:"))
        ]
        used = {e["source_id"] for e in edges} | {e["target_id"] for e in edges}
        entities = [e for e in entities if e["id"] in used]
        if len(claims) + len(entities) > max_nodes:
            buckets: dict[str, int] = {}
            for claim in claims:
                buckets[str(claim.get("domain_id") or "unknown")] = buckets.get(str(claim.get("domain_id") or "unknown"), 0) + 1
            return {
                "aggregated": True,
                "nodes": [{"id": key, "kind": "domain", "leaf_count": n} for key, n in sorted(buckets.items())],
                "edges": [],
                "claims": [],
                "entities": [],
            }
        return {"aggregated": False, "claims": claims, "entities": entities, "relations": edges}

    def graph_stats(self) -> dict[str, Any]:
        """Relation/entity histograms plus the count of unlinked claims."""
        with self.connect() as db:
            rel_kind = {r[0]: r[1] for r in db.execute("SELECT kind,COUNT(*) FROM relations GROUP BY kind")}
            rel_status = {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM relations GROUP BY status")}
            entity_kind = {r[0]: r[1] for r in db.execute("SELECT kind,COUNT(*) FROM entities GROUP BY kind")}
            linked = {r[0] for r in db.execute("SELECT source_id FROM relations UNION SELECT target_id FROM relations")}
            claim_ids = [r[0] for r in db.execute("SELECT id FROM claims WHERE status NOT IN ('retired','quarantined')")]
        orphans = [cid for cid in claim_ids if cid not in linked]
        return {
            "relations_by_kind": rel_kind,
            "relations_by_status": rel_status,
            "entities_by_kind": entity_kind,
            "orphan_claims": len(orphans),
        }

    @staticmethod
    def to_mermaid(graph: dict[str, Any]) -> str:
        """Render an export_graph payload as a Mermaid ``graph LR`` diagram."""
        def ident(node_id: str) -> str:
            return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", node_id)

        lines = ["graph LR"]
        for claim in graph.get("claims") or []:
            lines.append(f'  {ident(claim["id"])}["{claim["id"]}"]')
        for entity in graph.get("entities") or []:
            lines.append(f'  {ident(entity["id"])}("{entity["id"]}")')
        for edge in graph.get("relations") or []:
            lines.append(f'  {ident(edge["source_id"])} -->|{edge["kind"]}| {ident(edge["target_id"])}')
        return "\n".join(lines) + "\n"

    def relation_candidates(self, claim_id: str, *, limit: int = 8, embedder: Any = None) -> list[dict]:
        """Rank live claims by embedding similarity or token overlap."""
        root = self.get_claim(claim_id)
        if root is None or limit <= 0:
            return []
        candidates = [claim for claim in self.list_claims() if claim["id"] != claim_id and claim["status"] not in {"quarantined", "retired"}]
        if embedder is not None and candidates:
            import numpy as np

            vectors = np.asarray(embedder.embed([root["claim"]] + [c["claim"] for c in candidates]), dtype=float)
            norms = np.linalg.norm(vectors, axis=1)
            vectors = vectors / np.maximum(norms[:, None], 1e-12)
            scores = vectors[1:] @ vectors[0]
            by_id = {claim["id"]: float(score) for claim, score in zip(candidates, scores)}
            candidates.sort(key=lambda c: (
                -(by_id[c["id"]] + (0.1 if c.get("l2_id") == root.get("l2_id") else 0.0)), c["id"]
            ))
        else:
            from cuda_sft_qgen.knowledge.doc_index import tokenize

            words = set(tokenize(root["claim"]))
            candidates.sort(key=lambda c: (-(int(c.get("l2_id") == root.get("l2_id"))),
                                           -len(words & set(tokenize(c["claim"]))), c["id"]))
        return candidates[:limit]

    def record_question(self, question_id: Any, claims: list[tuple[str, int]] | None = None) -> None:
        """Bind a question to claim revisions so later audits can invalidate it."""
        if not isinstance(question_id, str):
            question = question_id
            question_id = question.id
            revision = getattr(question, "claim_revision", None) or 1
            claims = [(question.l3_id, int(revision))]
            for dep in getattr(question, "knowledge_dependencies", None) or []:
                if isinstance(dep, dict) and dep.get("claim_id"):
                    claims.append((dep["claim_id"], int(dep.get("revision") or 1)))
        with self.connect() as db:
            for claim_id, revision in claims or []:
                inserted = db.execute("INSERT OR IGNORE INTO question_claims VALUES(?,?,?)", (question_id, claim_id, revision)).rowcount
                current = db.execute("SELECT revision,status FROM claims WHERE id=?", (claim_id,)).fetchone()
                if inserted and current and (current["revision"] != revision or current["status"] in {"quarantined", "retired"}):
                    reason = "late_dependency_registration" if current["revision"] != revision else "audit:contradicted:dependency_unavailable"
                    db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)", (claim_id, revision, current["revision"], reason, time.time()))

    def impact_events(self, after_id: int = 0) -> list[dict]:
        """Claim revision/quarantine events after ``after_id`` (inclusive cursor)."""
        with self.connect() as db:
            events = [dict(r) for r in db.execute("SELECT * FROM impact_events WHERE id>? ORDER BY id", (after_id,))]
            for event in events:
                event["question_ids"] = [r[0] for r in db.execute("SELECT DISTINCT question_id FROM question_claims WHERE claim_id=? AND revision<=?", (event["claim_id"], event["old_revision"]))]
        return events

    def overlay_taxonomy(self, taxonomy: Taxonomy) -> Taxonomy:
        """Replace in-memory leaf claims/status with the store's current revisions."""
        domains = deepcopy(taxonomy.domains)
        domain_map = {d["id"]: d for d in domains}
        children = {(d["id"], c["id"]): c for d in domains for c in d.get("children", [])}
        index = {(d["id"], c["id"]): {item["id"]: item for item in c.get("leaves", [])}
                 for d in domains for c in d.get("children", [])}
        with self.connect() as db:
            rows = db.execute("SELECT * FROM claims ORDER BY id").fetchall()
        for row in rows:
            item = json.loads(row["payload"])
            domain_id = item.get("domain_id")
            l2_id = item.get("l2_id")
            if not domain_id or not l2_id:
                continue
            domain = domain_map.get(domain_id)
            if domain is None:
                domain = {
                    "id": domain_id,
                    "name": item.get("domain_name") or domain_id,
                    "children": [],
                }
                domains.append(domain)
                domain_map[domain_id] = domain
            key = (domain_id, l2_id)
            child = children.get(key)
            if child is None:
                child = {
                    "id": l2_id,
                    "name": item.get("l2_name") or l2_id,
                    "leaves": [],
                }
                domain.setdefault("children", []).append(child)
                children[key] = child
                index[key] = {}
            record = {k: item[k] for k in ("claim", "types", "difficulty", "kind", "hints", "sources", "applicability") if k in item}
            record.update(id=row["id"], revision=row["revision"], status=row["status"],
                          retired=row["status"] in {"quarantined", "retired"}, audit_due_at=row["audit_due_at"])
            index[key][row["id"]] = record
        for key, child in children.items():
            old_ids = [record["id"] for record in child.get("leaves", [])]
            extras = sorted(set(index[key]) - set(old_ids))
            child["leaves"] = [index[key][leaf_id] for leaf_id in [*old_ids, *extras]]
        return Taxonomy(taxonomy.version, taxonomy.name, _leaves_from_domains(domains), domains, taxonomy.index_path, dict(taxonomy.domain_files))

    def export_taxonomy(self, path: Path, taxonomy: Taxonomy) -> None:
        """Dump a store-overlaid taxonomy snapshot to a single YAML file."""
        snapshot = self.overlay_taxonomy(taxonomy)
        snapshot.domain_files = {}
        snapshot.index_path = Path(path)
        dump_taxonomy(snapshot, Path(path))

    def enqueue(self, kind: str, payload: dict, dedupe_key: str, *, delay: float = 0) -> int | None:
        """Enqueue a durable worker job; duplicate ``dedupe_key`` is ignored."""
        now = time.time()
        with self.connect() as db:
            cursor = db.execute("INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,status,available_at,created_at) VALUES(?,?,?,'queued',?,?)", (kind, _json(payload), dedupe_key, now + delay, now))
            return cursor.lastrowid if cursor.rowcount else None

    def enqueue_daily_audits(self, limit: int = 100, *, interval_seconds: float = 86400) -> int:
        """Queue audits for claims not reviewed within ``interval_seconds``."""
        cutoff = time.time() - interval_seconds
        with self.connect() as db:
            rows = db.execute(
                "SELECT id,revision,payload FROM claims WHERE audited_at<? AND status NOT IN ('retired','quarantined') "
                "AND (status!='legacy_unreviewed' OR audit_due_at>0) "
                "AND NOT EXISTS (SELECT 1 FROM jobs WHERE dedupe_key IN "
                "('audit:' || claims.id || ':' || claims.revision, 'compute:' || claims.id || ':' || claims.revision) "
                "AND status IN ('queued','running')) "
                "ORDER BY CASE status WHEN 'provisional' THEN 0 WHEN 'verified' THEN 1 ELSE 2 END,audited_at,updated_at LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        queued = 0
        for row in rows:
            kind = "compute_verify" if json.loads(row["payload"]).get("kind") in {"formula", "numeric"} else "audit"
            prefix = "compute" if kind == "compute_verify" else "audit"
            queued += self.enqueue(kind, {"claim_id": row["id"], "revision": row["revision"]},
                                   f"{prefix}:{row['id']}:{row['revision']}") is not None
        return queued

    def schedule_legacy_backlog(self, *, batch_per_l2: int = 20, due_days: float = 14,
                                now: float | None = None) -> int:
        """Queue a bounded batch of unreviewed legacy claims per L2."""
        if batch_per_l2 <= 0:
            return 0
        now = time.time() if now is None else float(now)
        due_at = now + due_days * 86400
        queued = 0
        counts: dict[tuple[str, str], int] = {}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT id,revision,payload FROM claims WHERE status='legacy_unreviewed' AND audit_due_at=0 "
                "AND NOT EXISTS (SELECT 1 FROM jobs WHERE dedupe_key IN "
                "('audit:' || claims.id || ':' || claims.revision, 'compute:' || claims.id || ':' || claims.revision) "
                "AND status IN ('queued','running')) "
                "ORDER BY json_extract(payload,'$.domain_id'),json_extract(payload,'$.l2_id'),id"
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload"])
                key = (str(payload.get("domain_id") or ""), str(payload.get("l2_id") or ""))
                if counts.get(key, 0) >= batch_per_l2:
                    continue
                counts[key] = counts.get(key, 0) + 1
                kind = "compute_verify" if payload.get("kind") in {"formula", "numeric"} else "audit"
                prefix = "compute" if kind == "compute_verify" else "audit"
                job_payload = {"claim_id": row["id"], "revision": row["revision"]}
                inserted = db.execute(
                    "INSERT OR IGNORE INTO jobs(kind,payload,dedupe_key,status,available_at,created_at) "
                    "VALUES(?,?,?,'queued',?,?)",
                    (kind, _json(job_payload), f"{prefix}:{row['id']}:{row['revision']}", now, now),
                ).rowcount
                if inserted:
                    payload["audit_due_at"] = due_at
                    db.execute("UPDATE claims SET audit_due_at=?,payload=? WHERE id=?",
                               (due_at, _json(payload), row["id"]))
                    queued += 1
        return queued

    def take_job(self, *, lease_seconds: int = 900) -> dict | None:
        """Lease the next due job, or None if the queue is idle."""
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE jobs SET status='queued' WHERE status='running' AND leased_at<?", (now - lease_seconds,))
            row = db.execute("SELECT * FROM jobs WHERE status='queued' AND available_at<=? ORDER BY CASE WHEN kind='expand' THEN 0 WHEN EXISTS (SELECT 1 FROM claims WHERE claims.id=json_extract(jobs.payload,'$.claim_id') AND claims.status='provisional') THEN 1 ELSE 2 END,id LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',leased_at=?,attempts=attempts+1 WHERE id=?", (now, row["id"]))
            result = dict(row)
            result["payload"] = json.loads(result["payload"])
            result["attempts"] += 1
            return result

    def finish_job(self, job_id: int, *, error: str | None = None) -> None:
        """Mark a leased job done, or requeue it with backoff on ``error``."""
        with self.connect() as db:
            row = db.execute("SELECT attempts,kind,payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            status = "done" if error is None else ("failed" if row[0] >= 3 else "queued")
            db.execute("UPDATE jobs SET status=?,error=?,available_at=? WHERE id=?", (status, error, time.time() + 60 * 2 ** row[0], job_id))
            if status == "failed" and row["kind"] in {"audit", "compute_verify"}:
                payload = json.loads(row["payload"])
                db.execute("UPDATE claims SET audited_at=? WHERE id=? AND revision=?", (time.time(), payload.get("claim_id"), payload.get("revision")))

    def stats(self) -> dict:
        """Claim-status and job-status histograms for CLI/gate reports."""
        with self.connect() as db:
            base = {"claims": {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM claims GROUP BY status")}, "jobs": {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}, "evidence": db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], "relations": db.execute("SELECT COUNT(*) FROM relations").fetchone()[0], "verified_by": {r[0]: r[1] for r in db.execute("SELECT verified_by,COUNT(*) FROM claims GROUP BY verified_by")}}
        base["graph"] = self.graph_stats()
        return base
