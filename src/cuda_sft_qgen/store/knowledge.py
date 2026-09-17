"""Transactional knowledge revisions, evidence, graph edges and durable work queue."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
from urllib.parse import urlparse

from cuda_sft_qgen.taxonomy import Taxonomy, _leaves_from_domains, dump_taxonomy

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


class KnowledgeStore:
    """SQLite knowledge graph: versioned claims, evidence, relations, and jobs.

    Verified claims are never overwritten by pack ingest. Audits may quarantine
    or revise a claim; impact events tell the question archive what to requeue.
    """

    def __init__(self, path: Path | str):
        """Open or create ``knowledge.sqlite3`` and ensure the schema exists."""
        self.path = Path(path)
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
        with self.connect() as db:
            for leaf in taxonomy.leaves:
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
        result.update(id=row["id"], revision=row["revision"], status=row["status"], audited_at=row["audited_at"])
        return result

    def list_claims(self, status: str | None = None, *, limit: int = 10000) -> list[dict]:
        """List claims, optionally filtered by status."""
        with self.connect() as db:
            rows = db.execute("SELECT * FROM claims" + (" WHERE status=?" if status else "") + " ORDER BY id LIMIT ?", (status, limit) if status else (limit,)).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload.update(id=row["id"], revision=row["revision"], status=row["status"], audited_at=row["audited_at"])
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
            db.execute("UPDATE claims SET revision=?,status=?,payload=?,fingerprint=?,updated_at=?,audited_at=0 WHERE id=?", (revision, status, _json(payload), self._fingerprint(payload["claim"]), now, claim_id))
            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)", (claim_id, revision, _json(payload), status, reason, now))
            db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)", (claim_id, row["revision"], revision, reason, now))
            db.execute("UPDATE relations SET status='stale' WHERE source_id=? OR target_id=?", (claim_id, claim_id))
        return revision

    def apply_audit(self, claim_id: str, revision: int, *, verdict: str, url: str = "", quote: str = "", document: str = "", reason: str = "", applicability: dict | None = None) -> bool:
        """Record an official-source audit; may verify, quarantine, or revise the claim."""
        if verdict not in {"supported", "contradicted", "insufficient"}:
            raise ValueError("invalid audit verdict")
        grounded = bool(official_url(url) and quote.strip() and quote.strip() in document)
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
                db.execute("INSERT OR IGNORE INTO evidence(claim_id,revision,url,quote,content_hash,fetched_at,verdict) VALUES(?,?,?,?,?,?,?)", (claim_id, revision, url, quote.strip(), digest, now, verdict))
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
            db.execute("UPDATE claims SET status=?,payload=?,audited_at=?,updated_at=? WHERE id=?", (status, _json(payload), now, now, claim_id))
            db.execute("INSERT INTO audits(claim_id,revision,verdict,reason,evidence_id,audited_at) VALUES(?,?,?,?,?,?)", (claim_id, revision, verdict, reason, evidence_id, now))
            if status != row["status"]:
                db.execute("INSERT INTO impact_events(claim_id,old_revision,new_revision,reason,created_at) VALUES(?,?,?,?,?)", (claim_id, revision, revision, f"audit:{verdict}:{reason}", now))
                if status == "quarantined":
                    db.execute("UPDATE relations SET status='stale' WHERE source_id=? OR target_id=?", (claim_id, claim_id))
        return True

    def add_evidence(self, claim_id: str, revision: int, *, url: str, quote: str, document: str, verdict: str = "relation") -> int:
        """Store an official-source quote that appears verbatim in ``document``.

        Raises:
            ValueError: If the URL is not official, the quote is missing from
                the document, or ``revision`` is stale.
        """
        if not official_url(url) or not quote.strip() or quote.strip() not in document:
            raise ValueError("evidence requires a retrieved official verbatim quote")
        digest = hashlib.sha256(document.encode()).hexdigest()
        now = time.time()
        with self.connect() as db:
            row = db.execute("SELECT revision FROM claims WHERE id=?", (claim_id,)).fetchone()
            if row is None or row[0] != revision:
                raise ValueError("stale evidence revision")
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?,?,?)", (digest, url, document, now))
            db.execute("INSERT OR IGNORE INTO evidence(claim_id,revision,url,quote,content_hash,fetched_at,verdict) VALUES(?,?,?,?,?,?,?)", (claim_id, revision, url, quote.strip(), digest, now, verdict))
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

    def relation_candidates(self, claim_id: str, *, limit: int = 8) -> list[dict]:
        """Rank other claims by shared L2 and token overlap for relate jobs."""
        root = self.get_claim(claim_id)
        if root is None:
            return []
        words = set(root["claim"].lower().split())
        candidates = [claim for claim in self.list_claims() if claim["id"] != claim_id and claim["status"] not in {"quarantined", "retired"}]
        candidates.sort(key=lambda claim: (claim.get("l2_id") == root.get("l2_id"), len(words & set(claim["claim"].lower().split()))), reverse=True)
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
        with self.connect() as db:
            rows = db.execute("SELECT * FROM claims ORDER BY id").fetchall()
        for row in rows:
            item = json.loads(row["payload"])
            domain_id = item.get("domain_id")
            if not domain_id:
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
            child = next((c for c in domain.get("children", []) if c["id"] == item["l2_id"]), None)
            if child is None:
                child = {
                    "id": item["l2_id"],
                    "name": item.get("l2_name") or item["l2_id"],
                    "leaves": [],
                }
                domain.setdefault("children", []).append(child)
            record = {k: item[k] for k in ("claim", "types", "difficulty", "kind", "hints", "sources", "applicability") if k in item}
            record.update(id=row["id"], revision=row["revision"], status=row["status"], retired=row["status"] in {"quarantined", "retired"})
            child["leaves"] = [r for r in child.get("leaves", []) if r["id"] != row["id"]] + [record]
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
            rows = db.execute("SELECT id,revision FROM claims WHERE audited_at<? AND status NOT IN ('retired','quarantined') AND NOT EXISTS (SELECT 1 FROM jobs WHERE dedupe_key='audit:' || claims.id || ':' || claims.revision AND status IN ('queued','running')) ORDER BY CASE status WHEN 'provisional' THEN 0 WHEN 'verified' THEN 1 ELSE 2 END,audited_at,updated_at LIMIT ?", (cutoff, limit)).fetchall()
        return sum(self.enqueue("audit", {"claim_id": r["id"], "revision": r["revision"]}, f"audit:{r['id']}:{r['revision']}") is not None for r in rows)

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
            if status == "failed" and row["kind"] == "audit":
                payload = json.loads(row["payload"])
                db.execute("UPDATE claims SET audited_at=? WHERE id=? AND revision=?", (time.time(), payload.get("claim_id"), payload.get("revision")))

    def stats(self) -> dict:
        """Claim-status and job-status histograms for CLI/gate reports."""
        with self.connect() as db:
            base = {"claims": {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM claims GROUP BY status")}, "jobs": {r[0]: r[1] for r in db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")}, "evidence": db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], "relations": db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]}
        base["graph"] = self.graph_stats()
        return base
