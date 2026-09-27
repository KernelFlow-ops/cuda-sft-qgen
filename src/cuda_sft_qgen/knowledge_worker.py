"""Background expansion and source-grounded audit; no runtime ContextVar access."""
from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from cuda_sft_qgen.keys import PRI_TAXONOMY
from cuda_sft_qgen.knowledge.doc_index import DocIndex
from cuda_sft_qgen.store.knowledge import KnowledgeStore, _norm_ws, official_url

logger = logging.getLogger(__name__)
DEFAULT_SOURCE = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
DEFAULT_INVENTORY_SOURCES = (
    ("guide", DEFAULT_SOURCE),
    ("headers", "https://docs.nvidia.com/cuda/cuda-runtime-api/index.html"),
    ("release_notes", "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"),
    ("nsight", "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"),
)
_TITLE_END = re.compile(r"(要求|约束|机制|用法|区别|原理|规则|语义|配置|流程)\s*$")
_ASSERT_MARK = re.compile(r"=|→|必须|不能|会|导致|等于|返回|\bmust\b|\bcannot\b|\breturns\b|\brequires\b|\bequals\b|\bis\s|\bare\s", re.I)


def needs_refine(leaf: Any) -> bool:
    """Identify legacy leaves that lack evidence hints or a testable claim."""
    claim = str(getattr(leaf, "claim", "") or "")
    return not getattr(leaf, "hints", None) or bool(_TITLE_END.search(claim)) or not bool(_ASSERT_MARK.search(claim))


class _DocumentText(HTMLParser):
    """Strip script/style and collect visible text from official HTML docs."""

    _BLOCKS = {"p", "li", "dt", "dd", "pre", "tr", *(f"h{i}" for i in range(1, 7))}

    def __init__(self):
        """Initialize the HTMLParser and visible-text buffers."""
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0
        self.block_tag: str | None = None
        self.block_parts: list[str] = []

    def _flush_block(self) -> None:
        text = " ".join(" ".join(self.block_parts).split())
        if text:
            if self.block_tag and re.fullmatch(r"h[1-6]", self.block_tag):
                self.parts.append("# " + text)
            self.parts.append(text)
        self.block_tag = None
        self.block_parts = []

    def handle_starttag(self, tag, attrs):
        """Ignore nested script/style sections."""
        if tag in {"script", "style"}:
            self.hidden += 1
            return
        if not self.hidden and tag in self._BLOCKS:
            if self.block_tag is not None:
                self._flush_block()
            self.block_tag = tag

    def handle_endtag(self, tag):
        """Leave a hidden script/style section."""
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if not self.hidden and tag == self.block_tag:
            self._flush_block()

    def handle_data(self, data):
        """Append visible text runs, collapsing inner whitespace."""
        if not self.hidden and data.strip():
            text = " ".join(data.split())
            if self.block_tag is not None:
                self.block_parts.append(text)
            elif len(text) >= 4:
                self.parts.append(text)

    def close(self) -> None:
        super().close()
        if self.block_tag is not None:
            self._flush_block()


def fetch_official_document(url: str) -> str:
    """GET an allowlisted NVIDIA docs URL and return visible text.

    Raises:
        ValueError: On non-official hosts, oversized bodies, or redirect loops.
    """
    if not official_url(url):
        raise ValueError("only configured NVIDIA official HTTPS documentation is allowed")
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        for _ in range(4):
            with client.stream("GET", url, headers={"User-Agent": "cuda-sft-qgen-knowledge-audit/1"}) as response:
                if response.is_redirect:
                    next_url = str(response.url.join(response.headers["location"]))
                    if not official_url(next_url):
                        raise ValueError("official documentation redirected outside allowlist")
                    url = next_url
                    continue
                response.raise_for_status()
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > 8_000_000:
                        raise ValueError("official document exceeds size budget")
                    chunks.append(chunk)
                text = b"".join(chunks).decode("utf-8", errors="replace")
                parser = _DocumentText()
                parser.feed(text)
                parser.close()
                return "\n".join(parser.parts)
    raise ValueError("too many documentation redirects")


def _passages(document: str, claim: str, limit: int = 24000) -> str:
    """Rank document lines by claim-token overlap and return a size-capped excerpt."""
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_:.-]{2,}", claim.lower()))
    blocks = document.splitlines()
    ranked = sorted(range(len(blocks)), key=lambda i: sum(token in blocks[i].lower() for token in tokens), reverse=True)
    selected: set[int] = set()
    size = 0
    for i in ranked:
        for j in range(max(0, i - 1), min(len(blocks), i + 3)):
            if j not in selected and size + len(blocks[j]) <= limit:
                selected.add(j)
                size += len(blocks[j])
        if size >= limit - 1000:
            break
    return "\n".join(blocks[i] for i in sorted(selected))


class KnowledgeWorker:
    """Background thread: expand provisional claims, audit official sources, relate."""
    def __init__(self, store: KnowledgeStore, llm: Any, *, embedder: Any = None,
                 settings: Any = None, taxonomy: Any = None, archive: Any = None,
                 official_fetcher: Callable[[str], str] | None = None):
        """Wire store, LLM, optional embedder/archive, and the official-docs fetcher.

        Does not start the thread; call :meth:`start`.
        """
        self.store = store
        self.llm = llm
        self.embedder = embedder
        self.settings = settings
        self.taxonomy = deepcopy(taxonomy)
        self.archive = archive
        self.fetcher = official_fetcher or fetch_official_document
        self.config = getattr(settings, "knowledge_cfg", None) or ((getattr(settings, "default_yaml", {}) or {}).get("knowledge") or {})
        raw_dir = Path(str(self.config.get("docs_dir") or "data/docs"))
        root = Path(getattr(settings, "root", None) or Path.cwd())
        docs_dir = raw_dir if raw_dir.is_absolute() else root / raw_dir
        self.doc_index = DocIndex.from_dir(docs_dir) if docs_dir.is_dir() else DocIndex([])
        self._retrieved_local = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_schedule = time.monotonic()
        self._jobs_run = 0
        self._documents: dict[str, tuple[float, str]] = {}
        self._taxonomy_index = None
        self._taxonomy_signature: tuple = ()

    def start(self) -> None:
        """Start the daemon poll loop if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._jobs_run >= max(0, int(self.config.get("worker_max_jobs_per_run", 20))):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="knowledge-audit", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal shutdown and wait for an active job to finish by default."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=None if timeout is None else max(0, timeout))

    close = stop

    def _loop(self) -> None:
        max_jobs = max(0, int(self.config.get("worker_max_jobs_per_run", 20)))
        while not self._stop.is_set() and self._jobs_run < max_jobs:
            try:
                if time.monotonic() - self._last_schedule >= 60:
                    if self._stop.is_set():
                        break
                    self.store.enqueue_daily_audits(int(self.config.get("audit_batch_size", 100)), interval_seconds=float(self.config.get("audit_interval_seconds", 86400)))
                    if self._stop.is_set():
                        break
                    self.store.schedule_legacy_backlog(
                        batch_per_l2=int(self.config.get("legacy_batch_per_l2", 20)),
                        due_days=float(self.config.get("legacy_due_days", 14)),
                    )
                    if self._stop.is_set():
                        break
                    self.enqueue_external_inventory()
                    self._last_schedule = time.monotonic()
                if self._stop.is_set():
                    break
                if self.run_once():
                    self._jobs_run += 1
                    if self._jobs_run >= max_jobs:
                        break
                    self._stop.wait(max(0.1, float(self.config.get("worker_poll_seconds", 5))))
                    continue
            except Exception:
                logger.exception("knowledge worker loop failed")
            self._stop.wait(max(0.1, float(self.config.get("worker_poll_seconds", 5))))

    def enqueue_expansion(self, *, accepted: list | None = None, rejected: list | None = None,
                          last_facet: str | None = None, review_round: int = 0) -> int | None:
        """Queue a pack expansion or generic expand job; returns the job id."""
        pack_id = self._next_pack_id()
        if pack_id:
            return self.store.enqueue(
                "expand_pack",
                {"pack_id": pack_id, "last_facet": last_facet, "review_round": review_round},
                f"pack:{pack_id}:{review_round}",
            )
        return self.store.enqueue("expand", {"last_facet": last_facet, "review_round": review_round}, "taxonomy:expansion")

    def _inventory_sources(self) -> list[dict[str, str]]:
        """Resolve typed, allowlisted documents used for coverage discovery."""
        configured = self.config.get("inventory_sources") or self.config.get("official_sources") or []
        if isinstance(configured, str):
            configured = [configured]
        rows: list[dict[str, str]] = []
        for item in configured:
            if isinstance(item, dict):
                url = str(item.get("url") or "")
                kind = str(item.get("kind") or "document")
            else:
                url = str(item)
                kind = "document"
            if official_url(url):
                rows.append({"kind": kind, "url": url})
        if not rows and bool(self.config.get("use_default_inventory_sources", True)):
            rows = [{"kind": kind, "url": url} for kind, url in DEFAULT_INVENTORY_SOURCES]
        return list({row["url"]: row for row in rows}.values())

    @staticmethod
    def _extract_inventory(
        document: str,
        url: str,
        *,
        source_kind: str = "document",
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Extract conservative concept candidates from headings and CUDA symbols."""
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in document.splitlines():
            heading = raw.lstrip().startswith(("#", "<h"))
            line = " ".join(raw.split()).strip(" #-:")
            if len(line) < 4 or len(line) > 180:
                continue
            symbols = re.findall(
                r"(?:__\w+|cuda[A-Z]\w+|CUDA\s+\d+(?:\.\d+)*|cu(?:BLAS|FFT|DNN|SPARSE)|"
                r"NVRTC|nvcc|nvJitLink|PTX|CUB|cp\.async(?:\.bulk)?(?:\.tensor)?|"
                r"mbarrier|wgmma|umma|tmem|TMA|sm_?\d+|Hopper|Blackwell|"
                r"Nsight|l1tex__\w+)",
                line,
                flags=re.I,
            )
            if not heading and not symbols:
                continue
            concept = line
            key = re.sub(r"[^a-z0-9]+", " ", concept.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({
                "id": "ext_" + re.sub(r"[^a-z0-9]+", "_", key)[:80].strip("_"),
                "concept": concept,
                "label": concept,
                "section": ("# " + concept) if heading else "",
                "tokens": sorted(set(symbols)),
                "url": url,
                "source_kind": source_kind,
                "source": "official_inventory",
                "specificity": len(set(symbols)) * 10
                + (3 if re.search(r"\d", concept) else 0)
                + (1 if heading else 0),
            })
        rows.sort(key=lambda row: (-int(row["specificity"]), str(row["id"])))
        return rows[:limit]

    def enqueue_external_inventory(self, *, force: bool = False) -> int | None:
        """Schedule a durable inventory refresh when enabled and stale."""
        if not self.config.get("external_inventory_enabled", False) and not force:
            return None
        stamp = float(self.store.get_metadata("external_inventory_at", 0) or 0)
        ttl = max(60.0, float(self.config.get("external_inventory_interval_seconds", 86400)))
        if not force and stamp and time.time() - stamp < ttl:
            return None
        return self.store.enqueue(
            "external_inventory",
            {"force": bool(force)},
            "taxonomy:external_inventory",
        )

    def refresh_external_inventory(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Refresh and persist the external concept inventory when configured.

        Inventory extraction is a recall aid only.  No claim is verified here;
        proposals enter the existing provisional audit queue later.
        """
        sources = self._inventory_sources()
        if not sources:
            return list(self.store.get_metadata("external_concept_inventory", []) or [])
        stamp = float(self.store.get_metadata("external_inventory_at", 0) or 0)
        ttl = max(60.0, float(self.config.get("external_inventory_interval_seconds", 86400)))
        cached = list(self.store.get_metadata("external_concept_inventory", []) or [])
        if cached and not force and time.time() - stamp < ttl:
            return cached
        inventory: list[dict[str, Any]] = []
        failures: list[str] = []
        for source in sources[: int(self.config.get("external_inventory_max_sources", 8) or 8)]:
            url = source["url"]
            try:
                cached_doc = self._documents.get(url)
                document = cached_doc[1] if cached_doc else self.fetcher(url)
                self._documents[url] = (time.time(), document)
                inventory.extend(self._extract_inventory(document, url, source_kind=source["kind"]))
            except Exception as exc:  # noqa: BLE001
                logger.warning("external inventory fetch failed for %s: %s", url, exc)
                failures.append(f"{url}:{type(exc).__name__}")
        # Stable ids make metadata diffs and critic prompts reproducible.
        unique = {str(item.get("id")): item for item in inventory if item.get("id")}
        inventory = list(unique.values())[: int(self.config.get("external_inventory_max_concepts", 500) or 500)]
        if not inventory:
            raise RuntimeError(
                "external inventory refresh produced no concepts"
                + (": " + "; ".join(failures) if failures else "")
            )
        self.store.set_metadata("external_concept_inventory", inventory)
        self.store.set_metadata("external_inventory_at", time.time())
        self.store.set_metadata("external_inventory_sources", sources)
        return inventory

    ensure_external_inventory = refresh_external_inventory

    # CLI-facing read/refresh interface.  Network work belongs here or in the
    # durable job, never in ordinary taxonomy expansion.
    def external_inventory_once(
        self,
        *,
        force: bool = False,
        apply: bool = True,
        max_leaves: int | None = None,
        max_calls: int | None = None,
    ) -> dict[str, Any]:
        """Refresh inventory, run coverage/critic, and optionally publish claims."""
        inventory = self.refresh_external_inventory(force=force)
        return self.process_external_inventory(
            inventory,
            apply=apply,
            max_leaves=max_leaves,
            max_calls=max_calls,
        )

    inventory_once = external_inventory_once
    schedule_external_inventory = enqueue_external_inventory

    def process_external_inventory(
        self,
        inventory: list[dict[str, Any]],
        *,
        apply: bool = True,
        max_leaves: int | None = None,
        max_calls: int | None = None,
    ) -> dict[str, Any]:
        """Run the critic against cached inventory and publish only provisional leaves."""
        if self.taxonomy is None:
            return {"status": "no_taxonomy", "coverage": {"total": len(inventory)}}
        from cuda_sft_qgen.taxonomy_manager import TaxonomyManager

        taxonomy = self.store.overlay_taxonomy(self.taxonomy)
        tax_cfg = (getattr(self.settings, "default_yaml", {}) or {}).get("taxonomy") or {}
        manager = TaxonomyManager(
            taxonomy,
            self.llm,
            embedder=self.embedder,
            embed_index=self._taxonomy_index,
            settings=self.settings,
            external_inventory=inventory,
            target_leaves_per_l2=int(tax_cfg.get("target_leaves_per_l2", 30) or 30),
        )
        if max_calls is not None:
            manager.max_llm_calls_per_tick = max(0, int(max_calls))
        manager._in_tick = True
        manager._llm_calls = 0
        try:
            report = manager.external_critic(
                inventory,
                max_new=min(
                    max(0, int(max_leaves if max_leaves is not None else manager.max_leaves_per_tick)),
                    20,
                ),
            )
        finally:
            manager._in_tick = False
        proposals = manager._validate_external_proposals(
            report.get("proposals") or [],
            max_new=min(max(0, int(max_leaves if max_leaves is not None else manager.max_leaves_per_tick)), 20),
        )
        existing = {leaf.l3_id for leaf in taxonomy.leaves}
        if proposals:
            manager._commit_leaves(proposals, apply=False)
        new_proposals = [dict(vars(leaf)) for leaf in taxonomy.leaves if leaf.l3_id not in existing]
        structure = {
            "new_axes": report.get("new_axes") or [],
            "split": report.get("split") or report.get("splits") or [],
            "merge": report.get("merge") or report.get("merges") or [],
            "health_notes": report.get("health_notes") or [],
        }
        if apply:
            self.store.set_metadata("taxonomy_structure_suggestions", structure)
        added = self.publish_proposals(new_proposals) if apply else 0
        return {
            "status": report.get("status") or "gap_detected",
            "coverage": report.get("coverage") or {},
            "proposed": len(new_proposals),
            "total_added": added,
            "proposals": new_proposals,
            "structure_suggestions": structure,
            "llm_calls": manager._llm_calls,
            "applied": apply,
        }

    def _packs_dir(self) -> Path | None:
        root = getattr(self.settings, "root", None)
        if root is None:
            return None
        path = Path(root) / "configs" / "knowledge" / "packs"
        return path if path.is_dir() else None

    def _next_pack_id(self) -> str | None:
        from cuda_sft_qgen.pack import list_packs, load_pack
        directory = self._packs_dir()
        if directory is None:
            return None
        packs = []
        for path in list_packs(directory):
            try:
                packs.append(load_pack(path))
            except ValueError:
                continue
        packs.sort(key=lambda p: int(p.get("priority") or 99))
        if not packs:
            return None
        last = self.store.get_metadata("last_expand_pack_id")
        ids = [p["id"] for p in packs]
        if last in ids:
            return ids[(ids.index(last) + 1) % len(ids)]
        return ids[0]

    def run_once(self) -> bool:
        """Lease and execute one job. Returns True if a job ran."""
        if self._stop.is_set():
            return False
        self.deliver_impacts()
        pool = getattr(self.llm, "pool", None)
        busy = getattr(pool, "llm_busy_for_taxonomy", None)
        # Test doubles and lightweight clients may expose arbitrary callable
        # attributes; only an explicit boolean ``True`` means taxonomy slots
        # are busy.
        if callable(busy):
            busy_value = busy()
            if isinstance(busy_value, bool) and busy_value:
                return False
        if self._stop.is_set():
            return False
        job = self.store.take_job()
        if job is None:
            return False
        try:
            from cuda_sft_qgen.observability import agent_role

            with agent_role("knowledge"):
                if job["kind"] == "audit":
                    self._audit(job["payload"])
                elif job["kind"] == "compute_verify":
                    self._compute_verify(job["payload"])
                elif job["kind"] == "external_inventory":
                    self.external_inventory_once(
                        force=bool(job["payload"].get("force")), apply=True
                    )
                elif job["kind"] in {"expand", "expand_pack"}:
                    self._expand(job["payload"])
                elif job["kind"] == "relate":
                    self._relate(job["payload"])
                else:
                    raise ValueError(f"unknown knowledge job: {job['kind']}")
            self.store.finish_job(job["id"])
        except Exception as exc:
            logger.warning("knowledge job %s failed: %s", job["id"], exc)
            self.store.finish_job(job["id"], error=str(exc)[:1000])
        return True

    def _complete_json(self, *args, **kwargs):
        from cuda_sft_qgen.observability import agent_role

        with agent_role("knowledge"):
            return self.llm.complete_json(*args, **kwargs)

    def _compute_verify(self, payload: dict) -> None:
        """Extract and deterministically check one formula or numeric claim."""
        claim = self.store.get_claim(payload["claim_id"])
        if (claim is None or claim["revision"] != payload["revision"]
                or claim["status"] in {"retired", "quarantined"}):
            return
        if claim.get("kind") not in {"formula", "numeric"}:
            raise ValueError("computation path only for formula/numeric")

        from cuda_sft_qgen import hardware_facts
        from cuda_sft_qgen.prompts import load_prompt

        facts_path = getattr(self.settings, "hardware_facts_path", None)
        if facts_path is None:
            root = Path(getattr(self.settings, "root", None) or Path.cwd())
            facts_path = root / "configs" / "hardware_facts.yaml"
        facts_text = hardware_facts.render_all_verified(hardware_facts.load(str(facts_path)))
        verified_claims = [
            str(item.get("claim") or "")
            for item in self.store.list_claims("verified")
            if item.get("l2_id") == claim.get("l2_id") and item.get("id") != claim["id"]
        ][:20]
        user = {
            "claim": claim["claim"],
            "kind": claim["kind"],
            "hardware_facts": facts_text,
            "verified_claims": verified_claims,
        }
        obj = self._complete_json([
            {"role": "system", "content": load_prompt("compute_extract")},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ], retries=1, priority=PRI_TAXONOMY)

        expression = str(obj.get("expression") or "").strip()
        trace = str(obj.get("trace") or "")
        if not expression:
            self.store.record_insufficient_audit(
                claim["id"], claim["revision"], reason=trace or "not_computable",
            )
            return

        trusted_sources = [_norm_ws(facts_text)]
        trusted_sources.extend(_norm_ws(text) for text in verified_claims)
        constants = obj.get("constants_used")
        if not isinstance(constants, list) or any(
            not isinstance(constant, str)
            or not _norm_ws(constant)
            or not any(_norm_ws(constant) in source for source in trusted_sources)
            for constant in constants
        ):
            self.store.record_insufficient_audit(
                claim["id"], claim["revision"], reason="unsupported_constant",
            )
            return

        variables = obj.get("variables")
        if not isinstance(variables, dict):
            self.store.record_insufficient_audit(
                claim["id"], claim["revision"], reason="invalid_variables",
            )
            return
        expected = str(obj.get("expected") or "").strip()
        if not expected:
            self.store.record_insufficient_audit(
                claim["id"], claim["revision"], reason="missing_expected_value",
            )
            return
        try:
            self.store.apply_computation_check(
                claim["id"], claim["revision"], expression=expression,
                variables=variables, expected=expected, trace=trace,
            )
        except ValueError as exc:
            self.store.record_insufficient_audit(
                claim["id"], claim["revision"], reason=f"computation_invalid:{exc}",
            )

    def refine_legacy_batch(self, l2_key: str, *, limit: int = 10, apply: bool = False) -> dict:
        """Rewrite one L2 batch into falsifiable provisional claims."""
        if self.taxonomy is None:
            raise ValueError("taxonomy is required for legacy refinement")
        from cuda_sft_qgen.taxonomy_expand import validate_leaf
        from cuda_sft_qgen.taxonomy_manager import TaxonomyManager

        taxonomy = self.store.overlay_taxonomy(self.taxonomy)
        selected = sorted(
            (leaf for leaf in taxonomy.leaves
             if f"{leaf.domain_id}.{leaf.l2_id}" == l2_key
             and leaf.status == "legacy_unreviewed" and needs_refine(leaf)),
            key=lambda leaf: leaf.l3_id,
        )[:max(0, int(limit))]
        ids = [leaf.l3_id for leaf in selected]
        rows: list[dict[str, Any]] = []
        if not selected:
            return {"l2": l2_key, "selected": 0, "valid_suggestions": 0, "applied": 0, "rows": rows}

        manager = TaxonomyManager(taxonomy, self.llm, settings=self.settings)
        result = manager.refine_leaf_quality(ids)
        suggestions = {str(item.get("l3_id")): item for item in result.get("suggestions") or []
                       if isinstance(item, dict) and item.get("l3_id")}
        all_ids = {leaf.l3_id for leaf in taxonomy.leaves}
        valid_count = applied = 0
        for leaf in selected:
            suggestion = suggestions.get(leaf.l3_id) or {}
            proposed = str(suggestion.get("suggested_claim") or "").strip()
            hints = suggestion.get("suggested_hints")
            reason = "missing_suggestion"
            if proposed and isinstance(hints, list) and hints and all(isinstance(hint, str) and hint.strip() for hint in hints):
                item = {"l3_id": leaf.l3_id, "claim": proposed, "kind": leaf.kind,
                        "types": list(leaf.types), "difficulty": list(leaf.difficulty), "hints": hints}
                reason = validate_leaf(
                    item, l2_id=leaf.l2_id, existing_ids=all_ids - {leaf.l3_id},
                    existing_claims=[other.claim for other in taxonomy.leaves if other.l3_id != leaf.l3_id],
                ) or ""
            elif proposed:
                reason = "missing_hints"
            valid = not reason
            if valid:
                valid_count += 1
                if apply:
                    revision = self.store.revise(
                        leaf.l3_id, {"claim": proposed, "hints": hints, "status": "provisional"},
                        reason="legacy_refine", expected_revision=leaf.revision,
                    )
                    kind = "compute_verify" if leaf.kind in {"formula", "numeric"} else "audit"
                    prefix = "compute" if kind == "compute_verify" else "audit"
                    self.store.enqueue(kind, {"claim_id": leaf.l3_id, "revision": revision},
                                       f"{prefix}:{leaf.l3_id}:{revision}")
                    applied += 1
            rows.append({"l3_id": leaf.l3_id, "old": leaf.claim, "new": proposed,
                         "hints": hints if isinstance(hints, list) else [], "valid": valid, "reason": reason})
        if applied:
            self.deliver_impacts()
        return {"l2": l2_key, "selected": len(selected), "valid_suggestions": valid_count,
                "applied": applied, "rows": rows}

    def retrieve_documents(self, claim: dict) -> list[tuple[str, str]]:
        """Prefer local documentation chunks, falling back to official URLs."""
        query = claim["claim"] + " " + " ".join(claim.get("hints") or [])
        hits = self.doc_index.search(query, k=6) if len(self.doc_index) else []
        self._retrieved_local = bool(hits)
        if hits:
            by_url: dict[str, list[str]] = {}
            for chunk, _score in hits:
                by_url.setdefault(chunk.url, []).append(chunk.text)
            return [(url, "\n\n".join(texts)) for url, texts in by_url.items()]
        urls = [str(url) for url in claim.get("sources", []) if official_url(str(url))]
        urls.extend(str(url) for url in self.config.get("official_sources", []) if official_url(str(url)))
        # URL suggestions are merely retrieval candidates, never evidence by themselves.
        if not urls:
            from cuda_sft_qgen.prompts import load_prompt
            suggestions = self._complete_json([
                {"role": "system", "content": load_prompt("url_suggest")},
                {"role": "user", "content": claim["claim"]},
            ], retries=1, priority=PRI_TAXONOMY)
            urls.extend(str(url) for url in suggestions.get("urls", []) if official_url(str(url)))
        urls = list(dict.fromkeys(urls or [DEFAULT_SOURCE]))[:3]
        retrieved: list[tuple[str, str]] = []
        failures = []
        for url in urls:
            try:
                cached = self._documents.get(url)
                cache_ttl = min(86400, max(0, float(self.config.get("audit_interval_seconds", 86400))))
                document = cached[1] if cached and time.time() - cached[0] < cache_ttl else self.fetcher(url)
                self._documents[url] = (time.time(), document)
                retrieved.append((url, document))
            except Exception as exc:
                failures.append(str(exc))
        if not retrieved:
            raise RuntimeError("official source retrieval failed: " + "; ".join(failures))
        return retrieved

    def _audit(self, payload: dict) -> None:
        claim = self.store.get_claim(payload["claim_id"])
        if claim is None or claim["revision"] != payload["revision"] or claim["status"] in {"retired", "quarantined"}:
            return
        retrieved = self.retrieve_documents(claim)
        related_claims = self.store.relation_candidates(claim["id"], limit=8, embedder=self.embedder)
        from cuda_sft_qgen.prompts import load_prompt
        review = self._complete_json([
            {"role": "system", "content": load_prompt("audit")},
            {"role": "user", "content": json.dumps({"claim": claim, "related_claims": related_claims, "documents": [{"url": url, "text": doc if self._retrieved_local else _passages(doc, claim["claim"])} for url, doc in retrieved]}, ensure_ascii=False)},
        ], retries=1, priority=PRI_TAXONOMY)
        verdict = str(review.get("verdict") or "insufficient")
        if verdict not in {"supported", "contradicted", "insufficient"}:
            verdict = "insufficient"
        url = str(review.get("url") or "")
        quote = str(review.get("quote") or "").strip()
        document = dict(retrieved).get(url, "")
        if verdict != "insufficient" and (not quote or _norm_ws(quote) not in _norm_ws(document)):
            verdict = "insufficient"
        if verdict == "supported":
            # A real quote can still support only half a generated assertion.
            # Require literal coverage of material identifiers and numbers
            # before asking an independent auditor about the semantic clauses.
            from cuda_sft_qgen.taxonomy_manager import TaxonomyManager
            material = TaxonomyManager._technical_tokens(claim["claim"])
            material.update(re.findall(r"\b\d+(?:\.\d+)?\b", claim["claim"]))
            quoted = TaxonomyManager._technical_tokens(quote)
            quoted.update(re.findall(r"\b\d+(?:\.\d+)?\b", quote))
            if material - quoted:
                verdict = "insufficient"
                review["reason"] = "quoted_evidence_missing_material_identifiers_or_numbers"
            else:
                confirmation = self._complete_json([
                    {"role": "system", "content": load_prompt("audit") +
                     "\nIndependently check EVERY material clause using only this retrieved excerpt. "
                     "Do not infer unstated constraints. Return supported only if all clauses follow."},
                    {"role": "user", "content": json.dumps({"claim": claim,
                     "documents": [{"url": url, "text": quote}]}, ensure_ascii=False)},
                ], retries=0, priority=PRI_TAXONOMY)
                second_quote = str(confirmation.get("quote") or "").strip()
                if (confirmation.get("verdict") != "supported" or confirmation.get("url") != url
                        or not second_quote or _norm_ws(second_quote) not in _norm_ws(quote)):
                    verdict = "insufficient"
                    review["reason"] = "independent_audit_not_confirmed"
        before = self.store.impact_events()
        event_cursor = before[-1]["id"] if before else 0
        applicability = review.get("applicability")
        applied = self.store.apply_audit(claim["id"], claim["revision"], verdict=verdict, url=url, quote=quote, document=document, reason=str(review.get("reason") or ""), applicability=applicability if isinstance(applicability, dict) else None)
        if not applied:
            return
        if verdict == "contradicted":
            self._invalidate(event_cursor)
            correction = str(review.get("corrected_claim") or "").strip()
            if correction and correction != claim["claim"]:
                revision = self.store.revise(claim["id"], {"claim": correction, "status": "provisional", "sources": [url], "applicability": review.get("applicability") or claim.get("applicability") or {}}, reason="source_grounded_correction", expected_revision=claim["revision"])
                self._invalidate(event_cursor)
                self.store.enqueue("audit", {"claim_id": claim["id"], "revision": revision}, f"audit:{claim['id']}:{revision}")
        if verdict == "supported":
            for edge in review.get("relations") or []:
                if not isinstance(edge, dict) or not edge.get("quote") or edge["quote"] not in document:
                    continue
                if edge.get("target_id") not in {item["id"] for item in related_claims}:
                    continue
                try:
                    evidence_id = self.store.add_evidence(claim["id"], claim["revision"], url=url, quote=edge["quote"], document=document)
                    self.store.add_relation(claim["id"], edge["target_id"], edge["kind"], evidence_id=evidence_id, status="verified", reason=edge.get("reason", ""))
                except (ValueError, KeyError, StopIteration):
                    logger.warning("discard invalid audited relation for %s", claim["id"])

    def _invalidate(self, after_id: int) -> None:
        self.deliver_impacts()

    def deliver_impacts(self) -> None:
        """Push undelivered knowledge impact events into the question archive."""
        if self.archive is None or not hasattr(self.archive, "invalidate_claim"):
            return
        after_id = int(self.store.get_metadata("delivered_impact_event", 0))
        for event in self.store.impact_events(after_id):
            if event["new_revision"] != event["old_revision"] or event["reason"].startswith("audit:contradicted:"):
                self.archive.invalidate_claim(event["claim_id"], event["old_revision"], event["reason"], event["id"], question_ids=event["question_ids"])
            self.store.set_metadata("delivered_impact_event", event["id"])

    def publish_proposals(self, proposals: list[dict]) -> int:
        """Insert provisional claims and enqueue an audit job for each new id."""
        added = 0
        for proposal in proposals:
            if self.store.add_claim(proposal, status="provisional"):
                claim_id = proposal.get("l3_id") or proposal["id"]
                self.store.enqueue("audit", {"claim_id": claim_id, "revision": 1}, f"audit:{claim_id}:1")
                added += 1
        return added

    def expand_once(self, *, apply: bool = False, mode: str = "tick", max_leaves: int | None = None,
                    max_calls: int | None = None, facet: str | None = None, domain_ids: list[str] | None = None,
                    l2_key: str | None = None, target_per_l2: int | None = None) -> dict:
        """Run one expansion tick (or pack/domain mode); enqueue audits for new claims."""
        if self.taxonomy is None:
            return {"total_added": 0, "proposals": []}
        from cuda_sft_qgen.taxonomy_manager import TaxonomyManager
        taxonomy = self.store.overlay_taxonomy(self.taxonomy)
        # Expansion must be CPU/local and durable; scheduled inventory jobs own
        # all network fetching and refreshes.
        inventory = list(self.store.get_metadata("external_concept_inventory", []) or [])
        tax_config = (getattr(self.settings, "default_yaml", {}) or {}).get("taxonomy") or {}
        manager = TaxonomyManager(taxonomy, self.llm, embedder=self.embedder, embed_index=self._taxonomy_index,
                                  settings=self.settings, external_inventory=inventory,
                                  target_leaves_per_l2=int(target_per_l2 or tax_config.get("target_leaves_per_l2", 30)))
        if not manager.enabled:
            return {"total_added": 0, "proposals": [], "reason": "disabled"}
        if max_calls is not None:
            manager.max_llm_calls_per_tick = max(0, max_calls)
        if facet:
            manager.facet_ids = [facet]
        cap = manager.max_leaves_per_tick if max_leaves is None else max(0, max_leaves)
        if cap <= 0 or max_calls == 0:
            return {"total_added": 0, "proposals": [], "proposed": 0, "applied": apply, "reason": "zero_budget"}
        signature = tuple((leaf.l3_id, leaf.revision, leaf.retired) for leaf in taxonomy.leaves)
        if manager.embed_index is not None and signature != self._taxonomy_signature:
            manager.embed_index.rebuild(taxonomy.leaves, save=False)
        self._taxonomy_index = manager.embed_index
        accepted = self.archive.load_accepted() if self.archive else []
        rejected = self.archive.load_rejected() if self.archive else []
        existing = {leaf.l3_id for leaf in taxonomy.leaves}
        if mode == "strategic":
            result = manager.expand_strategic(max_leaves_to_add=cap, max_llm_calls=max_calls,
                                               facets=[facet] if facet else None, domain_ids=domain_ids,
                                               accepted=accepted, rejected=rejected, apply=False)
        elif mode in {"gap", "l2"}:
            health = manager.analyze_health(accepted, rejected)
            target = l2_key or manager._pick_gap_l2(health, None)
            pair = manager._lookup_l2(target) if target else None
            result = {"target": target, "total_added": 0}
            if pair and cap > 0 and (max_calls is None or max_calls > 0):
                selected_facet = manager._facet_dict(facet or manager.choose_facet(health))
                if mode == "gap":
                    proposals = manager.gap_search(target_l2_key=target, health=health, rejected=rejected, max_new=min(cap, 20), facet=selected_facet)
                else:
                    proposals = manager._expand_l2(domain=pair[0], child=pair[1], facet=selected_facet, max_new=min(cap, 20))
                manager._commit_leaves(proposals, apply=False)
        else:
            result = manager.run_tick(accepted=accepted, rejected=rejected, apply=False,
                                      max_leaves=cap, last_facet=self.store.get_metadata("last_expansion_facet"))
        proposals = [dict(vars(leaf)) for leaf in taxonomy.leaves if leaf.l3_id not in existing]
        added = self.publish_proposals(proposals) if apply else 0
        result.update(total_added=added, proposals=proposals, proposed=len(proposals), applied=apply)
        self._taxonomy_signature = () if not apply else tuple((leaf.l3_id, leaf.revision, leaf.retired) for leaf in taxonomy.leaves)
        if apply:
            self.store.set_metadata("last_expansion_at", time.time())
            self.store.set_metadata("last_expansion_facet", result.get("facet") or facet)
            self.store.set_metadata("last_expansion_result", result)
        return result

    def refine_once(self, *, apply: bool = False) -> dict:
        """Optional claim-refine pass; no-op unless pipeline refine is enabled."""
        from cuda_sft_qgen.taxonomy_manager import TaxonomyManager
        taxonomy = self.store.overlay_taxonomy(self.taxonomy)
        manager = TaxonomyManager(taxonomy, self.llm, settings=self.settings)
        report = manager.analyze_health()
        result = manager.refine_leaf_quality(report.refinement_needed)
        patched = 0
        if apply:
            for suggestion in result.get("suggestions", []):
                claim_id = suggestion.get("l3_id")
                claim = self.store.get_claim(claim_id) if claim_id else None
                if not claim or not suggestion.get("suggested_claim"):
                    continue
                changes = {"claim": suggestion["suggested_claim"], "status": "provisional"}
                if suggestion.get("suggested_hints") is not None:
                    changes["hints"] = suggestion["suggested_hints"]
                revision = self.store.revise(claim_id, changes, reason="manager_refine", expected_revision=claim["revision"])
                self.store.enqueue("audit", {"claim_id": claim_id, "revision": revision}, f"audit:{claim_id}:{revision}")
                patched += 1
            self.deliver_impacts()
        return {**result, "patched": patched, "applied": apply}

    def _expand(self, payload: dict) -> None:
        last_expansion = float(self.store.get_metadata("last_expansion_at", 0))
        if time.time() - last_expansion < float(self.config.get("expansion_cooldown_seconds", 300)):
            return
        pack_id = payload.get("pack_id")
        if pack_id:
            result = self.expand_pack(pack_id, apply=True, max_leaves=int(self.config.get("expand_max_leaves", 8)))
            self.store.set_metadata("last_expand_pack_id", pack_id)
        else:
            result = self.expand_once(apply=True)
        logger.info("knowledge expansion completed added=%s facet=%s pack=%s", result.get("total_added"), result.get("facet"), pack_id)

    def expand_pack(self, pack_id: str, *, apply: bool = False, max_leaves: int = 8, facet: str | None = None) -> dict:
        """Expand L2 targets listed in a curriculum pack."""
        from cuda_sft_qgen.pack import list_packs, load_pack, curriculum_frontier
        directory = self._packs_dir()
        pack = None
        if directory is not None:
            for path in list_packs(directory):
                loaded = load_pack(path)
                if loaded["id"] == pack_id:
                    pack = loaded
                    break
        if pack is None:
            return {"total_added": 0, "reason": "unknown_pack", "pack": pack_id}
        frontier = curriculum_frontier(self.store, pack)
        added = 0
        details = []
        remaining = max(0, max_leaves)
        facets = list(pack.get("facets") or [])
        chosen = facet or (facets[0] if facets else None)
        for gap in frontier["l2"]:
            if remaining <= 0:
                break
            if gap["missing"] <= 0:
                continue
            l2_key = f"{gap['domain']}.{gap['l2']}"
            result = self.expand_once(
                apply=apply, mode="l2", l2_key=l2_key, facet=chosen,
                max_leaves=min(remaining, gap["missing"], 8),
            )
            n = int(result.get("total_added") or 0)
            added += n
            remaining -= n
            details.append({"l2": l2_key, **{k: result[k] for k in result if k != "proposals"}})
        return {"pack": pack_id, "total_added": added, "applied": apply, "frontier": frontier, "details": details, "facet": chosen}

    def _relate(self, payload: dict) -> None:
        claim = self.store.get_claim(payload.get("claim_id") or "")
        if claim is None or claim["status"] in {"retired", "quarantined"}:
            return
        from cuda_sft_qgen.prompts import load_prompt
        candidates = self.store.relation_candidates(claim["id"], limit=8, embedder=self.embedder)
        if not candidates:
            return
        review = self._complete_json([
            {"role": "system", "content": load_prompt("relate")},
            {"role": "user", "content": json.dumps({"claim": claim, "candidates": [
                {"id": c["id"], "claim": c["claim"], "l2_id": c.get("l2_id")} for c in candidates
            ]}, ensure_ascii=False)},
        ], retries=1, priority=PRI_TAXONOMY)
        allowed = {c["id"] for c in candidates}
        for edge in review.get("relations") or []:
            if not isinstance(edge, dict):
                continue
            target = edge.get("target_id")
            kind = edge.get("kind")
            if target not in allowed:
                continue
            try:
                self.store.add_relation(
                    claim["id"], target, kind,
                    status="provisional",
                    reason=str(edge.get("reason") or "relate_job")[:300],
                )
            except (ValueError, KeyError):
                logger.warning("discard invalid relate edge %s -> %s", claim["id"], target)
