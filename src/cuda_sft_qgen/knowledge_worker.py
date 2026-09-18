"""Background expansion and source-grounded audit; no runtime ContextVar access."""
from __future__ import annotations

from copy import deepcopy
from html.parser import HTMLParser
import json
import logging
import re
import threading
import time
from typing import Any, Callable

import httpx

from cuda_sft_qgen.keys import PRI_TAXONOMY
from cuda_sft_qgen.store.knowledge import KnowledgeStore, official_url

logger = logging.getLogger(__name__)
DEFAULT_SOURCE = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"


class _DocumentText(HTMLParser):
    """Strip script/style and collect visible text from official HTML docs."""

    def __init__(self):
        """Initialize the HTMLParser and visible-text buffers."""
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        """Ignore nested script/style sections."""
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        """Leave a hidden script/style section."""
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        """Append visible text runs, collapsing inner whitespace."""
        if not self.hidden and data.strip():
            self.parts.append(" ".join(data.split()))


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
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_schedule = time.monotonic()
        self._documents: dict[str, tuple[float, str]] = {}
        self._taxonomy_index = None
        self._taxonomy_signature: tuple = ()

    def start(self) -> None:
        """Start the daemon poll loop if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="knowledge-audit", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the loop to exit and join the worker thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0, timeout))

    close = stop

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if time.monotonic() - self._last_schedule >= 60:
                    self.store.enqueue_daily_audits(int(self.config.get("audit_batch_size", 100)), interval_seconds=float(self.config.get("audit_interval_seconds", 86400)))
                    self._last_schedule = time.monotonic()
                if self.run_once():
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

    def _packs_dir(self) -> Path | None:
        from pathlib import Path
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
        self.deliver_impacts()
        pool = getattr(self.llm, "pool", None)
        busy = getattr(pool, "llm_busy_for_taxonomy", None)
        if callable(busy) and busy():
            return False
        job = self.store.take_job()
        if job is None:
            return False
        try:
            if job["kind"] == "audit":
                self._audit(job["payload"])
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

    def _audit(self, payload: dict) -> None:
        claim = self.store.get_claim(payload["claim_id"])
        if claim is None or claim["revision"] != payload["revision"] or claim["status"] in {"retired", "quarantined"}:
            return
        urls = [str(url) for url in claim.get("sources", []) if official_url(str(url))]
        urls.extend(str(url) for url in self.config.get("official_sources", []) if official_url(str(url)))
        # URL suggestions are merely retrieval candidates, never evidence by themselves.
        if not urls:
            from cuda_sft_qgen.prompts import load_prompt
            suggestions = self.llm.complete_json([
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
        related_claims = self.store.relation_candidates(claim["id"])
        from cuda_sft_qgen.prompts import load_prompt
        review = self.llm.complete_json([
            {"role": "system", "content": load_prompt("audit")},
            {"role": "user", "content": json.dumps({"claim": claim, "related_claims": related_claims, "documents": [{"url": url, "text": _passages(doc, claim["claim"])} for url, doc in retrieved]}, ensure_ascii=False)},
        ], retries=1, priority=PRI_TAXONOMY)
        verdict = str(review.get("verdict") or "insufficient")
        if verdict not in {"supported", "contradicted", "insufficient"}:
            verdict = "insufficient"
        url = str(review.get("url") or "")
        quote = str(review.get("quote") or "").strip()
        document = dict(retrieved).get(url, "")
        if verdict != "insufficient" and (not quote or quote not in document):
            verdict = "insufficient"
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
        tax_config = (getattr(self.settings, "default_yaml", {}) or {}).get("taxonomy") or {}
        manager = TaxonomyManager(taxonomy, self.llm, embedder=self.embedder, embed_index=self._taxonomy_index,
                                  settings=self.settings, target_leaves_per_l2=int(target_per_l2 or tax_config.get("target_leaves_per_l2", 30)))
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
        candidates = self.store.relation_candidates(claim["id"], limit=8)
        if not candidates:
            return
        review = self.llm.complete_json([
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
