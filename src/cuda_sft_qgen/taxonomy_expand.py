"""LLM-driven L3 leaf expansion with id/claim validation and embed dedup."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from cuda_sft_qgen.keys import PRI_TAXONOMY
from cuda_sft_qgen.models import QuestionType
from cuda_sft_qgen.review.fingerprint import skeletonize, token_jaccard
from cuda_sft_qgen.taxonomy import LEAF_KINDS, Taxonomy, dump_taxonomy, merge_leaves
from cuda_sft_qgen.taxonomy_embed import TaxonomyEmbedIndex

logger = logging.getLogger(__name__)

_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_ARCH = re.compile(
    r"sm_?\d+|hopper|blackwell|ampere|volta|turing|ada|h100|b200|a100|cuda\s*\d",
    re.IGNORECASE,
)
_PROMO = re.compile(r"提升了性能|是一种.{0,12}技术|大大提高|革命性")
VALID_TYPES = set(QuestionType.__args__)  # type: ignore[attr-defined]


def normalize_leaf_item(item: dict[str, Any], *, l2_id: str) -> dict[str, Any]:
    """Coerce LLM leaf fields so a usable claim is not dropped for enum typos."""
    out = dict(item)
    l2 = str(l2_id or out.get("l2_id") or "")
    l3 = str(out.get("l3_id") or out.get("id") or "")
    if l2 and l3 and not l3.startswith(f"{l2}_"):
        out["l3_id"] = f"{l2}_{l3}"
    types = [str(t) for t in (out.get("types") or []) if str(t) in VALID_TYPES]
    if not types:
        types = ["conceptual"]
    out["types"] = types
    kind = str(out.get("kind") or "").strip()
    if kind not in LEAF_KINDS:
        out["kind"] = "mechanism"
    return out


def validate_leaf(
    item: dict[str, Any],
    *,
    l2_id: str,
    existing_ids: set[str],
    existing_claims: list[str],
) -> str | None:
    """Return a reject code (``bad_id``, ``dup_id``, ...) or None if the leaf is usable."""
    l3 = str(item.get("l3_id") or item.get("id") or "")
    claim = str(item.get("claim") or "").strip()
    kind = str(item.get("kind") or "").strip()
    if not l3 or not _SNAKE.match(l3):
        return "bad_id"
    if not l3.startswith(f"{l2_id}_"):
        return "bad_id"
    if l3 in existing_ids:
        return "dup_id"
    if len(claim) < 20:
        return "short_claim"
    difficulty = item.get("difficulty", [2])
    if not isinstance(difficulty, (list, tuple)) or not difficulty or any(type(value) is not int or not 1 <= value <= 4 for value in difficulty):
        return "bad_difficulty"
    for field in ("hints", "sources"):
        values = item.get(field, [])
        if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
            return f"bad_{field}"
    if not isinstance(item.get("applicability", {}), dict):
        return "bad_applicability"
    if _PROMO.search(claim) and not _ARCH.search(claim):
        return "promo"
    if kind and kind not in LEAF_KINDS:
        return "bad_kind"
    types = [str(t) for t in (item.get("types") or [])]
    if any(t not in VALID_TYPES for t in types):
        return "bad_type"
    if kind == "formula" and not re.search(r"[=→]|=>|->", claim):
        return "formula_no_eq"
    if kind == "feature" and not _ARCH.search(claim):
        return "feature_no_arch"
    skel = skeletonize(claim)
    for prev in existing_claims:
        if token_jaccard(skel, skeletonize(prev)) >= 0.88:
            return "near_dup"
    return None


def _existing_leaf_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    hints = item.get("hints") or []
    if isinstance(hints, str):
        n_hints = 0 if not hints.strip() else 1
    elif isinstance(hints, (list, tuple)):
        n_hints = len(hints)
    else:
        n_hints = 1
    return (n_hints, len(str(item.get("claim") or "")))


def existing_leaves_payload(child: dict[str, Any], *, cap: int = 40) -> list[dict[str, Any]]:
    """Compact active-leaf list for expand prompts, poorest claims first."""
    existing = sorted(
        [item for item in (child.get("leaves") or []) if not item.get("retired")],
        key=_existing_leaf_sort_key,
    )
    return [
        {"id": item.get("id"), "claim": item.get("claim"), "kind": item.get("kind")}
        for item in existing[:cap]
    ]


def _render(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def render_expand_user(
    *,
    domain: dict[str, Any],
    child: dict[str, Any],
    facet: dict[str, Any],
    max_new: int,
    near_dup_warnings: str,
    template: str,
) -> str:
    """Fill the expand.md user template for one L2 + facet."""
    payload = existing_leaves_payload(child)
    mapping = {
        "domain_id": str(domain.get("id") or ""),
        "domain_name": str(domain.get("name") or domain.get("id") or ""),
        "l2_id": str(child.get("id") or ""),
        "l2_name": str(child.get("name") or child.get("id") or ""),
        "facet_id": str(facet.get("id") or ""),
        "facet_must": str(facet.get("must") or ""),
        "facet_must_not": str(facet.get("must_not") or ""),
        "facet_example": str(facet.get("example_claim") or ""),
        "max_new_leaves": str(max_new),
        "max_new": str(max_new),
        "existing_leaves_json": json.dumps(payload, ensure_ascii=False),
        "existing_leaves": json.dumps(payload, ensure_ascii=False),
        "near_dup_warnings": near_dup_warnings or "",
        "gap_prompts": str(facet.get("gap_prompts") or ""),
    }
    return _render(template, mapping)


def expand_one_l2(
    taxonomy: Taxonomy,
    llm: Any,
    domain: dict[str, Any],
    child: dict[str, Any],
    *,
    target: int,
    max_per_call: int,
    system: str,
    embed_index: TaxonomyEmbedIndex | None = None,
    cosine_filter: bool = True,
) -> int:
    """Expand one L2 toward ``target`` leaves; returns the number added this call."""
    added_total = 0
    while len(child.get("leaves") or []) < target:
        existing = list(child.get("leaves") or [])
        need = min(max_per_call, target - len(existing))
        user = (
            "Expand ONLY this L2. Mix kinds. Prefix every l3_id with the l2_id.\n"
            + json.dumps(
                {
                    "domain_id": domain["id"],
                    "domain_name": domain.get("name"),
                    "l2_id": child["id"],
                    "l2_name": child.get("name"),
                    "existing_leaves": existing_leaves_payload(child),
                    "max_new_leaves": need,
                },
                ensure_ascii=False,
            )
        )
        try:
            obj = llm.complete_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                retries=1,
                priority=PRI_TAXONOMY,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("expand %s.%s failed: %s", domain["id"], child["id"], exc)
            break
        existing_ids = {leaf.l3_id for leaf in taxonomy.leaves}
        claims = [str(x.get("claim") or "") for x in existing]
        accepted: list[dict[str, Any]] = []
        for raw in obj.get("new_leaves") or []:
            if not isinstance(raw, dict):
                continue
            raw = dict(raw)
            raw["domain_id"] = domain["id"]
            raw["l2_id"] = child["id"]
            raw["l2_name"] = child.get("name")
            raw.setdefault("domain_name", domain.get("name"))
            reason = validate_leaf(
                raw,
                l2_id=str(child["id"]),
                existing_ids=existing_ids | {x.get("l3_id") or x.get("id") for x in accepted},
                existing_claims=claims + [str(x.get("claim") or "") for x in accepted],
            )
            if reason:
                logger.debug("drop leaf %s: %s", raw.get("l3_id"), reason)
                continue
            accepted.append(raw)
        if embed_index is not None and cosine_filter and accepted:
            try:
                keep, dropped, _gray = embed_index.filter_candidates(accepted)
                if dropped:
                    logger.debug("embed drop %s dups for %s", len(dropped), child.get("id"))
                accepted = keep
            except Exception as exc:  # noqa: BLE001
                logger.warning("expand embed filter failed: %s", exc)
        if not accepted:
            break
        n = merge_leaves(taxonomy, accepted, allow_new_domains=False)
        added_total += n
        if n == 0:
            break
        if embed_index is not None:
            try:
                embed_index.add_leaves(accepted)
            except Exception as exc:  # noqa: BLE001
                logger.warning("expand embed add failed: %s", exc)
        child = next(
            c for c in domain.get("children") or [] if c["id"] == child["id"]
        )
    return added_total


def expand_taxonomy(
    taxonomy: Taxonomy,
    llm: Any,
    *,
    per_l2: int = 30,
    max_per_call: int = 40,
    domain_ids: list[str] | None = None,
    system: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Expand all (or selected) L2 nodes; dump taxonomy when ``apply`` is True."""
    added = 0
    for domain in taxonomy.domains:
        if domain_ids and domain["id"] not in domain_ids:
            continue
        for child in list(domain.get("children") or []):
            added += expand_one_l2(
                taxonomy,
                llm,
                domain,
                child,
                target=per_l2,
                max_per_call=max_per_call,
                system=system,
            )
    if apply and added:
        taxonomy.version = int(taxonomy.version) + 1
        dump_taxonomy(taxonomy)
    return {"added": added, "total": len(taxonomy.leaves)}
