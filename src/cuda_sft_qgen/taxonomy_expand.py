from __future__ import annotations

import logging
import re
from typing import Any

from cuda_sft_qgen.models import QuestionType
from cuda_sft_qgen.review.fingerprint import skeletonize, token_jaccard
from cuda_sft_qgen.taxonomy import LEAF_KINDS, Taxonomy, dump_taxonomy, merge_leaves

logger = logging.getLogger(__name__)

_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_ARCH = re.compile(
    r"sm_?\d+|hopper|blackwell|ampere|volta|turing|cuda\s*\d",
    re.IGNORECASE,
)
_PROMO = re.compile(r"提升了性能|是一种.{0,12}技术|大大提高|革命性")
VALID_TYPES = set(QuestionType.__args__)  # type: ignore[attr-defined]


def validate_leaf(
    item: dict[str, Any],
    *,
    l2_id: str,
    existing_ids: set[str],
    existing_claims: list[str],
) -> str | None:
    l3 = str(item.get("l3_id") or item.get("id") or "")
    claim = str(item.get("claim") or "").strip()
    kind = str(item.get("kind") or "").strip()
    if not l3 or not _SNAKE.match(l3):
        return "bad_id"
    if not l3.startswith(f"{l2_id}_") and l3 in existing_ids:
        return "dup_id"
    if l3 in existing_ids:
        return "dup_id"
    if len(claim) < 20:
        return "short_claim"
    if _PROMO.search(claim) and not _ARCH.search(claim):
        return "promo"
    if kind and kind not in LEAF_KINDS:
        return "bad_kind"
    types = [str(t) for t in (item.get("types") or [])]
    if any(t not in VALID_TYPES for t in types):
        return "bad_type"
    if kind == "formula" and "=" not in claim and "→" not in claim and "=>" not in claim:
        if not re.search(r"\d", claim):
            return "formula_no_eq"
    if kind == "feature" and not _ARCH.search(claim):
        return "feature_no_arch"
    skel = skeletonize(claim)
    for prev in existing_claims:
        if token_jaccard(skel, skeletonize(prev)) >= 0.88:
            return "near_dup"
    return None


def expand_one_l2(
    taxonomy: Taxonomy,
    llm: Any,
    domain: dict[str, Any],
    child: dict[str, Any],
    *,
    target: int,
    max_per_call: int,
    system: str,
) -> int:
    added_total = 0
    while len(child.get("leaves") or []) < target:
        existing = list(child.get("leaves") or [])
        need = min(max_per_call, target - len(existing))
        user = (
            "Expand ONLY this L2. Mix kinds. Prefix every l3_id with the l2_id.\n"
            + __import__("json").dumps(
                {
                    "domain_id": domain["id"],
                    "domain_name": domain.get("name"),
                    "l2_id": child["id"],
                    "l2_name": child.get("name"),
                    "existing_leaf_ids": [x.get("id") for x in existing],
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
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("expand %s.%s failed: %s", domain["id"], child["id"], exc)
            break
        existing_ids = {leaf.l3_id for leaf in taxonomy.leaves}
        claims = [str(x.get("claim") or "") for x in existing]
        accepted: list[dict[str, Any]] = []
        for raw in obj.get("new_leaves") or []:
            raw = dict(raw)
            raw.setdefault("domain_id", domain["id"])
            raw.setdefault("l2_id", child["id"])
            raw.setdefault("l2_name", child.get("name"))
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
        if not accepted:
            break
        n = merge_leaves(taxonomy, accepted, allow_new_domains=False)
        added_total += n
        if n == 0:
            break
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
