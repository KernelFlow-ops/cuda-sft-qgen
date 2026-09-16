from __future__ import annotations

import json
import logging

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.taxonomy import dump_taxonomy, merge_leaves

logger = logging.getLogger(__name__)


def _sparsest_l2(taxonomy) -> tuple[dict, dict] | None:
    best: tuple[dict, dict] | None = None
    best_n = 10**9
    for domain in taxonomy.domains:
        for child in domain.get("children") or []:
            n = len(child.get("leaves") or [])
            if n < best_n:
                best = (domain, child)
                best_n = n
    return best


def architect_node(state: PipelineState) -> dict:
    rt = get_runtime()
    system = load_prompt("architect")
    pair = _sparsest_l2(rt.taxonomy)
    if pair is None:
        return {"status": "taxonomy_ready", "log": ["architect skipped: empty taxonomy"]}
    domain, child = pair
    cap = min(20, int(rt.settings.generation.get("expand_max_per_call", 20) or 20))
    user = (
        "Expand ONLY this L2. Mix kinds (mechanism/api/formula/feature/failure/numeric).\n"
        + json.dumps(
            {
                "domain_id": domain["id"],
                "domain_name": domain.get("name"),
                "l2_id": child["id"],
                "l2_name": child.get("name"),
                "existing_leaves": [
                    {"id": x.get("id"), "claim": x.get("claim"), "kind": x.get("kind")}
                    for x in (child.get("leaves") or [])
                ],
                "max_new_leaves": cap,
            },
            ensure_ascii=False,
        )
    )
    try:
        obj = rt.llm.complete_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            retries=rt.settings.max_parse_retries,
        )
        added = merge_leaves(rt.taxonomy, obj.get("new_leaves") or [])
        if added:
            dump_taxonomy(rt.taxonomy, rt.settings.taxonomy_path)
        msg = f"architect added {added} leaves (total={len(rt.taxonomy.leaves)})"
        logger.info(msg)
        return {"status": "taxonomy_ready", "log": [msg]}
    except Exception as exc:  # noqa: BLE001
        logger.warning("architect failed, keeping seed taxonomy: %s", exc)
        return {
            "status": "taxonomy_ready",
            "log": [f"architect skipped due to error: {exc}"],
        }
