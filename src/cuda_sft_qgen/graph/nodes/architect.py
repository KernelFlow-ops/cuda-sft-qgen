from __future__ import annotations

import json
import logging

from cuda_sft_qgen.graph.state import PipelineState
from cuda_sft_qgen.prompts import load_prompt
from cuda_sft_qgen.runtime import get_runtime
from cuda_sft_qgen.taxonomy import dump_taxonomy, merge_leaves

logger = logging.getLogger(__name__)


def architect_node(state: PipelineState) -> dict:
    rt = get_runtime()
    system = load_prompt("architect")
    user = (
        "Current taxonomy JSON (truncated domains ok to read fully):\n"
        + json.dumps(
            {"version": rt.taxonomy.version, "domains": rt.taxonomy.domains},
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
