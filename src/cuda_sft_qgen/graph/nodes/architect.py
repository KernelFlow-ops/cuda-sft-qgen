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


def _make_taxonomy_manager(rt):
    from cuda_sft_qgen.taxonomy_manager import manager_from_runtime

    return manager_from_runtime(rt)


def _diversity_from_state(state: PipelineState):
    payload = state.get("last_diversity") or {}
    if not isinstance(payload, dict):
        return None
    report = payload.get("report")
    if not report:
        return None
    try:
        from cuda_sft_qgen.models import DiversityReport

        return DiversityReport.model_validate(report)
    except Exception:  # noqa: BLE001
        return report


def taxonomy_health_node(state: PipelineState) -> dict:
    """Sidecar taxonomy health tick. Fail-open; never touches pending or steering."""
    try:
        rt = get_runtime()
        manager = _make_taxonomy_manager(rt)
        result = manager.run_tick(
            accepted=rt.store.load_accepted(),
            rejected=rt.store.load_rejected(),
            diversity_report=_diversity_from_state(state),
            apply=not rt.settings.fake_llm,
            last_facet=state.get("last_taxonomy_facet"),
            data_dir=rt.settings.data_dir,
            fake_llm=rt.settings.fake_llm,
        )
        if not isinstance(result, dict):
            result = {}
        score = result.get("health_score")
        added = result.get("total_added")
        mission = result.get("mission")
        facet = result.get("facet")
        status = result.get("status") or "taxonomy_health_checked"
        ticks = int(state.get("taxonomy_ticks") or 0) + 1
        msg = (
            f"taxonomy_manager status={status} health_score={score} "
            f"added={added} facet={facet}"
        )
        logger.info(msg)
        return {
            "status": status,
            "health_score": score,
            "taxonomy_ticks": ticks,
            "last_taxonomy_health": {
                "score": score,
                "added": added,
                "mission": mission,
            },
            "last_taxonomy_facet": facet,
            "log": [msg],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("taxonomy manager failed: %s", exc)
        return {
            "status": "taxonomy_health_checked",
            "log": [f"taxonomy manager error: {exc}"],
        }


taxonomy_manager_node = taxonomy_health_node

