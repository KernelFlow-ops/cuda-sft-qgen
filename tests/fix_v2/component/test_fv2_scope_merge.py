"""scripts/merge_scope.py on a COPY of the real taxonomy (task T32; scope v3 semantics)."""
import json
import shutil
import subprocess
import sys

import yaml

NEW_L2 = {"T06.block_scaled_fp4", "T09.forward_compat", "T13.paged_kv_cache", "T13.moe_grouped_gemm", "T31.topk_sampling"}


def _run(repo_root, index, apply=True):
    cmd = [sys.executable, str(repo_root / "scripts" / "merge_scope.py"), "--scope", str(repo_root / "configs" / "scope.yaml"),
           "--taxonomy", str(index)] + (["--apply"] if apply else [])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, check=True)
    return json.loads(out.stdout.strip())


def _leaf_ids(index):
    from cuda_sft_qgen.taxonomy import load_taxonomy
    return sorted(lf.l3_id for lf in load_taxonomy(index).leaves)


def test_merge_on_real_taxonomy_copy(tmp_path, repo_root):
    src = repo_root / "configs" / "taxonomy"
    dst = tmp_path / "taxonomy"
    shutil.copytree(src, dst)
    index = dst / "cuda.yaml"
    for path in (dst / "domains").glob("*.yaml"):
        domain = yaml.safe_load(path.read_text(encoding="utf-8"))
        domain.pop("weight", None)
        domain["children"] = [child for child in domain["children"]
                              if f"{domain['id']}.{child['id']}" not in NEW_L2]
        for child in domain["children"]:
            child.pop("arch", None)
            child.pop("target_leaves", None)
        path.write_text(yaml.safe_dump(domain, allow_unicode=True, sort_keys=False), encoding="utf-8")
    before = _leaf_ids(index)

    dry = _run(repo_root, index, apply=False)
    assert set(dry) == {"added_l1", "added_l2", "patched_l2", "weighted_l1"}
    assert _leaf_ids(index) == before and not yaml.safe_load((dst / "domains" / "T08_hopper_blackwell.yaml").read_text(encoding="utf-8")).get("weight")

    first = _run(repo_root, index)
    assert first["added_l1"] == [] and NEW_L2 <= set(first["added_l2"])
    assert "T08.tma" in first["patched_l2"] and len(first["weighted_l1"]) == 31
    assert _leaf_ids(index) == before                       # never touches leaves

    t08 = yaml.safe_load((dst / "domains" / "T08_hopper_blackwell.yaml").read_text(encoding="utf-8"))
    tma = next(c for c in t08["children"] if c["id"] == "tma")
    assert t08["weight"] == 1.3 and t08["name"] == "Hopper 与 Blackwell 新硬件"
    assert tma["arch"] == ["sm_90", "sm_90a", "sm_100a"] and tma["target_leaves"] == 14 and tma["leaves"]
    t06 = yaml.safe_load((dst / "domains" / "T06_tensor_core.yaml").read_text(encoding="utf-8"))
    new = next(c for c in t06["children"] if c["id"] == "block_scaled_fp4")
    assert new["leaves"] == [] and new["arch"] == ["sm_100a", "sm_120"]

    second = _run(repo_root, index)
    assert second == {"added_l1": [], "added_l2": [], "patched_l2": [], "weighted_l1": []}
