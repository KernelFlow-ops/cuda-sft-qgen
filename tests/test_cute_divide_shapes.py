"""Guard the hierarchy returned by CuTe zipped_divide and tiled_divide."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _leaves(path: Path) -> dict[str, dict]:
    taxonomy = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        leaf["id"]: leaf
        for child in taxonomy.get("children", [])
        for leaf in child.get("leaves", [])
    }


def _cutlass_include() -> Path | None:
    candidates: list[Path] = []
    for name in ("CUTLASS_PATH", "CUTLASS_ROOT", "CUTLASS_DIR"):
        value = os.getenv(name)
        if value:
            root = Path(value).expanduser()
            candidates.extend([root, root / "include"] if root.name == "include"
                              else [root / "include", root])
    candidates.extend([
        ROOT / "third_party" / "cutlass" / "include",
        ROOT / "vendor" / "cutlass" / "include",
        Path("/usr/local/cutlass/include"),
        Path("/opt/cutlass/include"),
    ])
    candidates.extend(Path("/usr/local").glob("cutlass*/include"))
    for candidate in candidates:
        if (candidate / "cute" / "layout.hpp").is_file():
            return candidate
    return None


def test_divide_taxonomy_claims_preserve_tiler_hierarchy():
    t20 = _leaves(ROOT / "configs/taxonomy/domains/T20_cute_algebra.yaml")
    t07 = _leaves(ROOT / "configs/taxonomy/domains/T07_cutlass_cute.yaml")
    zipped = t20["product_divide_zipped"]["claim"]
    tiled = t20["product_divide_tiled_api"]["claim"]

    assert "((_3,_2),(_4,_9))" in zipped
    assert "((_3,_2),_4,_9)" in zipped
    assert "保留 tiler 的嵌套 tile 模式" in tiled
    assert "只把 rest 各模提升为同级成员" in t20["zipped_layouts_vs_tiled"]["claim"]
    assert "只把 rest 各模提升为同级成员" in t07["cute_hier_zipped"]["claim"]
    mismatch = t20["zipped_layouts_mismatch"]["claim"]
    assert "get<1>(t)" in mismatch and "第一个 rest 模" in mismatch
    assert "get<1>(z)" in mismatch and "整个 rest 组" in mismatch
    assert "get<0,1>" not in mismatch
    relevant = [zipped, tiled, t20["zipped_layouts_vs_tiled"]["claim"],
                mismatch, t07["cute_hier_zipped"]["claim"]]
    assert all("tiler 拍扁" not in claim and "tiler 侧拍扁" not in claim for claim in relevant)


def test_cutlass_divide_shape_types_when_nvcc_is_available(tmp_path: Path):
    nvcc = shutil.which("nvcc")
    include = _cutlass_include()
    if not nvcc or include is None:
        pytest.skip("nvcc and CUTLASS cute/layout.hpp are required for the compile-time check")

    source = tmp_path / "cute_divide_shapes.cu"
    source.write_text(
        """#include <cute/layout.hpp>
#include <type_traits>

int main() {
  using namespace cute;
  constexpr auto layout = make_layout(make_shape(_12{}, Int<18>{}));
  constexpr auto tiler = make_shape(_3{}, _2{});
  constexpr auto zipped = zipped_divide(layout, tiler);
  constexpr auto tiled = tiled_divide(layout, tiler);
  constexpr auto expected_zipped = make_shape(
      make_shape(_3{}, _2{}), make_shape(_4{}, Int<9>{}));
  constexpr auto expected_tiled = make_shape(
      make_shape(_3{}, _2{}), _4{}, Int<9>{});
  static_assert(std::is_same_v<std::remove_cv_t<decltype(shape(zipped))>,
                               std::remove_cv_t<decltype(expected_zipped)>>);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(shape(tiled))>,
                               std::remove_cv_t<decltype(expected_tiled)>>);
}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [nvcc, "-std=c++17", f"-I{include}", "-c", str(source), "-o", str(tmp_path / "shape.o")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
