"""Mirror official NVIDIA docs into data/docs for the local BM25 index (fix/tasks/T34 step 5).

SKELETON placed by T16. Task T34 implements `mirror()`.
Usage: python scripts/mirror_docs.py --out data/docs --toolkit 12.x [--sources configs/knowledge/sources.yaml]
Writes <slug>.txt and <slug>.meta.json {url,title,toolkit_version,sha256}. Failures print `SKIP <url> <error>`;
without network the script exits 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def mirror(urls: list[str], out: Path, toolkit: str) -> int:
    """Return number of documents written."""
    raise NotImplementedError("implemented by task T34")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/docs"); ap.add_argument("--toolkit", default="12.x")
    ap.add_argument("--sources", default="configs/knowledge/sources.yaml")
    a = ap.parse_args()
    import yaml
    urls = [str(u) for u in (yaml.safe_load(Path(a.sources).read_text(encoding="utf-8")) or {}).get("urls") or []]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"written={mirror(urls, out, a.toolkit)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
