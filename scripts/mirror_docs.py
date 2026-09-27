"""Mirror official NVIDIA docs into data/docs for the local BM25 index (fix/tasks/T34 step 5).

SKELETON placed by T16. Task T34 implements `mirror()`.
Usage: python scripts/mirror_docs.py --out data/docs --toolkit 12.x [--sources configs/knowledge/sources.yaml]
Writes <slug>.txt and <slug>.meta.json {url,title,toolkit_version,sha256}. Failures print `SKIP <url> <error>`;
without network the script exits 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def mirror(urls: list[str], out: Path, toolkit: str) -> int:
    """Return number of documents written."""
    from cuda_sft_qgen.knowledge_worker import fetch_official_document

    written = 0
    for url in urls:
        try:
            text = fetch_official_document(url)
            if not text.strip():
                raise ValueError("empty document")
            parsed = urlparse(url)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", parsed.netloc + parsed.path).strip("_")
            (out / f"{slug}.txt").write_text(text, encoding="utf-8")
            meta = {"url": url, "title": slug, "toolkit_version": toolkit,
                    "sha256": hashlib.sha256(text.encode()).hexdigest()}
            (out / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {url} {type(exc).__name__}: {exc}")
    return written


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
