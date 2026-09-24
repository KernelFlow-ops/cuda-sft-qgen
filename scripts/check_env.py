"""Pre-flight check of the real-LLM environment (fix/05_ENV_AND_BUDGET.md §1–§2). Run from the repo root.

python fix/assets/scripts/check_env.py [--ping]
  --ping  additionally performs ONE tiny real chat call and ONE embedding call.
Exit 0 = usable; 1 = blocking problem. Warnings (e.g. REVIEW_MODEL equals NVIDIA_MODEL) do not fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--ping", action="store_true"); a = ap.parse_args()
    root = Path.cwd()
    if not (root / ".env").exists():
        print(json.dumps({"ok": False, "problems": [f"missing {root / '.env'} (copy .env.example and fill keys)"]}))
        return 1
    from dotenv import load_dotenv
    load_dotenv(root / ".env", override=False)
    keys = [os.getenv(k, "").strip() for k in ("NVIDIA_API_KEY", "NVIDIA_API_KEY_2", "NVIDIA_API_KEY_3")]
    keys += [k.strip() for k in os.getenv("NVIDIA_API_KEYS", "").replace(";", ",").split(",")]
    problems, warnings = [], []
    if not any(keys):
        problems.append("no NVIDIA_API_KEY")
    if os.getenv("FAKE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}:
        problems.append("FAKE_LLM must be 0")
    if os.getenv("EMBEDDING_PROVIDER", "nvidia").strip().lower() == "fake":
        problems.append("EMBEDDING_PROVIDER must not be fake")
    from cuda_sft_qgen.config import load_settings
    s = load_settings()
    if s.review_model == s.nvidia_model:
        warnings.append("REVIEW_MODEL equals NVIDIA_MODEL: chair is not model-independent")
    if int(os.getenv("MAX_TOKENS", "0") or 0) and int(os.getenv("MAX_TOKENS")) < 8192:
        warnings.append("MAX_TOKENS < 8192: structured writer output may be truncated")
    report = {"ok": not problems, "problems": problems, "warnings": warnings,
              "writer_model": s.nvidia_model, "review_model": s.review_model, "keys": sum(bool(k) for k in keys),
              "embedding": [s.embedding_provider, s.embedding_model]}
    if a.ping and not problems:
        import tempfile
        from cuda_sft_qgen.runtime import build_runtime
        rt = build_runtime(load_settings(data_dir=Path(tempfile.mkdtemp())))
        try:
            obj = rt.llm.complete_json([{"role": "user", "content": 'Return JSON {"ok": true}.'}], retries=1,
                                       temperature=0.0, thinking_level="none", max_tokens=32)
            vec = rt.embedder.embed(["warp shuffle"])
            report["ping"] = {"chat_ok": obj.get("ok") is True, "embedding_dim": int(len(vec[0]))}
            if not report["ping"]["chat_ok"]:
                problems.append("chat ping did not return {\"ok\": true}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"ping failed: {type(exc).__name__}: {exc}")
        finally:
            rt.close()
        report["ok"] = not problems
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
