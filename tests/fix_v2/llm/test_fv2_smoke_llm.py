"""Real API smoke (budget ≤ 5 requests): chat JSON, review model, embeddings. Gate for every later llm test."""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.llm


def test_chat_json_roundtrip(real_runtime, eval_out):
    obj = real_runtime.llm.complete_json(
        [{"role": "system", "content": "Return JSON only."},
         {"role": "user", "content": 'Return {"warp_size": <the CUDA warp size as an integer>}.'}],
        retries=1, temperature=0.0, thinking_level="none", max_tokens=64)
    assert int(obj["warp_size"]) == 32
    (eval_out / "smoke_chat.json").write_text(json.dumps(obj), encoding="utf-8")


def test_review_model_reachable(real_runtime, eval_out):
    reviewer = real_runtime.review_llm or real_runtime.llm
    obj = reviewer.complete_json([{"role": "system", "content": "Return JSON only."},
                                  {"role": "user", "content": 'Return {"ok": true}.'}],
                                 retries=1, temperature=0.0, thinking_level="none", max_tokens=32)
    assert obj.get("ok") is True
    s = real_runtime.settings
    (eval_out / "smoke_review_model.json").write_text(json.dumps({
        "writer_model": s.nvidia_model, "review_model": s.review_model,
        "independent_review": s.review_model != s.nvidia_model}), encoding="utf-8")


def test_embeddings_consistent(real_runtime):
    v = np.asarray(real_runtime.embedder.embed(["shared memory bank conflict", "shared memory bank conflict",
                                                "NCCL all-reduce bus bandwidth"]), dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert v.shape[0] == 3 and v.shape[1] >= 64
    assert float(v[0] @ v[1]) > 0.99 and float(v[0] @ v[2]) < float(v[0] @ v[1])
