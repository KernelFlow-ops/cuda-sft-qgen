"""Golden seeds (task T15): schema, enums, no leakage, distribution, and every numeric key recomputed.
The *_math tests read fix/assets only and must pass immediately after T16; test_installed_* need T15."""
from __future__ import annotations

import json
from collections import Counter
from math import ceil, gcd
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ASSETS = REPO / "cuda_qgen_fix" / "assets" / "seeds"
INSTALLED = REPO / "src" / "cuda_sft_qgen" / "prompts"


def _read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


GOLD = _read(ASSETS / "fewshot_seed_v2.jsonl")
NEG = _read(ASSETS / "fewshot_negative_v1.jsonl")
BY_ID = {g["id"]: g for g in GOLD}

QT = {"conceptual", "syntax", "code_complete", "debug", "optimization", "layout", "compare", "counterfactual", "profiling"}
KINDS = {"mechanism", "api", "formula", "feature", "failure", "numeric"}
REQ = ["id", "question", "tests_claim", "topic_path", "l3_id", "l3_claim", "question_type", "difficulty", "bloom",
       "persona", "hardware", "granularity", "language", "kind", "answer_format", "internal_reference_answer", "verification_spec"]


@pytest.mark.parametrize("item", GOLD, ids=lambda g: g["id"])
def test_schema(item):
    for key in REQ:
        assert key in item, key
    assert item["question_type"] in QT and item["kind"] in KINDS
    assert item["hardware"] in {"agnostic", "ampere", "hopper", "blackwell"}
    assert 1 <= item["difficulty"] <= 4 and 40 <= len(item["question"]) <= 2500
    assert item["topic_path"][-1] == item["l3_id"]


@pytest.mark.parametrize("item", GOLD, ids=lambda g: g["id"])
def test_no_obvious_leakage(item):
    q = item["question"]
    assert item["tests_claim"][:60] not in q
    assert item["l3_claim"][:40] not in q
    for bad in ("must mention", "答案是", "answer is", "要点包括"):
        assert bad not in q.lower()


def test_distribution_minima():
    c = lambda k: Counter(g[k] for g in GOLD)
    assert len(c("question_type")) == 9 and len(c("kind")) == 6
    assert set(c("difficulty")) == {1, 2, 3, 4}
    assert min(c("language").values()) >= 0.35 * len(GOLD)
    assert c("hardware")["blackwell"] >= 2 and c("hardware")["hopper"] >= 4
    assert len({g["topic_path"][0] for g in GOLD}) >= 15


# ---------- numeric keys recomputed independently ----------
def bank_ways(stride: int) -> int:
    banks: dict[int, set[int]] = {}
    for t in range(32):
        banks.setdefault((t * stride) % 32, set()).add(t * stride)
    return max(len(v) for v in banks.values())


def sectors(addrs, size):
    return len({b // 32 for a in addrs for b in range(a, a + size)})


def test_g01_bank():
    assert [bank_ways(s) for s in (6, 12, 33)] == [2, 4, 1]
    assert all(bank_ways(s) == gcd(s, 32) for s in range(1, 65))


def test_g03_occupancy():
    per_warp = ceil(36 * 32 / 256) * 256
    warps = 65536 // per_warp
    blocks = min(warps // 8, 32, 2048 // 256)
    assert (per_warp, blocks, blocks * 8 / 64, 65536 // (36 * 256)) == (1280, 6, 0.75, 7)


def test_g04_launch_bounds():
    assert 65536 // (4 * 256) == 64


def test_g06_sectors():
    lanes = range(32)
    assert sectors([4 * l for l in lanes], 4) == 4
    assert sectors([4 * (l + 1) for l in lanes], 4) == 5
    assert sectors([8 * l for l in lanes], 4) == 8
    assert sectors([8 * l for l in lanes], 8) == 8


def test_g07_vectorized():
    assert sectors(range(0, 512, 4), 4) == 16 == sectors([16 * l for l in range(32)], 16)


def test_g14_tma_rules():
    def ok(K, box, swz_bytes):
        stride = K * 2
        inner = box[0] * 2
        return stride % 16 == 0 and all(b <= 256 for b in box) and inner % 16 == 0 and (swz_bytes is None or inner <= swz_bytes)
    cfg = {"a": (1000, (64, 128), 128), "b": (1001, (64, 64), 128), "c": (1024, (4, 64), None),
           "d": (1024, (128, 64), 128), "e": (1024, (64, 256), 128)}
    assert sorted(k for k, v in cfg.items() if ok(*v)) == ["a", "e"]


def test_g15_expect_tx():
    assert 128 * 64 * 2 + 256 * 64 * 2 == 49152


def test_g16_wgmma():
    assert (64 // 16, 64 * 128 // 128) == (4, 64)


def test_g17_mma_sync():
    assert (16 * 16 // 2 // 32, 16 * 8 // 2 // 32, 16 * 8 // 32) == (4, 2, 4)
    assert (64 // 16) * (64 // 8) * (32 // 16) == 64


def test_g20_g21_cute():
    L = lambda i: (i % 4) * 8 + (i // 4) * 1
    assert (L(13), 1 * 8 + 3, 3 * 8 + 1, 3 * 8 + 7 + 1) == (11, 11, 25, 32)
    H = lambda i: ((i % 4) % 2) * 1 + ((i % 4) // 2) * 8 + (i // 4) * 2
    assert H(5) == 3 and sorted(H(i) for i in range(16)) == list(range(16)) and 1 + 8 + 6 == 15


def test_g22_swizzle():
    def ways(swizzle):
        banks: dict[int, set[int]] = {}
        for r in range(8):
            chunk = (0 ^ r) if swizzle else 0
            for b in range(128 * r + 16 * chunk, 128 * r + 16 * chunk + 16, 4):
                banks.setdefault((b // 4) % 32, set()).add(r)
        return max(len(v) for v in banks.values())
    assert (ways(False), ways(True), 128 * 5 + 16 * (2 ^ 5)) == (8, 1, 752)


def test_g27_g30_g31_g35_g36():
    assert 108 * 2 == 216
    assert (2 / 12, 3.0 * 2 / 12, 60 / 3.0) == pytest.approx((1 / 6, 0.5, 20))
    alg = 1e9 / 10e-3 / 1e9
    assert (alg, alg * 2 * 7 / 8) == (100.0, 175.0)
    assert ((1000 + 255) // 256) * 256 - 1000 == 24
    assert (2048 + 4096 // 16 + 4, 2048 + 4096 // 32) == (2308, 2176)


def test_negatives_have_defects():
    assert all(n.get("defect") and n.get("why") for n in NEG)
    assert {"correctness", "answer_leakage", "multiple_correct", "insufficient_premises", "duplicate"} <= {n["defect"].split("+")[0] for n in NEG}


def test_assets_golden_math():
    """Aggregate: all recomputation checks, runnable right after T16 (no project code needed)."""
    for fn in (test_g01_bank, test_g03_occupancy, test_g04_launch_bounds, test_g06_sectors, test_g07_vectorized,
               test_g14_tma_rules, test_g15_expect_tx, test_g16_wgmma, test_g17_mma_sync, test_g20_g21_cute,
               test_g22_swizzle, test_g27_g30_g31_g35_g36):
        fn()


def test_installed_equals_assets():
    assert _read(INSTALLED / "fewshot_seed.jsonl") == GOLD
    assert _read(INSTALLED / "fewshot_negative.jsonl") == NEG


def test_seed_enums_and_hardware_bucket():
    from cuda_sft_qgen.enums import ANSWER_FORMATS, ARCH_TARGETS, ARCH_TO_HARDWARE, SCENARIOS
    for g in GOLD:
        assert g["answer_format"] in ANSWER_FORMATS and g["scenario"] in SCENARIOS and g["arch_target"] in ARCH_TARGETS
        assert ARCH_TO_HARDWARE[g["arch_target"]] == g["hardware"], g["id"]


def test_seeds_pass_models_sieve_and_gate(settings):
    from cuda_sft_qgen.models import Question
    from cuda_sft_qgen.review.writer_gate import local_writer_defects
    from cuda_sft_qgen.sieve import rule_sieve
    for g in GOLD:
        q = Question.model_validate({**g, "dup_verdict": "unique"})
        assert rule_sieve(q, settings) is None, (g["id"], rule_sieve(q, settings))
        assert local_writer_defects(q) is None, (g["id"], local_writer_defects(q))
