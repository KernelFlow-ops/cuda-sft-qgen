from __future__ import annotations

from cuda_sft_qgen.keys import parse_api_keys


def test_parse_three_numbered_keys():
    env = {
        "NVIDIA_API_KEY": "nvapi-aaa",
        "NVIDIA_API_KEY_2": "nvapi-bbb",
        "NVIDIA_API_KEY_3": "nvapi-ccc",
    }
    assert parse_api_keys(env) == ["nvapi-aaa", "nvapi-bbb", "nvapi-ccc"]


def test_parse_skips_empty_and_duplicates():
    env = {
        "NVIDIA_API_KEY": "nvapi-aaa",
        "NVIDIA_API_KEY_2": "",
        "NVIDIA_API_KEY_3": "nvapi-aaa",
        "NVIDIA_API_KEYS": "nvapi-aaa, nvapi-ddd",
    }
    assert parse_api_keys(env) == ["nvapi-aaa", "nvapi-ddd"]


def test_parse_comma_list_only():
    env = {"NVIDIA_API_KEYS": "k1;k2,k3"}
    assert parse_api_keys(env) == ["k1", "k2", "k3"]


def test_split_even_saturates_capacity():
    from cuda_sft_qgen.embeddings import _split_even

    batches = _split_even([str(i) for i in range(20)], 6)
    assert len(batches) == 6
    assert sum(len(b) for b in batches) == 20
    assert max(len(b) for b in batches) - min(len(b) for b in batches) <= 1


def test_pool_capacity_three_keys():
    from cuda_sft_qgen.keys import build_key_pool

    pool = build_key_pool(
        ["k1", "k2", "k3"],
        base_url="https://example.invalid/v1",
        llm_per_key=2,
        embed_per_key=2,
        embed_oversub=1,
    )
    assert pool.n_keys == 3
    assert pool.llm_capacity == 6
    assert pool.embed_capacity == 6
    held = [pool.acquire("llm") for _ in range(6)]
    counts = {}
    for slot in held:
        counts[slot.index] = counts.get(slot.index, 0) + 1
    assert counts == {0: 2, 1: 2, 2: 2}
    for slot in held:
        pool.release("llm", slot)


def test_embed_oversub_doubles_capacity():
    from cuda_sft_qgen.keys import build_key_pool

    pool = build_key_pool(
        ["k1", "k2", "k3"],
        base_url="https://example.invalid/v1",
        llm_per_key=2,
        embed_per_key=2,
        embed_oversub=2,
    )
    assert pool.embed_capacity == 12
    held = [pool.acquire("embed") for _ in range(12)]
    assert pool.inflight["embed"] == 12
    for slot in held:
        pool.release("embed", slot)


def test_llm_cooldown_does_not_block_embed():
    from cuda_sft_qgen.keys import build_key_pool

    pool = build_key_pool(
        ["k1"],
        base_url="https://example.invalid/v1",
        llm_per_key=1,
        embed_per_key=1,
        embed_oversub=1,
    )
    slot = pool.acquire("llm")
    pool.mark_cooldown(slot, 30, kind="llm")
    pool.release("llm", slot)
    emb = pool.acquire("embed")
    assert emb.index == 0
    pool.release("embed", emb)
