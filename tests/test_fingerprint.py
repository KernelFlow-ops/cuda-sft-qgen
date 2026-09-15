from __future__ import annotations

from cuda_sft_qgen.review.fingerprint import (
    fingerprint,
    hamming64,
    skeletonize,
    token_jaccard,
)
from cuda_sft_qgen.review.normalize import exact_hash


def test_exact_hash_ignores_whitespace_and_case():
    a = "What is  __syncthreads ? "
    b = "what is __syncthreads?"
    assert exact_hash(a) == exact_hash(b)


def test_template_same_after_arch_number_api_swap():
    q1 = "在 H100 上调用 cudaMalloc 分配 16 个 float，会不会破坏 coalescing？"
    q2 = "在 B200 上调用 cudaMalloc 分配 32 个 half，会不会破坏 coalescing？"
    s1, s2 = skeletonize(q1), skeletonize(q2)
    assert "{ARCH}" in s1 and "{ARCH}" in s2
    assert "{N}" in s1 and "{N}" in s2
    assert "{API}" in s1 and "{API}" in s2
    jac = token_jaccard(s1, s2)
    ham = hamming64(fingerprint(q1).simhash, fingerprint(q2).simhash)
    assert jac >= 0.88 or ham <= 3


def test_slots_not_eaten_by_identifier_pass():
    text = skeletonize("Use cudaMalloc then cudaFree on sm_90 with 128 bytes.")
    assert "{API}" in text
    assert "{ARCH}" in text or "sm" not in text.lower()
    assert "{N}" in text


def test_different_claims_are_not_the_same_template():
    q1 = "为什么 __syncthreads 不能写在只被一半线程走到的分支里？"
    q2 = "CuTe composition(A,B) 之后 stride 如何合成？请给出 Shape 与 Stride。"
    jac = token_jaccard(skeletonize(q1), skeletonize(q2))
    assert jac < 0.75
