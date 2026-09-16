from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from cuda_sft_qgen.review.normalize import exact_hash, normalize_text

# Longest-first so "cudaMallocAsync" wins over "cudaMalloc".
_CUDA_APIS = tuple(
    sorted(
        {
            "cudaMallocAsync",
            "cudaMallocManaged",
            "cudaMallocHost",
            "cudaMalloc",
            "cudaFreeAsync",
            "cudaFree",
            "cudaMemcpyAsync",
            "cudaMemcpy",
            "cudaMemset",
            "cudaMemAdvise",
            "cudaMemPrefetchAsync",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaStreamWaitEvent",
            "cudaEventRecord",
            "cudaLaunchCooperativeKernel",
            "cudaLaunchKernel",
            "cudaDeviceEnablePeerAccess",
            "cudaGraphLaunch",
            "cudaGraphInstantiate",
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
            "__syncthreads",
            "__syncwarp",
            "__shfl_down_sync",
            "__shfl_xor_sync",
            "__shfl_sync",
            "__ballot_sync",
            "__activemask",
            "__launch_bounds__",
            "__restrict__",
            "__shared__",
            "__constant__",
            "__global__",
            "__device__",
            "__host__",
            "__ldg",
            "atomicAdd",
            "atomicCAS",
            "cp.async",
            "cp_async",
            "make_layout",
            "make_tensor",
            "make_shape",
            "make_stride",
            "composition",
            "complement",
            "coalesce",
            "filter",
            "dice",
            "tile_to_shape",
            "zipped_divide",
            "logical_divide",
            "logical_product",
            "blocked_product",
            "raked_product",
            "flat_product",
            "ranked_product",
            "partition_S",
            "partition_D",
            "local_partition",
            "tiled_partition",
            "coalesced_group",
            "partition_fragment",
            "local_tile",
            "TiledCopy",
            "TiledMMA",
            "ThrCopy",
            "ThrMMA",
            "Copy_Atom",
            "wmma::",
            "mma.sync",
            "wgmma.mma_async",
            "cute::Tensor",
            "cute::Layout",
            "Swizzle",
            "cutlass::gemm",
            "cutlass::conv",
            "cutlass::layout::RowMajor",
            "cutlass::layout::ColumnMajor",
            "cutlass::arch::Sm80",
            "cutlass::arch::Sm90",
            "cublasLtMatmul",
            "ncclAllReduce",
            "cub::DeviceReduce",
        },
        key=len,
        reverse=True,
    )
)

_ARCH_PATTERNS = [
    r"sm_?\d+",
    r"\bh100\b",
    r"\bh200\b",
    r"\ba100\b",
    r"\ba10\b",
    r"\bb100\b",
    r"\bb200\b",
    r"\bhopper\b",
    r"\bampere\b",
    r"\bblackwell\b",
    r"\bturing\b",
    r"\bvolta\b",
    r"\bada\b",
    r"compute_?\d+",
]

_SPEECH_ACTS = [
    (re.compile(r"\bwhat if\b|如果.{0,8}改|假如|倘若|counterfactual"), "what-if"),
    (re.compile(r"bug|错误|未定义|死锁|race|为什么崩溃|哪一行"), "debug"),
    (re.compile(r"补全|complete|fill in| impl "), "complete"),
    (re.compile(r"对比|versus|vs\.|区别|更适合"), "compare"),
    (re.compile(r"layout|stride|shape|swizzle|composition"), "layout-eq"),
    (re.compile(r"为什么|为何|\bwhy\b"), "why"),
    (re.compile(r"如何|怎样|\bhow\b"), "how"),
    (re.compile(r"nsight|ncu|nsys|counter|stall"), "profile"),
]


_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_NUMBER = re.compile(
    r"\b0x[0-9a-fA-F]+\b|\b\d+\.\d+\b|\b\d+\b"
)
_TOKEN = re.compile(r"\{[A-Z]+\}|[a-zA-Z_]+|[\u4e00-\u9fff]+")


def _replace_apis(text: str) -> str:
    for api in _CUDA_APIS:
        text = re.sub(re.escape(api), "{API}", text, flags=re.IGNORECASE)
    return text


def _replace_arch(text: str) -> str:
    for pat in _ARCH_PATTERNS:
        text = re.sub(pat, "{ARCH}", text, flags=re.IGNORECASE)
    return text


def speech_act(text: str) -> str:
    low = text.lower()
    for cre, label in _SPEECH_ACTS:
        if cre.search(low):
            return label
    if "?" in text or "？" in text:
        return "ask"
    return "other"


def _length_bucket(length: int) -> str:
    """Bucket length to reduce false matches between very different sized questions."""
    if length < 100:
        return "S"
    elif length < 250:
        return "M"
    elif length < 500:
        return "L"
    else:
        return "XL"


def skeletonize(text: str) -> str:
    text = normalize_text(text)
    text = _CODE_FENCE.sub(" {CODE} ", text)
    text = _replace_apis(text)
    text = _replace_arch(text)
    text = _NUMBER.sub("{N}", text)
    # Remaining identifiers (after API replacement) become {ID}, keep slots.
    def _id_sub(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.upper() in {"API", "ARCH", "CODE", "ID", "N"}:
            return token
        if len(token) <= 2:
            return token
        return "{ID}"

    # Do not eat existing {API}/{ARCH}/{N}/{CODE}/{ID} slots.
    text = re.sub(r"(?<!\{)\b[A-Za-z_][A-Za-z0-9_]{2,}\b(?!\})", _id_sub, text)
    text = re.sub(r"(?:\{ID\}\s*){2,}", "{ID} ", text)
    text = re.sub(r"\s+", " ", text).strip()
    act = speech_act(text)
    length_bucket = _length_bucket(len(text))
    return f"[{act}][{length_bucket}] {text}"


def skeleton_tokens(skeleton: str) -> set[str]:
    return {t for t in _TOKEN.findall(skeleton) if t}


def token_jaccard(a: str, b: str) -> float:
    sa, sb = skeleton_tokens(a), skeleton_tokens(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def simhash64(text: str) -> int:
    tokens = skeleton_tokens(text) or {text}
    acc = [0] * 64
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        for i in range(64):
            acc[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(64):
        if acc[i] >= 0:
            out |= 1 << i
    return out


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Fingerprint:
    exact: str
    skeleton: str
    simhash: int

    def hex_simhash(self) -> str:
        return f"{self.simhash:016x}"


def fingerprint(question: str) -> Fingerprint:
    skel = skeletonize(question)
    return Fingerprint(
        exact=exact_hash(question),
        skeleton=skel,
        simhash=simhash64(skel),
    )
