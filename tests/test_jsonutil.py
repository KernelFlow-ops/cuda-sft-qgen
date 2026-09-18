"""Tests for LLM JSON extraction, think-trace stripping, and truncated payloads."""

from __future__ import annotations

from cuda_sft_qgen.jsonutil import _scan_fields, normalize_language, parse_question_payload


SAMPLE_SYNCTHREADS = """{
  "question": "在下面的内核中，线程块大小为 256，且仅当 threadIdx.x < 64 时执行 __syncthreads()。请问该内核运行时会发生什么？

```cpp
__global__ void kernel() {
    if (threadIdx.x < 64) {
        // 一些计算
        __syncthreads();
    }
}
```",
  "tests_claim": "Verifies that __syncthreads must be reached by every thread in the block",
  "language": "zh"
}"""


SAMPLE_SMEM = """{
  "question": "下面的内核使用动态共享内存，并将其切分为两个数组：

```cpp
__global__ void kernel() {
    extern __shared__ char smem[];
    float*  A = reinterpret_cast<float*>(smem);
    int2*   B = reinterpret_cast<int2*>(smem + OFFSET);
}
```

`int2` 的对齐要求为 8 字节。请给出 OFFSET。",
  "tests_claim": "动态共享内存声明为 extern __shared__、多数组切分与对齐",
  "language": "zh"
}"""


SAMPLE_GRAPH = """{
  "question": "The program below uses CUDA graph capture. Identify the illegal line.

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
cudaMalloc(&p, 16);
cudaGraphEndCapture(stream, &graph);
```",
  "tests_claim": "stream capture 的禁止操作",
  "language": "en"
}"""


SAMPLE_INNER_QUOTES = """{
  "question": "Which call-object is legal? Consider a device lambda and the token "kernel" in comments.",
  "tests_claim": "device lambda capture rules",
  "language": "en"
}"""


def test_syncthreads_divergent_fence():
    obj = parse_question_payload(SAMPLE_SYNCTHREADS)
    assert obj is not None
    assert "__syncthreads" in obj["question"]
    assert "threadIdx.x < 64" in obj["question"]


def test_dynamic_smem_split():
    obj = parse_question_payload(SAMPLE_SMEM)
    assert obj is not None
    assert "extern __shared__" in obj["question"]
    assert "OFFSET" in obj["question"]


def test_graph_capture():
    obj = parse_question_payload(SAMPLE_GRAPH)
    assert obj is not None
    assert "cudaStreamBeginCapture" in obj["question"]


def test_inner_quotes_do_not_drop_question():
    obj = parse_question_payload(SAMPLE_INNER_QUOTES)
    assert obj is not None
    assert "kernel" in obj["question"]
    assert obj["tests_claim"]


def test_truncated_keeps_question():
    raw = '''{"question": "为什么 __syncthreads 不能写在只被一半线程走到的分支里？请说明死锁原因。"}'''
    obj = parse_question_payload(
        raw,
        fallback_claim="syncthreads deadlock",
        min_question_chars=24,
        keep_truncated=True,
    )
    assert obj is not None
    assert "json_truncated" in (obj.get("warnings") or [])
    assert obj["tests_claim"]
    assert obj["tests_claim"] != "syncthreads deadlock"
    assert "__syncthreads" in obj["tests_claim"]


def test_think_tag_stripped():
    raw = 'thinking...</think>{"question": "CUDA coalescing 对齐要求是什么？请给出事务大小。", "tests_claim": "c", "language": "zh"}'
    obj = parse_question_payload(raw)
    assert obj is not None
    assert "coalescing" in obj["question"]


def test_inner_cuda_fence_does_not_replace_json_payload():
    raw = (
        '```json\n'
        '{"question": "see\\n\\n```cuda\\n__global__ void k() {}\\n```\\n'
        'What happens to occupancy?", '
        '"tests_claim": "occupancy", "language": "zh"}\n'
        '```'
    )
    obj = parse_question_payload(raw)
    assert obj is not None
    assert "occupancy" in obj["question"]


def test_language_extra_keys_and_leak():
    raw = '''{
  "question": "CUDA coalescing 对齐要求是什么？请给出事务大小。",
  "tests_claim": "coalescing",
  "language": "en",
  "operator": "deepen"
}'''
    obj = parse_question_payload(raw)
    assert obj is not None
    assert obj["language"] == "en"
    scanned = _scan_fields(raw)
    assert scanned is not None
    assert scanned["language"] == "en"
    assert normalize_language('en",\n  "operator": "deepen') == "en"
    assert normalize_language("zh") == "zh"
