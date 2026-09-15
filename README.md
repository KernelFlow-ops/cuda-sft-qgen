# cuda-sft-qgen

用 LangGraph 多智能体流水线，为 NVIDIA CUDA 知识域生成 **SFT 问题集**（本阶段只出题、不出答案）。

覆盖 CUDA 语法、优化、CuTe/CUTLASS 布局、Hopper/Blackwell、工具链与库。出题先锁知识树再抽样；审题默认放行，只拦重复，分布问题通过 steering 回传给出题组，而不是删题。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # 填入最多 3 个 NVIDIA_API_KEY
```

`.env` 支持 `NVIDIA_API_KEY` / `NVIDIA_API_KEY_2` / `NVIDIA_API_KEY_3`。每个 key 的 LLM 并发默认 2、embedding 并发默认 2（3 个 key 合计 LLM 6 / embed 6）；某个 key 被 429 时冷却并换其他 key。`NVIDIA_BASE_URL`、`NVIDIA_MODEL`、`THINKING_LEVEL`（`none|medium|high`）、`REVIEW_BATCH_SIZE` 同样可配。

```bash
# 真实 API
cuda-sft-qgen generate --target 1000

# 不打 API 的结构自检
cuda-sft-qgen generate --fake --target 8 --batch 4

cuda-sft-qgen resume
cuda-sft-qgen stats
cuda-sft-qgen rebuild-taxonomy
```

产物在 `data/accepted.jsonl`、`data/rejected.jsonl`、`data/stats.json`。

## 流水线

```
Taxonomy Architect（可选，一次）
    → Coverage Planner（配额抽样任务单）
    → Writer / Evolver（并行出题）
    → Meaningfulness Sieve（只拦垃圾）
    → pending 凑满 REVIEW_BATCH_SIZE（默认 20）才进审题
    → Embedding + 三层查重 + 多样性统计
    → Review Chair（宽松；只写 steering，不因失衡删题）
    → Archivist → 下一轮 Planner 读 steering
```

重复判定：

1. L0 规范化后 SHA256
2. L1 骨架 SimHash / Jaccard（换数字、架构、API 仍算同一训练模板）
3. L2 embedding cosine（默认阈值 0.93 才拒；灰区默认过）

语义向量默认走同一 NVIDIA 中转 `nvidia/nemotron-3-embed-1b`。质量档可改：

```
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
```

## 思考等级

`THINKING_LEVEL=medium` 映射为

```python
extra_body={"chat_template_kwargs": {"enable_thinking": True, "medium_effort": True}}
```

流式读取 `delta.content`；`reasoning_content` 默认不落盘（`LOG_REASONING=1` 才打日志）。
