// Hand-placed: callers (top) → role resolution → shared pool → keys → provider → 429 handling.
const C = [95, 255, 415, 575, 735, 895];
module.exports = {
  id: '12-api-key-pool',
  manual: true,
  title: 'ApiKeyPool · 多 Key 优先级调度（跨 Agent 共享）',
  subtitle: 'keys.py · llm.py · runtime.build_runtime · config.AgentAPI',
  nodes: [
    { id: 'w', type: 'agent', label: 'Writer\n/ Evolver', badge: '优先级 0', at: [C[0], 70], w: 140, h: 70 },
    { id: 'gate', type: 'rule', label: 'Writer 自洽门\n（LLM 版）', badge: '0', at: [C[1], 70], w: 140, h: 70 },
    { id: 's', type: 'agent', label: 'Salvage', badge: '1', at: [C[2], 70], w: 140, h: 70 },
    { id: 'c', type: 'agent', label: 'Chair\n/ Verifier', badge: '2', at: [C[3], 70], w: 140, h: 70 },
    { id: 'sv', type: 'rule', label: 'LLM Sieve\n（默认关）', badge: '3', at: [C[4], 70], w: 140, h: 70 },
    { id: 't', type: 'agent', label: 'Knowledge\n/ Taxonomy', badge: '4', at: [C[5], 70], w: 140, h: 70 },

    { id: 'role', type: 'rule', label: '按角色解析 AgentAPI', sub: 'WRITER_* 为基准；EVOLVER_ / REVIEW_ / SALVAGE_ / KNOWLEDGE_*\n可覆盖供应商、地址、模型、key · 地址+key 相同的角色共用一个池', at: [495, 215], w: 700, maxSubW: 660 },

    { id: 'heap', type: 'rule', label: '优先级等待堆', sub: '数字小者先得 · 同级先来先得\nWriter 与 Review 共池时\nChair 预留 1 个槽位', at: [300, 385], w: 280 },
    { id: 'cap', type: 'rule', label: 'effective_cap', sub: '初始 = key 数 × 每 key 并发\n连续 3 次 429 → 降到 key 数\n每次成功 +1 回升', at: [690, 385], w: 280 },
    { id: 'k1', type: 'tool', label: 'key #1', sub: 'LLM 信号量 + Embed 信号量', at: [230, 512], w: 230 },
    { id: 'k2', type: 'tool', label: 'key #2', sub: '…', at: [495, 512], w: 200 },
    { id: 'kn', type: 'tool', label: 'key #n', sub: 'WRITER_API_KEY_1..7', at: [760, 512], w: 230 },

    { id: 'api', type: 'tool', label: '供应商 API（流式）', sub: 'OpenAI 兼容 / NVIDIA Integrate · 记录 usage', at: [495, 645], w: 440, maxSubW: 410 },
    { id: 'r429', type: 'decision', label: '429 /\n过载?', at: [495, 765] },
    { id: 'ok', type: 'accept', label: '返回文本', sub: '只推理不作答 → ReasoningBudgetExceeded', at: [200, 765], w: 330 },
    { id: 'cool', type: 'rule', label: '该 key 冷却后换 key', sub: '15s·2ⁿ（≤120s）\n每 key 每次调用最多 2 次', at: [790, 765], w: 250 },
    { id: 'rl', type: 'reject', label: 'RateLimitError', sub: '所有 key 用完配额 → brief 进 retry_tasks', at: [760, 885], w: 370, maxSubW: 340 },
    { id: 'emb', type: 'tool', label: 'Embedding 池（独立）', sub: 'EMBEDDING_* 可单独配置 · FIFO · 超配 ×2', at: [230, 885], w: 390, maxSubW: 360 },
    { id: 'n1', type: 'note', label: 'Taxonomy / 审计任务发现 Writer 或 Chair 在排队，就主动让出本轮：后台不抢前台槽位', at: [495, 1000], w: 420, maxW: 392 },
  ],
  groups: [
    { id: 'g_callers', label: '调用方（徽章 = 优先级，越小越先拿到槽位）', color: 'blue', box: [15, -4, 960, 120] },
    { id: 'g_pool', label: 'ApiKeyPool', color: 'violet', solid: true, box: [95, 290, 800, 272] },
  ],
  edges: [
    { from: 'g_callers', to: 'role', fromSide: 'bottom', toSide: 'top' },
    { from: 'role', to: 'g_pool', fromSide: 'bottom', toSide: 'top' },
    { from: 'heap', to: 'cap', label: '有空位' },
    { from: 'cap', to: 'k1', fromSide: 'bottom', toSide: 'top', fromOff: -60 },
    { from: 'cap', to: 'k2', fromSide: 'bottom', toSide: 'top', fromOff: -20 },
    { from: 'cap', to: 'kn', fromSide: 'bottom', toSide: 'top', fromOff: 70 },
    { from: 'g_pool', to: 'api', fromSide: 'bottom', toSide: 'top' },
    { from: 'api', to: 'r429' },
    { from: 'r429', to: 'ok', label: '否', color: 'accept' },
    { from: 'r429', to: 'cool', label: '是', color: 'warn' },
    { from: 'cool', to: 'rl', label: '配额用尽', color: 'reject' },
    { from: 'cool', to: 'g_pool', fromSide: 'right', toSide: 'right', via: [[960, 765], [960, 440]], label: '重新排队', style: 'dashed', color: 'warn', labelAt: [960, 640] },
  ],
};
