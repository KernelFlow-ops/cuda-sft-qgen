// Slide: Writer — layered prompt → LLM → three gates → pending / repair / retry / rejected.
const GX = 1010; const OX = 1330;
module.exports = {
  id: 's05-writer', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'p1', type: 'rule', label: '① 静态前缀', sub: 'writer.md / evolver.md', at: [220, 75], w: 380 },
    { id: 'p2', type: 'rule', label: '② persona + 质量卡', sub: 'bloom · 难度 · kind', at: [220, 190], w: 380 },
    { id: 'p3', type: 'rule', label: '③ 约束 + 思维链 + 硬件事实', sub: 'constraints · cot · policy', at: [220, 305], w: 380 },
    { id: 'p4', type: 'rule', label: '④ few-shot', sub: '正例 ≤5 · 负例 ≤2 · 反模板 ≤15', at: [220, 420], w: 380 },
    { id: 'p5', type: 'data', label: '⑤ 任务单 + 知识上下文', sub: 'CLAIM · ANGLE · TYPE · ARCH', at: [220, 535], w: 380 },

    { id: 'llm', type: 'agent', label: 'Writer / Evolver', sub: '6 个写手线程\n推理耗尽 → 加预算重试 1 次', at: [640, 300], w: 330, maxSubW: 300 },
    { id: 'retry', type: 'queue', label: 'retry', sub: '429 · 退避 15s→120s', at: [640, 520], w: 300 },

    { id: 'parse', type: 'rule', label: '解析 JSON', at: [GX, 90], w: 240 },
    { id: 'sieve', type: 'rule', label: '规则筛', at: [GX, 230], w: 240 },
    { id: 'gate', type: 'rule', label: '自洽门', at: [GX, 370], w: 240 },
    { id: 'pend', type: 'store', label: 'pending', sub: '等 Chair 审', at: [GX, 525], w: 240 },
    { id: 'salv', type: 'queue', label: '修题线', sub: '修 JSON 外壳', at: [OX, 90], w: 240 },
    { id: 'rej', type: 'reject', label: 'rejected', sub: '空泛 · 离题 · 泄漏', at: [OX, 230], w: 240 },
    { id: 'rw', type: 'queue', label: '修题线', sub: '带缺陷说明重写 1 次', at: [OX, 370], w: 240 },
  ],
  groups: [
    { id: 'g_p', label: '分层 prompt（字节稳定，可缓存）', color: 'blue', children: ['p1', 'p2', 'p3', 'p4', 'p5'] },
  ],
  edges: [
    { from: 'g_p', to: 'llm' },
    { from: 'llm', to: 'parse', fromSide: 'right', toSide: 'left', fromOff: -30, via: [[850, 270], [850, 90]] },
    { from: 'llm', to: 'retry', color: 'warn', style: 'dashed' },
    { from: 'parse', to: 'sieve' },
    { from: 'sieve', to: 'gate' },
    { from: 'gate', to: 'pend', label: '通过', color: 'accept' },
    { from: 'parse', to: 'salv', label: '坏了', color: 'pending' },
    { from: 'sieve', to: 'rej', label: '不过', color: 'reject' },
    { from: 'gate', to: 'rw', label: '缺陷', color: 'pending' },
  ],
};
