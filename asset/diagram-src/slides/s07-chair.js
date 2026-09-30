// Slide: Review Chair as a left-to-right funnel.
module.exports = {
  id: 's07-chair', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'pre1', type: 'rule', label: '语料 dup', sub: '→ reject', at: [160, 110], w: 260 },
    { id: 'pre2', type: 'rule', label: '非真实 Writer', sub: '→ pending', at: [160, 250], w: 260 },
    { id: 'pre3', type: 'rule', label: 'Writer 合同错', sub: '→ reject', at: [160, 390], w: 260 },

    { id: 'llm', type: 'agent', label: 'Chair LLM', sub: '独立模型 · temperature 0\n正确性 · 可答性 · claim 对齐\n硬件 · 泄漏 · 元数据 · 重复\n+ 价值分 1–5', at: [560, 250], w: 400, maxSubW: 360 },
    { id: 'rules', type: 'rule', label: '确定性规则', sub: '价值分 ≥3\n结构化题验证须 pass', at: [930, 180], w: 300 },
    { id: 'second', type: 'llm', label: '二审', sub: '高风险题 uncertain\n时再独立审一次', at: [930, 380], w: 300 },
    { id: 'mmr', type: 'rule', label: '整波精选', sub: 'MMR 多样性\n+ 批内去重', at: [1225, 250], w: 240 },
    { id: 'acc', type: 'accept', label: 'accept', sub: '+ 内部参考答案', at: [1520, 110], w: 200 },
    { id: 'rej', type: 'reject', label: 'reject', sub: '→ 可修的返修 1 次', at: [1520, 250], w: 200 },
    { id: 'pen', type: 'pending', label: 'pending', sub: '退避 30s·2ⁿ\n审满 3 次待人工', at: [1520, 390], w: 200 },
    { id: 'note', type: 'note', label: '解析失败、调用异常、证据不足 → 一律 pending；没有“默认 accept”这条路', at: [760, 560], w: 760, maxW: 720 },
  ],
  groups: [
    { id: 'g_pre', label: '预检查（不调 LLM）', color: 'gray', children: ['pre1', 'pre2', 'pre3'] },
  ],
  edges: [
    { from: 'g_pre', to: 'llm' },
    { from: 'llm', to: 'rules', fromSide: 'right', toSide: 'left', fromOff: -70 },
    { from: 'llm', to: 'second', fromSide: 'right', toSide: 'left', fromOff: 70, color: 'pending' },
    { from: 'rules', to: 'mmr', fromSide: 'right', toSide: 'left', toOff: -30 },
    { from: 'second', to: 'mmr', fromSide: 'right', toSide: 'left', toOff: 30 },
    { from: 'mmr', to: 'acc', color: 'accept' },
    { from: 'mmr', to: 'rej', color: 'reject' },
    { from: 'mmr', to: 'pen', color: 'pending' },
  ],
};
