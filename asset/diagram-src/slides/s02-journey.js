// Slide: one question's journey (U-shaped: generation row → review row).
const X = [140, 440, 740, 1040, 1340];
const Y1 = 95; const Y2 = 400;
module.exports = {
  id: 's02-journey', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'kg', type: 'store', label: '知识图谱', sub: 'claim · 证据 · 关系', at: [X[0], Y1], w: 250 },
    { id: 'planner', type: 'agent', label: 'Planner', sub: '配额 · steering\n10% 探索新叶子', at: [X[1], Y1], w: 250 },
    { id: 'writer', type: 'agent', label: 'Writer / Evolver', sub: '分层 prompt\nfew-shot · 种子题', at: [X[2], Y1], w: 250 },
    { id: 'gate', type: 'rule', label: '规则筛 + 自洽门', sub: '拦确定缺陷\n可重写 1 次', at: [X[3], Y1], w: 250 },
    { id: 'pend', type: 'store', label: 'pending', sub: 'SQLite 持久化', at: [X[4], Y1], w: 250 },
    { id: 'score', type: 'agent', label: 'Review Score', sub: 'L0 / L1 / L2 查重\n+ 近邻原文', at: [X[4], Y2], w: 250 },
    { id: 'ver', type: 'llm', label: 'Verifier', sub: '盲解投票 · 编译\n沙箱运行', at: [X[3], Y2], w: 250 },
    { id: 'chair', type: 'agent', label: 'Review Chair', sub: '独立模型逐题审\n7 项检查', at: [X[2], Y2], w: 250 },
    { id: 'arch', type: 'agent', label: 'Archivist', sub: '单事务提交\n登记知识依赖', at: [X[1], Y2], w: 250 },
    { id: 'out', type: 'data', label: 'training.jsonl', sub: '只导出审查通过、\n未失效的题', at: [X[0], Y2], w: 250 },
    { id: 'note', type: 'note', label: '任何一步失败或不确定 → 回到 pending 退避重审，永不默认放行', at: [740, 590], w: 700, maxW: 660 },
  ],
  edges: [
    { from: 'kg', to: 'planner' },
    { from: 'planner', to: 'writer' },
    { from: 'writer', to: 'gate' },
    { from: 'gate', to: 'pend' },
    { from: 'pend', to: 'score', label: '到期即审' },
    { from: 'score', to: 'ver' },
    { from: 'ver', to: 'chair' },
    { from: 'chair', to: 'arch' },
    { from: 'arch', to: 'out' },
    { from: 'chair', to: 'planner', fromSide: 'top', toSide: 'bottom', fromOff: -60, toOff: 60, via: [[680, 250], [500, 250]], label: 'steering 反馈', style: 'dashed', color: 'feedback', labelAt: [590, 250] },
    { from: 'arch', to: 'kg', fromSide: 'top', toSide: 'bottom', fromOff: -60, via: [[380, 262], [140, 262]], label: '失效追溯', style: 'dashed', color: 'reject', labelAt: [240, 262], back: false },
  ],
};
