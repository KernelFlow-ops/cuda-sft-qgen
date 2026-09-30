// Slide: one coordinator, three worker lanes, one background knowledge thread.
const LX = 1040;
module.exports = {
  id: 's03-lanes', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'coord', type: 'rule', label: '并发协调器', sub: '每 0.25s 一轮\n刷新 → 回收 → 熔断检查\n→ 派发三条工作线\n队列容量 40 · 崩溃可续跑', at: [260, 285], w: 400, h: 260 },
    { id: 'gen', type: 'agent', label: 'generation-stage', sub: 'Planner → Writer / Evolver', badge: '6 线程', at: [LX, 70], w: 560, maxSubW: 400, badgeScale: 1.3 },
    { id: 'rev', type: 'agent', label: 'review-stage', sub: 'Score → Verifier → Chair → Archivist', badge: '4 线程', at: [LX, 215], w: 560, maxSubW: 400, badgeScale: 1.3 },
    { id: 'rep', type: 'agent', label: 'repair-stage', sub: 'Salvage / Rewrite / Review Repair', badge: '2 线程', at: [LX, 360], w: 560, maxSubW: 400, badgeScale: 1.3 },
    { id: 'kw', type: 'agent', label: 'KnowledgeWorker（守护线程）', sub: '扩叶子 · 官方文档审计 · 关系 · 计算校验', at: [LX, 505], w: 560, maxSubW: 400 },
    { id: 'pool', type: 'tool', label: 'ApiKeyPool：Writer 0 > Salvage 1 > Chair 2 > Sieve 3 > Taxonomy 4', sub: 'Chair 预留 1 个槽位 · 连续 429 自动降并发 · 后台任务主动让路', at: [690, 655], w: 1300, maxSubW: 1100 },
  ],
  edges: [
    { from: 'coord', to: 'gen', fromSide: 'right', toSide: 'left', fromOff: -90, label: '有余量就派发', style: 'dashed', color: 'data', labelAt: [600, 120] },
    { from: 'coord', to: 'rev', fromSide: 'right', toSide: 'left', fromOff: -30, label: '≥1 题到期即审', style: 'dashed', color: 'data', labelAt: [600, 230] },
    { from: 'coord', to: 'rep', fromSide: 'right', toSide: 'left', fromOff: 30, label: '坏 JSON / 缺陷', style: 'dashed', color: 'data', labelAt: [600, 330] },
    { from: 'coord', to: 'kw', fromSide: 'right', toSide: 'left', fromOff: 90, label: '每 5 轮审题入队', style: 'dashed', color: 'feedback', labelAt: [600, 440] },
  ],
};
