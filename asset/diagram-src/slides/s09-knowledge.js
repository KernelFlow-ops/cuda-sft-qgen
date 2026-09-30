// Slide: knowledge claims — how a leaf becomes verified evidence, and how corrections propagate.
module.exports = {
  id: 's09-knowledge', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'tm', type: 'agent', label: 'TaxonomyManager', sub: '体检 → 选 facet → 扩叶子\n语义去重 ≥0.92', at: [190, 120], w: 340 },
    { id: 'legacy', type: 'store', label: '3311 个遗留叶子', sub: 'legacy_unreviewed\n每 L2 分批送审', at: [190, 420], w: 340 },
    { id: 'prov', type: 'pending', label: 'provisional', sub: 'Planner 只给 10% 探索份额', at: [600, 270], w: 320 },
    { id: 'audit', type: 'agent', label: '官方文档审计', sub: '① 引文逐字出现在文档中\n② 覆盖全部关键标识与数字\n③ 独立 LLM 仅凭引文复核', at: [1000, 270], w: 400, maxSubW: 370 },
    { id: 'ver', type: 'accept', label: 'verified', sub: '存证据 + 关系边', at: [1400, 110], w: 240 },
    { id: 'ins', type: 'pending', label: 'insufficient', sub: '保持原状态', at: [1400, 270], w: 240 },
    { id: 'quar', type: 'reject', label: 'quarantined', sub: '关系标 stale', at: [1400, 430], w: 240 },
    { id: 'rev', type: 'rule', label: '按官方原文改写新版本', at: [1000, 540], w: 360 },
    { id: 'inv', type: 'reject', label: '相关 accepted 题退回 pending', sub: 'impact_events → invalidate', at: [1400, 590], w: 400, maxSubW: 360 },
  ],
  edges: [
    { from: 'tm', to: 'prov', label: '新叶子' },
    { from: 'legacy', to: 'audit', fromSide: 'right', toSide: 'bottom', toOff: -120, label: '分批送审', labelAt: [620, 420] },
    { from: 'prov', to: 'audit' },
    { from: 'audit', to: 'ver', color: 'accept' },
    { from: 'audit', to: 'ins', color: 'pending' },
    { from: 'audit', to: 'quar', color: 'reject' },
    { from: 'quar', to: 'rev', fromSide: 'left', toSide: 'top', toOff: 60, label: '有更正', labelAt: [1180, 430] },
    { from: 'rev', to: 'audit', fromSide: 'top', toSide: 'bottom', fromOff: -60, toOff: -60, label: '新版本再审', style: 'dashed', labelAt: [940, 420] },
    { from: 'quar', to: 'inv', color: 'reject' },
  ],
};
