// Slide: two independent checks before the Chair — dedup and verification.
module.exports = {
  id: 's06-score-verify', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'q', type: 'data', label: '候选题', at: [300, 60], w: 240 },
    { id: 'l0', type: 'rule', label: 'L0 · 完全相同', sub: '归一化文本 SHA256', at: [300, 175], w: 420 },
    { id: 'l1', type: 'rule', label: 'L1 · 模板骨架', sub: 'SimHash 汉明 ≤3 · Jaccard ≥0.91', at: [300, 295], w: 420, maxSubW: 390 },
    { id: 'l2', type: 'rule', label: 'L2 · 语义近邻', sub: 'embedding 余弦 ≥0.94 · 跨语言 claim ≥0.92', at: [300, 415], w: 420, maxSubW: 390 },
    { id: 'dup', type: 'reject', label: 'dup', at: [150, 545], w: 130 },
    { id: 'gray', type: 'pending', label: 'gray', at: [300, 545], w: 130 },
    { id: 'uniq', type: 'accept', label: 'unique', at: [450, 545], w: 140 },

    { id: 'scalar', type: 'llm', label: '数值 / 符号', sub: 'K=5 盲解投票\n多数 = Writer 答案', at: [870, 165], w: 330 },
    { id: 'open', type: 'llm', label: '开放题', sub: '3 份作答 + rubric\n最低分 ≥ 0.6', at: [1240, 165], w: 330 },
    { id: 'compile', type: 'tool', label: '编译题', sub: 'nvcc 沙箱\n编译结果 = 预期', at: [870, 350], w: 330 },
    { id: 'runtime', type: 'tool', label: '运行题', sub: 'GPU 沙箱 + sanitizer\n输出 = 预期', at: [1240, 350], w: 330 },
    { id: 'pass', type: 'accept', label: 'pass', at: [880, 545], w: 150 },
    { id: 'fail', type: 'reject', label: 'fail', at: [1055, 545], w: 150 },
    { id: 'pen', type: 'pending', label: 'pending', at: [1240, 545], w: 170 },
  ],
  groups: [
    { id: 'g_score', label: 'Review Score · 三层查重', color: 'orange', children: ['q', 'l0', 'l1', 'l2', 'dup', 'gray', 'uniq'] },
    { id: 'g_ver', label: 'Verifier · 按题目合同独立验证', color: 'blue', children: ['scalar', 'open', 'compile', 'runtime'] },
  ],
  edges: [
    { from: 'q', to: 'l0' },
    { from: 'l0', to: 'l1' },
    { from: 'l1', to: 'l2' },
    { from: 'l2', to: 'dup', fromOff: -150, color: 'reject' },
    { from: 'l2', to: 'gray', color: 'pending' },
    { from: 'l2', to: 'uniq', fromOff: 150, color: 'accept' },
    { from: 'g_score', to: 'g_ver', fromSide: 'right', toSide: 'left', style: 'dashed', color: 'data' },
    { from: 'g_ver', to: 'pass', fromSide: 'bottom', toSide: 'top', fromOff: -175, color: 'accept' },
    { from: 'g_ver', to: 'fail', fromSide: 'bottom', toSide: 'top', fromOff: 0, color: 'reject' },
    { from: 'g_ver', to: 'pen', fromSide: 'bottom', toSide: 'top', fromOff: 185, color: 'pending' },
  ],
};
