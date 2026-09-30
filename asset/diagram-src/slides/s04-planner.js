// Slide: Planner — four signals in, three sampling steps, one brief out.
module.exports = {
  id: 's04-planner', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'quota', type: 'rule', label: '配额残差', sub: '目标占比 − 当前占比', at: [175, 70], w: 320 },
    { id: 'steer', type: 'rule', label: 'Chair 的 steering', sub: '欠配 ×3 · 超配 ×0.05', at: [175, 215], w: 320 },
    { id: 'health', type: 'rule', label: '叶子覆盖与健康分', sub: '每叶子 ≤12 题硬上限', at: [175, 360], w: 320 },
    { id: 'kg', type: 'store', label: '知识图谱', sub: 'claim 版本 · 相关 claim', at: [175, 510], w: 320 },

    { id: 'leaf', type: 'agent', label: '① 选叶子', sub: '90% 活跃叶子 · 10% 探索新叶子', at: [690, 110], w: 420, maxSubW: 380 },
    { id: 'axes', type: 'agent', label: '② 联合采样 7 个轴', sub: '题型 · 难度 · bloom · arch_target\n答案格式 · 粒度 · 场景', at: [690, 285], w: 420, maxSubW: 380 },
    { id: 'role', type: 'agent', label: '③ Writer 还是 Evolver', sub: '30% 以已通过题为种子改结构', at: [690, 460], w: 420, maxSubW: 380 },

    { id: 'brief', type: 'data', label: 'QuestionBrief', sub: '叶子 claim · 角度卡\n题型 / 难度 / bloom\narch_target · 答案格式\n知识上下文（≤6 claim）\n反模板 · 禁止复述旧题干', at: [1210, 285], w: 360, maxSubW: 320 },
    { id: 'note', type: 'note', label: '命中禁用组合：先 5 次拒绝采样，再精确枚举违规最少的组合', at: [1210, 555], w: 380, maxW: 350 },
  ],
  groups: [
    { id: 'g_in', label: '输入信号', color: 'yellow', children: ['quota', 'steer', 'health', 'kg'] },
    { id: 'g_s', label: '逐条抽样 × n', color: 'blue', children: ['leaf', 'axes', 'role'] },
  ],
  edges: [
    { from: 'g_in', to: 'g_s', label: '加权', labelAt: [405, 300] },
    { from: 'leaf', to: 'axes' },
    { from: 'axes', to: 'role' },
    { from: 'g_s', to: 'brief' },
    { from: 'brief', to: 'note', style: 'dotted', arrow: false, color: 'muted' },
  ],
};
