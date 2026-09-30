// Hand-placed: main chain on the left, four verification modes in a 2×2 grid.
const MX = 250;            // main chain x
const A = 40; const B = 520; const GW = 440; const GH = 372;
const R1 = 640; const R2 = 1062;
const mode = (x0, y0) => ({
  n1: [x0 + 150, y0 + 98], n2: [x0 + 150, y0 + 200], d: [x0 + 150, y0 + 306],
  p1: [x0 + 350, y0 + 256], p2: [x0 + 350, y0 + 330], top: [x0 + 350, y0 + 98],
});
const s = mode(A, R1); const o = mode(B, R1); const c = mode(A, R2); const r = mode(B, R2);

module.exports = {
  id: '07-verifier',
  manual: true,
  title: 'Verifier · 独立验证（Chair 内部调用）',
  subtitle: 'review/verifier.py · review/cute_shapes.py',
  nodes: [
    { id: 'in', type: 'data', label: '候选题', sub: '题干 + verification_spec + 私有 draft_target', at: [MX, 60], w: 330, maxSubW: 300 },
    { id: 'cute', type: 'rule', label: 'CuTe 形状检查', sub: 'check_cute_shapes', at: [MX, 170], w: 220 },
    { id: 'f_cute', type: 'reject', label: 'fail', sub: '形状矛盾 → Chair 强制 reject', at: [600, 138], w: 250 },
    { id: 'p_cute', type: 'pending', label: 'pending', sub: '形状写法不支持', at: [600, 212], w: 250 },
    { id: 'on', type: 'decision', label: '验证已开启且\n有结构化合同?', at: [MX, 298] },
    { id: 'dis', type: 'start', label: 'disabled', sub: '功能关闭 / 旧版无合同题\n不额外调用 LLM', at: [600, 298], w: 230 },
    { id: 'valid', type: 'decision', label: '合同有效?', at: [MX, 420] },
    { id: 'p_spec', type: 'pending', label: 'pending', sub: '合同字段缺失或非法', at: [600, 420], w: 230 },
    { id: 'mode', type: 'decision', label: 'spec.mode', at: [MX, 530] },
    { id: 'budget', type: 'note', label: '全局预算：每次 Solver / 评分调用先 reserve，单次运行 ≤600 次；耗尽后题目保持 pending，且不消耗审题次数', at: [870, 330], w: 250, maxW: 222 },

    { id: 'solve', type: 'llm', label: 'K 个盲解 Solver', sub: '只看公开题干 · K=5\n温度 0.2 · 并行投票', at: s.n1, w: 250 },
    { id: 'canon', type: 'rule', label: '答案归一化', sub: '安全 AST + Fraction\nSymPy 化简 · 保留单位', at: s.n2, w: 250 },
    { id: 'vote', type: 'decision', label: '多数 ≥60%\n= Writer 答案?', at: s.d },
    { id: 's_pass', type: 'accept', label: 'pass', at: s.p1, w: 120 },
    { id: 's_pen', type: 'pending', label: 'pending', sub: '票数不齐 / 分歧', at: s.p2, w: 150 },

    { id: 'resp', type: 'llm', label: '3 份独立作答', sub: '只看公开题干 · 温度 0.3', at: o.n1, w: 250 },
    { id: 'judge', type: 'llm', label: 'Rubric 评分', sub: '每份 0–1 分\nRIP = 三份最低分', at: o.n2, w: 250 },
    { id: 'rip', type: 'decision', label: 'RIP ≥ 0.6?', at: o.d },
    { id: 'o_pass', type: 'accept', label: 'pass', at: o.p1, w: 120 },
    { id: 'o_pen', type: 'pending', label: 'pending', sub: '分数不足', at: o.p2, w: 150 },

    { id: 'tu', type: 'decision', label: '完整编译单元?\n#include + main', at: [c.n1[0], c.n1[1] + 6] },
    { id: 'nvcc', type: 'tool', label: 'nvcc -arch=sm_XX', sub: 'bwrap 沙箱 · 无网络\nprlimit 限 CPU/文件/fd', at: c.n2, w: 250 },
    { id: 'cmp', type: 'decision', label: '编译结果\n= 预期?', at: c.d },
    { id: 'c_pen', type: 'pending', label: 'pending', sub: '不完整 / 环境问题\n诊断未出现', at: c.top, w: 160 },
    { id: 'c_pass', type: 'accept', label: 'pass', at: c.p1, w: 120 },
    { id: 'c_fail', type: 'reject', label: 'fail', sub: '与合同相反', at: c.p2, w: 150 },

    { id: 'gpu', type: 'decision', label: '本机 GPU\n架构匹配?', at: [r.n1[0], r.n1[1] + 6] },
    { id: 'run', type: 'tool', label: '沙箱内编译并运行', sub: '只暴露 GPU 设备节点\n可选 compute-sanitizer', at: r.n2, w: 250 },
    { id: 'out', type: 'decision', label: '输出 / 诊断\n= 预期?', at: r.d },
    { id: 'r_pen', type: 'pending', label: 'pending', sub: '架构不符 / 超时\n非零退出', at: r.top, w: 160 },
    { id: 'r_pass', type: 'accept', label: 'pass', at: r.p1, w: 120 },
    { id: 'r_fail', type: 'reject', label: 'fail', sub: 'stdout 矛盾', at: r.p2, w: 150 },

    { id: 'ev', type: 'data', label: 'VerificationEvidence', sub: 'status：pass · fail · pending · disabled\nreason + 票数 / 评分 / 工具日志', at: [480, 1540], w: 420, maxSubW: 390 },
    { id: 'chair', type: 'agent', label: 'Review Chair', sub: '写入审题 payload，供第二次审题复用', at: [480, 1650], w: 320 },
    { id: 'note', type: 'note', label: '超时不能证明死锁；一次干净运行不能证明没有竞态', at: [850, 1545], w: 210, maxW: 185 },
  ],
  groups: [
    { id: 'g_all', label: '按 spec.mode 分派', color: 'gray', box: [A - 22, R1 - 58, B + GW - A + 44, R2 + GH - R1 + 80] },
    { id: 'g_s', label: 'scalar / symbolic · 盲解投票', color: 'blue', box: [A, R1, GW, GH] },
    { id: 'g_o', label: 'open · 多答 + Rubric', color: 'violet', box: [B, R1, GW, GH] },
    { id: 'g_c', label: 'compile · 编译预期', color: 'gray', box: [A, R2, GW, GH] },
    { id: 'g_r', label: 'runtime · 运行预期', color: 'orange', box: [B, R2, GW, GH] },
  ],
  edges: [
    { from: 'in', to: 'cute' },
    { from: 'cute', to: 'f_cute', fromSide: 'right', toSide: 'left', fromOff: -12, label: '矛盾', color: 'reject', labelAt: [420, 138] },
    { from: 'cute', to: 'p_cute', fromSide: 'right', toSide: 'left', fromOff: 12, label: '不支持', color: 'pending', labelAt: [420, 212] },
    { from: 'cute', to: 'on', label: '通过 / 无关' },
    { from: 'on', to: 'dis', label: '否' },
    { from: 'on', to: 'valid', label: '是' },
    { from: 'valid', to: 'p_spec', color: 'pending' },
    { from: 'valid', to: 'mode', label: '是' },
    { from: 'mode', to: 'g_s', fromSide: 'bottom', toSide: 'top', toOff: MX - (A + GW / 2) },
    { from: 'mode', to: 'g_o', fromSide: 'right', toSide: 'top', fromOff: -6 },
    { from: 'mode', to: 'g_c', fromSide: 'left', toSide: 'left', via: [[8, 530], [8, R2 + GH / 2]] },
    { from: 'mode', to: 'g_r', fromSide: 'right', toSide: 'right', fromOff: 8, via: [[1000, 538], [1000, R2 + GH / 2]] },

    { from: 'solve', to: 'canon' }, { from: 'canon', to: 'vote' },
    { from: 'vote', to: 's_pass', fromSide: 'right', toSide: 'left', color: 'accept' },
    { from: 'vote', to: 's_pen', fromSide: 'bottom', toSide: 'left', color: 'pending' },
    { from: 'resp', to: 'judge' }, { from: 'judge', to: 'rip' },
    { from: 'rip', to: 'o_pass', fromSide: 'right', toSide: 'left', color: 'accept' },
    { from: 'rip', to: 'o_pen', fromSide: 'bottom', toSide: 'left', color: 'pending' },
    { from: 'tu', to: 'nvcc', label: '是' },
    { from: 'tu', to: 'c_pen', fromSide: 'right', toSide: 'left', color: 'pending' },
    { from: 'nvcc', to: 'cmp' },
    { from: 'cmp', to: 'c_pass', fromSide: 'right', toSide: 'left', color: 'accept' },
    { from: 'cmp', to: 'c_fail', fromSide: 'bottom', toSide: 'left', color: 'reject' },
    { from: 'gpu', to: 'run', label: '是' },
    { from: 'gpu', to: 'r_pen', fromSide: 'right', toSide: 'left', color: 'pending' },
    { from: 'run', to: 'out' },
    { from: 'out', to: 'r_pass', fromSide: 'right', toSide: 'left', color: 'accept' },
    { from: 'out', to: 'r_fail', fromSide: 'bottom', toSide: 'left', color: 'reject' },

    { from: 'g_all', to: 'ev', label: '任一模式的结论', toSide: 'top' },
    { from: 'ev', to: 'chair' },
    { from: 'ev', to: 'note', style: 'dotted', arrow: false, color: 'muted' },
  ],
};
