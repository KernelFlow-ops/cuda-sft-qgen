// Slide: four design principles as hand-drawn cards.
const Y = 185; const X = [205, 590, 975, 1360];
const card = (stroke, fill, hatch) => ({ type: 'rule', stroke, fill, hatch, h: 320, w: 360, scale: 1.25, maxSubW: 250 });
module.exports = {
  id: 's10-principles', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'p1', ...card('#1864ab', '#e7f5ff', '#74c0fc'), label: '① 每题独立审查', sub: 'unique 也要审\n审题模型与出题分开\n数值题盲解投票\n代码题沙箱编译运行', at: [X[0], Y] },
    { id: 'p2', ...card('#d9480f', '#fff4e6', '#ffa94d'), label: '② 不确定就 pending', sub: '解析失败 · 调用异常\n证据不足 · 预算耗尽\n都不会被默认放行', at: [X[1], Y] },
    { id: 'p3', ...card('#2b8a3e', '#ebfbee', '#8ce99a'), label: '③ SQLite 单一事实源', sub: '审题结果单事务提交\n崩溃后 resume 续跑\n索引与 JSONL 可重建', at: [X[2], Y] },
    { id: 'p4', ...card('#5f3dc4', '#f3f0ff', '#b197fc'), label: '④ 知识可追溯', sub: 'claim 带版本与\n官方原文逐字证据\n被驳斥 → 相关题失效', at: [X[3], Y] },
    { id: 's1', type: 'rule', scale: 1.15, label: '知识树：31 个 L1 · 404 个 L2 · 3311 个 L3 叶子', at: [395, 440], w: 700 },
    { id: 's2', type: 'rule', scale: 1.15, label: '1 个协调器 · 3 条工作线 · 1 个后台知识线程', at: [1170, 440], w: 700 },
  ],
  edges: [],
};
