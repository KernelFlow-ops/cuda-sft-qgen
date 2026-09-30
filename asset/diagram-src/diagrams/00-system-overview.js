// Hand-placed layout: the README hero image.
const Y_TOP = 70;
const Y_BAR = 205;
const Y_MAIN = 400;
const Y_LOW = 590;
const Y_BOTTOM = 715;
const X = { planner: 85, writer: 270, pend: 455, score: 640, chair: 825, arch: 1010 };

module.exports = {
  id: '00-system-overview',
  manual: true,
  title: 'cuda-sft-qgen · 多 Agent 系统总览',
  subtitle: '一个协调器 · 三条工作线 · 一个后台知识线程 —— 只出题，不出答案',
  nodes: [
    { id: 'cli', type: 'start', label: 'cuda-sft-qgen generate / resume', at: [150, Y_TOP] },
    { id: 'boot', type: 'agent', label: 'Bootstrap', sub: '从 SQLite 恢复进度', at: [400, Y_TOP], w: 150 },
    { id: 'coord', type: 'rule', label: '并发协调器 run_concurrent', sub: '派发三条工作线 · 回收结果 · 熔断 · 背压 · 断点续跑', at: [340, Y_BAR], w: 640, maxSubW: 560 },

    { id: 'planner', type: 'agent', label: 'Planner', sub: '配额 · steering\n· 知识上下文', at: [X.planner, Y_MAIN], w: 140, h: 92 },
    { id: 'writer', type: 'agent', label: 'Writer\n/ Evolver', sub: '分层 prompt\n规则筛 · 自洽门', at: [X.writer, Y_MAIN], w: 140, h: 92 },
    { id: 'pend', type: 'store', label: 'pending', sub: 'SQLite 持久化', at: [X.pend, Y_MAIN], w: 140, h: 92 },
    { id: 'score', type: 'agent', label: 'Review Score', sub: '近邻检索\n三层查重', at: [X.score, Y_MAIN], w: 140, h: 92 },
    { id: 'chair', type: 'agent', label: 'Review Chair', sub: '独立审题\n+ Verifier', at: [X.chair, Y_MAIN], w: 140, h: 92 },
    { id: 'arch', type: 'agent', label: 'Archivist', sub: '单事务归档\n登记知识依赖', at: [X.arch, Y_MAIN], w: 140, h: 92 },

    { id: 'repair', type: 'agent', label: 'Salvage / Rewrite / Review Repair', sub: 'repair-stage 线程\n修 JSON 外壳 · 带缺陷说明重写一次', at: [362, Y_LOW], w: 340 },
    { id: 'qdb', type: 'store', label: 'questions.sqlite3', sub: 'accepted · rejected · pending\n生成任务 · 重试队列', at: [X.arch, Y_LOW], w: 210 },
    { id: 'out', type: 'data', label: 'training.jsonl', sub: '只导出独立审查通过的题', at: [X.arch, Y_BOTTOM], w: 230 },

    { id: 'tm', type: 'agent', label: 'TaxonomyManager', sub: '体检 · 扩叶子', at: [840, Y_TOP + 8], w: 160 },
    { id: 'audit', type: 'agent', label: 'Auditor', sub: '官方原文逐字审计', at: [1015, Y_TOP + 8], w: 150 },
    { id: 'kdb', type: 'store', label: 'knowledge.sqlite3', sub: 'claim 版本 · 证据 · 关系', at: [927, Y_BAR + 12], w: 220 },

    { id: 'pool', type: 'tool', label: 'ApiKeyPool · 所有 LLM 调用共享', sub: '优先级 Writer 0 > Salvage 1 > Chair 2 > Sieve 3 > Taxonomy 4', at: [300, Y_BOTTOM], w: 520, maxSubW: 480 },
    { id: 'note', type: 'note', label: '每道题都要独立审查；失败、不确定只会 pending，绝不默认放行', at: [712, Y_BOTTOM], w: 262, maxW: 236 },
  ],
  groups: [
    { id: 'g_gen', label: 'generation-stage', color: 'blue', children: ['planner', 'writer'] },
    { id: 'g_rev', label: 'review-stage', color: 'violet', children: ['score', 'chair', 'arch'] },
    { id: 'g_kw', label: 'KnowledgeWorker 守护线程', color: 'green', children: ['tm', 'audit', 'kdb'] },
  ],
  edges: [
    { from: 'cli', to: 'boot' },
    { from: 'boot', to: 'coord', toOff: 60 },
    { from: 'coord', to: 'planner', fromSide: 'bottom', fromOff: X.planner - 20 - 340, toSide: 'top', toOff: -20, label: '派发生成', style: 'dashed', color: 'data', labelAt: [112, 250] },
    { from: 'coord', to: 'score', fromSide: 'bottom', fromOff: X.score - 340, toSide: 'top', label: '≥1 题到期即审', style: 'dashed', color: 'data', labelAt: [582, 250] },
    { from: 'coord', to: 'tm', fromSide: 'right', toSide: 'left', via: [[700, Y_BAR], [700, Y_TOP + 8]], label: '每 5 轮审题入队', style: 'dashed', color: 'feedback', labelW: 70, labelAt: [652, 140] },
    { from: 'tm', to: 'kdb', fromSide: 'bottom', toSide: 'top', toOff: -40 },
    { from: 'audit', to: 'kdb', fromSide: 'bottom', toSide: 'top', toOff: 40 },
    { from: 'kdb', to: 'planner', fromSide: 'bottom', toSide: 'top', toOff: 25, via: [[927, 284], [X.planner + 25, 284]], label: '知识上下文 · claim 版本', style: 'dashed', color: 'data', labelAt: [470, 284], labelW: 220 },
    { from: 'planner', to: 'writer', label: 'brief' },
    { from: 'writer', to: 'pend' },
    { from: 'pend', to: 'score', label: '到期' },
    { from: 'score', to: 'chair' },
    { from: 'chair', to: 'arch' },
    { from: 'writer', to: 'repair', fromSide: 'bottom', toSide: 'top', toOff: -92, label: '坏 JSON / 确定缺陷', color: 'pending', labelAt: [200, 516], labelW: 200 },
    { from: 'repair', to: 'pend', fromSide: 'top', fromOff: 93, toSide: 'bottom', label: '修好入 pending', color: 'accept', labelAt: [515, 516], labelW: 200 },
    { from: 'chair', to: 'planner', fromSide: 'bottom', toSide: 'bottom', via: [[X.chair, 482], [X.planner, 482]], label: 'steering 反馈（按存活题统计）', style: 'dashed', color: 'feedback', labelAt: [650, 482], labelW: 260 },
    { from: 'arch', to: 'qdb', label: '单事务提交' },
    { from: 'qdb', to: 'out', label: 'export 复核' },
    { from: 'kdb', to: 'qdb', fromSide: 'right', toSide: 'right', via: [[1150, Y_BAR + 12], [1150, Y_LOW]], label: 'claim 被驳斥 → 相关题退回 pending', style: 'dashed', color: 'reject', labelW: 80, labelAt: [1150, 400] },
  ],
};
