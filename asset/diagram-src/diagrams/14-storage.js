// Hand-placed: two authoritative SQLite files side by side, derived files below.
const L = [145, 400]; const R = [680, 950];
const Y = [255, 420, 580, 700];
module.exports = {
  id: '14-storage',
  manual: true,
  title: '数据与存储 · SQLite 是唯一权威',
  subtitle: 'store/archive.py · store/knowledge.py · data_dir/',
  nodes: [
    { id: 'lock', type: 'rule', label: 'pipeline.lock', sub: '同一 data_dir 同时只允许一个进程', at: [275, 55], w: 330 },
    { id: 'tax', type: 'rule', label: 'configs/taxonomy', sub: '31 个 L1 · 404 个 L2 · 3311 个 L3', at: [R[1], 55], w: 330 },

    { id: 'q_tasks', type: 'store', label: 'generation_tasks', sub: 'pending → complete\nrepair_pending（审题返修）', at: [L[0], Y[0]], w: 220 },
    { id: 'q_retry', type: 'store', label: 'retry_tasks', sub: '失败转入 · due_at\n租约 lease_until', at: [L[1], Y[0]], w: 200 },
    { id: 'q_questions', type: 'store', label: 'questions', sub: 'status = accepted / rejected / pending\npayload = 完整 Question JSON', at: [275, Y[1]], w: 360, maxSubW: 330 },
    { id: 'q_meta', type: 'store', label: 'metadata', sub: 'run_state · steering\nknowledge_cursor', at: [L[0], Y[2]], w: 220 },
    { id: 'q_corr', type: 'store', label: 'corrections', sub: '知识失效事件 → 题目', at: [L[1], Y[2]], w: 200 },
    { id: 'q_exp', type: 'store', label: 'export_members', sub: '每次导出包含哪些题', at: [275, Y[3]], w: 240 },

    { id: 'k_jobs', type: 'store', label: 'jobs', sub: 'expand · audit · relate\ncompute_verify · inventory', at: [R[0], Y[0]], w: 200 },
    { id: 'k_claims', type: 'store', label: 'claims + revisions', sub: '5 种状态：legacy_unreviewed\nprovisional · verified\nquarantined · retired', at: [R[1], Y[0]], w: 230 },
    { id: 'k_audit', type: 'store', label: 'audits', sub: 'supported / contradicted\n/ insufficient', at: [R[0], Y[1]], w: 200 },
    { id: 'k_evid', type: 'store', label: 'evidence + documents', sub: '官方原文逐字引文\n文档按 sha256 存档', at: [R[1], Y[1]], w: 230 },
    { id: 'k_qc', type: 'store', label: 'question_claims', sub: '题目 ↔ claim 版本', at: [R[0], Y[2]], w: 200 },
    { id: 'k_rel', type: 'store', label: 'relations + entities', sub: '13 种边：requires · derives\nsupported_by · applies_to …', at: [R[1], Y[2]], w: 230 },
    { id: 'k_imp', type: 'store', label: 'impact_events', sub: 'claim 版本变化 · 被驳斥', at: [R[0], Y[3]], w: 200 },

    { id: 'jsonl', type: 'rule', label: 'JSONL 投影', sub: 'accepted / rejected / pending\nretry_queue · corrections', at: [165, 885], w: 260 },
    { id: 'npz', type: 'rule', label: '派生索引', sub: 'embeddings.npz · gist_embeddings.npz\nembed_cache.npz', at: [450, 885], w: 290 },
    { id: 'stats', type: 'rule', label: '快照 / 日志', sub: 'stats.json · taxonomy_health.json\nsalvage_raw.jsonl', at: [735, 885], w: 260 },
    { id: 'train', type: 'data', label: 'training.jsonl', sub: '+ .manifest.json\n+ .curriculum.json', at: [R[1] + 20, 885], w: 190 },
  ],
  groups: [
    { id: 'g_q', label: 'questions.sqlite3 · Archive（WAL）', color: 'green', solid: true, box: [20, 140, 510, 615] },
    { id: 'g_k', label: 'knowledge.sqlite3 · KnowledgeStore（WAL）', color: 'cyan', solid: true, box: [562, 140, 518, 615] },
    { id: 'g_d', label: '派生文件：可删、可从 SQLite 重建', color: 'gray', box: [20, 805, 870, 150] },
  ],
  edges: [
    { from: 'q_tasks', to: 'q_retry', color: 'warn' },
    { from: 'q_tasks', to: 'q_questions', fromSide: 'bottom', toSide: 'top', toOff: -100, label: '出题成功' },
    { from: 'q_retry', to: 'q_questions', fromSide: 'bottom', toSide: 'top', toOff: 110, label: 'drain', style: 'dashed' },
    { from: 'q_corr', to: 'q_questions', fromSide: 'top', toSide: 'bottom', toOff: 110, label: 'accepted → pending', color: 'pending' },
    { from: 'q_questions', to: 'q_meta', fromSide: 'bottom', toSide: 'top', fromOff: -110, label: '同一事务', style: 'dashed' },
    { from: 'q_questions', to: 'q_exp', fromSide: 'bottom', toSide: 'top', style: 'dashed' },
    { from: 'tax', to: 'k_claims', label: '首次导入' },
    { from: 'k_jobs', to: 'k_claims' },
    { from: 'k_claims', to: 'k_evid', label: '审计' },
    { from: 'k_evid', to: 'k_rel', label: 'supported' },
    { from: 'k_claims', to: 'k_imp', fromSide: 'left', toSide: 'right', fromOff: 32, via: [[808, Y[0] + 32], [808, Y[3]]], label: 'revise / 隔离', color: 'reject', labelAt: [808, 500] },
    { from: 'q_questions', to: 'k_qc', fromSide: 'right', toSide: 'left', fromOff: 30, via: [[546, Y[1] + 30], [546, Y[2]]], label: '归档时登记', style: 'dashed', labelAt: [546, 505] },
    { from: 'k_qc', to: 'k_imp' },
    { from: 'k_imp', to: 'q_corr', fromSide: 'left', toSide: 'right', via: [[520, Y[3]], [520, Y[2] + 15]], label: 'invalidate', color: 'reject', labelAt: [520, 645] },
    { from: 'g_q', to: 'g_d', fromSide: 'bottom', toSide: 'top', fromOff: -130, toOff: -305 },
    { from: 'q_exp', to: 'train', fromSide: 'bottom', toSide: 'top', via: [[275, 780], [R[1] + 20, 780]], label: 'export：逐题复核后写出', labelAt: [640, 780], labelW: 220 },
  ],
};
