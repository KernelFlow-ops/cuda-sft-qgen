// Slide: question lifecycle, simplified to the main states and loops.
const Y = 280;
module.exports = {
  id: 's08-lifecycle', manual: true, scale: 1.3, legend: false, margin: 26,
  nodes: [
    { id: 'brief', type: 'data', label: 'brief', sub: '先落库', at: [120, Y], w: 190 },
    { id: 'gen', type: 'agent', label: '出题中', at: [410, Y], w: 220 },
    { id: 'pend', type: 'pending', label: 'pending', sub: 'SQLite', at: [720, Y], w: 220 },
    { id: 'rev', type: 'agent', label: '审题中', at: [1030, Y], w: 220 },
    { id: 'acc', type: 'accept', label: 'accepted', at: [1360, 150], w: 220 },
    { id: 'rej', type: 'reject', label: 'rejected', at: [1360, 410], w: 220 },
    { id: 'exp', type: 'store', label: 'training.jsonl', at: [1360, 40], w: 260 },
    { id: 'fix', type: 'queue', label: '修题 / 重试', sub: '坏 JSON · 429', at: [410, 75], w: 260 },
    { id: 'rr', type: 'queue', label: '审题返修', sub: '最多 1 次', at: [1030, 520], w: 240 },
    { id: 'exh', type: 'pending', label: 'review_exhausted', sub: '等人工 / --retry-reviews', at: [720, 520], w: 300 },
  ],
  edges: [
    { from: 'brief', to: 'gen' },
    { from: 'gen', to: 'pend', label: '过关' },
    { from: 'pend', to: 'rev', fromOff: -14, toOff: -14, label: '到期' },
    { from: 'rev', to: 'pend', fromOff: 16, toOff: 16, label: '不确定', color: 'pending', style: 'dashed' },
    { from: 'rev', to: 'acc', color: 'accept', label: '全部 pass' },
    { from: 'rev', to: 'rej', color: 'reject', label: '任一 fail' },
    { from: 'acc', to: 'exp', label: 'export' },
    { from: 'gen', to: 'fix', fromOff: -40, toOff: -40, color: 'pending' },
    { from: 'fix', to: 'gen', fromOff: 40, toOff: 40, style: 'dashed' },
    { from: 'fix', to: 'pend', fromSide: 'right', toSide: 'top', label: '修好', color: 'accept' },
    { from: 'pend', to: 'exh', label: '审满 3 次', color: 'pending' },
    { from: 'rej', to: 'rr', fromSide: 'bottom', toSide: 'right', label: '可修缺陷', color: 'feedback', style: 'dashed' },
    { from: 'rr', to: 'gen', fromSide: 'bottom', toSide: 'bottom', via: [[1030, 618], [410, 618]], label: '新 id 重新出题', style: 'dashed', color: 'feedback', labelAt: [560, 618] },
    { from: 'acc', to: 'pend', fromSide: 'left', toSide: 'top', fromOff: -20, via: [[790, 130]], label: 'claim 被驳斥', color: 'reject', style: 'dashed', labelAt: [1000, 130] },
  ],
};
