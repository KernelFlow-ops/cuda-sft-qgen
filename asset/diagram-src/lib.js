// Hand-drawn diagram renderer: ELK layered layout + rough.js strokes + resvg PNG.
// A diagram is plain data (nodes / groups / edges); see diagrams/*.js.
'use strict';

const fs = require('fs');
const path = require('path');
const fontkit = require('fontkit');
const rough = require('roughjs');
const ELK = require('elkjs/lib/elk.bundled.js');
const { Resvg } = require('@resvg/resvg-js');

const FONT_DIR = process.env.DIAGRAM_FONT_DIR || path.join(__dirname, 'fonts');
const FONTS = {
  hand: { file: 'Xiaolai-Regular.ttf', family: 'Xiaolai' },
  mono: { file: 'XiaolaiMono-Regular.ttf', family: 'Xiaolai Mono' },
};
const fontCache = {};
function font(kind) {
  if (!fontCache[kind]) fontCache[kind] = fontkit.openSync(path.join(FONT_DIR, FONTS[kind].file));
  return fontCache[kind];
}
const widthCache = new Map();
function textWidth(str, size, kind = 'hand') {
  const key = kind + '|' + str;
  let w = widthCache.get(key);
  if (w === undefined) {
    const f = font(kind);
    w = f.layout(str).advanceWidth / f.unitsPerEm;
    widthCache.set(key, w);
  }
  return w * size;
}

const CJK = '⺀-鿿豈-﫿＀-￯　-〿①-⓿';
const TOKEN = new RegExp(`[${CJK}]|[^\\s${CJK}]+\\s*|\\s+`, 'g');
const NO_LINE_START = /^[，。、；：！？）》」』】,.;:!?)\]]/;

function wrap(text, size, maxW, kind = 'hand') {
  const out = [];
  for (const para of String(text).split('\n')) {
    const tokens = para.match(TOKEN) || [''];
    let line = '';
    for (const t of tokens) {
      const cand = line + t;
      if (line.trim() && textWidth(cand.trimEnd(), size, kind) > maxW && !NO_LINE_START.test(t)) {
        out.push(line.trimEnd());
        line = t.trimStart();
      } else {
        line = cand;
      }
    }
    out.push(line.trimEnd());
  }
  return out;
}

function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------- palette
const INK = '#1e1e1e';
const PALETTE = {
  llm:      { stroke: '#1864ab', fill: '#e7f5ff', hatch: '#74c0fc', tag: 'LLM', legend: 'LLM 调用' },
  rule:     { stroke: '#d9480f', fill: '#fff9db', hatch: '#ffd43b', legend: '确定性规则 / 代码' },
  store:    { stroke: '#2b8a3e', fill: '#ebfbee', hatch: '#8ce99a', legend: '持久化存储' },
  queue:    { stroke: '#5f3dc4', fill: '#f3f0ff', hatch: '#b197fc', legend: '队列 / 任务' },
  decision: { stroke: '#a61e4d', fill: '#fff0f6', hatch: '#f783ac', legend: '判断' },
  data:     { stroke: '#0b7285', fill: '#e3fafc', hatch: '#66d9e8', legend: '数据契约' },
  tool:     { stroke: '#495057', fill: '#f1f3f5', hatch: '#adb5bd', legend: '外部工具 / 服务' },
  accept:   { stroke: '#2b8a3e', fill: '#b2f2bb', solid: true, legend: 'accept' },
  reject:   { stroke: '#c92a2a', fill: '#ffc9c9', solid: true, legend: 'reject' },
  pending:  { stroke: '#d9480f', fill: '#ffd8a8', solid: true, legend: 'pending' },
  start:    { stroke: '#343a40', fill: '#f8f9fa', solid: true, legend: '起止' },
  note:     { stroke: '#e0a800', fill: '#fff3bf', solid: true, legend: '说明' },
  agent:    { stroke: '#1864ab', fill: '#d0ebff', hatch: '#4dabf7', tag: 'Agent', legend: 'Agent' },
};
const GROUP_COLORS = {
  blue: { stroke: '#1971c2', fill: '#f4f9ff', text: '#1864ab' },
  green: { stroke: '#2f9e44', fill: '#f4fcf5', text: '#2b8a3e' },
  orange: { stroke: '#e8590c', fill: '#fff8f1', text: '#d9480f' },
  violet: { stroke: '#7048e8', fill: '#f8f6ff', text: '#5f3dc4' },
  pink: { stroke: '#c2255c', fill: '#fff5f9', text: '#a61e4d' },
  gray: { stroke: '#868e96', fill: '#fafafa', text: '#495057' },
  cyan: { stroke: '#1098ad', fill: '#f2fcfd', text: '#0b7285' },
  yellow: { stroke: '#f08c00', fill: '#fffcf0', text: '#e67700' },
};
const EDGE_COLORS = {
  default: '#495057', accept: '#2b8a3e', reject: '#c92a2a', pending: '#e8590c',
  feedback: '#7048e8', data: '#1971c2', warn: '#e67700', muted: '#adb5bd',
};

// ---------------------------------------------------------------- sizing
const S = {
  label: 17, sub: 13.3, edge: 13.6, group: 16, title: 32, subtitle: 15, note: 14,
  lh: 1.32, padX: 15, padY: 10, maxLabelW: 230, maxSubW: 240,
};

function hashSeed(str) {
  let h = 2166136261;
  for (const c of String(str)) h = Math.imul(h ^ c.codePointAt(0), 16777619);
  return (h >>> 0) % 2 ** 31 || 7;
}

function measureNode(n) {
  const scale = n.scale || 1;
  const labelSize = S.label * scale;
  const subSize = S.sub * scale;
  const maxW = (n.maxW || S.maxLabelW) * scale;
  const labelKind = n.mono ? 'mono' : 'hand';
  const labelLines = n.label ? wrap(n.label, labelSize, maxW, labelKind) : [];
  const subKind = n.subHand ? 'hand' : 'mono';
  const subLines = n.sub ? wrap(n.sub, subSize, (n.maxSubW || S.maxSubW) * scale, subKind) : [];
  if (n.type === 'note') {
    const noteSize = S.note * scale;
    const lines = wrap(n.label, noteSize, (n.maxW || 210) * scale, 'hand');
    const w = Math.max(...lines.map((l) => textWidth(l, noteSize))) + 26;
    const h = lines.length * noteSize * S.lh + 22;
    return { w, h, labelLines: lines, labelSize: noteSize, subLines: [], subSize, labelKind: 'hand', subKind };
  }
  const lw = Math.max(0, ...labelLines.map((l) => textWidth(l, labelSize, labelKind)));
  const sw = Math.max(0, ...subLines.map((l) => textWidth(l, subSize, subKind)));
  const contentW = Math.max(lw, sw);
  const contentH = labelLines.length * labelSize * S.lh + (subLines.length ? subLines.length * subSize * S.lh + 3 : 0);
  let w = contentW + 2 * S.padX * scale;
  let h = contentH + 2 * S.padY * scale;
  const shape = n.shape || defaultShape(n.type);
  if (shape === 'diamond') { w = contentW * 1.42 + 34; h = contentH * 1.55 + 30; }
  else if (shape === 'cylinder') { h += 20; w += 6; }
  else if (shape === 'stadium') { w += h * 0.45; }
  else if (shape === 'para') { w += 26; }
  else if (shape === 'ellipse') { w = contentW * 1.28 + 26; h = contentH * 1.35 + 22; }
  else if (shape === 'queue') { w += 30; }
  else if (shape === 'doc') { h += 8; }
  if (PALETTE[n.type] && PALETTE[n.type].tag && shape !== 'diamond') h += 6;
  if (n.minW) w = Math.max(w, n.minW);
  if (n.minH) h = Math.max(h, n.minH);
  return { w: Math.ceil(w), h: Math.ceil(h), labelLines, labelSize, subLines, subSize, labelKind, subKind, shape };
}

function defaultShape(type) {
  switch (type) {
    case 'decision': return 'diamond';
    case 'store': return 'cylinder';
    case 'queue': return 'queue';
    case 'data': return 'para';
    case 'start': return 'stadium';
    case 'accept': case 'reject': case 'pending': return 'stadium';
    case 'tool': return 'rect';
    case 'note': return 'note';
    default: return 'rect';
  }
}

// ---------------------------------------------------------------- layout
const elk = new ELK();

async function layout(spec) {
  const measured = new Map();
  const byId = new Map();
  const groupIds = new Set();
  for (const n of spec.nodes) {
    if (byId.has(n.id)) throw new Error(`duplicate node id ${n.id} in ${spec.id}`);
    byId.set(n.id, n);
    measured.set(n.id, measureNode(n));
  }
  const groups = spec.groups || [];
  const parentOf = new Map();
  for (const g of groups) {
    groupIds.add(g.id);
    for (const c of g.children) {
      if (parentOf.has(c)) throw new Error(`${c} in two groups (${spec.id})`);
      parentOf.set(c, g.id);
    }
  }
  const elkNode = (n) => {
    const m = measured.get(n.id);
    const node = { id: n.id, width: m.w, height: m.h };
    if (n.layer !== undefined) node.layoutOptions = { 'elk.layered.layering.layerChoiceConstraint': String(n.layer) };
    if (n.layerFirst) node.layoutOptions = { ...(node.layoutOptions || {}), 'elk.layered.layering.layerConstraint': 'FIRST' };
    if (n.layerLast) node.layoutOptions = { ...(node.layoutOptions || {}), 'elk.layered.layering.layerConstraint': 'LAST' };
    return node;
  };
  const groupNode = (g) => {
    const labelW = textWidth(g.label || '', S.group) + 40;
    const children = g.children.map((cid) => {
      if (groupIds.has(cid)) return groupNode(groups.find((x) => x.id === cid));
      const n = byId.get(cid);
      if (!n) throw new Error(`group ${g.id} child ${cid} missing`);
      return elkNode(n);
    });
    const opts = {
      'elk.padding': `[top=${g.label ? 46 : 18},left=18,bottom=18,right=18]`,
      'elk.nodeSize.constraints': 'MINIMUM_SIZE',
      'elk.nodeSize.minimum': `(${Math.ceil(labelW)},60)`,
      ...(g.direction ? { 'elk.direction': g.direction } : {}),
      ...(g.layoutOptions || {}),
    };
    return { id: g.id, children, layoutOptions: opts };
  };
  const topGroups = groups.filter((g) => !parentOf.has(g.id));
  const rootChildren = [
    ...spec.nodes.filter((n) => !parentOf.has(n.id)).map(elkNode),
    ...topGroups.map(groupNode),
  ];
  const edges = (spec.edges || []).map((e, i) => {
    // Feedback edges are laid out target→source so they do not drive layering.
    const edge = e.back
      ? { id: `e${i}`, sources: [e.to], targets: [e.from] }
      : { id: `e${i}`, sources: [e.from], targets: [e.to] };
    if (e.label) {
      const lines = wrap(e.label, S.edge, e.labelW || 150);
      edge.labels = [{
        text: e.label,
        width: Math.ceil(Math.max(...lines.map((l) => textWidth(l, S.edge))) + 10),
        height: Math.ceil(lines.length * S.edge * 1.25 + 4),
      }];
      edge._lines = lines;
    }
    return edge;
  });
  const graph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': spec.direction || 'DOWN',
      'elk.hierarchyHandling': 'INCLUDE_CHILDREN',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.json.shapeCoords': 'ROOT',
      'elk.json.edgeCoords': 'ROOT',
      'elk.layered.spacing.nodeNodeBetweenLayers': String(spec.layerGap || 40),
      'elk.spacing.nodeNode': String(spec.nodeGap || 34),
      'elk.spacing.edgeNode': '20',
      'elk.spacing.edgeEdge': '14',
      'elk.layered.spacing.edgeNodeBetweenLayers': '20',
      'elk.layered.spacing.edgeEdgeBetweenLayers': '12',
      'elk.spacing.edgeLabel': '4',
      'elk.spacing.componentComponent': '48',
      'elk.layered.nodePlacement.strategy': spec.placement || 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.semiInteractive': 'true',
      'elk.layered.considerModelOrder.strategy': spec.modelOrder || 'NODES_AND_EDGES',
      'elk.edgeLabels.placement': 'CENTER',
      'elk.edgeLabels.inline': spec.inlineLabels === false ? 'false' : 'true',
      'elk.layered.edgeLabels.sideSelection': 'SMART_DOWN',
      'elk.separateConnectedComponents': spec.separate === false ? 'false' : 'true',
      ...(spec.layoutOptions || {}),
    },
    children: rootChildren,
    edges,
  };
  const res = await elk.layout(graph);
  const pos = new Map();
  const walk = (node) => {
    for (const c of node.children || []) {
      pos.set(c.id, { x: c.x, y: c.y, w: c.width, h: c.height });
      walk(c);
    }
  };
  walk(res);
  const outEdges = [];
  const collect = (node) => {
    for (const e of node.edges || []) outEdges.push(e);
    for (const c of node.children || []) collect(c);
  };
  collect(res);
  return { res, pos, measured, byId, edges: outEdges, specEdges: spec.edges || [], groupIds };
}

// ---------------------------------------------------------------- manual layout
// Nodes carry `at: [cx, cy]` (optionally `w`/`h`); edges are routed orthogonally
// between box sides, or through explicit `via` points.
function sideAnchor(box, side, off = 0) {
  const cx = box.x + box.w / 2; const cy = box.y + box.h / 2;
  if (side === 'top') return { x: cx + off, y: box.y };
  if (side === 'bottom') return { x: cx + off, y: box.y + box.h };
  if (side === 'left') return { x: box.x, y: cy + off };
  return { x: box.x + box.w, y: cy + off };
}

function autoSides(a, b) {
  const dx = b.x + b.w / 2 - (a.x + a.w / 2);
  const dy = b.y + b.h / 2 - (a.y + a.h / 2);
  const vGap = dy > 0 ? b.y - (a.y + a.h) : a.y - (b.y + b.h);
  const hGap = dx > 0 ? b.x - (a.x + a.w) : a.x - (b.x + b.w);
  if (vGap >= hGap) return dy > 0 ? ['bottom', 'top'] : ['top', 'bottom'];
  return dx > 0 ? ['right', 'left'] : ['left', 'right'];
}

function routeManual(a, b, e) {
  let [fs, ts] = autoSides(a, b);
  if (e.fromSide) fs = e.fromSide;
  if (e.toSide) ts = e.toSide;
  const p0 = sideAnchor(a, fs, e.fromOff || 0);
  const p3 = sideAnchor(b, ts, e.toOff || 0);
  const vert = (s) => s === 'top' || s === 'bottom';
  if (e.via) {
    const via = e.via.map(([x, y]) => ({ x, y }));
    const first = via[0]; const last = via[via.length - 1];
    // Snap anchors onto the via line when it stays on the box side; else add an elbow.
    const snap = (p, box, side, target) => {
      if (vert(side)) {
        if (target.x >= box.x + 4 && target.x <= box.x + box.w - 4) return [{ x: target.x, y: p.y }];
        return [p, { x: p.x, y: target.y }];
      }
      if (target.y >= box.y + 4 && target.y <= box.y + box.h - 4) return [{ x: p.x, y: target.y }];
      return [p, { x: target.x, y: p.y }];
    };
    const head = snap(p0, a, fs, first);
    const tail = snap(p3, b, ts, last).reverse();
    return [...head, ...via, ...tail];
  }
  if (vert(fs) && vert(ts)) {
    if (Math.abs(p0.x - p3.x) < 1) return [p0, { x: p0.x, y: p3.y }];
    const my = e.bend ?? (p0.y + p3.y) / 2;
    return [p0, { x: p0.x, y: my }, { x: p3.x, y: my }, p3];
  }
  if (!vert(fs) && !vert(ts)) {
    if (Math.abs(p0.y - p3.y) < 1) return [p0, { x: p3.x, y: p0.y }];
    const mx = e.bend ?? (p0.x + p3.x) / 2;
    return [p0, { x: mx, y: p0.y }, { x: mx, y: p3.y }, p3];
  }
  return vert(fs) ? [p0, { x: p0.x, y: p3.y }, p3] : [p0, { x: p3.x, y: p0.y }, p3];
}

function layoutManual(spec) {
  const measured = new Map();
  const byId = new Map();
  const pos = new Map();
  for (const n of spec.nodes) {
    const n2 = n.w ? { ...n, minW: n.w, maxW: n.maxW || n.w - 2 * S.padX } : n;
    const m = measureNode(n2);
    if (n.h) m.h = Math.max(m.h, n.h);
    measured.set(n.id, m);
    byId.set(n.id, n);
    const [cx, cy] = n.at;
    pos.set(n.id, { x: cx - m.w / 2, y: cy - m.h / 2, w: m.w, h: m.h });
  }
  const groupIds = new Set();
  const res = { children: [] };
  for (const g of spec.groups || []) {
    groupIds.add(g.id);
    if (g.box) {
      const [x, y, w, h] = g.box;
      pos.set(g.id, { x, y, w, h });
    } else {
      const boxes = g.children.map((c) => pos.get(c));
      const pad = g.pad || [44, 16, 16, 16];
      const x0 = Math.min(...boxes.map((b) => b.x)) - pad[3];
      const y0 = Math.min(...boxes.map((b) => b.y)) - pad[0];
      const x1 = Math.max(...boxes.map((b) => b.x + b.w)) + pad[1];
      const y1 = Math.max(...boxes.map((b) => b.y + b.h)) + pad[2];
      pos.set(g.id, { x: x0, y: y0, w: x1 - x0, h: y1 - y0 });
    }
    res.children.push({ id: g.id, children: [] });
  }
  const edges = (spec.edges || []).map((e, i) => {
    const a = pos.get(e.from); const b = pos.get(e.to);
    if (!a || !b) throw new Error(`edge ${e.from}->${e.to} missing endpoint in ${spec.id}`);
    const pts = routeManual(a, b, e);
    const out = { id: `e${i}`, sections: [{ startPoint: pts[0], bendPoints: pts.slice(1, -1), endPoint: pts[pts.length - 1] }] };
    if (e.label) {
      const lines = wrap(e.label, S.edge, e.labelW || 150);
      const width = Math.ceil(Math.max(...lines.map((l) => textWidth(l, S.edge))) + 10);
      const height = Math.ceil(lines.length * S.edge * 1.25 + 4);
      let cx; let cy;
      if (e.labelAt) [cx, cy] = e.labelAt;
      else {
        let best = null;
        for (let k = 0; k < pts.length - 1; k++) {
          const len = Math.hypot(pts[k + 1].x - pts[k].x, pts[k + 1].y - pts[k].y);
          if (!best || len > best.len) best = { len, k };
        }
        const p = pts[best.k]; const q = pts[best.k + 1];
        cx = (p.x + q.x) / 2 + (e.labelDx || 0); cy = (p.y + q.y) / 2 + (e.labelDy || 0);
      }
      out.labels = [{ text: e.label, x: cx - width / 2, y: cy - height / 2, width, height }];
      out._lines = lines;
    }
    return out;
  });
  return { res, pos, measured, byId, edges, specEdges: spec.edges || [], groupIds };
}

// ---------------------------------------------------------------- drawing
const gen = rough.generator();

function toSvg(drawable, extra = '') {
  const dash = drawable.options && drawable.options.strokeLineDash;
  const dashAttr = dash && dash.length ? ` stroke-dasharray="${dash.join(' ')}"` : '';
  return gen.toPaths(drawable).map((p) => {
    const isFill = p.fill && p.fill !== 'none';
    return `<path d="${p.d}" stroke="${p.stroke}" stroke-width="${p.strokeWidth}" fill="${p.fill || 'none'}" stroke-linecap="round" stroke-linejoin="round"${isFill ? '' : dashAttr}${extra}/>`;
  }).join('');
}

function rrectPath(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  return `M${x + r},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} L${x + r},${y + h} Q${x},${y + h} ${x},${y + h - r} L${x},${y + r} Q${x},${y} ${x + r},${y} Z`;
}

function shapePath(shape, x, y, w, h) {
  switch (shape) {
    case 'diamond':
      return `M${x + w / 2},${y} L${x + w},${y + h / 2} L${x + w / 2},${y + h} L${x},${y + h / 2} Z`;
    case 'stadium':
      return rrectPath(x, y, w, h, h / 2);
    case 'para': {
      const k = 14;
      return `M${x + k},${y} L${x + w},${y} L${x + w - k},${y + h} L${x},${y + h} Z`;
    }
    case 'doc': {
      const wv = 7;
      return `M${x},${y} L${x + w},${y} L${x + w},${y + h - wv} Q${x + w * 0.75},${y + h - 2 * wv} ${x + w / 2},${y + h - wv} Q${x + w * 0.25},${y + h} ${x},${y + h - wv} Z`;
    }
    case 'ellipse':
      return null;
    default:
      return rrectPath(x, y, w, h, 11);
  }
}

function drawShape(node, box, seed) {
  const pal = PALETTE[node.type] || PALETTE.rule;
  const { x, y, w, h } = box;
  const shape = node.shape || defaultShape(node.type);
  const stroke = node.stroke || pal.stroke;
  const fill = node.fill || pal.fill;
  const hatch = node.hatch || pal.hatch;
  const sw = node.strokeWidth || (node.type === 'agent' ? 2.4 : 1.85);
  const base = { seed, roughness: node.roughness ?? 1.15, bowing: 1.1, stroke, strokeWidth: sw };
  const dash = node.dashed ? { strokeLineDash: [9, 6] } : {};
  let out = '';
  if (shape === 'note') {
    const tilt = node.tilt ?? ((seed % 5) - 2) * 0.55;
    const cx = x + w / 2; const cy = y + h / 2;
    out += `<g transform="rotate(${tilt} ${cx} ${cy})">`;
    out += `<rect x="${x + 4}" y="${y + 5}" width="${w}" height="${h}" fill="#000" opacity="0.07" rx="3"/>`;
    out += toSvg(gen.rectangle(x, y, w, h, { seed, roughness: 0.7, stroke: '#e8c547', strokeWidth: 1.2, fill: pal.fill, fillStyle: 'solid' }));
    out += `<path d="M${x + w - 16},${y + h} L${x + w},${y + h - 16} L${x + w},${y + h} Z" fill="#f5d76e" opacity="0.8"/>`;
    out += '</g>';
    return out;
  }
  if (shape === 'ellipse') {
    out += toSvg(gen.ellipse(x + w / 2, y + h / 2, w, h, { ...base, stroke: 'none', fill, fillStyle: 'solid' }));
    if (hatch && !pal.solid) out += `<g opacity="0.55">${toSvg(gen.ellipse(x + w / 2, y + h / 2, w, h, { seed: seed + 1, stroke: 'none', fill: hatch, fillStyle: 'hachure', hachureGap: 9, fillWeight: 0.9, hachureAngle: -41 }))}</g>`;
    out += toSvg(gen.ellipse(x + w / 2, y + h / 2, w, h, { ...base, ...dash }));
    return out;
  }
  if (shape === 'cylinder') {
    const ry = 9;
    const body = `M${x},${y + ry} L${x},${y + h - ry} Q${x},${y + h + ry * 0.35} ${x + w / 2},${y + h + ry * 0.2} Q${x + w},${y + h + ry * 0.35} ${x + w},${y + h - ry} L${x + w},${y + ry}`;
    const bodyFill = `${body} Q${x + w},${y - ry * 0.3} ${x + w / 2},${y - ry * 0.2} Q${x},${y - ry * 0.3} ${x},${y + ry} Z`;
    out += `<path d="${bodyFill}" fill="${fill}" stroke="none"/>`;
    if (hatch) out += `<g opacity="0.5">${toSvg(gen.path(bodyFill, { seed: seed + 1, stroke: 'none', fill: hatch, fillStyle: 'hachure', hachureGap: 9, fillWeight: 0.9, hachureAngle: -41 }))}</g>`;
    out += toSvg(gen.path(body, { ...base }));
    out += toSvg(gen.ellipse(x + w / 2, y + ry, w, ry * 2, { ...base, seed: seed + 2, fill: '#ffffff', fillStyle: 'solid' }));
    return out;
  }
  const d = shapePath(shape, x, y, w, h);
  out += `<path d="${d}" fill="${fill}" stroke="none"/>`;
  if (hatch && !pal.solid && !node.noHatch) {
    out += `<g opacity="0.5">${toSvg(gen.path(d, { seed: seed + 1, stroke: 'none', fill: hatch, fillStyle: 'hachure', hachureGap: node.type === 'agent' ? 7 : 9.5, fillWeight: 0.9, hachureAngle: -41, roughness: 1 }))}</g>`;
  }
  out += toSvg(gen.path(d, { ...base, ...dash }));
  if (shape === 'queue') {
    for (let i = 1; i <= 3; i++) {
      const lx = x + w - 8 * i - 2;
      out += toSvg(gen.line(lx, y + 4, lx, y + h - 4, { seed: seed + 10 + i, roughness: 0.8, stroke, strokeWidth: 1.3 }));
    }
  }
  return out;
}

function textBlock(lines, size, kind, cx, top, color, anchor = 'middle', weight = 0) {
  const f = font(kind);
  const asc = f.ascent / f.unitsPerEm;
  const desc = -f.descent / f.unitsPerEm;
  const lh = size * S.lh;
  let out = '';
  lines.forEach((line, i) => {
    const baseline = top + i * lh + lh / 2 + (asc - desc) * size / 2 - size * 0.04;
    const stroke = weight ? ` stroke="${color}" stroke-width="${weight}" paint-order="stroke"` : '';
    out += `<text x="${cx.toFixed(1)}" y="${baseline.toFixed(1)}" font-family="${FONTS[kind].family}" font-size="${size.toFixed(2)}" fill="${color}" text-anchor="${anchor}"${stroke}>${esc(line)}</text>`;
  });
  return out;
}

function drawNode(node, box, m) {
  const seed = hashSeed(node.id);
  let out = drawShape(node, box, seed);
  const pal = PALETTE[node.type] || {};
  const shape = m.shape || node.shape || defaultShape(node.type);
  const cx = box.x + box.w / 2;
  if (shape === 'note') {
    const tilt = node.tilt ?? ((seed % 5) - 2) * 0.55;
    const top = box.y + 11;
    out += `<g transform="rotate(${tilt} ${cx} ${box.y + box.h / 2})">`;
    out += textBlock(m.labelLines, m.labelSize, 'hand', box.x + 13, top, '#5c4400', 'start');
    out += '</g>';
    return out;
  }
  const labelH = m.labelLines.length * m.labelSize * S.lh;
  const subH = m.subLines.length ? m.subLines.length * m.subSize * S.lh + 3 : 0;
  const tagShift = pal.tag && shape !== 'diamond' ? 6 : 0;
  let top = box.y + (box.h - labelH - subH) / 2 + tagShift / 2;
  if (shape === 'cylinder') top += 8;
  const color = node.textColor || INK;
  out += textBlock(m.labelLines, m.labelSize, m.labelKind, cx, top, color, 'middle', node.bold === false ? 0 : 0.35);
  if (m.subLines.length) {
    out += textBlock(m.subLines, m.subSize, m.subKind, cx, top + labelH + 3, node.subColor || '#5c636a');
  }
  const k = S.label / 17;
  if (pal.tag && shape !== 'diamond' && !node.noTag) {
    const tag = node.tag || pal.tag;
    const fs = 10.5 * k;
    const tw = textWidth(tag, fs, 'mono') + 10 * k;
    const th = 16 * k;
    const tx = box.x + 7 * k; const ty = box.y - th / 2;
    out += toSvg(gen.path(rrectPath(tx, ty, tw, th, 7 * k), { seed: seed + 5, roughness: 0.6, stroke: pal.stroke, strokeWidth: 1.2, fill: '#ffffff', fillStyle: 'solid' }));
    out += `<text x="${tx + tw / 2}" y="${ty + th * 0.75}" font-family="${FONTS.mono.family}" font-size="${fs.toFixed(2)}" fill="${pal.stroke}" text-anchor="middle">${esc(tag)}</text>`;
  }
  if (node.badge) {
    const fs = 11 * k * (node.badgeScale || 1);
    const bw = textWidth(node.badge, fs, 'hand') + 12 * k;
    const bh = 18 * k * (node.badgeScale || 1);
    const bx = box.x + box.w - bw + 6 * k; const by = box.y - bh / 2;
    out += toSvg(gen.path(rrectPath(bx, by, bw, bh, bh / 2), { seed: seed + 7, roughness: 0.6, stroke: '#343a40', strokeWidth: 1.1, fill: node.badgeFill || '#fff3bf', fillStyle: 'solid' }));
    out += `<text x="${bx + bw / 2}" y="${by + bh * 0.72}" font-family="${FONTS.hand.family}" font-size="${fs.toFixed(2)}" fill="#343a40" text-anchor="middle">${esc(node.badge)}</text>`;
  }
  return out;
}

function drawGroup(g, box) {
  const col = GROUP_COLORS[g.color || 'gray'];
  const seed = hashSeed(g.id);
  const d = rrectPath(box.x, box.y, box.w, box.h, 16);
  let out = `<path d="${d}" fill="${g.fill || col.fill}" stroke="none"/>`;
  out += toSvg(gen.path(d, { seed, roughness: 1.3, bowing: 1.5, stroke: col.stroke, strokeWidth: g.solid ? 2 : 1.6, strokeLineDash: g.solid ? undefined : [10, 7], disableMultiStroke: true }));
  return out;
}

function drawGroupLabel(g, box) {
  const col = GROUP_COLORS[g.color || 'gray'];
  if (!g.label) return '';
  const halo = g.fill || col.fill;
  const f = font('hand');
  const base = box.y + 10 + S.group * S.lh / 2 + (f.ascent + f.descent) / f.unitsPerEm * S.group / 2;
  let out = `<text x="${box.x + 16}" y="${base.toFixed(1)}" font-family="Xiaolai" font-size="${S.group}" fill="${col.text}" stroke="${halo}" stroke-width="6" stroke-linejoin="round" paint-order="stroke">${esc(g.label)}</text>`;
  out += `<text x="${box.x + 16}" y="${base.toFixed(1)}" font-family="Xiaolai" font-size="${S.group}" fill="${col.text}" stroke="${col.text}" stroke-width="0.3">${esc(g.label)}</text>`;
  if (g.sub) {
    const lw = textWidth(g.label, S.group) + 26;
    const fm = font('mono');
    const b2 = box.y + 11.5 + S.sub * S.lh / 2 + (fm.ascent + fm.descent) / fm.unitsPerEm * S.sub / 2;
    out += `<text x="${box.x + lw}" y="${b2.toFixed(1)}" font-family="Xiaolai Mono" font-size="${S.sub}" fill="#868e96" stroke="${halo}" stroke-width="5" stroke-linejoin="round" paint-order="stroke">${esc(g.sub)}</text>`;
  }
  return out;
}

function roundedPolyline(pts, r = 10) {
  pts = pts.filter((p, i) => i === 0 || Math.hypot(p.x - pts[i - 1].x, p.y - pts[i - 1].y) > 0.5);
  if (pts.length < 2) return '';
  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const p0 = pts[i - 1]; const p1 = pts[i]; const p2 = pts[i + 1];
    const d1 = Math.hypot(p1.x - p0.x, p1.y - p0.y); const d2 = Math.hypot(p2.x - p1.x, p2.y - p1.y);
    const rr = Math.min(r, d1 / 2, d2 / 2);
    const a = { x: p1.x - (p1.x - p0.x) / d1 * rr, y: p1.y - (p1.y - p0.y) / d1 * rr };
    const b = { x: p1.x + (p2.x - p1.x) / d2 * rr, y: p1.y + (p2.y - p1.y) / d2 * rr };
    d += ` L${a.x},${a.y} Q${p1.x},${p1.y} ${b.x},${b.y}`;
  }
  const last = pts[pts.length - 1];
  d += ` L${last.x},${last.y}`;
  return d;
}

function arrowHead(tip, from, color, seed, size = 11) {
  const ang = Math.atan2(tip.y - from.y, tip.x - from.x);
  const spread = 0.46;
  const p1 = { x: tip.x - size * Math.cos(ang - spread), y: tip.y - size * Math.sin(ang - spread) };
  const p2 = { x: tip.x - size * Math.cos(ang + spread), y: tip.y - size * Math.sin(ang + spread) };
  return toSvg(gen.linearPath([[p1.x, p1.y], [tip.x, tip.y], [p2.x, p2.y]], { seed, roughness: 0.5, stroke: color, strokeWidth: 2 }));
}

function drawEdge(e, spec, idx) {
  const se = spec;
  const color = EDGE_COLORS[se.color || 'default'] || se.color;
  const seed = hashSeed(`${se.from}->${se.to}#${idx}`);
  let out = '';
  for (const s of e.sections || []) {
    const pts = [s.startPoint, ...(s.bendPoints || []), s.endPoint];
    if (se.back) pts.reverse();
    const d = roundedPolyline(pts, 12);
    const dash = se.style === 'dashed' ? [8, 6] : se.style === 'dotted' ? [2.5, 5] : undefined;
    out += toSvg(gen.path(d, { seed, roughness: 0.75, bowing: 0.8, stroke: color, strokeWidth: se.width || 1.7, strokeLineDash: dash, disableMultiStroke: true }));
    if (se.arrow !== false) {
      const tip = pts[pts.length - 1];
      let from = pts[pts.length - 2];
      out += arrowHead(tip, from, color, seed + 3);
    }
    if (se.both) out += arrowHead(pts[0], pts[1], color, seed + 4);
  }
  for (const l of e.labels || []) {
    const lines = e._lines || wrap(l.text, S.edge, 150);
    const lx = l.x + l.width / 2;
    const f = font('hand');
    let t = `<rect x="${(l.x - 1).toFixed(1)}" y="${(l.y + 1).toFixed(1)}" width="${(l.width + 2).toFixed(1)}" height="${(l.height - 1).toFixed(1)}" rx="5" fill="#ffffff" opacity="0.92"/>`;
    lines.forEach((line, i) => {
      const lh = S.edge * 1.25;
      const baseline = l.y + 2 + i * lh + lh / 2 + (f.ascent + f.descent) / f.unitsPerEm * S.edge / 2;
      t += `<text x="${lx.toFixed(1)}" y="${baseline.toFixed(1)}" font-family="Xiaolai" font-size="${S.edge}" fill="${color === EDGE_COLORS.default ? '#343a40' : color}" text-anchor="middle" stroke="#ffffff" stroke-width="4.2" stroke-linejoin="round" paint-order="stroke">${esc(line)}</text>`;
    });
    out += t;
  }
  return out;
}

function legendSvg(types, x, y, maxW) {
  let out = '';
  let cx = x; let cy = y;
  const items = types.filter((t) => PALETTE[t] && PALETTE[t].legend);
  for (const t of items) {
    const pal = PALETTE[t];
    const label = pal.legend;
    const w = 30 + textWidth(label, 13) + 22;
    if (cx + w > x + maxW) { cx = x; cy += 26; }
    const node = { id: `legend-${t}`, type: t };
    const shape = defaultShape(t);
    const box = shape === 'diamond' ? { x: cx, y: cy - 2, w: 26, h: 20 } : { x: cx, y: cy, w: 26, h: 16 };
    const n2 = { ...node, shape: shape === 'cylinder' || shape === 'queue' || shape === 'note' ? 'rect' : shape === 'para' ? 'para' : shape };
    out += drawShape({ ...n2, strokeWidth: 1.3, roughness: 0.7 }, box, hashSeed(t));
    out += `<text x="${cx + 34}" y="${cy + 13}" font-family="Xiaolai" font-size="13" fill="#495057">${esc(label)}</text>`;
    cx += w;
  }
  return { svg: out, height: cy - y + 22 };
}

async function renderDiagram(spec, outPath, opts = {}) {
  // Optional global scale (slides): enlarge every font size and padding for this render.
  const saved = { ...S };
  if (spec.scale) for (const k of ['label', 'sub', 'edge', 'group', 'note', 'padX', 'padY', 'maxLabelW', 'maxSubW']) S[k] *= spec.scale;
  try {
    return await renderDiagramInner(spec, outPath, opts);
  } finally {
    Object.assign(S, saved);
  }
}

async function renderDiagramInner(spec, outPath, opts = {}) {
  const L = spec.manual ? layoutManual(spec) : await layout(spec);
  const { res, pos } = L;
  const margin = spec.margin ?? 34;
  const titleH = spec.title ? (spec.subtitle ? 74 : 54) : 0;
  let minX = Infinity; let minY = Infinity; let maxX = -Infinity; let maxY = -Infinity;
  for (const [, b] of pos) { minX = Math.min(minX, b.x); minY = Math.min(minY, b.y - 10); maxX = Math.max(maxX, b.x + b.w); maxY = Math.max(maxY, b.y + b.h); }
  for (const e of L.edges) for (const s of e.sections || []) for (const p of [s.startPoint, ...(s.bendPoints || []), s.endPoint]) { minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x); minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y); }
  for (const e of L.edges) for (const l of e.labels || []) { minX = Math.min(minX, l.x); maxX = Math.max(maxX, l.x + l.width); maxY = Math.max(maxY, l.y + l.height); }
  const contentW = maxX - minX;
  const titleW = spec.title ? Math.max(textWidth(spec.title, S.title), spec.subtitle ? textWidth(spec.subtitle, S.subtitle, 'mono') : 0) : 0;
  const usedTypes = [...new Set(spec.nodes.map((n) => n.type))];
  const legendOrder = ['agent', 'llm', 'rule', 'decision', 'data', 'queue', 'store', 'tool', 'accept', 'reject', 'pending', 'note'];
  const legendTypes = spec.legend === false ? [] : legendOrder.filter((t) => usedTypes.includes(t) && t !== 'start');
  const W = Math.ceil(Math.max(contentW + 2 * margin, titleW + 2 * margin, spec.minWidth || 0));
  const legend = legendTypes.length ? legendSvg(legendTypes, margin, 0, W - 2 * margin) : { svg: '', height: 0 };
  const legendH = legendTypes.length ? legend.height + 22 : 0;
  const offX = margin - minX + Math.max(0, (W - 2 * margin - contentW) / 2);
  const offY = margin + titleH - minY + 8;
  const H = Math.ceil(maxY - minY + 2 * margin + titleH + legendH + 8);
  let body = '';
  // groups first (outermost first), then edges, then nodes
  const groupList = [];
  const walk = (node, depth) => {
    for (const c of node.children || []) {
      if (L.groupIds.has(c.id)) groupList.push({ id: c.id, depth });
      walk(c, depth + 1);
    }
  };
  walk(res, 0);
  groupList.sort((a, b) => a.depth - b.depth);
  for (const { id } of groupList) body += drawGroup(spec.groups.find((g) => g.id === id), pos.get(id));
  L.edges.forEach((e) => {
    const idx = Number(e.id.slice(1));
    body += drawEdge(e, L.specEdges[idx], idx);
  });
  for (const { id } of groupList) body += drawGroupLabel(spec.groups.find((g) => g.id === id), pos.get(id));
  for (const n of spec.nodes) body += drawNode(n, pos.get(n.id), L.measured.get(n.id));
  let head = '';
  if (spec.title) {
    head += `<text x="${margin}" y="${margin + 26}" font-family="Xiaolai" font-size="${S.title}" fill="${INK}" stroke="${INK}" stroke-width="0.5" paint-order="stroke">${esc(spec.title)}</text>`;
    if (spec.subtitle) head += `<text x="${margin + 2}" y="${margin + 54}" font-family="Xiaolai Mono" font-size="${S.subtitle}" fill="#868e96">${esc(spec.subtitle)}</text>`;
  }
  const legendBlock = legendTypes.length ? `<g transform="translate(0 ${H - legendH + 4})">${legend.svg}</g>` : '';
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`
    + `<rect width="${W}" height="${H}" fill="${spec.background || '#ffffff'}"/>`
    + head
    + `<g transform="translate(${offX.toFixed(1)} ${offY.toFixed(1)})">${body}</g>`
    + legendBlock
    + '</svg>';
  const zoom = opts.zoom || spec.zoom || 2;
  const r = new Resvg(svg, {
    font: {
      fontFiles: [path.join(FONT_DIR, FONTS.hand.file), path.join(FONT_DIR, FONTS.mono.file)],
      loadSystemFonts: false,
      defaultFontFamily: 'Xiaolai',
    },
    fitTo: { mode: 'zoom', value: zoom },
    shapeRendering: 2,
    textRendering: 1,
  });
  const png = r.render().asPng();
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, png);
  if (opts.svgPath) fs.writeFileSync(opts.svgPath, svg);
  return { width: W, height: H, bytes: png.length };
}

module.exports = { renderDiagram, layout, textWidth, wrap, PALETTE, GROUP_COLORS, S };
