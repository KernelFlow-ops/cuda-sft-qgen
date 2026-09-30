// Build the 10-slide overview deck from the rendered hand-drawn diagrams.
'use strict';
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ASSET = process.env.ASSET_DIR || path.resolve(__dirname, '..');
const OUT = process.env.DECK_OUT || path.join(ASSET, 'cuda-sft-qgen-agents.pptx');
const FONT = 'Microsoft YaHei';
const INK = '1E1E1E';
const MUTED = '5C636A';
const ACCENT = '1971C2';
const BG = 'F4F6F9';
const TOTAL = 10;

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function fit(file, box) {
  const { w, h } = pngSize(file);
  const scale = Math.min(box.w / w, box.h / h);
  const iw = w * scale; const ih = h * scale;
  return { x: box.x + (box.w - iw) / 2, y: box.y + (box.h - ih) / 2, w: iw, h: ih };
}

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 x 7.5 in
pres.author = 'cuda-sft-qgen';
pres.title = 'cuda-sft-qgen · 多 Agent 结构';

function paperCard(slide, r, pad = 0.14) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: r.x - pad, y: r.y - pad, w: r.w + 2 * pad, h: r.h + 2 * pad,
    rectRadius: 0.12, fill: { color: 'FFFFFF' }, line: { color: 'E3E7EC', width: 0.75 },
    shadow: { type: 'outer', blur: 10, offset: 3, angle: 90, color: '000000', opacity: 0.10 },
  });
}

function pageNo(slide, n) {
  slide.addText(`${String(n).padStart(2, '0')} / ${TOTAL}`, {
    x: 11.93, y: 7.05, w: 0.95, h: 0.3, fontFace: FONT, fontSize: 10, color: '8A9199',
    align: 'right', margin: 0, isTextBox: true,
  });
  slide.addText('cuda-sft-qgen', {
    x: 0.45, y: 7.05, w: 3, h: 0.3, fontFace: FONT, fontSize: 10, color: '8A9199',
    margin: 0, isTextBox: true,
  });
}

function contentSlide(n, title, caption, image, notes) {
  const slide = pres.addSlide();
  slide.background = { color: BG };
  slide.addText(title, {
    x: 0.45, y: 0.32, w: 12.4, h: 0.62, fontFace: FONT, fontSize: 28, bold: true, color: INK,
    margin: 0, isTextBox: true,
  });
  slide.addText(caption, {
    x: 0.45, y: 0.96, w: 12.4, h: 0.42, fontFace: FONT, fontSize: 15, color: MUTED,
    margin: 0, isTextBox: true,
  });
  const box = { x: 0.62, y: 1.72, w: 12.1, h: 5.05 };
  const r = fit(path.join(ASSET, image), box);
  paperCard(slide, r);
  slide.addImage({ path: path.join(ASSET, image), x: r.x, y: r.y, w: r.w, h: r.h, altText: title });
  pageNo(slide, n);
  if (notes) slide.addNotes(notes);
  return slide;
}

// 1 — title
{
  const slide = pres.addSlide();
  slide.background = { color: BG };
  slide.addText('CUDA · SFT 问题集 · 多 Agent', {
    x: 0.6, y: 1.05, w: 3.9, h: 0.42, fontFace: FONT, fontSize: 13, bold: true, color: ACCENT,
    fill: { color: 'E7F5FF' }, align: 'center', margin: 0, isTextBox: true,
  });
  slide.addText('cuda-sft-qgen', {
    x: 0.6, y: 1.7, w: 5.6, h: 0.9, fontFace: FONT, fontSize: 44, bold: true, color: INK,
    margin: 0, isTextBox: true,
  });
  slide.addText('用多 Agent 流水线，为 CUDA 知识域批量生成高质量训练问题', {
    x: 0.6, y: 2.65, w: 5.3, h: 0.95, fontFace: FONT, fontSize: 20, color: INK,
    margin: 0, valign: 'top', isTextBox: true,
  });
  slide.addText('只出题，不出答案 · 每题独立审查 · 不确定就 pending', {
    x: 0.6, y: 3.75, w: 5.3, h: 0.45, fontFace: FONT, fontSize: 14, color: MUTED,
    margin: 0, isTextBox: true,
  });
  const chips = [['31', 'L1 领域'], ['3311', 'L3 知识叶子'], ['3 + 1', '工作线 + 知识线程']];
  chips.forEach(([big, small], i) => {
    const x = 0.6 + i * 1.78;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 4.65, w: 1.62, h: 1.25, rectRadius: 0.1, fill: { color: 'FFFFFF' },
      line: { color: 'E3E7EC', width: 0.75 },
      shadow: { type: 'outer', blur: 8, offset: 2, angle: 90, color: '000000', opacity: 0.08 },
    });
    slide.addText(big, {
      x, y: 4.78, w: 1.62, h: 0.6, fontFace: FONT, fontSize: 26, bold: true, color: ACCENT,
      align: 'center', margin: 0, isTextBox: true,
    });
    slide.addText(small, {
      x, y: 5.38, w: 1.62, h: 0.4, fontFace: FONT, fontSize: 11.5, color: MUTED,
      align: 'center', margin: 0, isTextBox: true,
    });
  });
  const r = fit(path.join(ASSET, '00-system-overview.png'), { x: 6.25, y: 0.75, w: 6.6, h: 6.0 });
  paperCard(slide, r);
  slide.addImage({ path: path.join(ASSET, '00-system-overview.png'), x: r.x, y: r.y, w: r.w, h: r.h, altText: '系统总览' });
  pageNo(slide, 1);
  slide.addNotes('cuda-sft-qgen 是一个 LangGraph 多 Agent 流水线：由版本化知识图谱驱动出题，本阶段只产出问题。右图是系统总览：协调器调度生成、审题、修题三条工作线，后台 KnowledgeWorker 负责扩充知识树和官方文档审计。');
}

contentSlide(2, '一道题的旅程', '上排生成、下排审题：任何一步失败或不确定，都回到 pending 重审', 'slides/s02-journey.png',
  'Planner 从知识图谱抽题单；Writer/Evolver 用分层 prompt 出题，规则筛和自洽门先拦掉确定缺陷；题目写入 SQLite pending。审题线依次做三层查重、独立验证和 Chair 审查，Archivist 单事务归档。Chair 按存活题分布给 Planner 反馈 steering；知识被驳斥时，相关题目会被追溯失效。');
contentSlide(3, '一个协调器 · 三条工作线', '按队列余量派发：第一道题到期就开审，生成与审题互不阻塞', 'slides/s03-lanes.png',
  '协调器每 0.25 秒一轮：刷新知识与计数、回收完成的 future、检查熔断（连续 8 个审题错误或 20 个 exhausted），再按余量派发。生成来源优先级：未完成的 generation_tasks > 到期 retry_tasks > Planner 新题单。所有 LLM 调用共享 ApiKeyPool，按 Writer 0 > Salvage 1 > Chair 2 > Sieve 3 > Taxonomy 4 的优先级拿槽位。');
contentSlide(4, 'Planner · 抽题单', '四路信号加权，逐条抽样：选叶子 → 联合采样各轴 → 决定 Writer 或 Evolver', 'slides/s04-planner.png',
  '配额残差 = max(目标占比 − 当前占比, 0.02)；steering 把欠配 L1 权重乘 3、超配乘 0.05。叶子按已有题量加权，12 题封顶。各轴受 compat_matrix 约束，命中禁用组合先拒绝采样再精确枚举。30% 题单以已通过题为种子交给 Evolver，并附带 ≤6 个相关 claim 的知识上下文。');
contentSlide(5, 'Writer / Evolver · 出题', '字节稳定的分层 prompt，出题后先过解析、规则筛、自洽门三道关', 'slides/s05-writer.png',
  '静态前缀和半静态块可缓存；few-shot 选同叶子、同 kind、同答案格式的正例，并加入质量负例和反模板。坏 JSON 交给修题线只修外壳；规则筛拦空泛、离题和泄漏；自洽门只拦确定缺陷，并带缺陷说明重写一次。429 和推理预算耗尽进入 retry 队列指数退避。');
contentSlide(6, '查重与独立验证', 'Review Score 只打分不放行；Verifier 按题目合同独立验证', 'slides/s06-score-verify.png',
  '三层查重：L0 归一化文本 SHA256、L1 模板骨架 SimHash/Jaccard、L2 embedding 余弦，另有跨语言 claim 查重。Verifier 在 Chair 内调用：数值题 5 个盲解 solver 投票并与 Writer 答案比对；开放题 3 份作答按 rubric 取最低分；编译/运行题在 bubblewrap 沙箱中执行。超时、缺工具、架构不符都只会得到 pending。');
contentSlide(7, 'Review Chair · 独立审题', '预检查 → 独立模型审查 → 确定性规则 → 二审 → 整波精选', 'slides/s07-chair.png',
  'Chair 用 temperature 0 的独立模型逐题审查（unique 也审），检查正确性、可答性、claim 对齐、硬件、泄漏、元数据、重复，并给出价值分。最低价值分 < 3 拒绝；结构化题验证未 pass 保持 pending；高风险题型 uncertain 时再独立审一次。整波再做 MMR 多样性精选与批内去重，并生成 steering。');
contentSlide(8, '题目生命周期', '每个状态都落在 SQLite：崩溃可续跑，失败可追溯', 'slides/s08-lifecycle.png',
  'brief 先写入 generation_tasks；出题失败进入修题或 retry；pending 到期才审；不确定退避 30s·2ⁿ，审满 3 次标记 review_exhausted。可修的 reject 用原 brief 新 id 返修一次。已通过题依赖的 claim 被驳斥或改版，会自动退回 pending 重审。');
contentSlide(9, '知识图谱 · 可追溯的 claim', '新叶子先 provisional，只有官方原文逐字证据能把它变成 verified', 'slides/s09-knowledge.png',
  'TaxonomyManager 按 11 个 facet 轮转扩叶子并做语义去重；遗留叶子按 L2 分批送审。审计必须通过三道证据关：引文逐字出现在官方文档中、覆盖全部关键标识与数字、独立 LLM 仅凭引文复核。被驳斥的 claim 隔离并按原文改写新版本，impact 事件让相关题目失效。');
contentSlide(10, '四条设计底线', '宁可慢一点留在 pending，也不让一道未经证实的题进入训练集', 'slides/s10-principles.png',
  '每题独立审查；不确定就 pending；SQLite 是唯一事实源；知识可追溯。详细的逐 Agent 结构图见 README 和 asset/ 目录。');

pres.writeFile({ fileName: OUT }).then((f) => console.log('wrote', f));
