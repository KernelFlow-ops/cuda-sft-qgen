# 结构图源码

`asset/` 下的手绘风格结构图和演示文稿都由这里的脚本生成。每张图是一个数据文件（节点、分组、连线），代码改动后改对应文件再重新渲染即可。

| 路径 | 内容 |
|---|---|
| `diagrams/NN-*.js` | README 使用的 15 张结构图，输出到 `asset/NN-*.png` |
| `slides/sNN-*.js` | 演示文稿用的横版简化图，输出到 `asset/slides/` |
| `lib.js` | 渲染器：ELK 分层布局（或手工坐标）+ rough.js 手绘线条 + resvg 输出 PNG |
| `build-deck.js` | 用 pptxgenjs 生成 `asset/cuda-sft-qgen-agents.pptx` |
| `quantize.py` | 把 PNG 量化为 256 色调色板，体积约降到 1/3 |

## 重新生成

需要 Node.js 18+ 和带 Pillow 的 Python。

```bash
cd asset/diagram-src
npm install
sh fetch-fonts.sh        # 下载小赖手写体 Xiaolai / Xiaolai Mono（SIL OFL）
node build.js            # 全部结构图；只渲染某几张：node build.js 08 07
node build-slides.js     # 演示文稿配图
python quantize.py       # 压缩 PNG
node build-deck.js       # 生成 PPTX
```

## 数据格式

```js
module.exports = {
  id: '08-review-chair',           // 输出文件名
  title: '…', subtitle: '…',       // 图内标题（演示文稿配图不设）
  manual: true,                    // 可选：手工坐标，节点用 at: [cx, cy]
  scale: 1.3,                      // 可选：整体放大字号（演示文稿配图用）
  nodes: [{ id, type, label, sub, at, w, h }],
  groups: [{ id, label, color, children | box }],
  edges: [{ from, to, label, style, color, back, fromSide, toSide, via }],
};
```

节点 `type` 决定颜色和形状：`agent`、`llm`、`rule`、`decision`、`data`、`queue`、`store`、`tool`、`accept`、`reject`、`pending`、`start`、`note`。自动布局时，回边加 `back: true`，避免反馈边影响分层。
