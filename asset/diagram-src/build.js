// Render every diagram spec in diagrams/ (or the ids given on the command line).
'use strict';
const fs = require('fs');
const path = require('path');
const { renderDiagram } = require('./lib');

const OUT = process.env.OUT_DIR || path.join(__dirname, '..');

(async () => {
  const only = process.argv.slice(2);
  const dir = path.join(__dirname, process.env.SPEC_DIR || 'diagrams');
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.js')).sort();
  for (const f of files) {
    delete require.cache[require.resolve(path.join(dir, f))];
    const mod = require(path.join(dir, f));
    const specs = Array.isArray(mod) ? mod : [mod];
    for (const spec of specs) {
      if (only.length && !only.some((o) => spec.id.includes(o))) continue;
      const t0 = Date.now();
      const r = await renderDiagram(spec, path.join(OUT, `${spec.id}.png`));
      console.log(`${spec.id}: ${r.width}x${r.height} ${(r.bytes / 1024).toFixed(0)}KB ${Date.now() - t0}ms`);
    }
  }
})().catch((e) => { console.error(e); process.exit(1); });
