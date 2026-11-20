// Build the development-record documents:  node build.js [m1 m2 ...]   (default: all)
// Needs the npm package "docx" (v9) on NODE_PATH or in a local node_modules.
const fs = require('fs');
const path = require('path');
const { build, Packer } = require('./lib');

const outDir = path.resolve(__dirname, '..');
const dir = path.join(__dirname, 'content');
const want = process.argv.slice(2);
const files = fs.readdirSync(dir).filter((f) => f.endsWith('.js')).sort()
  .filter((f) => !want.length || want.includes(path.basename(f, '.js')));

(async () => {
  for (const f of files) {
    const doc = require(path.join(dir, f));
    const buf = await Packer.toBuffer(build(doc));
    fs.writeFileSync(path.join(outDir, doc.file), buf);
    console.log('wrote', doc.file, (buf.length / 1024).toFixed(0) + ' KB');
  }
})().catch((e) => { console.error(e); process.exit(1); });
