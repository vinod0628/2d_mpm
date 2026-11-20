// Builder for the Python MPM Studio method documents (docs/python/*.docx); adapted from docs/matlab/_build/lib.js.
// Each content file (content/*.js) exports { file, code, short, title, subtitle, blocks }.
// Block types:
//   ['h1', text] ['h2', text] ['h3', text]   headings (number them in the text)
//   ['p', text]                             paragraph; inline markup below
//   ['eq', latexish, id?]                   numbered display equation; refer to it as [[id]]
//   ['bullets', [text, ...]]  ['steps', [text, ...]]   bulleted / numbered list
//   ['table', {head: [...], rows: [[...], ...], widths: [...], caption}]
//   ['img', file, widthPx?, caption]        PNG from _build/figures (scaled to ≤ 600 px = page width)
//   ['note', text]                          shaded remark box
//   ['refs', [text, ...]]                   reference list (hanging indent)
// Inline markup: $math$, **bold**, *italic*, `code`.
// Math markup (to native Word equations / OMML): x_{i}, x^{2}, x_{i}^{2}, \frac{a}{b}, \sqrt{a},
//   \sum_{i=1}^{N}{body}, \rm{exp} (upright), \bf{w} (bold), \txt{if} (normal text), {..} groups,
//   \\ escapes the next character.
const fs = require('fs');
const path = require('path');
const d = require('docx');
const dm = require('docx/math');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, TabStopType, Footer, Header, PageNumber,
} = d;

const FONT = 'Times New Roman';
const HFONT = 'Arial';
const MONO = 'Consolas';
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440;
const CONTENT_W = PAGE_W - 2 * MARGIN;            // 9026 DXA

// ------------------------------------------------------------------ math parser
const SYM = {
  alpha: 'α', beta: 'β', gamma: 'γ', Gamma: 'Γ', delta: 'δ', Delta: 'Δ', epsilon: 'ε', eta: 'η',
  theta: 'θ', kappa: 'κ', lambda: 'λ', Lambda: 'Λ', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', rho: 'ρ',
  sigma: 'σ', Sigma: 'Σ', tau: 'τ', phi: 'φ', Phi: 'Φ', psi: 'ψ', Psi: 'Ψ', omega: 'ω', Omega: 'Ω',
  infty: '∞', le: '≤', ge: '≥', in: '∈', cdot: '·', times: '×', approx: '≈', ne: '≠', to: '→',
  partial: '∂', nabla: '∇', odot: '⊙', ldots: '…', prod: '∏', sim: '∼', pm: '±', propto: '∝',
  R: 'ℝ', T: 'ᵀ', quad: ' ', ell: 'ℓ', Theta: 'Θ', zeta: 'ζ', chi: 'χ',
};
function parseMath(src) {
  const ch = Array.from(src);
  let i = 0;
  // letters are math italic; digits, operators and punctuation are upright (style plain);
  // an ASCII hyphen becomes a minus sign
  const runs = (t) => [].concat(run(t));
  const run = (t) => {
    t = t.replace(/-/g, '−');
    const parts = t.split(/([A-Za-zα-ωϕϵ̀-ͯ]+)/u).filter((s) => s !== '');
    if (parts.length === 1) {
      return /^[A-Za-zα-ωϕϵ̀-ͯ]/u.test(parts[0]) ? new d.MathRun(parts[0]) : new d.MathRun({ text: parts[0], style: 'plain' });
    }
    return parts.map((s) => (/^[A-Za-zα-ωϕϵ̀-ͯ]/u.test(s) ? new d.MathRun(s) : new d.MathRun({ text: s, style: 'plain' })));
  };
  function readName() {
    let n = '';
    while (i < ch.length && /[a-zA-Z]/.test(ch[i])) n += ch[i++];
    return n;
  }
  function argText() {
    if (ch[i] !== '{') return ch[i++];
    i++; let t = '', depth = 1;
    while (i < ch.length) {
      if (ch[i] === '{') depth++;
      if (ch[i] === '}' && --depth === 0) { i++; break; }
      t += ch[i++];
    }
    return t;
  }
  function arg() {
    if (ch[i] === '{') { i++; return seq(true).flat(); }
    if (ch[i] === '\\') { const items = []; atom(items); return items.flat(); }
    return runs(ch[i++]);
  }
  function scripts() {
    let sub = null, sup = null;
    for (let k = 0; k < 2; k++) {
      if (ch[i] === '_' && !sub) { i++; sub = arg(); }
      else if (ch[i] === '^' && !sup) { i++; sup = arg(); }
    }
    return { sub, sup };
  }
  function attach(base, s) {
    if (s.sub && s.sup) return [new d.MathSubSuperScript({ children: base, subScript: s.sub, superScript: s.sup })];
    if (s.sub) return [new d.MathSubScript({ children: base, subScript: s.sub })];
    if (s.sup) return [new d.MathSuperScript({ children: base, superScript: s.sup })];
    return base;
  }
  // parse one backslash command into items (array of component-arrays)
  function atom(items) {
    i++;                                          // skip backslash
    const name = readName();
    if (name === '') { items.push(runs(ch[i++])); return; }
    switch (name) {
      case 'frac': { const a = arg(), b = arg(); items.push([new d.MathFraction({ numerator: a, denominator: b })]); return; }
      case 'sqrt': { items.push([new d.MathRadical({ children: arg() })]); return; }
      case 'sum': {
        const s = scripts(); const body = arg();
        const o = { children: body }; if (s.sub) o.subScript = s.sub; if (s.sup) o.superScript = s.sup;
        items.push([new d.MathSum(o)]); return;
      }
      case 'hat': case 'bar': case 'tilde':           // native Word accents (m:acc)
        items.push([new dm.MathAccent({ accent: name, children: arg() })]); return;
      case 'cal': items.push([new d.MathRun({ text: argText(), script: 'script' })]); return;   // script letter (m:scr)
      case 'rm': items.push([new d.MathRun({ text: argText(), style: 'plain' })]); return;
      case 'bf': items.push([new d.MathRun({ text: argText(), style: 'bold' })]); return;
      case 'txt': items.push([new d.MathRun({ text: argText(), normalText: true })]); return;
      default:
        if (SYM[name]) { items.push(runs(SYM[name])); return; }
        if (['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'max', 'min', 'sign', 'diag', 'det', 'expit', 'logit'].includes(name)) {
          items.push([new d.MathRun({ text: name, style: 'plain' })]); return;          // upright function name
        }
        throw new Error('unknown math command \\' + name + ' in: ' + src);
    }
  }
  function seq(untilBrace) {
    const items = []; let buf = '';
    // a script binds to the last character only, so that character becomes its own item
    const flush = (splitLast) => {
      if (!buf) return;
      const a = Array.from(buf);
      const last = splitLast ? a.pop() : null;
      if (a.length) items.push(runs(a.join('')));
      if (last !== null) items.push(runs(last));
      buf = '';
    };
    while (i < ch.length) {
      const c = ch[i];
      if (c === '}' && untilBrace) { i++; break; }
      if (c === '{') { flush(); i++; items.push(seq(true).flat()); continue; }
      if (c === '\\') { flush(); atom(items); continue; }
      if (c === '_' || c === '^') {
        flush(true);
        const base = items.pop() || runs('');
        const s = scripts();
        items.push(attach(base, s));
        continue;
      }
      buf += c; i++;
    }
    flush();
    return items;
  }
  return seq(false).flat();
}

function mathObj(src) { return new d.Math({ children: parseMath(src) }); }

// ------------------------------------------------------------------ inline text
function inline(text, base = {}) {
  const out = [];
  const re = /(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t[0] === '$') out.push(mathObj(t.slice(1, -1)));
    else if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t[0] === '*') out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 19, ...base }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

// ------------------------------------------------------------------ blocks
const border = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function table(spec) {
  const widths = spec.widths;
  const sum = widths.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - CONTENT_W) > 2) throw new Error(`table widths sum ${sum} != ${CONTENT_W}: ${spec.head}`);
  const mkCell = (txt, w, head) => new TableCell({
    borders: cellBorders,
    width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: 'DCE6F0', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: String(txt).split('\n').map((line) => new Paragraph({
      spacing: { before: 0, after: 0, line: 240 },
      // a leading zero-width space keeps a math-only cell inline (left-aligned) instead of display
      children: [...(line.startsWith('$') ? [new TextRun({ text: '​', size: 19 })] : []),
        ...inline(line, { size: 19, bold: head ? true : undefined })],
    })),
  });
  const rows = [new TableRow({ tableHeader: true, children: spec.head.map((h, k) => mkCell(h, widths[k], true)) })];
  for (const r of spec.rows) rows.push(new TableRow({ children: r.map((c, k) => mkCell(c, widths[k], false)) }));
  const out = [];
  if (spec.caption) {
    out.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 },
      children: inline(spec.caption, { bold: true, size: 20 }) }));
  }
  out.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows }));
  out.push(new Paragraph({ spacing: { before: 0, after: 80 }, children: [] }));
  return out;
}

function build(doc) {
  // equation numbering pass
  let n = 0; const ids = {};
  for (const b of doc.blocks) if (b[0] === 'eq') { n++; if (b[2]) ids[b[2]] = n; b.num = n; }
  const sub = (s) => s.replace(/\[\[([\w-]+)\]\]/g, (_, id) => {
    if (!(id in ids)) throw new Error('unknown equation id ' + id + ' in ' + doc.file);
    return '(' + ids[id] + ')';
  });
  const rec = (x) => (typeof x === 'string' ? sub(x) : Array.isArray(x) ? x.map(rec)
    : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, rec(v)])) : x);

  const children = [];
  children.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun(doc.title)] }));
  children.push(new Paragraph({ spacing: { after: 60 }, children: inline(doc.subtitle, { italics: true, color: '444444' }) }));
  children.push(new Paragraph({
    spacing: { after: 240 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2E5A88', space: 4 } },
    children: inline(`MPM Studio (Python 3.10) · method record · ${doc.date || '2026-11-19'}`, { size: 18, color: '666666' }),
  }));
  let listInst = 0;
  for (const raw of doc.blocks) {
    const b = rec(raw); b.num = raw.num;
    switch (b[0]) {
      case 'h1': children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b[1])] })); break;
      case 'h2': children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b[1])] })); break;
      case 'h3': children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(b[1])] })); break;
      case 'p': children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: inline(b[1]) })); break;
      case 'eq':
        children.push(new Paragraph({
          spacing: { before: 100, after: 100 },
          tabStops: [{ type: TabStopType.CENTER, position: Math.round(CONTENT_W / 2) }, { type: TabStopType.RIGHT, position: CONTENT_W }],
          children: [new TextRun('\t'), mathObj(b[1]), new TextRun(`\t(${b.num})`)],
        }));
        break;
      case 'bullets':
        for (const t of b[1]) children.push(new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { after: 40 }, children: inline(t) }));
        break;
      case 'steps':
        listInst++;
        for (const t of b[1]) children.push(new Paragraph({ numbering: { reference: 'steps', level: 0, instance: listInst }, spacing: { after: 40 }, children: inline(t) }));
        break;
      case 'table': children.push(...table(b[1])); break;
      case 'img': {                                   // PNG figure, centred, with caption below
        const buf = fs.readFileSync(path.join(__dirname, 'figures', b[1]));
        const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);   // IHDR
        const width = Math.min(b[2] || 600, 600);
        children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
          children: [new d.ImageRun({ type: 'png', data: buf, transformation: { width, height: Math.round((h * width) / w) } })] }));
        if (b[3]) children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: inline(b[3], { size: 19 }) }));
        break;
      }
      case 'note':
        children.push(new Paragraph({
          shading: { fill: 'F3F3F3', type: ShadingType.CLEAR, color: 'auto' },
          border: { left: { style: BorderStyle.SINGLE, size: 18, color: '2E5A88', space: 6 } },
          spacing: { before: 120, after: 120 }, indent: { left: 120 },
          children: inline(b[1], { size: 20 }),
        }));
        break;
      case 'code':                                   // monospace lines in a shaded box
        b[1].forEach((line, i) => children.push(new Paragraph({
          shading: { fill: 'F5F5F5', type: ShadingType.CLEAR, color: 'auto' },
          indent: { left: 200 }, keepNext: i < b[1].length - 1,
          spacing: { before: i === 0 ? 80 : 0, after: i === b[1].length - 1 ? 120 : 0, line: 240 },
          children: [new TextRun({ text: line === '' ? ' ' : line, font: MONO, size: 17 })],
        })));
        break;
      case 'refs':
        for (const t of b[1]) children.push(new Paragraph({ indent: { left: 400, hanging: 400 }, spacing: { after: 60 }, children: inline(t, { size: 20 }) }));
        break;
      default: throw new Error('unknown block ' + b[0]);
    }
  }

  return new Document({
    creator: 'MPM Studio',
    title: doc.title,
    styles: {
      default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { after: 120, line: 276 } } } },
      paragraphStyles: [
        { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: HFONT, size: 36, bold: true, color: '1F3B5C' }, paragraph: { spacing: { before: 0, after: 80 }, keepNext: true } },
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: HFONT, size: 28, bold: true, color: '1F3B5C' }, paragraph: { spacing: { before: 320, after: 120 }, keepNext: true, outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: HFONT, size: 24, bold: true, color: '2E5A88' }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
        { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: HFONT, size: 22, bold: true, italics: true, color: '2E5A88' }, paragraph: { spacing: { before: 200, after: 80 }, keepNext: true, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
        { reference: 'steps', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: `${doc.code} · ${doc.short}`, size: 16, color: '888888', font: HFONT })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '888888' })] })] }) },
      children,
    }],
  });
}

module.exports = { build, Packer, CONTENT_W };
