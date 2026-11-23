const { R } = require('../refs');
const C = require('../common');

const { f, g } = C;
const J = C.load('index_overlay_carlin');
const K = C.load('index_overlay_curnamona');
const ev = (X) => X.reference.tables.evidence;
const auc = (X) => X.reference.curves.training.auc;
const pa = (X) => X.reference.curves.training;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
let tn = 3;
const T = () => ++tn;
const evRows = (X) => ev(X).map((r) => [r.layer, r.rule, r.scores, f(r.weight, 3), f(r.mean_score, 3),
  r.mean_score_at_deposits == null ? '—' : f(r.mean_score_at_deposits, 3)]);
const ahpRows = (X) => X.ahp.layers.map((n, i) => [n].concat(X.ahp.matrix[i].map((v) => (v >= 1 ? g(v, 3) : `1/${g(1 / v, 3)}`)))
  .concat([f(X.ahp.ahp[i], 3), `${f(X.ahp.fuzzy1_tfn[i][0], 3)} / ${f(X.ahp.fuzzy1_tfn[i][1], 3)} / ${f(X.ahp.fuzzy1_tfn[i][2], 3)}`, f(X.ahp.fuzzy1[i], 3)]));
const ruleTxt = (r) => ({ classes: 'score per class', positive: 'linear, high values favourable', negative: 'linear, low values favourable', ge: 'binary ≥', le: 'binary ≤' }[r] || r);

// knowledge-driven tools on the same demo data (all-deposit ROC AUC of their selected settings)
const KD = [['Index overlay (this document)', 'index_overlay'], ['TOPSIS / VIKOR', 'topsis'], ['Fuzzy logic overlay', 'fuzzy_logic'],
  ['Logistic fuzzification + geometric average', 'logistic_geometric'], ['Boolean logic (condition count)', 'boolean_logic']];
const kdRows = () => KD.filter(([, k]) => C.has(`${k}_carlin`) && C.has(`${k}_curnamona`)).map(([name, k]) => {
  const a = C.load(`${k}_carlin`), b = C.load(`${k}_curnamona`);
  return [name, f(a.reference.curves.training.auc), f(b.reference.curves.training.auc),
    `${f(a.reference.curves.training.pa_prediction, 0)} / ${f(b.reference.curves.training.pa_prediction, 0)}`];
});

const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Au)', 'Curnamona (REE)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Study area mask / deposits (validation) / absence points', 'study_mask / deposits / all background cells', 'study_mask / deposits / all background cells'],
    ['Evidence layers (ticked): scoring and scores / range', ev(J).map((r) => `${r.layer}: ${ruleTxt(r.rule)}, ${r.scores}`).join('; '),
      ev(K).map((r) => `${r.layer}: ${ruleTxt(r.rule)}, ${r.scores}`).join('; ')],
    ['Weights', J.selected.weighting || 'equal', K.selected.weighting || 'equal'],
    ['AHP judgements (if AHP / fuzzy AHP is chosen)', 'Table 6.5', 'Table 6.9'],
    ['Missing evidence', 'use the layers with data (weights renormalised)', 'use the layers with data (weights renormalised)'],
    ['Outputs / prefix', 'index / index', 'index / index'],
    ['Result (all deposits)', `ROC AUC ${f(auc(J))}; P–A ${f(pa(J).pa_prediction, 0)} % of the deposits in ${f(pa(J).pa_area, 0)} % of the area`,
      `ROC AUC ${f(auc(K))}; P–A ${f(pa(K).pa_prediction, 0)} % / ${f(pa(K).pa_area, 0)} %`],
  ] }];

module.exports = {
  file: 'index_overlay.docx',
  code: 'IO',
  short: 'Index overlay (AHP / fuzzy AHP)',
  title: 'Index Overlay with AHP and Fuzzy-AHP Weights for Mineral Prospectivity Mapping',
  subtitle: 'Knowledge-driven method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Index overlay (Bonham-Carter, 1994) is the most widely used knowledge-driven integration method: every evidence map is converted to expert scores and the maps are combined by their weighted mean. It is simple, fast and transparent, which explains its continued use in greenfields exploration and as the baseline of method comparisons. Its weights are often derived from pairwise comparisons by the analytic hierarchy process (AHP; Saaty, 1977, 1980) or, to express uncertain judgements, by fuzzy AHP (Buckley, 1985). MPM Studio implements binary and multi-class index overlay with equal, custom, AHP or fuzzy-AHP weights in one tool: *Knowledge-driven ▸ Overlay ▸ Index Overlay (AHP / Fuzzy AHP weights)*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/knowledge_driven/index_overlay/model.py`', 'Scoring rules, weighted mean, AHP (eigenvector, CI, CR), fuzzy AHP (Buckley)'],
      ['`python/logic/knowledge_driven/index_overlay/plugin.py`, `dialogs.py`', 'Registration, outputs, dialog with the pairwise-comparison matrix and live consistency ratio'],
      ['`python/tests/test_index_overlay.py`', 'Saaty example, consistent matrix, CR warning, fuzzy-AHP limits, scores, missing data, Carlin model'],
    ] }],
    ['h2', '1.1 Why one tool for index overlay, AHP and fuzzy AHP'],
    ['p', 'Published comparisons rank fuzzy AHP and fuzzy logic above plain index overlay and AHP: at Feyz-Abad fuzzy AHP 80 %, fuzzy logic 75 %, index overlay 70 % and AHP 60 % prediction rate; at Seridune (porphyry Cu) fuzzy AHP AUC 0.862 against 0.806 for fuzzy logic (Abedi et al., 2017); on the Southwest Indian Ridge (seafloor sulfides) 0.887 against 0.813 (Ma et al., 2020). AHP and fuzzy AHP are weighting schemes, not integration models — "AHP models" are index overlays with AHP weights — so MPM Studio offers them as weight options of the overlay; their effect on the demo maps is measured in Section 6. A Dempster–Shafer model with expert beliefs was dropped from the plan: it is rarely used as a purely knowledge-driven method and no comparative accuracy was found.'],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Index overlay'],
    ['p', 'In a **binary** index overlay each map is a 0/1 pattern (favourable or not) and the index counts the weighted favourable patterns. In a **multi-class** index overlay each class of each map has a score that expresses its favourability; continuous maps are rescaled linearly between two values in the favourable direction. Weighting the maps expresses their relative importance in the deposit model. The result is a relative favourability index, not a probability (Bonham-Carter, 1994).'],
    ['h2', '2.2 Analytic hierarchy process'],
    ['p', 'AHP replaces the direct assignment of n weights by n(n − 1)/2 pairwise judgements on Saaty\'s 1–9 scale (1 equal, 3 moderate, 5 strong, 7 very strong, 9 extreme importance; reciprocals for the inverse). For perfectly consistent judgements $a_{ij}=w_{i}/w_{j}$ the weights are the principal eigenvector of the matrix with eigenvalue n; real judgements give $λ_{max}>n$, and the consistency ratio CR compares the inconsistency with that of random matrices — CR < 0.1 is acceptable (Saaty, 1977, 1980).'],
    ['h2', '2.3 Fuzzy AHP'],
    ['p', 'Experts rarely mean "exactly 3". Fuzzy AHP expresses each judgement as a triangular fuzzy number (a − δ, a, a + δ) and propagates the vagueness to fuzzy weights; Buckley (1985) uses the fuzzy geometric mean of each row, which — unlike Chang\'s extent analysis — never gives zero weights. The crisp weight is the centroid of the fuzzy weight. With δ = 0 the method reduces to the geometric-mean approximation of AHP.'],
    ['h1', '3 Mathematical basis'],
    ['eq', 'S(\\bf{x})=\\frac{\\sum_{i}{W_{i}s_{i}(\\bf{x})}}{\\sum_{i}{W_{i}}}', 'io-s'],
    ['p', 'Scores: class score $s_{i}=σ_{ij}$ (expert, 0–1); linear $s_{i}=\\rm{clip}((x-x_{lo})/(x_{hi}-x_{lo}),0,1)$ (high values favourable) or $1-$ that (low values favourable); binary $s_{i}=1[x≥t]$ or $1[x≤t]$. Missing evidence: the cell is excluded, or the weights are renormalised over the layers with data. AHP:'],
    ['eq', '\\bf{A}\\bf{w}=λ_{max}\\bf{w},      CI=\\frac{λ_{max}-n}{n-1},      CR=\\frac{CI}{RI(n)}', 'io-ahp'],
    ['p', 'with Saaty\'s random index RI(3 … 10) = 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49. Fuzzy AHP (Buckley, 1985): judgement $a≥1$ → $\\tilde{a}=(\\max(1,a-δ),a,\\min(9,a+δ))$, $a<1$ → reciprocal of $\\tilde{(1/a)}$; row fuzzy geometric mean and fuzzy weight'],
    ['eq', '\\tilde{r}_{i}=(\\prod_{j}{\\tilde{a}_{ij}})^{1/n},      \\tilde{w}_{i}=(\\frac{l_{i}}{\\sum{u}},\\frac{m_{i}}{\\sum{m}},\\frac{u_{i}}{\\sum{l}}),      w_{i}∝\\frac{l+m+u}{3}', 'io-fahp'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Scores** — each ticked layer is converted to 0–1 scores by its rule (classes without a score count as missing).',
      '**Weights** — equal, custom, or AHP / fuzzy AHP from the pairwise matrix (the dialog shows the weights, λmax, CI and CR while the judgements are typed).',
      '**Overlay** — weighted mean [[io-s]] over the layers with data (default) or only where all layers have data.',
      '**Validation** — ROC, success rate and P–A plot of all deposits (nothing is trained).',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2300, 4326], rows: [
      ['Scoring', 'categorical: score per class; continuous: linear (dist… layers: low values favourable)', 'Score per class (0–1; classes without a score = no data), linear rescaling between two values, or binary condition ≥ / ≤ a threshold.'],
      ['Scores / range / threshold', 'class = 0.5; 2nd–98th percentile; upper / lower quartile', 'Edit the class scores; the linear range should span the geologically meaningful variation (e.g. the reach of fault control).'],
      ['Weights', 'equal', 'Equal, custom, AHP, fuzzy AHP; with AHP keep CR < 0.1.'],
      ['Pairwise matrix', 'all 1', 'Upper triangle: importance of the row layer over the column layer (3, 5, 1/3 …); the lower triangle is filled in.'],
      ['Fuzziness δ', '1', 'Spread of every fuzzy judgement (fuzzy AHP).'],
      ['Missing evidence', 'renormalise', 'Use the layers with data; "exclude" keeps only cells with every layer (e.g. only 37 % of Carlin, where the host map has gaps).'],
      ['Outputs / prefix', 'index / index', 'Optional score map of every layer.'],
    ] }],
    ['h1', '6 Demo data: deposit models, results and their geological meaning'],
    ...C.demoData(6, ['Validation protocol (knowledge-driven)', 'all 34 deposit cells (nothing is trained)', 'all 7 occurrences (nothing is trained)']),
    ...C.protocolKD(6, C.MISSING_RENORM),
    ['h2', '6.5 Carlin (Au)'],
    C.searchTable(J, `Table 6.${T()}. Carlin — configurations (all deposits).`),
    ['table', { caption: `Table 6.${T()}. Carlin — AHP judgements (row versus column), AHP and fuzzy-AHP weights (δ = 1; CR = ${f(J.ahp.cr, 3)}).`,
      head: ['Layer'].concat(J.ahp.layers).concat(['AHP w', 'Fuzzy w (l / m / u)', 'Fuzzy w']), widths: [1700, 900, 900, 900, 900, 900, 1926, 900],
      rows: ahpRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — scores and weights of the selected model.`, head: ['Layer', 'Rule', 'Scores / range', 'Weight', 'Mean score', 'Mean score at deposits'], widths: [1600, 1100, 2526, 1000, 1300, 1500], rows: evRows(J) }],
    C.curvesTable(J, `Table 6.${T()}. Carlin — validation with all deposits.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — index overlay with the known deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success rate, (c) P–A plot, (d) layer weights.'],
    ['p', `Selected: **${J.selected.label}**, ROC AUC **${f(auc(J))}**; at the P–A intersection ${f(pa(J).pa_prediction, 0)} % of the deposits lie in ${f(pa(J).pa_area, 0)} % of the area. The full min–max range (${f(cand(J, 'multi-class, equal weights, min'))}) beats the 2–98 % range (${f(cand(J, 'multi-class, equal weights, 2'))}) here, for two reasons visible in the data: (1) six of the 34 deposits lie above the 98th percentile of Sb — with the robust range they all get the score 1, tied with the richest 2 % of the area, so the strongest Sb anomalies are no longer distinguished, whereas the full range keeps ranking them; (2) the full range of the distance to faults (0–32 km, 98th percentile 13.5 km) gives almost every cell a fault score close to 1 (mean 0.89, at deposits 0.94), which quietly lowers the influence of this weak proxy — consistent with the fuzzy weights of evidence, where the faults add no information once host rock and Sb are known. AHP weights (${f(cand(J, 'multi-class, AHP'))}) and fuzzy AHP (δ = 1: ${f(cand(J, 'multi-class, fuzzy AHP weights (δ = 1'))}; δ = 2: ${f(cand(J, 'multi-class, fuzzy AHP weights (δ = 2'))}) improve on equal weights with the same range; their weights are almost identical (Table 6.5). Binary patterns lose the gradation inside and outside the favourable zones (${f(cand(J, 'binary (upper / lower quartile), equal'))}).`],
    ['h2', '6.6 Curnamona (REE)'],
    C.searchTable(K, `Table 6.${T()}. Curnamona — configurations (all occurrences).`),
    ['table', { caption: `Table 6.${T()}. Curnamona — AHP judgements, AHP and fuzzy-AHP weights (δ = 1; CR = ${f(K.ahp.cr, 3)}).`,
      head: ['Layer'].concat(K.ahp.layers).concat(['AHP w', 'Fuzzy w (l / m / u)', 'Fuzzy w']), widths: [1700, 900, 900, 900, 900, 900, 1926, 900],
      rows: ahpRows(K) }],
    ['table', { caption: `Table 6.${T()}. Curnamona — scores and weights of the selected model.`, head: ['Layer', 'Rule', 'Scores / range', 'Weight', 'Mean score', 'Mean score at deposits'], widths: [1800, 1100, 2326, 1000, 1300, 1500], rows: evRows(K) }],
    C.curvesTable(K, `Table 6.${T()}. Curnamona — validation with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — index overlay with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) layer weights.'],
    ['p', `Selected: **${K.selected.label}**, ROC AUC **${f(auc(K))}**. Here the robust range is better, because a few extreme radiometric values over granite cores would otherwise compress the useful contrast; the AHP weights (${f(cand(K, 'multi-class, AHP'))}, fuzzy ${f(cand(K, 'multi-class, fuzzy AHP weights (δ = 1'))}) give Th and U together 70 % of the weight and are slightly worse than equal weights: Nb and the silica index carry information that the judgements undervalue. Binary overlay: ${f(cand(K, 'binary (upper / lower quartile), equal'))}.`],
    ['h2', '6.7 Comparison with the other knowledge-driven tools'],
    ['table', { caption: `Table 6.${T()}. Knowledge-driven tools on the demo data (selected settings; ROC AUC and P–A deposits % with all deposits).`,
      head: ['Tool', 'Carlin AUC', 'Curnamona AUC', 'P–A deposits % (Carlin / Curnamona)'], widths: [3826, 1500, 1700, 2000], rows: kdRows() }],
    ['p', 'With continuous rescaling and a geologically chosen range, index overlay is as accurate as the fuzzy and logistic–geometric overlays on the demo projects — better on Carlin, about equal on Curnamona. Its advantage over the fuzzy γ operator on Carlin comes from the compensatory weighted mean: the Carlin deposits are not all strong in every proxy (a few lie outside the mapped host — mean host score 0.91 at the deposits — and many have only moderate As), so an operator that lets one weak proxy pull the result down misses them.'],
    ['h2', '6.8 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the index overlay dialog; Section 6.9 explains the reasoning.'],
    setupTable(`Table 6.${T()}. Index overlay dialog settings for the demo projects.`),
    ['h2', '6.9 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal settings of the demo projects.`,
      head: ['Element', 'Carlin (Au)', 'Curnamona (REE)', 'Geological and ore-forming reasoning'],
      widths: [1500, 1800, 1800, 3926],
      rows: [
        ['Scoring', 'multi-class, linear', 'multi-class, linear', 'Hydrothermal footprints and radiometric signatures are gradational; binary patterns discard that information and were 0.04–0.09 AUC worse.'],
        ['Host / categorical scores', 'favourable host 1, other 0', '—', 'Reactive silty carbonate is the chemical trap of Carlin-type gold.'],
        ['Range', J.selected.robust === false ? 'data min–max' : '2–98 %', K.selected.robust === false ? 'data min–max' : '2–98 %', 'The linear range must keep the anomalies that host the deposits distinguishable: in Carlin the richest Sb values (six deposits above the 98th percentile) must not saturate; in Curnamona the extreme radiometric values of a few granite cores must not compress the contrast of the rest of the granite terrain.'],
        ['Weights', J.selected.weighting || 'equal', K.selected.weighting || 'equal', `Equal weights treat every mineral-system component as necessary. AHP / fuzzy AHP are worthwhile when the expert can rank the processes with confidence (Carlin AHP with the robust range: ${f(cand(J, 'multi-class, AHP'))} vs equal ${f(cand(J, 'multi-class, equal weights, 2'))}); they did not help when the ranking undervalued independent proxies (Curnamona).`],
        ['Fuzziness δ', '1', '1', 'Fuzzy AHP gave almost the same weights as AHP for consistent judgements (CR 0.004): fuzziness matters when judgements are uncertain or inconsistent.'],
        ['Missing evidence', 'renormalise', 'renormalise', 'Gaps in one map (unmapped host units) should not erase the evidence of the other maps.'],
      ],
    }],
    ['h2', '6.10 Discussion'],
    ['bullets', [
      `**Accuracy.** ROC AUC of all deposits: Carlin ${f(auc(J))}, Curnamona ${f(auc(K))}.`,
      '**Simplicity.** Every number in the map can be traced to a score and a weight — the method of choice when the result must be explained to non-specialists.',
      '**AHP and fuzzy AHP.** Structured, documented weights; on the demo data they changed the AUC by less than ±0.02, i.e. the scoring of the evidence matters more than the weights.',
      '**Limits.** Compensatory: a high score in one layer can offset the absence of a necessary component; use the Boolean or fuzzy AND / γ operators, or a permissive-tract gate, where a component is strictly required.',
    ]],
    ['h1', '7 References'],
    ['refs', R('abedi2017', 'bonham1994', 'buckley1985', 'carranza2008', 'chung2003', 'cline2005', 'deepseam', 'delong1988',
      'elburg2012', 'fawcett2006', 'hofstra2000', 'ma2020', 'muntean2011', 'neumann2000', 'raines2001', 'saaty1977',
      'saaty1980', 'yousefi2015')],
  ],
};
