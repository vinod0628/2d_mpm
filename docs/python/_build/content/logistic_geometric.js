const { R } = require('../refs');
const C = require('../common');

const { f, g } = C;
const J = C.load('logistic_geometric_carlin');
const K = C.load('logistic_geometric_curnamona');
const ev = (X) => X.reference.tables.evidence;
const auc = (X) => X.reference.curves.training.auc;
const pa = (X) => X.reference.curves.training;
const cand = (X, lab) => { const r = X.search.find((s) => s.label === lab); return r ? r.mean_test_auc : null; };
let tn = 3;
const T = () => ++tn;
const evRows = (X) => ev(X).map((r) => [r.layer, r.direction, r.function, f(r.mean, 3), f(r.weight, 3),
  r.normalised_density == null ? '—' : f(r.normalised_density, 2)]);

module.exports = {
  file: 'logistic_geometric.docx',
  code: 'LGFA',
  short: 'Logistic fuzzification + geometric average',
  title: 'Logistic-Function Fuzzification and Geometric Average for Mineral Prospectivity Mapping',
  subtitle: 'Knowledge-driven method of MPM Studio (Yousefi and Carranza, 2015) — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Yousefi and Carranza (2015a, b) proposed a knowledge-driven workflow that removes most subjective thresholds from fuzzy modelling: continuous evidence is fuzzified by a logistic function whose inflection and slope follow from the range of the data, and the fuzzified layers are combined by their geometric average — a multi-criteria decision-making operator. The expert only decides whether high or low values of each layer are favourable and scores the classes of categorical layers. The prediction–area (P–A) plot (Yousefi and Carranza, 2015c) evaluates the result and can weight the layers. MPM Studio: *Knowledge-driven ▸ Fuzzy ▸ Logistic Fuzzification + Geometric Average*.'],
    ['table', { caption: 'Table 1.1. Source files.', head: ['File', 'Role'], widths: [4300, 4726], rows: [
      ['`python/logic/knowledge_driven/logistic_geometric/model.py`', 'Logistic parameters, fuzzification, (weighted) geometric average, P–A weights'],
      ['`python/logic/knowledge_driven/logistic_geometric/plugin.py`, `dialogs.py`', 'Registration, outputs, dialogs'],
      ['`python/tests/test_logistic_geometric.py`', 'min → 0.01 / max → 0.99, geometric means, P–A weights, Carlin model'],
    ] }],
    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Data-range logistic fuzzification'],
    ['p', 'Linear or piecewise fuzzification requires the expert to set the values at which membership becomes 0 and 1. The logistic function is smooth, bounded and monotone; Yousefi and Carranza (2015a) set its inflection at the middle of the data range and its slope so that the minimum and maximum map to ≈ 0.01 and ≈ 0.99. Every layer then spans the full fuzzy range, its shape is the same for all layers, and no threshold has to be chosen — only the direction (positive or negative evidence).'],
    ['h2', '2.2 Geometric average'],
    ['p', 'The algebraic product shrinks as layers are added and the minimum ignores all but the weakest layer. The geometric average (Yousefi and Carranza, 2015b) keeps the AND-like character (one very low score pulls the result down) while every layer contributes and the result does not depend on the number of layers. Weights turn it into a weighted geometric mean.'],
    ['h2', '2.3 P–A weights'],
    ['p', 'In the P–A plot the prediction rate (share of deposits above a threshold) and the occupied area (share of cells above it) are plotted against the threshold; at their intersection a share Pr of the deposits occupies a share Oa = 1 − Pr of the area. The normalised density Nd = Pr/Oa and the weight W = ln Nd measure how well an evidence map concentrates the deposits (Yousefi and Carranza, 2015c). Using them as weights makes the model data-guided.'],
    ['h1', '3 Mathematical basis'],
    ['eq', 'F(x)=\\frac{1}{1+e^{-s(x-i)}},      i=\\frac{x_{min}+x_{max}}{2},      s=±\\frac{9.2}{x_{max}-x_{min}}', 'lg-f'],
    ['p', 'With these parameters $F(x_{min})=1/(1+e^{4.6})≈0.01$ and $F(x_{max})≈0.99$ (9.2 = 2 ln 99); the sign of $s$ is negative for negative evidence.'],
    ['eq', 'G(\\bf{x})=(\\prod_{k=1}^{n}{F_{k}^{w_{k}}})^{1/\\sum{w_{k}}},      w_{k}=1  or  w_{k}=\\ln\\frac{Pr_{k}}{Oa_{k}}', 'lg-g'],
    ['p', 'Layers with $Nd≤1$ (no better than random) receive weight 0.'],
    ['h1', '4 Algorithm'],
    ['steps', [
      '**Study area** — mask ∩ cells with all evidence.',
      '**Fuzzification** — continuous layers by [[lg-f]] with the entered range (default: data minimum and maximum; the 2–98 % range is more robust to outliers); categorical layers by class scores.',
      '**Weights** — equal, custom, or P–A weights from each fuzzified layer and the deposits.',
      '**Combination** — weighted geometric mean [[lg-g]] (computed as the exponential of the weighted mean logarithm).',
      '**Validation** — ROC, success rate and P–A plot of all deposits.',
    ]],
    ['h1', '5 How to use'],
    ['table', { caption: 'Table 5.1. Settings.', head: ['Setting', 'Default', 'Meaning and guidance'], widths: [2400, 2200, 4426], rows: [
      ['Evidence type', 'positive / score per class', 'Positive: high values favourable; negative: low values favourable; categorical: score per class (0 < score ≤ 1).'],
      ['Range (min, max) / class scores', 'data minimum and maximum', 'Range of the logistic function; narrow it (e.g. 2–98 %) for skewed layers.'],
      ['Weights', 'equal', 'Equal (geometric average), P–A (ln Nd, needs deposits) or custom.'],
      ['Outputs / prefix', 'geometric average / lgfa', 'Optional fuzzified maps.'],
    ] }],
    ['h1', '6 Demo data: deposit models, results and their geological meaning'],
    ...C.demoData(6),
    ...C.protocolKD(6),
    ['h2', '6.5 Carlin (Au)'],
    C.searchTable(J, `Table 6.${T()}. Carlin — configurations (all 32 deposit cells).`),
    ['table', { caption: `Table 6.${T()}. Carlin — fuzzification and weights of the selected model.`, head: ['Layer', 'Type', 'Logistic parameters / scores', 'Mean F', 'Weight', 'Nd'], widths: [1700, 1100, 2926, 1100, 1100, 1100], rows: evRows(J) }],
    C.curvesTable(J, `Table 6.${T()}. Carlin — validation with all deposits.`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — geometric average of the fuzzified evidence with the known deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success rate, (c) P–A plot, (d) weights of the evidence layers.'],
    ['p', `The threshold-free model reaches a ROC AUC of **${f(auc(J))}**; at the P–A intersection ${f(pa(J).pa_prediction, 0)} % of the deposits lie in ${f(pa(J).pa_area, 0)} % of the area. P–A weights lowered the AUC (${f(cand(J, 'all layers, data range, P–A weights'))} vs ${f(cand(J, 'all layers, data range, equal weights (default)'))}): the single-layer P–A densities favour layers that are rich but spatially narrow, which then dominate the product. The distance to faults is almost uninformative with the min–max rule (mean F close to 1): its range (0 to > 30 km) is so skewed that the inflection lies at ≈ 16 km, beyond the reach of fault control — a known weakness of the data-range rule for strongly skewed evidence.`],
    ['h2', '6.6 Curnamona (REE)'],
    C.searchTable(K, `Table 6.${T()}. Curnamona — configurations (all 7 occurrences).`),
    ['table', { caption: `Table 6.${T()}. Curnamona — fuzzification and weights of the selected model.`, head: ['Layer', 'Type', 'Logistic parameters / scores', 'Mean F', 'Weight', 'Nd'], widths: [1900, 1100, 2726, 1100, 1100, 1100], rows: evRows(K) }],
    C.curvesTable(K, `Table 6.${T()}. Curnamona — validation with all occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — geometric average of the fuzzified evidence with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) weights.'],
    ['p', `Selected: **${K.selected.label}**, ROC AUC **${f(auc(K))}**. The 2–98 % range is preferable here because the radiometric channels have a few extreme values over outcropping granite cores; with the full range those outliers would compress the useful contrast of the rest of the area into the lower half of the logistic curve.`],
    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal settings of the demo projects.`,
      head: ['Element', 'Carlin (Au)', 'Curnamona (REE)', 'Geological and ore-forming reasoning'],
      widths: [1500, 1800, 1800, 3926],
      rows: [
        ['Positive evidence', 'As, Sb', 'Th, U, Nb', 'Pathfinder enrichment and radiometric intensity increase towards the hydrothermal centre / the U–Th–REE granite.'],
        ['Negative evidence', 'distance to faults', 'silica index', 'Favourability decreases away from fluid conduits; quartz-rich cover hides the basement.'],
        ['Class scores', 'favourable host 0.9, other 0.1', '—', 'Reactive carbonate is the trap of Carlin-type gold; outside it gold is rarely deposited (a small non-zero score keeps the geometric average defined).'],
        ['Range', J.selected.robust ? '2–98 %' : 'data min–max', K.selected.robust ? '2–98 %' : 'data min–max', 'The logistic transition should span the geologically meaningful variation; outliers (granite cores, isolated high samples) and long distance tails compress it.'],
        ['Weights', J.selected.weighting === 'pa' ? 'P–A' : 'equal', K.selected.weighting === 'pa' ? 'P–A' : 'equal', 'Every component of the mineral system is necessary; equal weights treat them as such. P–A weights reward layers that are individually selective, which is not the same as being geologically necessary.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy without thresholds.** ROC AUC of all deposits: Carlin ${f(auc(J))}, Curnamona ${f(auc(K))}.`,
      '**Objectivity.** Only the direction of each layer and the class scores are expert choices; the shape of every membership follows from the data — the results are reproducible by other workers.',
      '**Skewed evidence.** The min–max rule should be combined with a sensible range (percentiles or a geological limit such as the reach of fault control).',
      '**Weights.** P–A weighting is a data-guided option; here equal weights were at least as good, consistent with the mineral-systems view that all components are required.',
    ]],
    ['h1', '7 References'],
    ['refs', R('bonham1994', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006', 'hofstra2000',
      'mccuaig2010', 'muntean2011', 'neumann2000', 'raines2001', 'yousefi2015', 'yousefi2015a', 'yousefi2015b', 'zimmermann1980')],
  ],
};
