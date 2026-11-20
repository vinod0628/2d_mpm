const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm, esc } = C;
const J = C.load('wofe_carlin');
const K = C.load('wofe_curnamona');

const nm = (s) => s.replace('@q8', ' (8 classes)');
const val = (v, d = 3) => (v === null || v === undefined ? '—' : f(v, d));

// weights table of one evidence layer (all classes / thresholds)
function weightsTable(L, caption) {
  if (L.type === 'categorical') {
    return ['table', {
      caption,
      head: ['Class', 'Label', 'Area (km²)', 'Deposits', 'W⁺', 's(W⁺)', 'Weight used', 's(weight)'],
      widths: [800, 2226, 1200, 900, 1000, 1000, 1000, 900],
      rows: L.table.map((r) => [String(r.cls), esc(r.label), f(r.area, 1), String(r.deposits), val(r['W+']),
        val(r['s(W+)']), f(r.weight, 3), f(r.s_weight, 3)]),
    }];
  }
  return ['table', {
    caption,
    head: ['Class', 'Range', 'Area (km²)', 'Dep.', 'Cum. dep.', 'W⁺', 's(W⁺)', 'W⁻', 's(W⁻)', 'C', 'Stud(C)'],
    widths: [600, 1800, 900, 500, 600, 800, 700, 800, 700, 800, 826],
    rows: L.table.map((r) => {
      const b = (s) => (r.selected ? `**${s}**` : s);
      return [b(String(r.cls)), b(esc(r.label)), f(r.area, 1), String(r.deposits), String(r.cum_deposits),
        val(r['W+']), val(r['s(W+)']), val(r['W-']), val(r['s(W-)']), b(val(r.C)), b(val(r['Stud(C)'], 2))];
    }),
  }];
}

function genTable(X, caption) {
  const rows = [];
  for (const L of X.reference.tables.layers) {
    for (const r of L.gen) rows.push([nm(L.name), esc(r.label), f(r.area, 1), String(r.deposits), f(r.weight, 3), f(r.s_weight, 3)]);
  }
  return ['table', {
    caption,
    head: ['Evidence layer', 'Generalised class', 'Area (km²)', 'Deposits', 'Weight', 's(weight)'],
    widths: [2000, 3026, 1100, 900, 1000, 1000],
    rows,
  }];
}

function selTable(X, caption) {
  return ['table', {
    caption,
    head: ['Evidence layer', 'Weight type', 'Selected threshold', 'C', 'Stud(C)', 'W⁺ / W⁻'],
    widths: [2100, 1500, 2526, 900, 900, 1100],
    rows: X.reference.tables.layers.map((L) => {
      const inP = L.gen.find((r) => r.gen_class === 2) || L.gen[L.gen.length - 1];
      const out = L.gen.find((r) => r.gen_class === 1);
      return [nm(L.name), L.type, L.type === 'categorical' ? 'one weight per class' : esc(inP.label.replace('pattern: ', '')),
        val(L.contrast), val(L.studentized, 2),
        L.type === 'categorical' ? '—' : `${f(inP.weight, 2)} / ${f(out.weight, 2)}`];
    }),
  }];
}

function ciTable(X, caption) {
  const ci = X.reference.tables.ci;
  return ['table', {
    caption,
    head: ['Statistic', 'Value'],
    widths: [5026, 4000],
    rows: Object.entries(ci).map(([k, v]) => [k, Number.isInteger(v) ? String(v) : g(v, 4)]),
  }];
}

function pairTable(X, caption) {
  return ['table', {
    caption,
    head: ['Layer 1', 'Layer 2', 'χ²', 'df', 'p', 'Min. expected', 'n', 'Note'],
    widths: [1700, 1700, 800, 500, 800, 1100, 500, 1926],
    rows: X.reference.tables.pairwise.map((r) => [nm(r.layer_1), nm(r.layer_2), val(r.chi2, 2), String(r.dof),
      val(r.p_value, 3), val(r.min_expected, 2), String(r.n), r.note || '']),
  }];
}

const layer = (X, name) => X.reference.tables.layers.find((L) => L.name === name);
const thr = (X, name) => {
  const L = layer(X, name);
  if (!L || !L.gen) return '—';
  const r = L.gen.find((x) => x.gen_class === 2);
  return r ? r.label.replace('pattern: ', '') : '—';
};
let tn = 3;
const T = () => ++tn;
const sj = J.selected, sk = K.selected;
const lodoMean = (X) => { const s = X.search.find((x) => x.label === X.selected.label); return s ? s.mean_test_auc : null; };

module.exports = {
  file: 'weights_of_evidence.docx',
  code: 'WofE',
  short: 'Weights of evidence',
  title: 'Weights of Evidence (WofE) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven Bayesian method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Weights of evidence is a Bayesian, data-driven method that combines binary or multi-class evidence maps into a map of the posterior probability that a unit cell contains a deposit (Bonham-Carter et al., 1989; Agterberg et al., 1990; Bonham-Carter, 1994). Each class of every evidence layer receives a weight — the log-ratio of its frequency at deposits and elsewhere — estimated from the known deposits; the weights are added to the prior log-odds under the assumption that the layers are conditionally independent given the deposits. The weights are easy to interpret geologically (“how much more likely is a deposit in this lithology?”), the variances of the weights and of the posterior are known analytically, and the method needs few deposits. MPM Studio implements the method of Bonham-Carter (1994) with cumulative generalisation of ordered evidence, missing-data variance and the conditional-independence tests (menu *Data-driven ▸ Statistical ▸ Weights of Evidence*).'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4200, 4826],
      rows: [
        ['`python/logic/data_driven/wofe/model.py`', 'Weights, generalisation, posterior, variances, missing-data variance, CI tests'],
        ['`python/logic/data_driven/wofe/plugin.py`', 'Registration; run in the worker thread; outputs and evaluation'],
        ['`python/logic/data_driven/wofe/dialogs.py`', 'Parameter and results dialogs'],
        ['`python/logic/common/training.py`', 'Train / test split, thinning, training / test points'],
        ['`python/tests/test_wofe.py`', 'Hand-calculated weights, Bonham-Carter (1994) examples, missing-data variance, CI tests'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Bayes’ rule in odds form'],
    ['p', 'The study area is divided into $N$ unit cells, $N_{D}$ of which contain a deposit ($D$). Before any evidence is considered, the probability of a deposit in a cell is the prior $P(D)=N_{D}/N$. A binary evidence pattern $B$ (e.g. “within 4 km of a fault”) changes this probability; Bayes’ rule in odds form, $O(D|B)=O(D)·P(B|D)/P(B|\\bar{D})$, shows that the ratio of the frequencies of the pattern at deposits and at non-deposit cells — the likelihood ratio — is all that is needed. Its logarithm is the weight. A positive weight means the pattern is more frequent at deposits than elsewhere; a contrast $C=W^{+}-W^{-}$ summarises the spatial association of the pattern with the deposits (Bonham-Carter et al., 1989).'],
    ['h2', '2.2 Combining evidence and conditional independence'],
    ['p', 'For several layers, the posterior log-odds are the prior log-odds plus the sum of the weights of the classes present in a cell. This is exact only if the layers are **conditionally independent** given the deposits — that is, if among deposit cells (and among non-deposit cells) the layers are statistically independent. Correlated evidence (two geochemical elements from the same alteration halo, a distance-to-fault map and a fault-controlled lithology) is counted twice, which inflates the posterior probabilities. The overall test compares the predicted number of deposits $T=\\sum{P_{post}}$ with the observed number $N_{D}$ (Bonham-Carter, 1994); Agterberg and Cheng (2002) give a one-tailed significance test. Weighted logistic regression (logistic_regression.docx) is the remedy when the dependence is strong (Agterberg et al., 1993).'],
    ['h2', '2.3 Generalisation of ordered evidence'],
    ['p', 'Ordered (interval) evidence — distance to faults, geochemical concentration, radiometric intensity — is classified, and the classes are accumulated from one end: **ascending** patterns ($x≤t$, e.g. proximity to faults) or **descending** patterns ($x≥t$, e.g. anomalous concentration). The contrast of every cumulative pattern is computed, and the threshold $t$ with the largest contrast among the statistically significant patterns becomes the binary pattern used in the model (Bonham-Carter, 1994; Carranza, 2009). This turns a continuous map into the binary “favourable / not favourable” decision that the deposits support best, and the threshold itself is geological information (the distance to which fault control extends, the anomaly threshold).'],
    ['h2', '2.4 Uncertainty'],
    ['p', 'Because the weights are estimated from a finite number of deposits, they have variances (Bishop et al., 1975; Bonham-Carter et al., 1989), and so does the posterior probability. A second source of uncertainty is **missing evidence**: where a layer has no data (an unmapped area), its weight is set to 0, and the uncertainty about the class the cell would have is added as the missing-data variance (Agterberg et al., 1990; Bonham-Carter, 1994). The ratio of posterior to its total standard deviation is a confidence map: cells with $P/s(P)>1.645$ have a posterior significantly above zero.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Weights of a binary pattern'],
    ['p', 'With areas in unit cells, $N\\{B∩D\\}$ the number of deposits in the pattern and $N\\{B∩\\bar{D}\\}$ the pattern area without deposits:'],
    ['eq', 'W^{+}=\\ln\\frac{N\\{B∩D\\}/N\\{D\\}}{N\\{B∩\\bar{D}\\}/N\\{\\bar{D}\\}},      W^{-}=\\ln\\frac{N\\{\\bar{B}∩D\\}/N\\{D\\}}{N\\{\\bar{B}∩\\bar{D}\\}/N\\{\\bar{D}\\}}', 'wofe-w'],
    ['eq', 's^{2}(W^{+})=\\frac{1}{N\\{B∩D\\}}+\\frac{1}{N\\{B∩\\bar{D}\\}},      s^{2}(W^{-})=\\frac{1}{N\\{\\bar{B}∩D\\}}+\\frac{1}{N\\{\\bar{B}∩\\bar{D}\\}}', 'wofe-var'],
    ['eq', 'C=W^{+}-W^{-},      s(C)=\\sqrt{s^{2}(W^{+})+s^{2}(W^{-})},      \\rm{Stud}(C)=C/s(C)', 'wofe-c'],
    ['p', 'Stud(C) is approximately a standard normal deviate; Stud(C) ≥ 1.96 means that the contrast differs from 0 at the 95 % level (two-sided). If a count is 0 the weight is undefined; MPM Studio either sets the weight to 0 (“no evidence”) or, on request, replaces the zero deposit count by 0.5 (Haldane, 1956), which gives a finite negative weight. Cumulative patterns always use the 0.5 correction so that the pattern that contains all deposits — frequently the best one — has a defined $W^{-}$; a cumulative pattern without any deposits is never selected.'],
    ['h2', '3.2 Categorical (multi-class) layers'],
    ['p', 'For a nominal layer (lithology), every class $j$ is a pattern $B_{j}$ against the rest of the layer; its weight is $W^{+}_{j}$ of [[wofe-w]] with variance $s^{2}(W^{+}_{j})$, and a cell receives the weight of its class (Bonham-Carter, 1994, ch. 9).'],
    ['h2', '3.3 Cumulative generalisation'],
    ['p', 'For classes $c_{1}<…<c_{m}$ of an ordered layer and threshold index $k$, the ascending pattern is $B_{k}=\\{x≤c_{k}\\}$ (descending: $\\{x≥c_{m-k+1}\\}$). With cumulative areas $A_{k}$ and cumulative deposit counts $n_{k}$, [[wofe-w]]–[[wofe-c]] are evaluated for $k=1…m-1$ and the threshold is'],
    ['eq', 'k^{*}=\\rm{arg max}_{k}\\{C_{k} : \\rm{Stud}(C_{k})≥s_{min}, n_{k}>0\\}', 'wofe-thr'],
    ['p', '(criterion *maximum contrast*, default $s_{min}=1.96$) or $k^{*}=\\rm{arg max}_{k}\\rm{Stud}(C_{k})$ (criterion *maximum studentized contrast*). If no threshold reaches $s_{min}$, the most significant one is used with a warning. Cells inside the pattern receive $W^{+}_{k^{*}}$, cells outside $W^{-}_{k^{*}}$.'],
    ['h2', '3.4 Posterior probability'],
    ['eq', '\\ln O(D|B_{1}…B_{n})=\\ln\\frac{N\\{D\\}}{N-N\\{D\\}}+\\sum_{i=1}^{n}{W_{i}(\\bf{x})},      P_{post}=\\frac{O}{1+O}', 'wofe-post'],
    ['p', 'where $W_{i}(\\bf{x})$ is the weight of the (generalised) class of layer $i$ at the cell, and 0 where the layer is missing.'],
    ['h2', '3.5 Variances of the posterior'],
    ['eq', 's^{2}(P_{post})=P_{post}^{2}[\\frac{1}{N\\{D\\}}+\\sum_{i=1}^{n}{s^{2}(W_{i}(\\bf{x}))}]', 'wofe-pvar'],
    ['p', 'For a cell where layer $i$ is missing, the missing-data variance sums, over the classes $j$ of that layer with area share $A_{j}/A$, the squared change of the posterior if the class were known (Bonham-Carter, 1994; ArcSDM *missingdatavar*):'],
    ['eq', '\\rm{Var}_{MD}=\\sum_{j}{(P_{j}-P_{post})^{2}\\frac{A_{j}}{A}},      O_{j}=O_{post}·e^{W_{j}}', 'wofe-md'],
    ['eq', 's_{tot}(P)=\\sqrt{s^{2}(P_{post})+\\sum_{i}{\\rm{Var}_{MD,i}}},      \\rm{confidence}=P_{post}/s_{tot}(P)', 'wofe-tot'],
    ['h2', '3.6 Conditional-independence tests'],
    ['p', 'Over the unique conditions $u$ (cells with the same combination of generalised classes, area $A_{u}$, posterior $P_{u}$ and standard deviation $s_{u}$):'],
    ['eq', 'T=\\sum_{u}{A_{u}P_{u}},      s(T)=\\sqrt{\\sum_{u}{(A_{u}s_{u})^{2}}},      \\rm{CI}=N\\{D\\}/T,      z=\\frac{T-N\\{D\\}}{s(T)}', 'wofe-ci'],
    ['p', 'CI < 0.85 suggests that conditional independence is violated (Bonham-Carter, 1994); the one-tailed p-value $1-Φ(z)$ tests $T>N\\{D\\}$ (Agterberg and Cheng, 2002). As in ArcSDM, the probability of conditional dependence $100Φ(z)$ % and the overall CI % are also reported. Pairwise $χ^{2}$ tests on the generalised classes of the deposit cells identify which pairs of layers are dependent (Bonham-Carter, 1994).'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Input check** — evidence must be integer rasters (classes); real-valued layers are reclassified by the user (*Data ▸ Reclassify raster*); grids are aligned to the first layer (nearest neighbour) if requested.',
      '**Study area** — the mask (or the cells with evidence); with *exclude*, cells missing any evidence are removed; the unit-cell factor $A_{c}/A_{u}$ converts cell counts to unit cells.',
      '**Training deposits** — the deposit points are split into training and test deposits (spatially stratified, default 20 % held out), optionally thinned; one deposit per cell.',
      '**Weights per layer** — counts per class by `bincount`; categorical: [[wofe-w]] per class; cumulative: cumulative sums in the chosen direction, [[wofe-w]]–[[wofe-c]] for every threshold, selection [[wofe-thr]], two generalised classes.',
      '**Posterior** — sum of the prior log-odds and the cell weights [[wofe-post]]; variance [[wofe-pvar]]; missing-data variance [[wofe-md]] per layer; total standard deviation and confidence [[wofe-tot]].',
      '**Tests** — unique conditions, [[wofe-ci]], Agterberg–Cheng test, pairwise χ² on deposit cells.',
      '**Outputs and evaluation** — `post.tif`, `post_std.tif` (total std), `conf.tif`, optional weight rasters, training / test point shapefiles; ROC, success-rate and P–A curves for training and test deposits.',
    ]],
    ['table', {
      caption: 'Table 4.1. Differences from ArcSDM and verification.',
      head: ['Item', 'MPM Studio'],
      widths: [3000, 6026],
      rows: [
        ['Formulas', 'Same as ArcSDM (Kemp et al., 2001; ArcSDM, 2025) and Bonham-Carter (1994); weights reproduce the hand-calculated examples in `test_wofe.py`'],
        ['Deposits per cell', 'One training deposit per cell (ArcSDM counts duplicate points in a cell twice)'],
        ['Zero-deposit classes', 'Weight 0 or the 0.5-deposit (Haldane) correction instead of ArcSDM’s 0.99 / 0.001 substitutions'],
        ['Cumulative threshold', 'Largest C with Stud(C) ≥ minimum; patterns without deposits excluded; fallback to the largest Stud(C) with a warning'],
        ['Missing data', 'Missing-data variance and total std as ArcSDM (adopted)'],
        ['CI tests', 'Overall ratio, Agterberg–Cheng with s(T) summed over unique conditions (as in the paper), ArcSDM’s probability of conditional dependence and overall CI %'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Statistical ▸ Weights of Evidence…*. Continuous evidence must first be classified (for example into 8–10 quantile classes or geologically meaningful intervals) with *Data ▸ Reclassify raster*; the dialog lists real-valued layers but disables them.'],
    ['table', {
      caption: 'Table 5.1. Training data.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2400, 1600, 5026],
      rows: [
        ['Deposits (points)', '—', 'Known deposits / occurrences of one deposit type (coherent deposit types, Carranza et al., 2008).'],
        ['Study area mask', 'cells with evidence', 'Area in which the prior, the weights and the posterior are defined.'],
        ['Hold out for testing', '20 %, spatially stratified', 'Deposits withheld from training to measure the prediction rate; *random* is also available; 0 % uses all deposits.'],
        ['Random seed', '0', 'Reproducible split.'],
        ['Thin deposits', '0 (off)', 'Minimum distance between training deposits; removes clusters that would dominate the weights.'],
        ['Absence points (ROC)', 'all background cells', 'Optional known barren points as ROC negatives.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2200, 6826],
      rows: [
        ['Use', 'Tick the evidence layers; choose layers that represent different components of the mineral system, so that they are as independent as possible.'],
        ['Classes', 'Number of classes inside the study area (“—” for real values).'],
        ['Weight type', '*Categorical* (one weight per class; nominal data such as lithology), *Cumulative ascending* (≤ threshold; proximity: distances) or *Cumulative descending* (≥ threshold; anomalies: concentrations, intensities).'],
        ['Generalisation', 'Cumulative types: *maximum contrast C* (among thresholds with Stud(C) ≥ minimum) or *maximum studentized contrast*.'],
        ['Min Stud(C)', '1.96 (95 %); lower values allow larger contrasts with fewer deposits, higher values require stronger significance.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.3. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2400, 2200, 4426],
      rows: [
        ['Unit cell area', '1 km²', 'Area of a unit cell (≈ deposit footprint); every class should have more unit cells than deposits.'],
        ['Missing evidence', 'weight 0 + missing-data variance', 'Keep cells without data (weight 0, uncertainty added) or exclude them from the study area.'],
        ['Classes without deposits', 'weight 0', 'Weight 0 (no evidence) or 0.5-deposit correction (negative weight; penalises classes without deposits).'],
        ['Resample evidence', 'on', 'Align grids to the first layer (nearest neighbour).'],
        ['Outputs', 'posterior + std + confidence', 'Optional generalised weight rasters; prefix `wofe`.'],
      ],
    }],
    ['p', 'The results dialog shows the training deposits, the prior, the CI ratio and the training / test AUC, with tabs *Summary*, one tab per evidence layer (full weights table, selected threshold and generalised weights) and *Conditional independence* (overall and Agterberg–Cheng tests, pairwise χ²). *AUC / ROC* opens the evaluation dialog and *Export tables (Excel)* writes all tables.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin WofE model.`),
    ['p', 'Distance to faults is used in its 10 published distance classes; log As and log Sb are reclassified into 8 equal-area (quantile) classes so that every class has enough area and the cumulative thresholds are evenly spaced in area. Host rock is a categorical layer (favourable silty-carbonate units vs other units; unmapped units are missing evidence).'],
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (spatial 70 / 30 splits, seeds 100–104).`),
    ['p', `The selected setting is **${sj.label}**. Over 10 independent splits (seeds 0–9) it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — WofE posterior probability per km² with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) contrast C of the cumulative thresholds.'],
    selTable(J, `Table 6.${T()}. Carlin — selected thresholds (reference run).`),
    ...J.reference.tables.layers.map((L, i) => weightsTable(L, `Table 6.${T()}. Carlin — weights of ${nm(L.name)} (${L.type}; selected threshold in bold).`)),
    genTable(J, `Table 6.${T()}. Carlin — generalised weights used in the posterior.`),
    ciTable(J, `Table 6.${T()}. Carlin — conditional-independence tests.`),
    pairTable(J, `Table 6.${T()}. Carlin — pairwise χ² tests on the training deposit cells.`),
    ['p', `The CI ratio of ${f(J.reference.tables.ci['CI ratio N(D)/T'], 2)} (predicted ${f(J.reference.tables.ci['Predicted deposits T'], 1)} vs observed ${J.reference.tables.ci['Observed deposits N(D)']} deposits) shows that the posterior probabilities are inflated: As and Sb record the same hydrothermal halo, and both are concentrated near faults. The ranking of the cells — and therefore the ROC and success-rate curves — is less affected than the absolute probabilities; for calibrated probabilities use logistic regression with the same evidence. The pairwise tests have too few deposits per cell of the 2 × 2 tables (expected counts < 5) to be conclusive.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona WofE model.`),
    ['p', 'All four layers are continuous rasters reclassified into 8 quantile classes; radiometric Th and U and log Nb are used as descending patterns (enrichment), the ASTER silica index as an ascending pattern (low quartz content of the cover).'],
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — WofE posterior probability per km² with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) contrast C of the cumulative thresholds.'],
    selTable(K, `Table 6.${T()}. Curnamona — selected thresholds (reference run).`),
    ...K.reference.tables.layers.map((L, i) => weightsTable(L, `Table 6.${T()}. Curnamona — weights of ${nm(L.name)} (${L.type}; selected threshold in bold).`)),
    genTable(K, `Table 6.${T()}. Curnamona — generalised weights used in the posterior.`),
    ciTable(K, `Table 6.${T()}. Curnamona — conditional-independence tests.`),
    ['p', `With only seven occurrences the weights have large standard deviations (s(W⁻) ≈ 1 where only one occurrence lies outside a pattern) and the CI ratio (${f(K.reference.tables.ci['CI ratio N(D)/T'], 2)}) reflects the strong correlation of the radiometric channels (Th and U both map the same granites). The Agterberg–Cheng test is not significant (p = ${f(K.reference.tables.ci['Agterberg-Cheng p (one-tailed, T > N(D))'], 2)}) because s(T) is large — with so few deposits, the test has little power.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal WofE parameters for the demo projects and their geological reasoning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1500, 1500, 4426],
      rows: [
        ['Evidence layers', 'host rock, distance to faults, As, Sb', 'Th, U, Nb, silica index', 'One layer per component of the mineral system — Carlin: fluid pathway (faults), chemical trap (reactive silty carbonate), fluid footprint (As–Sb halo); Curnamona: metal source (U–Th–REE granite), REE–HFSE enrichment (Nb), exposure (low-silica, non-quartzose cover). Layers of different components are the most nearly conditionally independent.'],
        ['Weight type: faults', 'cumulative ascending', '—', 'Prospectivity decreases with distance from the fluid conduits; the threshold answers “up to which distance do faults control mineralisation?”.'],
        ['Selected fault threshold', thr(J, 'dist_faults_class'), '—', 'Ore fluids rose along high-angle faults and spread laterally through permeable, reactive beds; a corridor of a few km around mapped faults captures the deposits that lie on unmapped splays and in the lateral outflow zones (Cline et al., 2005).'],
        ['Weight type: geochemistry / radiometrics', 'cumulative descending', 'descending (Th, U, Nb); ascending (silica)', 'Anomalies are “high value” patterns; the threshold is a data-driven anomaly threshold. For the silica index, low values (less quartz-rich cover, more exposed granite and alteration) are favourable.'],
        ['Selected anomaly thresholds', `As: ${thr(J, 'as_q8')}; Sb: ${thr(J, 'sb_q8')} (log10 ppm)`, `Th: ${thr(K, 'rad_th@q8')}; U: ${thr(K, 'rad_u@q8')}; Nb: ${thr(K, 'geochem_nb_log_idw@q8')}; silica: ${thr(K, 'aster_silica_index@q8')}`, 'Carlin: Sb has the narrower, stronger threshold (top quantiles) — Sb is deposited in the inner part of the Carlin halo, while As forms a broader halo and also occurs in unmineralised black shales, so its optimal pattern is wider and its contrast lower. Curnamona: U is the most selective (top class) because U is mobile and enriched in the hydrothermally overprinted granite and breccias hosting the REE minerals, while Th (immobile, in monazite / allanite) marks the whole granite body.'],
        ['Weight type: host rock', 'categorical', '—', 'Lithology is nominal. The silty-carbonate units of the lower plate are the chemical trap: carbonate dissolution created permeability and the iron-bearing host provided the sulfidation sites for gold-bearing pyrite (Hofstra and Cline, 2000).'],
        ['Generalisation', 'max C, Stud(C) ≥ ' + (sj.min_stud == null ? '—' : sj.min_stud), 'max C, Stud(C) ≥ ' + (sk.min_stud == null ? '—' : sk.min_stud), 'Maximum contrast among significant thresholds gives the most discriminating pattern that the deposits still support. With 24–27 Carlin deposits the 95 % level is attainable and gave the best held-out AUC (Table 6.5). ' + (sk.min_stud >= 1.96 ? 'Even with 7 Curnamona occurrences (s(C) ≥ 0.8) the radiometric patterns reach the 95 % level because their contrasts are large (C ≈ 3–4: the occurrences sit on the most radioactive granite); Nb and the silica index do not (Stud(C) < 1.96), so for them the most significant threshold is used (fallback with a warning). The default gave the best LODO AUC: it keeps the strong radiometric thresholds strict while the weaker layers still contribute.' : 'With 7 Curnamona occurrences s(C) ≥ 0.8, so a 95 % requirement would reject even strong associations — a lower level keeps the geologically expected patterns and the LODO test confirms they predict unseen occurrences.')],
        ['Classes without deposits', 'weight 0', 'weight 0', 'An empty class is absence of evidence, not evidence of absence: with few deposits, many favourable cells simply have not been explored. The 0.5-deposit correction would assign a strong negative weight from zero counts (Table 6.5 shows it is not better).'],
        ['Missing evidence', 'weight 0 + MD variance', 'weight 0 + MD variance', 'Unmapped host-rock units keep their area (no information, weight 0) and the missing-data variance shows the extra uncertainty there instead of silently dropping parts of the belt.'],
        ['Classes of continuous evidence', '8 quantiles (As, Sb); 10 distance classes', '8 quantiles', 'Quantile classes have equal area, so each cumulative step adds the same area and the thresholds are evenly resolved; 8 classes give ≈ 3 deposits per class at Carlin.'],
        ['Unit cell', '1 km²', '1 km²', 'Of the order of the footprint of a deposit and its inner alteration zone; every class is much larger than its number of deposits.'],
        ['Hold-out', '20 % (GUI default; 30 % in the tests)', 'LODO', 'Enough training deposits for stable weights and a spatially stratified test set; with 7 occurrences only leave-one-out is meaningful.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodoMean(K))}. WofE ranks slightly below logistic regression and random forest on Carlin (logistic_regression.docx, random_forest.docx), because the binary generalisation discards the gradation within the patterns and correlated layers are double-counted.`,
      '**Interpretability.** The weights tables and thresholds are direct, testable geological statements (fault corridor width, anomaly thresholds), which is the main strength of the method for exploration and for papers.',
      '**Conditional independence.** CI ratios below 0.85 in both projects call for caution with the absolute probabilities; options are to combine As and Sb into one pathfinder layer, to drop one of two correlated layers, or to use logistic regression.',
      '**Few deposits.** The method remains usable with seven occurrences, but the weights’ standard deviations and the confidence map must be reported with the posterior.',
    ]],

    ['h1', '7 References'],
    ['refs', R('agterberg1990', 'agterberg1992', 'agterberg1993', 'agtcheng2002', 'bishop1975', 'bonham1989', 'bonham1994',
      'carranza2008', 'carranza2009', 'chung2003', 'cline2005', 'delong1988', 'deepseam', 'elburg2012', 'fawcett2006',
      'haldane1956', 'hofstra2000', 'kemp2001', 'muntean2011', 'neumann2000', 'raines2001', 'roberts2017', 'stevens2004',
      'yousefi2015', 'arcsdm')],
  ],
};
