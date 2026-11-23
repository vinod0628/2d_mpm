const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm, esc } = C;
const J = C.load('ebf_carlin');
const K = C.load('ebf_curnamona');
const tab = (X) => X.reference.tables;
const st = (X) => tab(X).stats;
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, start) => { const r = X.search.find((s) => s.label.startsWith(start)); return r ? r.mean_test_auc : null; };
const sel = (X) => X.selected;
const zero = (X) => ((sel(X).zero_deposit || 'zero') === 'zero' ? 'Bel = 0 (original EBF)' : '0.5-deposit correction');
const ncls = (X) => sel(X).n_classes || 10;
const score = (X) => ({ bel: 'belief Bel', pls: 'plausibility Pls', mid: 'mid-point (Bel + Pls) / 2' })[sel(X).score || 'bel'];
const other = (name) => (C.has(name) ? C.load(name) : null);
const lay = (X, name) => tab(X).layers.find((r) => r.layer === name);
const cls = (X, name) => tab(X)[`classes_${name}`.slice(0, 31)] || [];
const DEF = '10 quantile classes, Bel = 0 without deposits (original EBF), score Bel (default)';
const HAL = '10 quantile classes, 0.5-deposit correction, score Bel';
let tn = 3;
const T = () => ++tn;
const no = {};                  // table numbers referred to in the text

const layerTable = (X, cap) => ['table', {
  caption: cap,
  head: ['Layer', 'Classes', 'Largest Bel', 'Most favourable class (area %)', 'Mean Unc', 'AUC of Bel alone', 'Conflict 1 − β'],
  widths: [1800, 800, 1000, 2226, 1000, 1100, 1100],
  rows: tab(X).layers.map((r) => [r.layer, `${r.classes} (${r.classing === 'values' ? 'values' : 'quantiles'})`, f(r.bel_max),
    `${esc(r.best_class)} (${f(r.best_class_area_pct, 1)})`, f(r.mean_unc), f(r.single_layer_auc),
    r.conflict_when_added == null ? '—' : f(r.conflict_when_added)]),
}];
const classTable = (X, name, cap) => ['table', {
  caption: cap,
  head: ['Class', 'Area %', 'Deposits', 'W_CD', 'W_CD̄', 'Bel', 'Dis', 'Unc', 'Pls'],
  widths: [2026, 900, 900, 900, 900, 850, 850, 850, 850],
  rows: cls(X, name).map((r) => [esc(r.label), f(r.area_pct, 1), String(r.deposits), f(r.W_CD, 2), f(r.W_CDbar, 2),
    f(r.Bel), f(r.Dis), f(r.Unc), f(r.Pls)].map((v) => (r.favourable ? `**${v}**` : v))),
}];
const pairTable = (X, cap) => ['table', {
  caption: cap,
  head: ['Layer 1', 'Layer 2', 'χ²', 'df', 'p', 'Min. expected', 'n', 'Note'],
  widths: [1700, 1700, 800, 500, 800, 1100, 500, 1926],
  rows: tab(X).pairwise.map((r) => [r.layer_1, r.layer_2, f(r.chi2, 2), String(r.dof), f(r.p_value, 3),
    f(r.min_expected, 2), String(r.n), r.note || '']),
}];
const occ = (X) => X.search.find((s) => s.label === X.selected.label).lodo;
const lodoOf = (key, i) => { const o = other(`${key}_curnamona`); if (!o) return null;
  const r = o.search.find((s) => s.label === o.selected.label).lodo.find((x) => x.deposit === i); return r ? r.lodo_auc : null; };
const auc = (key, X) => { const o = other(`${key}_${X}`); if (!o) return null;
  return o.robust_summary ? o.robust_summary.test_auc_mean : o.search.find((s) => s.label === o.selected.label).mean_test_auc; };
const best = (X) => tab(X).layers.slice().sort((a, b) => b.single_layer_auc - a.single_layer_auc)[0];

const setupTable = (cap) => ['table', {
  caption: cap, head: ['Dialog field', 'Carlin (Au)', 'Curnamona (REE)'], widths: [2626, 3200, 3200], rows: [
    ['Project (*Project ▸ Open demo project*)', 'Carlin-type Au, Nevada (USGS)', 'REE, Curnamona, South Australia'],
    ['Deposits (points) / study area mask', 'deposits / study_mask', 'deposits / study_mask'],
    ['Hold out for testing', '30 % spatially stratified (validation runs); 0 % for the final map', '0 % (leave-one-deposit-out in this document)'],
    ['Thin deposits / absence points', 'off / all background cells', 'off / all background cells'],
    ['Evidence layers (ticked) and Classes', J.evidence.map((e) => `${e.name}: ${e.kind === 'values' ? 'class values' : 'quantile classes'}`).join('; '),
      K.evidence.map((e) => `${e.name}: ${e.kind === 'values' ? 'class values' : 'quantile classes'}`).join('; ')],
    ['Quantiles', String(ncls(J)), String(ncls(K))],
    ['Missing evidence', 'vacuous belief (Unc = 1, keep area)', 'vacuous belief (Unc = 1, keep area)'],
    ['Classes without deposits', zero(J), zero(K)],
    ['Prospectivity score', score(J), score(K)],
    ['Outputs, prefix', 'Dis, Unc and Pls maps on; layer belief maps off; ebf', 'Dis, Unc and Pls maps on; layer belief maps off; ebf'],
    ['Result', `test AUC ${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)} (10 splits)`, `LODO AUC ${f(lodo(K))}`],
  ] }];

const compare = () => {
  const rows = [['**Evidential belief functions (this document)**', `**${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}**`, `**${f(lodo(K))}**`]];
  const add = (key, label) => {
    const a = other(`${key}_carlin`), b = other(`${key}_curnamona`);
    if (!a && !b) return;
    rows.push([label, a && a.robust_summary ? pm(a.robust_summary.test_auc_mean, a.robust_summary.test_auc_sd) : '—',
      b ? f(lodo(b)) : '—']);
  };
  add('wofe', 'Weights of evidence (binary generalisation)');
  add('logistic_regression', 'Logistic regression');
  add('random_forest', 'Random forest');
  add('fuzzy_wofe', 'Fuzzy weights of evidence (hybrid)');
  add('dc_fuzzy', 'Data-calibrated fuzzy memberships (hybrid)');
  add('neuro_fuzzy', 'Neuro-fuzzy (hybrid)');
  return ['table', { caption: `Table 6.${(no.compare = T())}. Comparison with the related statistical methods on the same demo data and protocol.`,
    head: ['Method', 'Carlin: test AUC (10 splits)', 'Curnamona: mean LODO AUC'], widths: [4626, 2200, 2200], rows }];
};

module.exports = {
  file: 'ebf.docx',
  code: 'EBF',
  short: 'Evidential belief functions',
  title: 'Evidential Belief Functions (EBF) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven Dempster–Shafer method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Evidential belief functions apply the Dempster–Shafer theory of evidence (Dempster, 1967; Shafer, 1976) to mineral prospectivity mapping (Moon, 1990; An et al., 1994). In the data-driven form of Carranza and Hale (2003) and Carranza et al. (2005), every class of every evidence layer receives four numbers estimated from the known deposits: the **belief** Bel (support for the proposition “this cell contains a deposit”), the **disbelief** Dis (support against it), the **uncertainty** Unc = 1 − Bel − Dis (ignorance: support for neither) and the **plausibility** Pls = Bel + Unc (support that is not refuted). The layers are combined with Dempster’s rule of combination. Unlike a single probability, the pair Bel ≤ Pls brackets the support for deposits, and the uncertainty map shows where the evidence is weak, contradictory or missing. MPM Studio implements the method in *Data-driven ▸ Statistical ▸ Evidential Belief Functions*.'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4200, 4826],
      rows: [
        ['`python/logic/data_driven/ebf/model.py`', 'Classes of every layer, class belief functions, Dempster’s rule, missing evidence, layer statistics, independence tests'],
        ['`python/logic/data_driven/ebf/plugin.py`', 'Registration; run in the worker thread; outputs, tables and evaluation'],
        ['`python/logic/data_driven/ebf/dialogs.py`', 'Parameter dialog and results dialog (Bel / Unc / Dis bar chart per class)'],
        ['`python/logic/common/training.py`', 'Train / test split, thinning, training / test points'],
        ['`python/tests/test_ebf.py`', 'Hand-calculated belief functions, link to WofE, binary layers, Dempster’s rule, missing evidence, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Belief functions'],
    ['p', 'For a cell, the frame of discernment has two propositions: $D$ (a deposit is present) and $\\bar{D}$ (no deposit). A basic probability assignment distributes a unit mass over the subsets $\\{D\\}$, $\\{\\bar{D}\\}$ and $\\{D,\\bar{D}\\}$; the mass on the whole frame is the uncommitted part — the evidence says nothing about the cell. Belief is the mass committed to $D$, disbelief the mass committed to $\\bar{D}$, uncertainty the uncommitted mass, and plausibility everything not committed against $D$. The unknown probability of a deposit lies in $[\\rm{Bel},\\rm{Pls}]$ (Shafer, 1976). This separation of “evidence against” from “no evidence” is the main difference from Bayesian methods, where the lack of evidence must be expressed as a probability too (weights of evidence uses a weight of 0 for missing data).'],
    ['h2', '2.2 Data-driven belief functions'],
    ['p', 'An et al. (1994) and Carranza and Hale (2003) estimate the belief functions of the classes of an evidence layer from the spatial association between the classes and the known deposits. For class $C_{j}$, $W_{C_{j}D}$ is the ratio of the proportion of deposits in the class to the proportion of non-deposit cells in the class, and $W_{C_{j}\\bar{D}}$ the same ratio for the deposits and non-deposit cells outside the class. These ratios are exactly the likelihood ratios of weights of evidence ($W_{C_{j}D}=e^{W^{+}}$, $W_{C_{j}\\bar{D}}=e^{W^{-}}$; Bonham-Carter, 1994). Normalised over the classes of the layer, they give Bel and Dis of each class: a class rich in deposits relative to its area gets a high belief and a low disbelief; a class with fewer deposits than expected gets the opposite; classes in between keep much uncommitted mass.'],
    ['h2', '2.3 Dempster’s rule and independence'],
    ['p', 'Dempster’s rule combines two belief functions by multiplying their masses and re-distributing the products: products that agree ($D$ with $D$, $D$ with the frame, …) form the new masses; products that contradict ($D$ in one layer with $\\bar{D}$ in the other) are the conflict $1-β$ and are removed by renormalisation (Dempster, 1967; Shafer, 1976). The rule is commutative and associative, so the order of the layers does not matter, and the mass on the frame shrinks with every informative layer — uncertainty is reduced by agreeing evidence. Like weights of evidence, the rule assumes that the layers are independent sources of evidence; the pairwise χ² tests on the deposit cells report dependent pairs.'],
    ['h2', '2.4 Missing evidence, binary layers and empty classes'],
    ['p', 'A cell where a layer has no data receives the **vacuous** belief function (Bel = 0, Dis = 0, Unc = 1) for that layer. It is the identity element of Dempster’s rule, so the cell is decided by the other layers and its uncertainty stays high — the natural Dempster–Shafer representation of an unmapped area (An et al., 1994). Two properties of the data-driven belief functions matter in practice. (i) A **binary** layer has no uncertainty: with two classes, Dis of one class equals Bel of the other and Unc = 0 (proved in Section 3.2). (ii) A class **without deposits** has $W_{C_{j}D}=0$ and Bel = 0; if all but one class of a layer are empty, the remaining class gets Bel = 1 and Dis = 0 — certainty — and in Dempster’s rule any cell with Bel = 1 keeps Bel = 1 whatever the other layers say. MPM Studio warns when this happens and offers the 0.5-deposit (Haldane, 1956) correction, which keeps every belief below 1 so that all layers can still rank the cells.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Class belief functions'],
    ['p', 'With cells counted on known evidence — $N(T)$ cells, $N(D)$ deposit cells, $N(C_{j})$ cells and $N(C_{j}∩D)$ deposit cells of class $C_{j}$ — (Carranza and Hale, 2003):'],
    ['eq', 'W_{C_{j}D}=\\frac{N(C_{j}∩D)/N(D)}{[N(C_{j})-N(C_{j}∩D)]/[N(T)-N(D)]},      W_{C_{j}\\bar{D}}=\\frac{[N(D)-N(C_{j}∩D)]/N(D)}{[N(T)-N(D)-N(C_{j})+N(C_{j}∩D)]/[N(T)-N(D)]}', 'ebf-w'],
    ['eq', '\\rm{Bel}_{j}=\\frac{W_{C_{j}D}}{\\sum_{k}{W_{C_{k}D}}},      \\rm{Dis}_{j}=\\frac{W_{C_{j}\\bar{D}}}{\\sum_{k}{W_{C_{k}\\bar{D}}}},      \\rm{Unc}_{j}=1-\\rm{Bel}_{j}-\\rm{Dis}_{j},      \\rm{Pls}_{j}=\\rm{Bel}_{j}+\\rm{Unc}_{j}', 'ebf-bel'],
    ['p', 'A class is called favourable when $W_{C_{j}D}>1$ (more deposits than expected from its area). With the 0.5-deposit correction, $N(C_{j}∩D)=0$ and $N(D)-N(C_{j}∩D)=0$ are replaced by 0.5 in the numerators of [[ebf-w]]; an empty non-deposit count in a denominator (a class made only of deposit cells) is always set to 0.5. If, for a very small class, $\\rm{Bel}_{j}+\\rm{Dis}_{j}>1$, both are divided by their sum (Unc = 0) and a message is shown.'],
    ['h2', '3.2 Binary layers'],
    ['p', 'For two classes $A$ and $B=\\bar{A}$, the non-deposit cells outside $A$ are those of $B$, so $W_{A\\bar{D}}=W_{BD}$ and $W_{B\\bar{D}}=W_{AD}$, and [[ebf-bel]] gives'],
    ['eq', '\\rm{Dis}_{A}=\\frac{W_{BD}}{W_{AD}+W_{BD}}=\\rm{Bel}_{B},      \\rm{Unc}_{A}=1-\\rm{Bel}_{A}-\\rm{Bel}_{B}=0', 'ebf-binary'],
    ['p', 'The same holds with the 0.5 correction, which is applied symmetrically. Binary evidence is therefore all-or-nothing in EBF; multi-class evidence keeps uncertainty in the classes whose deposit density is close to the average.'],
    ['h2', '3.3 Dempster’s rule of combination'],
    ['p', 'For two layers with belief functions $(\\rm{Bel}_{1},\\rm{Dis}_{1},\\rm{Unc}_{1})$ and $(\\rm{Bel}_{2},\\rm{Dis}_{2},\\rm{Unc}_{2})$ at a cell (An et al., 1994; Carranza and Hale, 2003):'],
    ['eq', 'β=1-\\rm{Bel}_{1}\\rm{Dis}_{2}-\\rm{Dis}_{1}\\rm{Bel}_{2}', 'ebf-beta'],
    ['eq', '\\rm{Bel}_{12}=\\frac{\\rm{Bel}_{1}\\rm{Bel}_{2}+\\rm{Bel}_{1}\\rm{Unc}_{2}+\\rm{Bel}_{2}\\rm{Unc}_{1}}{β},      \\rm{Dis}_{12}=\\frac{\\rm{Dis}_{1}\\rm{Dis}_{2}+\\rm{Dis}_{1}\\rm{Unc}_{2}+\\rm{Dis}_{2}\\rm{Unc}_{1}}{β},      \\rm{Unc}_{12}=\\frac{\\rm{Unc}_{1}\\rm{Unc}_{2}}{β}', 'ebf-comb'],
    ['p', 'The nine products of the masses sum to 1; the two contradicting ones are the conflict $1-β$, so $\\rm{Bel}_{12}+\\rm{Dis}_{12}+\\rm{Unc}_{12}=1$. The vacuous function (0, 0, 1) leaves the other function unchanged, and $\\rm{Bel}_{1}=1$ gives $\\rm{Bel}_{12}=1$ for any second layer ([[ebf-comb]] with $\\rm{Dis}_{1}=\\rm{Unc}_{1}=0$) — the saturation discussed in Section 2.4. More than two layers are combined one after the other; the result does not depend on the order. The prospectivity score is Bel (default), Pls or $(\\rm{Bel}+\\rm{Pls})/2$.'],
    ['h2', '3.4 Layer diagnostics'],
    ['p', 'For every layer the results report the largest class belief, the range of the class beliefs, the most favourable class, the mean uncertainty, the ROC AUC of the layer’s belief alone (training deposits against all other cells) and the mean conflict $1-β$ when the layer is combined with the layers before it. Pairwise independence is tested with χ² on the deposit cells, using favourable / unfavourable classes ($W_{C_{j}D}>1$) as a 2 × 2 table (Bonham-Carter, 1994).'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Input check** — class rasters use their values as classes (at most 1000); continuous rasters are divided into equal-area classes inside the study area (quantiles of the known values; tied values collapse classes, empty classes are dropped). Grids are aligned to the first layer (nearest neighbour for class rasters, bilinear for continuous ones) if requested.',
      '**Study area** — the mask, or the cells where any layer has data; with *exclude*, cells missing any evidence are removed.',
      '**Training deposits** — split into training and test deposits (spatially stratified, default 20 % held out), optionally thinned; one deposit per cell.',
      '**Class belief functions** — counts per class by `bincount` on the known cells of each layer; [[ebf-w]], [[ebf-bel]] with the chosen rule for classes without deposits.',
      '**Combination** — start from the vacuous function in every cell; combine the layers one by one with [[ebf-beta]]–[[ebf-comb]]; cells without evidence of a layer use the vacuous function for it.',
      '**Diagnostics** — layer table, conflict, pairwise χ² on deposit cells; messages for empty classes, Bel = 1 certainty and dependent pairs.',
      '**Outputs and evaluation** — score raster (`bel`, `pls` or `mid`), optional `dis`, `unc`, `pls` / `bel` and per-layer belief maps, tables, training / test point shapefiles; ROC, success-rate and P–A curves of the score.',
    ]],
    ['table', {
      caption: 'Table 4.1. Implementation choices and verification.',
      head: ['Item', 'MPM Studio'],
      widths: [3000, 6026],
      rows: [
        ['Formulas', 'Carranza and Hale (2003); An et al. (1994) for Dempster’s rule. `test_ebf.py` checks the class functions by hand, $\\ln W_{CD}=W^{+}$ and $\\ln W_{C\\bar{D}}=W^{-}$ against the WofE code, the binary-layer identity, the rule against a hand calculation, its commutativity, associativity and vacuous identity.'],
        ['Counts', 'Raster cells (one training deposit per cell); the ratios in [[ebf-w]] do not depend on a unit-cell area as long as it is the same for all classes.'],
        ['Missing evidence', 'Vacuous belief function (Unc = 1) or exclusion of the cells; the vacuous choice is the Dempster–Shafer representation of “no information”.'],
        ['Classes without deposits', 'Bel = 0 (original) or a symmetric 0.5-deposit correction; a warning when a class reaches Bel = 1.'],
        ['Bel + Dis > 1', 'Possible only for tiny classes with the 0.5 correction in a denominator; rescaled to Unc = 0 and reported.'],
        ['Continuous evidence', 'Quantile classes in the dialog (no separate reclassification needed); class rasters keep their classes.'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Statistical ▸ Evidential Belief Functions…*. Tick the evidence layers; class rasters (lithology, distance classes) use *class values*, continuous rasters *quantile classes* (proposed automatically).'],
    ['table', {
      caption: 'Table 5.1. Training data.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2400, 1600, 5026],
      rows: [
        ['Deposits (points)', '—', 'Known deposits / occurrences of one deposit type.'],
        ['Study area mask', 'layer named *mask*, else cells with evidence', 'Area in which the belief functions are estimated and combined.'],
        ['Hold out for testing', '20 %, spatially stratified', 'Deposits withheld to measure the prediction rate; 0 % uses all deposits.'],
        ['Random seed', '42', 'Reproducible split.'],
        ['Thin deposits', 'off', 'Minimum distance between training deposits.'],
        ['Absence points (ROC)', 'all background cells', 'Optional known barren points as ROC negatives.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2200, 6826],
      rows: [
        ['Use', 'Tick one layer per component of the mineral system (as independent as possible).'],
        ['Values', 'Number of distinct values in a sample (“real” for floating-point rasters).'],
        ['Classes', '*class values* (one belief function per value; class rasters, at most 1000 values) or *quantile classes* (equal-area classes of the values in the study area; any numeric raster).'],
        ['Quantiles', 'Number of quantile classes (2–50, default 10). Fewer classes with few deposits: every class should hold several deposits, otherwise its belief rests on chance.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.3. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2400, 2200, 4426],
      rows: [
        ['Missing evidence', 'vacuous belief (Unc = 1, keep area)', 'Cells without data of a layer are decided by the other layers and keep a high uncertainty; or exclude them from the study area.'],
        ['Classes without deposits', 'Bel = 0 (original EBF)', 'Or the 0.5-deposit correction: small positive Bel instead of 0; recommended when a layer has a class holding all deposits (Bel = 1 warning).'],
        ['Prospectivity score', 'belief Bel', 'Bel is the usual EBF map; Pls = 1 − Dis also counts uncommitted support; the mid-point lies between them.'],
        ['Outputs', 'score + Dis, Unc, Pls maps', 'Optional belief map of every layer; prefix `ebf`.'],
        ['Resample evidence', 'on', 'Align grids to the first layer.'],
      ],
    }],
    ['table', { caption: 'Table 5.4. Outputs.', head: ['Output', 'Content'], widths: [2300, 6726], rows: [
      ['`bel` / `pls` / `mid`', 'Prospectivity score (the first layer added to the map).'],
      ['`dis`, `unc`, `pls` (or `bel`)', 'The other integrated belief functions: support against deposits, ignorance, upper bound.'],
      ['`bel_<layer>` (optional)', 'Class belief of every layer.'],
      ['Tables', '`summary`, `layers` (classes, largest Bel, belief contrast, most favourable class, mean Unc, AUC of Bel alone, conflict), `classes_<layer>` (cells, deposits, area %, W_CD, W_CD̄, W⁺, Bel, Dis, Unc, Pls, favourable), `pairwise` (χ² tests).'],
    ] }],
    ['p', 'The results dialog shows the training deposits, the number of layers, the mean uncertainty, the mean belief at the training deposits and the training / test AUC, with tabs *Summary*, *Layers* (AUC of every layer’s belief as a bar chart), one tab per layer (class table and a stacked Bel / Unc / Dis bar per class) and *Independence*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),
    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin EBF model.`),
    ['p', 'Host rock is a binary class raster (favourable silty-carbonate units / other units; unmapped units are missing evidence), distance to faults is used in its 10 published distance classes, and log As and log Sb are divided into quantile classes in the dialog.'],
    C.searchTable(J, `Table 6.${(no.searchJ = T())}. Carlin — candidate settings (5 spatial 70 / 30 splits).`),
    C.robustTable(J, `Table 6.${T()}. Carlin — selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    layerTable(J, `Table 6.${(no.layJ = T())}. Carlin — layers (reference run).`),
    classTable(J, 'geology_host', `Table 6.${(no.hostJ = T())}. Carlin — belief functions of geology_host (favourable classes in bold).`),
    classTable(J, 'sb_log_idw', `Table 6.${(no.sbJ = T())}. Carlin — belief functions of sb_log_idw.`),
    ['img', J.figures.map, 520, `Figure 6.1. Carlin — EBF ${score(J)} with training (green) and test (blue) deposits.`],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) belief (solid) and plausibility (dashed) of the classes of every layer.'],
    ['img', 'ebf_carlin_unc.png', 600, 'Figure 6.3. Carlin — (a) uncertainty Unc and (b) plausibility Pls of the reference run with all deposits (triangles).'],
    ['p', `The selected setting is **${sel(J).label}**; over 10 splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (training ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}). With the original rule (Bel = 0 for classes without deposits) the mean test AUC is only ${f(cand(J, DEF))} (Table 6.${no.searchJ}). The reason is the binary host layer: every training deposit on mapped geology lies in the favourable host class, so the original rule gives that class Bel = 1 and Dis = 0, and by Dempster’s rule every host cell keeps Bel = 1 — distance to faults, As and Sb can no longer rank the ${f(cls(J, 'geology_host')[1].area_pct, 0)} % of the mapped area that is host rock. With the 0.5-deposit correction the host class has Bel = ${f(cls(J, 'geology_host')[1].Bel)} (Table 6.${no.hostJ}), and the other layers rank the host cells (${f(cand(J, HAL))}). Sb is the strongest single layer (AUC of its belief ${f(lay(J, 'sb_log_idw').single_layer_auc)}): its belief rises through the upper three deciles to ${f(lay(J, 'sb_log_idw').bel_max)} in the top decile, which holds ${cls(J, 'sb_log_idw').slice(-1)[0].deposits} of the ${st(J).deposits} training deposits (Table 6.${no.sbJ}). Distance to faults has flat beliefs (largest ${f(lay(J, 'dist_faults_class').bel_max)}, within 1 km) and the highest uncertainty, because deposits occur up to 8 km from the mapped faults. The conflict when a layer is added stays small (≤ ${f(Math.max(...tab(J).layers.filter((r) => r.conflict_when_added != null).map((r) => r.conflict_when_added)))}): the layers rarely contradict each other. Figure 6.3 shows the specific product of EBF: the uncertainty is zero on the mapped geology (a binary layer has Unc = 0) and high beneath the post-mineral cover, where the host layer is missing (${int(lay(J, 'geology_host').missing_cells)} cells, ${f(100 * lay(J, 'geology_host').missing_cells / st(J).cells, 0)} % of the study area). There the faults and the geochemical halo alone decide, and Pls stays high: covered ground is plausible but not proven — the covered-extension targets of the Carlin trends. The mean belief at the training deposits is ${f(st(J).deposit_mean_bel)}, with a mean uncertainty of ${f(st(J).deposit_mean_unc)}.`],
    pairTable(J, `Table 6.${T()}. Carlin — pairwise χ² tests on the training deposit cells.`),
    ['p', 'Host rock cannot be tested (all training deposits on mapped geology are in the favourable class). Distance to faults and Sb are dependent among the deposits: the deposits in the anomalous Sb classes lie in the favourable fault corridors, as expected where the Sb halo formed along the fluid conduits. Dempster’s rule counts this shared information twice, which sharpens the belief map more than the ranking (the pair adds little conflict). Combining the two into one layer, or dropping one, is the remedy if calibrated beliefs are needed.'],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona EBF model.`),
    C.searchTable(K, `Table 6.${(no.searchK = T())}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    layerTable(K, `Table 6.${(no.layK = T())}. Curnamona — layers (reference run).`),
    classTable(K, 'rad_u', `Table 6.${(no.uK = T())}. Curnamona — belief functions of rad_u.`),
    ['img', K.figures.map, 520, `Figure 6.4. Curnamona — EBF ${score(K)} with the seven REE occurrences.`],
    ['img', K.figures.curves, 600, 'Figure 6.5. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) class belief and plausibility of every layer.'],
    ['img', 'ebf_curnamona_unc.png', 600, 'Figure 6.6. Curnamona — (a) uncertainty Unc and (b) plausibility Pls.'],
    ['p', `The selected setting is **${sel(K).label}**, with a mean LODO AUC of **${f(lodo(K))}** (Table 6.${no.searchK}). U is the strongest layer (AUC of its belief ${f(lay(K, 'rad_u').single_layer_auc)}): the top octile of U holds ${cls(K, 'rad_u').slice(-1)[0].deposits} of the ${st(K).deposits} occurrences and gets Bel = ${f(lay(K, 'rad_u').bel_max)} (Table 6.${no.uK}). Th follows (${f(lay(K, 'rad_th').single_layer_auc)}); Nb and the silica index are weak (${f(lay(K, 'geochem_nb_log_idw').single_layer_auc)} each), but their beliefs follow the expected directions without being told — high Nb and **low** silica (the two lowest silica octiles have the largest belief, ${f(lay(K, 'aster_silica_index').bel_max)}). Since no layer is binary, the two rules for empty classes give almost the same result at 10 classes (${f(cand(K, DEF))} vs ${f(cand(K, HAL))}). The number of classes matters more with seven occurrences: 5 classes ${f(cand(K, '5 quantile classes, 0.5'))}, 8 classes ${f(cand(K, '8 quantile'))}, 10 classes ${f(cand(K, HAL))} and 20 classes ${f(cand(K, '20 quantile'))}. Leave-one-out ranks six occurrences at the 91st–96th percentile. Moolawatana (${f(occ(K).find((x) => x.deposit === 4).lodo_auc)}) is the exception — a Cu–REE occurrence with U only at the 56th percentile and a more siliceous surface (64th percentile of the silica index) — and none of the methods ranks it high (WofE ${f(lodoOf('wofe', 4))}, LR ${f(lodoOf('logistic_regression', 4))}, fuzzy WofE ${f(lodoOf('fuzzy_wofe', 4))}). Because every occurrence lies in a favourable class of every layer, the pairwise χ² tests cannot be computed (one class among the deposits).`],
    compare(),
    ['p', `On Carlin, EBF is the most accurate of the classical statistical methods (WofE ${f(auc('wofe', 'carlin'))}, logistic regression ${f(auc('logistic_regression', 'carlin'))}) and close to the best hybrids (fuzzy WofE ${f(auc('fuzzy_wofe', 'carlin'))}, neuro-fuzzy ${f(auc('neuro_fuzzy', 'carlin'))}). It uses the same likelihood ratios as WofE ($W_{CD}=e^{W^{+}}$) but keeps all ten classes of every continuous layer instead of one binary threshold, so the gradation of the As–Sb halo and of the fault distance is not discarded; and the normalisation over the classes bounds the contribution of each layer, which limits the double counting of correlated layers. On Curnamona it is slightly below WofE (${f(auc('wofe', 'curnamona'))}), logistic regression (${f(auc('logistic_regression', 'curnamona'))}) and fuzzy WofE (${f(auc('fuzzy_wofe', 'curnamona'))}): with six training occurrences per LODO run most of the eight classes of each layer are empty, while the binary WofE patterns pool the classes and estimate fewer quantities from the same occurrences.`],
    ['h2', '6.7 Recommended setup for the demo projects'],
    ['p', 'The settings below reproduce the selected results from the dialog; Section 6.8 explains the reasoning.'],
    setupTable(`Table 6.${T()}. EBF dialog settings for the demo projects.`),
    ['h2', '6.8 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal EBF parameters for the demo projects and their geological reasoning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1500, 1600, 1600, 4326],
      rows: [
        ['Evidence layers', 'host rock, distance to faults, As, Sb', 'Th, U, Nb, silica index', 'One layer per component of the mineral system, as for WofE — Carlin: chemical trap (reactive silty carbonate), fluid pathway (high-angle faults), fluid footprint (As–Sb halo); Curnamona: U–Th–REE granite source (Th, U), REE–HFSE enrichment (Nb), exposure (low-silica, non-quartzose cover). Layers of different components come closest to the independent sources that Dempster’s rule assumes.'],
        ['Classes', 'host: class values (binary); faults: 10 published distance classes; As, Sb: quantiles', 'quantiles for all four layers', 'Class rasters keep their geological classes (lithological units, mapped distance buffers). Continuous geochemistry and radiometrics need equal-area classes, so that every class has the same area and its belief compares deposit densities directly.'],
        ['Quantiles', String(ncls(J)), String(ncls(K)), `Carlin (24 training deposits): 5–20 classes give ${f(cand(J, '20 quantile'))}–${f(cand(J, '5 quantile classes, 0.5'))}, within 0.015; 10 classes (≈ 2–3 deposits per class) resolve the outward decay of the Sb–As halo, while 5 classes merge the inner halo with the weakly anomalous fringe. Curnamona (6 training occurrences per LODO run): 8 classes is best (${f(cand(K, '8 quantile'))}). With 5 classes the top 20 % of U form one class (${f(cand(K, '5 quantile classes, 0.5'))}), which dilutes the highly radioactive, hydrothermally overprinted granite that hosts the occurrences; 20 classes leave most classes empty (${f(cand(K, '20 quantile'))}), so the beliefs rest on single occurrences.`],
        ['Classes without deposits', zero(J), zero(K), `Carlin: essential (${f(cand(J, HAL))} vs ${f(cand(J, DEF))}). A class without deposits is absence of evidence rather than evidence of absence — much favourable ground is simply unexplored — yet the original rule turns it into certainty: the host class gets Bel = 1 and overrides the other layers, so the fault and halo information inside the host rocks is lost. Curnamona: no binary layer, so the two rules tie (${f(cand(K, HAL))} vs ${f(cand(K, DEF))}); the correction is kept for consistency.`],
        ['Missing evidence', 'vacuous (Unc = 1)', 'vacuous (no missing data)', `Carlin: geology is unmapped under the post-mineral cover in ${f(100 * lay(J, 'geology_host').missing_cells / st(J).cells, 0)} % of the study area. Excluding these cells would remove most of the belt, including covered ground next to known deposits. The vacuous belief lets faults and geochemistry decide there and flags the cells with high uncertainty (Figure 6.3) — the honest representation of “host rock unknown”, and exactly where covered-extension targets lie.`],
        ['Prospectivity score', score(J), score(K), `Bel, Pls and the mid-point rank the cells almost identically on both projects (Carlin ${f(cand(J, HAL))} / ${f(cand(J, '10 quantile classes, 0.5-deposit correction, score Pls'))} / ${f(cand(J, '10 quantile classes, 0.5-deposit correction, score (Bel'))}), because almost all deposits lie where the uncertainty is low. Bel is the conservative map (proven support); Pls is the map for greenfield targeting under cover, where it is high but Bel is low.`],
        ['Hold-out', '20 % (GUI default; 30 % in the tests)', 'LODO', 'Enough training deposits for stable class beliefs plus a spatially stratified test set; with seven occurrences only leave-one-out is meaningful.'],
      ],
    }],
    ['h2', '6.9 Discussion'],
    ['bullets', [
      `**Accuracy.** Carlin test AUC ${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)} (10 splits), the best classical statistical method on these data; Curnamona mean LODO AUC ${f(lodo(K))}, slightly below WofE and logistic regression with seven occurrences.`,
      '**Uncertainty as a map.** The uncertainty map separates “evidence against deposits” (Dis) from “no evidence” (Unc). This is the main reason to use EBF: unmapped or covered ground is flagged as uncertain instead of being scored as unfavourable.',
      '**Binary layers and empty classes.** A binary layer carries no uncertainty, and with the original rule a class that holds every deposit becomes a certainty that overrides all other layers. Use the 0.5-deposit correction and multi-class evidence where possible; the dialog warns when Bel = 1 occurs.',
      '**Independence.** Dempster’s rule assumes independent sources, like weights of evidence; correlated layers (fault corridors and the Sb halo at Carlin) are counted twice. The beliefs are therefore relative support, not calibrated probabilities.',
      '**Few deposits.** Each class belief rests on the deposits in that class; with fewer than ~10 deposits use 5–8 classes and read the class tables before the map.',
    ]],
    ['h1', '7 References'],
    ['refs', R('an1994', 'bonham1994', 'carranza2003', 'carranza2005', 'carranza2009', 'chung2003', 'cline2005', 'deepseam',
      'delong1988', 'dempster1967', 'elburg2012', 'fawcett2006', 'haldane1956', 'hofstra2000', 'moon1990', 'muntean2011',
      'neumann2000', 'raines2001', 'roberts2017', 'shafer1976', 'stevens2004', 'yousefi2015')],
  ],
};
