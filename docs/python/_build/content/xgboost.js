const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('xgboost_carlin');
const K = C.load('xgboost_curnamona');
const RFJ = C.has('random_forest_carlin') ? C.load('random_forest_carlin') : null;

const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const imp = (X) => X.reference.tables.importance;
const top = (X) => imp(X)[0];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const dir = (v) => (v === null || v === undefined || Number.isNaN(v) ? '—' : (v >= 0 ? '+' : '') + f(v, 2));
const impRows = (X) => imp(X).map((r) => [String(r.rank), r.layer, f(r.mean_abs_shap, 3), f(r.shap_pct, 1),
  f(r.shap_deposits, 3), dir(r.direction), f(r.gain_pct, 1), pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const sensRows = (X) => X.sensitivity.map((s) => [g(s.learning_rate, 2), String(s.trees), f(s.cv_auc), f(s.train_auc),
  s.test_auc == null ? '—' : f(s.test_auc), f(s.seconds, 1)]);
const monoTxt = (X) => {
  const m = P(X).monotone || {};
  const k = Object.keys(m).filter((x) => m[x]);
  return k.length ? k.map((x) => `${x} ${m[x] > 0 ? '↑' : '↓'}`).join(', ') : 'none';
};
const layerDir = (X, name) => { const r = imp(X).find((x) => x.layer === name); return r ? r.direction : null; };
const lrRange = (X) => {
  const sv = X.sensitivity;
  const te = sv.map((x) => x.test_auc).filter((v) => v != null);
  const tr = sv.map((x) => x.trees);
  return { e0: g(sv[0].learning_rate, 2), t0: sv[0].trees, e1: g(sv[sv.length - 1].learning_rate, 2), t1: sv[sv.length - 1].trees,
    lo: te.length ? f(Math.min(...te)) : '—', hi: te.length ? f(Math.max(...te)) : '—', tmin: Math.min(...tr), tmax: Math.max(...tr) };
};
const candAuc = (X, lab) => { const r = X.search.find((x) => x.label === lab); return r ? r.mean_test_auc : null; };
let tn = 3;
const T = () => ++tn;

function depthReason(X) {
  const d = P(X).max_depth;
  return `Depth ${d}: every tree can combine up to ${d} evidence layers in one rule. ${d <= 3
    ? 'Ore formation needs a few processes to coincide (fluid pathway, chemical trap, metal source or its geochemical footprint), so interactions of two or three proxies are geologically meaningful; deeper trees would model coincidences of many layers that a few dozen deposits cannot support.'
    : 'Deeper trees model higher-order coincidences of the proxies; with few deposits they are justified only if the held-out AUC confirms them.'}`;
}
function ratioReason(X) {
  const d = X.dataset_info;
  const r = P(X).neg_ratio;
  return `1 : ${f(r, 0)}. One deposit cell per ${int(d.study_km2 / d.deposit_cells)} km²: barren ground is far more varied than mineralised ground. ${r > 1
    ? 'Several barren samples per deposit describe that variety; boosting then concentrates on the barren cells that resemble deposits (largest gradients), i.e. on the geologically “near-miss” ground.'
    : 'An equal number keeps the classes balanced; larger ratios did not improve the held-out ranking here.'}`;
}
function monoReason(X, ds) {
  const m = P(X).monotone || {};
  const on = Object.keys(m).some((k) => m[k]);
  if (ds === 'carlin') {
    return on ? 'Applied (distance to faults ↓, As ↑, Sb ↑): fluid flow is focused along faults and decreases away from them, and the As–Sb halo strengthens towards the ore zone; the constraint forbids geologically impossible wiggles (e.g. a second favourable band far from faults) that a few deposits could otherwise create.'
      : `Not applied: the unconstrained model ranked the held-out deposits at least as well, and its SHAP directions already follow geology (distance to faults ${dir(layerDir(X, 'dist_faults'))}, Sb ${dir(layerDir(X, 'sb_log_idw'))}). The constraint remains recommended when the map must be defensible to explorers, because it guarantees a monotone response.`;
  }
  return on ? 'Applied (Th ↑, U ↑, Nb ↑, silica index ↓): the U–Th–REE granite source and its HFSE enrichment raise prospectivity monotonically, quartz-rich cover lowers it.'
    : `Not applied: the constrained model (LODO ${f(candAuc(X, 'depth 3, 1:5, geological monotone constraints'))}) ranked the occurrences as well as the selected setting (${f(lodo(X))}) — a difference far below the LODO spread — so the simpler default was kept; the SHAP directions (U ${dir(layerDir(X, 'rad_u'))}, Th ${dir(layerDir(X, 'rad_th'))}, Nb ${dir(layerDir(X, 'geochem_nb_log_idw'))}) show the expected positive relations.`;
}

module.exports = {
  file: 'xgboost.docx',
  code: 'XGB',
  short: 'XGBoost',
  title: 'XGBoost (Extreme Gradient Boosting) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven machine-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'XGBoost (Chen and Guestrin, 2016) is a regularised, second-order implementation of gradient-boosted decision trees (Friedman, 2001). Instead of averaging independent deep trees, as a random forest does, it builds many shallow trees one after another, each correcting the errors of the ensemble so far. In mineral prospectivity mapping it learns the multivariate signature of known deposits from deposit cells and random non-deposit cells (Parsa, 2021). MPM Studio (menu *Data-driven ▸ Machine learning ▸ XGBoost*) chooses the number of trees by cross-validated early stopping, supports **monotone constraints** that keep the response geologically plausible, handles missing evidence natively, explains the model with **TreeSHAP** contributions (Lundberg et al., 2020) — as an importance table and as contribution maps — and runs on an NVIDIA GPU automatically when one is present.'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4100, 4926],
      rows: [
        ['`python/logic/data_driven/xgboost_mpm/model.py`', 'Training samples, cross-validated early stopping, booster, SHAP, importance, device detection'],
        ['`python/logic/data_driven/xgboost_mpm/plugin.py`', 'Registration; run in the worker thread; outputs (score, s.d., SHAP maps) and tables'],
        ['`python/logic/data_driven/xgboost_mpm/dialogs.py`', 'Parameter dialog (monotone column, device) and results dialog (boosting curve, importance)'],
        ['`python/logic/common/features.py`', 'Evidence encoding (values, one-hot classes; missing: exclude, impute or keep as NaN)'],
        ['`python/logic/common/training.py`', 'Train / test split, non-deposit sampling, training / test points'],
        ['`python/tests/test_xgboost.py`', 'Early stopping, SHAP additivity, monotone constraint, missing values, GPU fallback, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Boosting as gradient descent in function space'],
    ['p', 'Friedman (2001) showed that boosting is a gradient descent in the space of functions: starting from a constant, each step adds a small regression tree that approximates the negative gradient of the loss of the current model, scaled by a learning rate (shrinkage). With the logistic loss the model is additive on the **log-odds** scale, like logistic regression, but each term is a tree that can represent thresholds and interactions. Small learning rates and many trees generalise better than few large steps; subsampling the training rows for each tree (stochastic gradient boosting, Friedman, 2002) reduces variance further.'],
    ['h2', '2.2 XGBoost'],
    ['p', 'XGBoost (Chen and Guestrin, 2016) uses the second-order Taylor expansion of the loss (gradient and Hessian), so the optimal leaf values and the split gain have closed forms; it adds L1 / L2 penalties on the leaf weights and a minimum loss reduction per split to the objective, subsamples rows and columns, and learns a default direction for missing values at every split (sparsity-aware split finding). Its histogram algorithm runs on CPUs and NVIDIA GPUs.'],
    ['h2', '2.3 Interpretation with SHAP'],
    ['p', 'Shapley additive explanations (Lundberg and Lee, 2017) attribute a prediction to the input features with the game-theoretic Shapley values — the only additive attribution that is locally accurate, consistent and gives zero to features that are not used. For tree ensembles TreeSHAP computes them exactly in polynomial time (Lundberg et al., 2020). For every cell, the contributions of the evidence layers plus a constant add up to the predicted log-odds; mapped over the study area they show **why** a cell is prospective (e.g. “Sb anomaly +1.2, host rock +0.8, far from faults −0.4”).'],
    ['h2', '2.4 Monotone constraints'],
    ['p', 'Geological knowledge often fixes the direction of an effect: prospectivity should not increase away from the fluid conduits, nor decrease with a pathfinder element. XGBoost can enforce a non-decreasing or non-increasing response to chosen variables by allowing only splits whose child weights respect the order. This injects process knowledge into a data-driven model, prevents implausible patterns learned from a few deposits and makes the map easier to defend.'],
    ['h2', '2.5 XGBoost in mineral prospectivity mapping'],
    ['p', 'Parsa (2021) used XGBoost with data augmentation to map carbonate-hosted Zn–Pb potential in western Iran; gradient boosting is now a standard benchmark beside random forest and SVM in MPM. As for the other machine-learning methods, non-deposit samples are drawn away from known deposits (Carranza et al., 2008), several random draws are averaged (Zuo and Wang, 2020), and the score ranks prospectivity but is not a probability per unit cell.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Model and objective'],
    ['p', 'With training samples $(\\bf{x}_{i},y_{i})$, $y_{i}∈\\{0,1\\}$ (deposit = 1), the model after $K$ trees is'],
    ['eq', 'F_{K}(\\bf{x})=F_{0}+η\\sum_{k=1}^{K}{f_{k}(\\bf{x})},      P(\\bf{x})=\\frac{1}{1+e^{-F_{K}(\\bf{x})}}', 'xgb-model'],
    ['p', 'where $f_{k}$ is a regression tree with $T_{k}$ leaves and leaf weights $w_{kt}$, and $η$ is the learning rate. Tree $k$ minimises the regularised objective'],
    ['eq', '\\cal{L}^{(k)}=\\sum_{i}{\\ell(y_{i},F_{k-1}(\\bf{x}_{i})+f_{k}(\\bf{x}_{i}))}+γT_{k}+\\frac{1}{2}λ\\sum_{t}{w_{kt}^{2}}+α\\sum_{t}{|w_{kt}|}', 'xgb-obj'],
    ['p', 'with the logistic loss $\\ell=-[y\\ln P+(1-y)\\ln(1-P)]$.'],
    ['h2', '3.2 Second-order approximation, leaf weights and split gain'],
    ['p', 'With gradients and Hessians of the loss at the current model,'],
    ['eq', 'g_{i}=P_{i}-y_{i},      h_{i}=P_{i}(1-P_{i})', 'xgb-gh'],
    ['p', 'and $G_{t}=\\sum_{i∈t}{g_{i}}$, $H_{t}=\\sum_{i∈t}{h_{i}}$ over the samples of leaf $t$ (sample weights multiply $g$ and $h$; *balanced* weights deposits by $n_{0}/n_{1}$), the optimal leaf weight (for $α=0$) and the gain of splitting a node into $L$ and $R$ are'],
    ['eq', 'w_{t}^{*}=-\\frac{G_{t}}{H_{t}+λ},      \\rm{Gain}=\\frac{1}{2}[\\frac{G_{L}^{2}}{H_{L}+λ}+\\frac{G_{R}^{2}}{H_{R}+λ}-\\frac{(G_{L}+G_{R})^{2}}{H_{L}+H_{R}+λ}]-γ', 'xgb-gain'],
    ['p', 'A split is made only if the gain is positive and both children have a Hessian sum of at least the minimum child weight. Samples with $P_{i}$ far from $y_{i}$ have large $|g_{i}|$: each new tree concentrates on the cells the ensemble still misranks. Splits are searched over histogram bins of every column (`hist`); a missing value is sent to the side that gives the larger gain.'],
    ['h2', '3.3 Monotone constraints'],
    ['p', 'For a variable constrained to be increasing, a split on it is accepted only if $w_{L}≤w_{R}$, and the weights of all leaves below are bounded by the midpoint of the parent’s children, which makes $F_{K}$ non-decreasing in that variable for any fixed values of the others (decreasing analogously).'],
    ['h2', '3.4 Number of trees by cross-validated early stopping'],
    ['p', 'The samples are split into $k$ stratified folds; all folds are boosted together and the held-out AUC of fold $j$ after $m$ trees, $A_{j}(m)$, is recorded. The number of trees is'],
    ['eq', 'K^{*}=\\rm{arg max}_{m≤M}\\bar{A}(m),      \\bar{A}(m)=\\frac{1}{k}\\sum_{j=1}^{k}{A_{j}(m)}', 'xgb-es'],
    ['p', 'where boosting stops at $M$ once $\\bar{A}$ has not improved for the patience (default 50 rounds). The held-out predictions after $K^{*}$ trees give the cross-validated AUC, the confusion matrix and the permutation importance; the final booster is trained on all samples with $K^{*}$ trees.'],
    ['h2', '3.5 SHAP values, repeats and importance'],
    ['eq', 'F_{K}(\\bf{x})=φ_{0}+\\sum_{j=1}^{p}{φ_{j}(\\bf{x})},      φ_{j}=\\sum_{S⊆\\{1..p\\}∖\\{j\\}}{\\frac{|S|!(p-|S|-1)!}{p!}[v(S∪\\{j\\})-v(S)]}', 'xgb-shap'],
    ['p', 'with $v(S)$ the expected model output when only the features in $S$ are known (TreeSHAP conditional expectation along the trees). The contribution of an evidence layer is the sum over its columns. Reported per layer: mean $|φ|$ over the training samples and its share, the mean $φ$ of the deposits, the effect direction (Spearman correlation of value and $φ$), the share of the total split gain and the permutation importance (drop of the cross-validated AUC). With $R$ random non-deposit samples (one booster each),'],
    ['eq', 'S(\\bf{x})=\\frac{1}{R}\\sum_{r=1}^{R}{P_{r}(\\bf{x})},      s(\\bf{x})=\\sqrt{\\frac{1}{R}\\sum_{r}{(P_{r}(\\bf{x})-S(\\bf{x}))^{2}}},      \\bar{φ}_{j}(\\bf{x})=\\frac{1}{R}\\sum_{r}{φ_{j,r}(\\bf{x})}', 'xgb-score'],

    ['h1', '4 Algorithm'],
    ['h2', '4.1 Steps'],
    ['steps', [
      '**Device** — *auto*: a one-off probe trains a 1-tree booster with `device=cuda`; if the GPU is used, all boosters run on CUDA, otherwise on the CPU (no setting has to be changed between PCs).',
      '**Encoding** — evidence resampled to one grid; feature matrix of the study cells; missing evidence excluded, imputed or kept as NaN.',
      '**Training samples** — training deposits (after the spatially stratified hold-out and optional thinning) + $r·n_{d}$ non-deposit cells (distance from deposits, even spacing).',
      '**Early stopping** — $k$ folds boosted in lockstep in blocks of 25 trees (training continuation) until [[xgb-es]] stops; $K^{*}$, held-out predictions.',
      '**Final booster** — all samples, $K^{*}$ trees; probability of every study cell in chunks of 250 000 cells; optional TreeSHAP contributions [[xgb-shap]].',
      '**Repeats** — steps 3–5 for $R$ random draws; score, s.d. and mean SHAP maps [[xgb-score]].',
      '**Importance and validation** — sample 1: SHAP, gain and permutation importance, training and CV confusion matrices, AUC.',
      '**Outputs and evaluation** — `score.tif`, `score_std.tif`, `shap_<layer>.tif`, training / test point shapefiles, tables; ROC, success-rate and P–A curves of the training and test deposits.',
    ]],
    ['h2', '4.2 Library and own code'],
    ['table', {
      caption: 'Table 4.1. Components and verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3000, 6026],
      rows: [
        ['Boosting, SHAP, GPU', 'xgboost 3.2 (`hist`, CUDA built in; Chen and Guestrin, 2016); TreeSHAP contributions add up to the logit of the score (test)'],
        ['Early stopping across folds', 'Own code (lockstep boosting, mean held-out AUC curve, patience rule; test of the rule)'],
        ['Monotone constraints', 'xgboost; the response to a constrained layer is monotone (test)'],
        ['Device detection', 'Own code (probe with CUDA, read the device from the booster configuration; GPU request without GPU → CPU + warning, test)'],
        ['Sampling, split, evaluation', 'Shared MPM Studio modules'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Machine learning ▸ XGBoost…*. The training-data section is the same as for the other data-driven methods.'],
    ['table', {
      caption: 'Table 5.1. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2200, 6826],
      rows: [
        ['Use', 'Tick the evidence layers.'],
        ['Variable type', '*Continuous* or *Categorical (one-hot)*; trees are invariant to monotone transformations, so log-transforming is not needed (but harmless).'],
        ['Monotone constraint', 'Continuous layers: *none*, *increasing* (score may only rise with the value, e.g. pathfinder elements) or *decreasing* (e.g. distance to faults or to an intrusion).'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Training samples.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1500, 4926],
      rows: [
        ['Non-deposit samples per deposit', '1', 'Ratio $r$.'],
        ['Min. distance from deposits', '4 cells', 'Keep barren samples outside the halo of known deposits.'],
        ['Min. spacing between samples', 'auto', 'Even spread of the barren samples.'],
        ['Random samples (repeats)', '5', '$R$ boosters, scores averaged.'],
        ['Cross-validation', '5-fold', 'Folds for early stopping, CV AUC and importance (10, 3).'],
      ],
    }],
    ['table', {
      caption: 'Table 5.3. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1500, 4926],
      rows: [
        ['Max. number of trees', '500', 'Upper limit $M$; the cross-validation usually stops earlier.'],
        ['Early stopping (rounds)', '50', 'Patience of [[xgb-es]]; off = use all trees.'],
        ['Learning rate η', '0.05', 'Shrinkage; smaller values need more trees (Table: learning-rate sensitivity).'],
        ['Max. tree depth', '3', 'Order of the interactions.'],
        ['Min. child weight', '1', 'Minimum Hessian sum per leaf; larger = more conservative.'],
        ['Min. split loss γ', '0', 'Minimum gain per split.'],
        ['Row / column subsample', '0.8 / 0.8', 'Stochastic boosting; de-correlates the trees.'],
        ['L2 λ / L1 α', '1 / 0', 'Penalties on the leaf weights [[xgb-obj]].'],
        ['Class weight', 'none', '*balanced* weights deposits by $n_{0}/n_{1}$.'],
        ['Missing evidence', 'exclude', 'Exclude, impute, or keep as missing (learned default direction; the whole study area is mapped).'],
        ['Device', 'auto', 'NVIDIA GPU (CUDA) if detected, else CPU; the dialog shows the detected device.'],
        ['Outputs / prefix', 'score + std / xgb', 'Optional SHAP contribution map per evidence layer.'],
      ],
    }],
    ['p', 'The results dialog shows the training samples, the cross-validated AUC, the number of trees, the device and the ROC AUC of the training and test deposits; tabs *Summary*, *Boosting curve*, *Variable importance* (table and SHAP bar chart), *Sample validation* and *Repeats*; *ROC / AUC…* and *Export tables (Excel)…*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin XGBoost model.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (3 repeats, 5-fold early stopping, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}${RFJ ? ` (random forest: test ${f(RFJ.robust_summary.test_auc_mean)}, training ${f(RFJ.robust_summary.train_auc_mean)})` : ''}. Early stopping keeps the ensemble small (${st(J).trees} trees in the reference run); the training–test gap (${f(J.robust_summary.train_auc_mean - J.robust_summary.test_auc_mean)}) is similar to that of the random forest, i.e. both tree ensembles fit their training deposits more closely than the regularised linear models do, and only the held-out AUC measures prediction.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — XGBoost prospectivity score (mean of 3 boosters) with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) SHAP importance (red: the score falls with the value).'],
    ['table', {
      caption: `Table 6.${T()}. Carlin — variable importance of the reference booster (SHAP on the training samples; direction = Spearman correlation of value and SHAP; permutation = CV AUC drop).`,
      head: ['Rank', 'Layer', 'Mean |SHAP|', 'SHAP %', 'Mean SHAP of deposits', 'Direction', 'Gain %', 'Permutation'],
      widths: [700, 1826, 1000, 900, 1300, 1000, 900, 1400],
      rows: impRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — model summary of the reference run.`,
      head: ['Item', 'Value'], widths: [4526, 4500], rows: sumRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — validation of the training samples (sample 1).`,
      head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — learning rate (selected setting, seed 42).`,
      head: ['η', 'Trees K*', 'CV AUC', 'Training AUC', 'Test AUC', 'Time (s)'],
      widths: [1200, 1300, 1600, 1600, 1600, 1726],
      rows: sensRows(J),
    }],
    ['p', `SHAP ranks **${top(J).layer}** first (${f(top(J).shap_pct, 1)} % of the mean |SHAP|). The effect directions agree with the Carlin mineral system: higher Sb raises the score (${dir(layerDir(J, 'sb_log_idw'))}), larger distance to faults lowers it (${dir(layerDir(J, 'dist_faults'))}). The learning-rate table shows the classical trade-off (Friedman, 2001): the number of trees falls from ${lrRange(J).t0} at η = ${lrRange(J).e0} to ${lrRange(J).t1} at η = ${lrRange(J).e1}, while the test AUC stays between ${lrRange(J).lo} and ${lrRange(J).hi}; η = 0.05 is a safe compromise between run time and smoothness of the score map.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona XGBoost model.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — XGBoost prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) SHAP importance.'],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — variable importance of the reference booster.`,
      head: ['Rank', 'Layer', 'Mean |SHAP|', 'SHAP %', 'Mean SHAP of deposits', 'Direction', 'Gain %', 'Permutation'],
      widths: [700, 1826, 1000, 900, 1300, 1000, 900, 1400],
      rows: impRows(K),
    }],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — learning rate (selected setting, all occurrences).`,
      head: ['η', 'Trees K*', 'CV AUC', 'Training AUC', 'Test AUC', 'Time (s)'],
      widths: [1200, 1300, 1600, 1600, 1600, 1726],
      rows: sensRows(K),
    }],
    ['p', `With seven occurrences the selected setting reaches a mean LODO AUC of **${f(lodo(K))}**. SHAP ranks **${top(K).layer}** first; the directions of U (${dir(layerDir(K, 'rad_u'))}), Th (${dir(layerDir(K, 'rad_th'))}) and Nb (${dir(layerDir(K, 'geochem_nb_log_idw'))}) are those expected for REE mineralisation derived from U–Th–REE-rich A-type granite. With 5–6 deposits per training fold the held-out AUC curve is flat and noisy: early stopping chooses between ${lrRange(K).tmin} and ${lrRange(K).tmax} trees across the learning rates (Table 6.${tn}), and the importances are indicative only.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal XGBoost parameters for the demo projects and their meaning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component (Carlin: pathway, trap, footprint; Curnamona: granite source, HFSE enrichment, exposure). Boosting adds the proxies on the log-odds scale like logistic regression, but each through a flexible, thresholded response.'],
        ['Max. tree depth', String(P(J).max_depth), String(P(K).max_depth), depthReason(J)],
        ['Learning rate η', g(P(J).learning_rate, 2), g(P(K).learning_rate, 2), 'Small steps make every tree a minor correction, so no single deposit cluster dominates the model; the sensitivity tables show that larger η gives similar accuracy with fewer, coarser trees.'],
        ['Number of trees', `${st(J).trees} (early stopping)`, `${st(K).trees} (early stopping)`, 'Chosen on held-out deposits: boosting stops when further trees only fit the particular training deposits (local noise of the mineral system) rather than its regional signature.'],
        ['Non-deposit ratio', `1 : ${f(P(J).neg_ratio, 0)}`, `1 : ${f(P(K).neg_ratio, 0)}`, ratioReason(J)],
        ['Monotone constraints', monoTxt(J), monoTxt(K), monoReason(J, 'carlin') + ' ' + monoReason(K, 'curnamona')],
        ['Row / column subsample', '0.8 / 0.8', '0.8 / 0.8', 'Each tree sees a random 80 % of the samples and layers, so correlated proxies of one process (As and Sb from the same fluids; Th and U from the same granite) share the credit instead of one masking the other.'],
        ['Regularisation λ / min. child weight', '1 / 1', '1 / 1', 'Leaf values are shrunk towards zero and leaves must hold a minimum of information; with 24–34 or 7 deposits this prevents leaves built on a single occurrence.'],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'Width of the alteration / geochemical halo: cells inside it belong to the mineral system and must not be barren examples (Cline et al., 2005; Elburg et al., 2012).'],
        ['Repeats', '3 (5 in the GUI)', '3 (5 in the GUI)', 'Several random barren samples remove the dependence on one draw and give the score s.d.'],
        ['Device', st(J).device, st(K).device, 'auto: identical results on CPU and GPU up to floating-point rounding; the GPU shortens large grids and many repeats.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodo(K))}. The companion documents report weights of evidence, logistic regression, random forest and SVM with the same protocol.`,
      '**Interpretability.** SHAP values make the boosted trees as interpretable as a linear model at the cell level: the contribution maps show which process proxy makes each target prospective, which is valuable for exploration decisions and for papers.',
      '**Geological constraints.** Monotone constraints let the geologist fix the direction of an effect while the data fix its shape and strength — a simple hybrid of knowledge- and data-driven modelling.',
      '**Score, not probability.** The score depends on the deposit / barren ratio; it ranks cells but is not a probability per unit cell.',
      '**Few deposits.** With seven occurrences early stopping, importances and SHAP directions are uncertain; report the LODO spread.',
    ]],

    ['h1', '7 References'],
    ['refs', R('carranza2008', 'chen2016', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'friedman2001', 'friedman2002', 'hofstra2000', 'lundberg2017', 'lundberg2020', 'muntean2011', 'neumann2000',
      'parsa2021', 'raines2001', 'roberts2017', 'stevens2004', 'yousefi2015', 'zuo2020', 'arcsdm')],
  ],
};
