const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('elm_carlin');
const K = C.load('elm_curnamona');
const MJ = C.has('mlp_carlin') ? C.load('mlp_carlin') : null;

const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const imp = (X) => X.reference.tables.importance;
const top = (X) => imp(X)[0];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, lab) => { const r = X.search.find((s) => s.label === lab); return r ? r.mean_test_auc : null; };
const impRows = (X) => imp(X).map((r) => [String(r.rank), r.layer, String(r.columns), pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const tuneRows = (X) => (X.reference.tables.tuning || []).map((r) => [g(r.C, 3), f(r.log2_C, 0), f(r.cv_auc, 4), r.selected ? 'selected' : '']);
const aName = { sigmoid: 'sigmoid', tanh: 'tanh', relu: 'ReLU', rbf: 'Gaussian RBF' };
let tn = 3;
const T = () => ++tn;

function cReason(X) {
  const c = st(X).C;
  return `C = ${g(c, 3)}. ${c <= 1 ? 'Strong regularisation: the output weights are shrunk, so the ELM represents a smooth favourability surface that a few deposits can support; the CV curve falls for large C, where the ELM starts to fit individual deposits (Figure 6.2d).' : 'Moderate regularisation: the data support a more detailed response; larger C over-fits (Figure 6.2d).'} Geologically the smooth surface corresponds to the gradational footprint of a mineral system (alteration halo, geochemical dispersion, radiometric signature of a granite).`;
}
function hiddenReason(X) {
  return `${P(X).hidden} random neurons. With ridge output weights the number of neurons matters little (held-out AUC: 30 neurons ${f(cand(X, '30 sigmoid, tuned C, 1:3'))}, 100 neurons ${f(cand(X, '100 sigmoid, tuned C, 1:3 non-deposits'))}, 300 neurons ${f(cand(X, '300 sigmoid, tuned C, 1:3'))}): C, not L, controls the complexity; L only needs to be large enough to represent the response of a few proxies.`;
}
function ratioReason(X) {
  const d = X.dataset_info;
  const r = P(X).neg_ratio;
  return `1 : ${f(r, 0)}. One deposit cell per ${int(d.study_km2 / d.deposit_cells)} km²: barren ground is far more varied than mineralised ground; ${r > 1 ? 'several barren samples per deposit describe that variety.' : 'equal numbers keep the least-squares targets balanced; larger ratios did not help here.'}`;
}

module.exports = {
  file: 'elm.docx',
  code: 'ELM',
  short: 'Extreme learning machine',
  title: 'Extreme Learning Machine (ELM) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven machine-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'The extreme learning machine (Huang et al., 2006) is a single-hidden-layer feed-forward network whose hidden layer is generated at random and never trained; only the output weights are learned, in closed form, by regularised least squares. Training is a single linear solve — orders of magnitude faster than back-propagation — with one regularisation constant C. Chen and Wu (2017) used ELM regression for mineral prospectivity mapping. MPM Studio (menu *Data-driven ▸ Machine learning ▸ Extreme Learning Machine*) trains an ensemble of ELMs on deposits and random non-deposit cells, tunes C by cross-validation, calibrates the output to a 0–1 score and reports the ensemble spread as uncertainty.'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4000, 5026],
      rows: [
        ['`python/logic/data_driven/elm/model.py`', 'Random hidden layers, ridge output weights, C tuning, ensemble, calibration, importance'],
        ['`python/logic/data_driven/elm/plugin.py`, `dialogs.py`', 'Registration, outputs, parameter and results dialogs'],
        ['`python/tests/test_elm.py`', 'Ridge solution in both forms, RBF layer, tuning, ensemble, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Random hidden layers'],
    ['p', 'Huang et al. (2006) proved that a single-hidden-layer network with $L$ hidden neurons whose input weights and biases are chosen at random can interpolate $N$ distinct samples when $L≥N$ and approximate any continuous function — provided the output weights are chosen optimally. Because the hidden layer is fixed, the output weights are the solution of a linear least-squares problem: there is no iterative training, no local minima and no learning rate.'],
    ['h2', '2.2 Regularisation'],
    ['p', 'The minimum-norm least-squares solution over-fits when the hidden layer is large. Huang et al. (2012) added a ridge penalty: minimising $‖\\bf{β}‖^{2}/2+C‖\\bf{Hβ}-\\bf{t}‖^{2}/2$ gives smooth outputs, with $C$ controlling the trade-off between fit and smoothness — the same role as the penalty of an SVM or the L2 weight decay of an MLP.'],
    ['h2', '2.3 Ensembles and calibration'],
    ['p', 'A single ELM depends on its random hidden layer. Averaging an ensemble of ELMs with different random weights stabilises the output and reduces its variance (Liu and Wang, 2010); the spread of the members is a measure of model uncertainty. The least-squares output approximates the 0/1 targets but is not bounded; a monotone logistic (Platt) mapping fitted to the cross-validated outputs turns it into a score in [0, 1] without changing the ranking.'],
    ['h2', '2.4 ELM in mineral prospectivity mapping'],
    ['p', 'Chen and Wu (2017) trained ELM regression on deposit and non-deposit locations and found it competitive with SVM and random forest at a fraction of the training time. The fast training makes repeated sampling of the barren class and large ensembles cheap — attractive for the many-run validation protocols needed with few deposits. Training data follow the common protocol of the machine-learning methods (Carranza et al., 2008; Zuo and Wang, 2020).'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Hidden layer'],
    ['eq', 'H_{ij}=g(\\bf{a}_{j}^{T}\\bf{z}_{i}+b_{j}),      \\bf{a}_{j}∼U(-1,1)^{p},  b_{j}∼U(-1,1)', 'elm-h'],
    ['p', 'with $g$ the sigmoid, $\\rm{tanh}$ or ReLU; for the Gaussian RBF layer $H_{ij}=\\exp(-‖\\bf{z}_{i}-\\bf{c}_{j}‖^{2}/σ^{2})$ with centres $\\bf{c}_{j}$ drawn from the training samples and $σ$ the median distance between centres.'],
    ['h2', '3.2 Output weights'],
    ['eq', '\\bf{β}=(\\bf{H}^{T}\\bf{H}+\\bf{I}/C)^{-1}\\bf{H}^{T}\\bf{t}  (N≥L),      \\bf{β}=\\bf{H}^{T}(\\bf{H}\\bf{H}^{T}+\\bf{I}/C)^{-1}\\bf{t}  (N<L)', 'elm-beta'],
    ['p', 'The two forms are algebraically identical (Huang et al., 2012); the smaller system is solved. The output of one ELM is $f(\\bf{z})=\\bf{h}(\\bf{z})^{T}\\bf{β}$.'],
    ['h2', '3.3 Ensemble, calibration, tuning'],
    ['eq', '\\bar{f}(\\bf{z})=\\frac{1}{N_{e}}\\sum_{e}{f_{e}(\\bf{z})},      P(\\bf{z})=\\frac{1}{1+\\exp(A\\bar{f}(\\bf{z})+B)}', 'elm-ens'],
    ['p', '$A<0$ and $B$ are fitted to the cross-validated ensemble outputs (Platt sigmoid; same algorithm as the SVM). $C$ is chosen from $\\{2^{-6},2^{-4},…,2^{12}\\}$ by the 5-fold cross-validated AUC; the score map is the mean of $P$ over the members and the $R$ random barren samples, the uncertainty map the s.d. of the members’ calibrated outputs.'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Encoding and scaling** — feature matrix; z-scores with study-area statistics.',
      '**Training samples** — training deposits + $r·n_{d}$ non-deposit cells.',
      '**Tuning (sample 1)** — CV AUC of the ensemble for every $C$; best $C$ (ties → smaller).',
      '**Cross-validation and calibration** — held-out ensemble outputs → CV AUC, Platt sigmoid [[elm-ens]], permutation importance.',
      '**Ensemble** — $N_{e}$ ELMs with new random hidden layers, output weights [[elm-beta]] on all samples.',
      '**Prediction and repeats** — calibrated outputs for every study cell; mean and s.d. over members and samples.',
    ]],
    ['table', {
      caption: 'Table 4.1. Implementation and verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3000, 6026],
      rows: [
        ['Hidden layers, output weights', 'Own NumPy code; both forms of [[elm-beta]] equal the augmented least-squares solution (test)'],
        ['Calibration', 'Platt / Lin et al. routine shared with the SVM; monotone'],
        ['Ensemble, tuning, importance', 'Own code; reproducible with a seed (test)'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Machine learning ▸ Extreme Learning Machine…*. Training data, evidence and training samples as for the other machine-learning methods.'],
    ['table', {
      caption: 'Table 5.1. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1500, 4926],
      rows: [
        ['Hidden neurons', '100', '$L$; 30–300 give similar results when C is tuned.'],
        ['Activation', 'sigmoid', 'sigmoid, tanh, ReLU or Gaussian RBF (random centres).'],
        ['Tune C / C', 'on / 1', 'Cross-validated choice of the regularisation; off = the value entered.'],
        ['ELMs per sample', '10', 'Ensemble size $N_{e}$.'],
        ['Repeats / cross-validation', '5 / 5-fold', 'Random barren samples; folds for tuning, calibration and importance.'],
        ['Outputs / prefix', 'score + s.d. / elm', ''],
      ],
    }],
    ['p', 'Results dialog: samples, CV AUC, C, number of ELMs, ROC AUC of training and test deposits; tabs *Summary*, *Regularisation*, *Variable importance*, *Sample validation*, *Repeats*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin ELM.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (3 repeats × 10 ELMs, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}${MJ ? ` (MLP: test ${f(MJ.robust_summary.test_auc_mean)})` : ''}; a whole run with ${3 * 10} ELMs takes ${f(J.robust_summary.seconds_mean, 1)} s.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — ELM prospectivity score with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) cross-validated AUC against C.'],
    ['table', { caption: `Table 6.${T()}. Carlin — regularisation (5-fold CV AUC, sample 1).`, head: ['C', 'log₂ C', 'CV AUC', ''], widths: [2200, 1800, 2500, 2526], rows: tuneRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — permutation importance (CV AUC drop).`, head: ['Rank', 'Layer', 'Columns', 'CV AUC drop'], widths: [900, 3726, 1400, 3000], rows: impRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — model summary.`, head: ['Item', 'Value'], widths: [4526, 4500], rows: sumRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — validation of the training samples (sample 1).`, head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J) }],
    ['p', `The permutation importance ranks **${top(J).layer}** first. The regularisation curve has the classic shape: too small C under-fits (the response cannot follow the host-rock contrast), too large C fits individual deposits and the held-out AUC falls.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona ELM.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — ELM prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) cross-validated AUC against C.'],
    ['table', { caption: `Table 6.${T()}. Curnamona — permutation importance.`, head: ['Rank', 'Layer', 'Columns', 'CV AUC drop'], widths: [900, 3726, 1400, 3000], rows: impRows(K) }],
    ['p', `With seven occurrences the selected setting reaches a mean LODO AUC of **${f(lodo(K))}**, and the importance ranks **${top(K).layer}** first.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal ELM parameters for the demo projects and their meaning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component; z-scores give every proxy the same scale for the random hidden weights.'],
        ['Hidden neurons', String(P(J).hidden), String(P(K).hidden), hiddenReason(J)],
        ['Activation', aName[P(J).activation], aName[P(K).activation], 'Sigmoid neurons give smooth, saturating responses — favourability that rises with an anomaly or towards a conduit and then levels off, as hydrothermal footprints do.'],
        ['C (regularisation)', g(st(J).C, 3), g(st(K).C, 3), cReason(J)],
        ['ELMs per sample × repeats', '10 × 3', '10 × 3', 'Random hidden layers differ; the ensemble average represents the stable, geologically meaningful signal, the spread shows where it is uncertain.'],
        ['Non-deposit ratio', `1 : ${f(P(J).neg_ratio, 0)}`, `1 : ${f(P(K).neg_ratio, 0)}`, ratioReason(J)],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'Width of the alteration / geochemical halo (Cline et al., 2005; Elburg et al., 2012).'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodo(K))}.`,
      '**Speed.** One linear solve per ELM: ensembles of tens of networks and many barren samples cost seconds, so the uncertainty of the random sampling can be explored cheaply.',
      '**Regularisation is the key parameter.** With tuned C the results are insensitive to the number of neurons and to the activation.',
      '**Score, not probability.** The calibrated output ranks cells; it depends on the deposit / barren ratio.',
    ]],

    ['h1', '7 References'],
    ['refs', R('carranza2008', 'chen2017', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'hofstra2000', 'huang2006', 'huang2012', 'lin2007', 'liu2010', 'muntean2011', 'neumann2000', 'platt1999',
      'raines2001', 'roberts2017', 'stevens2004', 'yousefi2015', 'zuo2020')],
  ],
};
