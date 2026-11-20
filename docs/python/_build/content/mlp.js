const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('mlp_carlin');
const K = C.load('mlp_curnamona');
const RFJ = C.has('random_forest_carlin') ? C.load('random_forest_carlin') : null;
const LRJ = C.has('logistic_regression_carlin') ? C.load('logistic_regression_carlin') : null;

const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const imp = (X) => X.reference.tables.importance;
const top = (X) => imp(X)[0];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const cand = (X, lab) => { const r = X.search.find((s) => s.label === lab); return r ? r.mean_test_auc : null; };
const impRows = (X) => imp(X).map((r) => [String(r.rank), r.layer, String(r.columns), pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const arch = (X) => P(X).hidden.join('-');
function actReason() {
  const t = (a) => (a === 'tanh' ? 'tanh (smooth, saturating)' : a === 'relu' ? 'ReLU (piecewise linear)' : a);
  const relu = 'ReLU gives piecewise-linear responses with thresholds — favourability rising only above an anomaly threshold or within a distance of a conduit, the threshold behaviour of fluid–rock reaction once a reactive unit is reached.';
  const tanh = `tanh gives smooth, saturating responses — favourability that rises with an anomaly or towards a conduit and then levels off, as gradational hydrothermal halos do (Carlin: tanh ${f(cand(J, '32-16 tanh, 1:3'))} vs ReLU ${f(cand(J, '32-16 ReLU, α 1e-4, 1:3 non-deposits'))} held-out AUC).`;
  return `Carlin: ${t(P(J).activation)}; Curnamona: ${t(P(K).activation)}. ` + (P(J).activation === 'tanh' || P(K).activation === 'tanh' ? tanh + ' ' + relu : relu);
}
let tn = 3;
const T = () => ++tn;

function sizeReason(X) {
  const big = cand(X, '64-32 ReLU, 1:3');
  const small = cand(X, '8 ReLU (one small layer), 1:3');
  const def = cand(X, '32-16 ReLU, α 1e-4, 1:3 non-deposits');
  return `${arch(X)} neurons (${st(X).weights} weights). The held-out AUC of the one-layer network with 8 neurons (${f(small)}), the 32-16 network (${f(def)}) and the 64-32 network (${f(big)}) differ little: the deposit signature is a smooth function of a few proxies, so additional capacity mainly memorises individual deposits. Two hidden layers allow interactions (Sb anomaly counts most inside the reactive host), one layer already approximates any smooth response (Hornik et al., 1989).`;
}
function alphaReason(X) {
  const strong = cand(X, '32-16 ReLU, α 0.01 (strong L2), 1:3');
  const base = cand(X, '32-16 ReLU, α 1e-4, 1:3 non-deposits');
  const cmp = Math.abs(strong - base) < 0.005 ? `a 100× stronger penalty changed the held-out AUC by less than 0.005 (${f(strong)} vs ${f(base)}) — early stopping already limits the effective complexity` : `a 100× stronger penalty gave ${f(strong)} (vs ${f(base)})`;
  return `α = ${g(P(X).alpha, 2)}. Weight decay keeps the response smooth; ${cmp}. Geologically, a smooth response is expected because ore-forming processes produce gradational footprints (halo intensity, distance decay from conduits) rather than abrupt jumps.`;
}
function ratioReason(X) {
  const d = X.dataset_info;
  const r = P(X).neg_ratio;
  return `1 : ${f(r, 0)}. One deposit cell per ${int(d.study_km2 / d.deposit_cells)} km²: barren ground is more varied than mineralised ground; ${r > 1 ? 'several barren samples per deposit describe that variety and stabilise the networks, which otherwise learn the few barren samples by heart.' : 'equal numbers keep every mini-batch balanced; larger ratios did not improve the held-out ranking.'}`;
}

module.exports = {
  file: 'mlp.docx',
  code: 'MLP',
  short: 'Neural network (MLP)',
  title: 'Artificial Neural Network (Multilayer Perceptron, MLP) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven machine-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Artificial neural networks were among the first machine-learning methods used for mineral prospectivity mapping (Brown et al., 2000; Porwal et al., 2003). A multilayer perceptron (MLP) maps the evidence vector of a cell through one or more hidden layers of neurons to an output between 0 and 1, learning non-linear responses and interactions from deposit and non-deposit samples. MPM Studio’s MLP (menu *Data-driven ▸ Machine learning ▸ Neural Network (MLP)*) follows the design of eis_toolkit’s `train_MLP_classifier` (EIS Toolkit, 2024) — dense layers, ReLU, the Adam optimiser, binary cross-entropy, early stopping — but is implemented with scikit-learn (no TensorFlow) and adds two elements needed for small MPM training sets: the number of epochs is chosen on the **cross-validated AUC**, and an **ensemble** of networks is averaged.'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4000, 5026],
      rows: [
        ['`python/logic/data_driven/mlp/model.py`', 'Scaling, lockstep cross-validation with early stopping, ensemble, prediction, importance'],
        ['`python/logic/data_driven/mlp/plugin.py`', 'Registration; run in the worker thread; outputs'],
        ['`python/logic/data_driven/mlp/dialogs.py`', 'Parameter and results dialogs (learning curve, importance)'],
        ['`python/logic/common/features.py`, `training.py`', 'Evidence encoding, train / test split, non-deposit sampling'],
        ['`python/tests/test_mlp.py`', 'Training, early stopping, ensemble, importance, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 The multilayer perceptron'],
    ['p', 'Each neuron computes a weighted sum of its inputs plus a bias and applies a non-linear activation; neurons are arranged in layers, each fully connected to the next. With one hidden layer of sufficient size an MLP can approximate any continuous function (Hornik et al., 1989). The output neuron uses the logistic sigmoid, so the network output is a value in [0, 1] that is trained like a probability. The weights are learned by back-propagation of the error gradient (Rumelhart et al., 1986) with a stochastic gradient optimiser — here Adam (Kingma and Ba, 2015), which adapts the step size of every weight from running estimates of the gradient’s first and second moments.'],
    ['h2', '2.2 Over-fitting, regularisation and early stopping'],
    ['p', 'A network with hundreds of weights trained on a few dozen deposits can memorise them. Three measures counter this: an L2 penalty (weight decay) that shrinks the weights, **early stopping** — training is stopped when the performance on held-out samples stops improving (Prechelt, 1998) — and averaging an **ensemble** of networks trained from different random initial weights, which reduces the variance of the prediction (Hansen and Salamon, 1990). The spread of the ensemble is a measure of model uncertainty.'],
    ['h2', '2.3 The MLP in mineral prospectivity mapping'],
    ['p', 'Brown et al. (2000) introduced back-propagation networks for gold prospectivity in the Yilgarn; Porwal et al. (2003) mapped base-metal potential in the Aravalli Province and showed that networks capture interactions between evidence layers that weights of evidence cannot. As for the other machine-learning methods, training samples are deposits and random non-deposit locations away from known deposits (Carranza et al., 2008), and the output ranks prospectivity but is not a probability per unit cell.'],
    ['h2', '2.4 Relation to eis_toolkit'],
    ['p', 'eis_toolkit’s `train_MLP_classifier` builds a Keras Sequential network of Dense (+ optional Dropout) layers, compiled with Adam, binary cross-entropy and early stopping on the validation loss with `validation_split`. MPM Studio keeps this architecture and its defaults but (i) uses scikit-learn, so no TensorFlow is needed; (ii) uses a stratified cross-validation instead of Keras’ `validation_split`, which takes the *last* rows unshuffled — with samples ordered deposits-first it would validate on barren cells only; (iii) stops on the held-out AUC rather than the loss, which matches the ranking purpose of a prospectivity map; (iv) uses L2 instead of dropout; and (v) averages an ensemble.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Forward pass'],
    ['eq', '\\bf{h}_{0}=\\bf{z},      \\bf{h}_{l}=g(\\bf{W}_{l}\\bf{h}_{l-1}+\\bf{b}_{l}),  l=1…L,      P(\\bf{z})=σ(\\bf{w}_{o}^{T}\\bf{h}_{L}+b_{o})', 'mlp-forward'],
    ['p', 'with $\\bf{z}$ the z-scored evidence (one-hot columns for categorical layers) and $g$ the activation: ReLU $g(u)=\\rm{max}(0,u)$, $\\rm{tanh}$ or the logistic sigmoid.'],
    ['h2', '3.2 Loss and gradient'],
    ['eq', 'E(θ)=-\\frac{1}{n}\\sum_{i}{[y_{i}\\ln P_{i}+(1-y_{i})\\ln(1-P_{i})]}+\\frac{α}{2n}\\sum_{l}{‖\\bf{W}_{l}‖^{2}}', 'mlp-loss'],
    ['p', 'The gradient is obtained by back-propagation: with $δ_{o}=P-y$ at the output and $\\bf{δ}_{l}=g\'(\\bf{a}_{l})⊙\\bf{W}_{l+1}^{T}\\bf{δ}_{l+1}$, $∂E/∂\\bf{W}_{l}=\\bf{δ}_{l}\\bf{h}_{l-1}^{T}$ (averaged over the mini-batch, plus $α\\bf{W}_{l}/n$).'],
    ['h2', '3.3 Adam'],
    ['eq', '\\bf{m}_{t}=β_{1}\\bf{m}_{t-1}+(1-β_{1})\\bf{g}_{t},  \\bf{v}_{t}=β_{2}\\bf{v}_{t-1}+(1-β_{2})\\bf{g}_{t}^{2},      θ_{t}=θ_{t-1}-η\\frac{\\hat{\\bf{m}}_{t}}{\\sqrt{\\hat{\\bf{v}}_{t}}+ε}', 'mlp-adam'],
    ['p', 'with bias-corrected moments $\\hat{\\bf{m}}_{t}=\\bf{m}_{t}/(1-β_{1}^{t})$, $\\hat{\\bf{v}}_{t}=\\bf{v}_{t}/(1-β_{2}^{t})$, $β_{1}=0.9$, $β_{2}=0.999$, learning rate $η$ (Kingma and Ba, 2015); mini-batches of 32 samples, reshuffled every epoch.'],
    ['h2', '3.4 Number of epochs and ensemble'],
    ['eq', 'E^{*}=\\rm{arg max}_{e}\\bar{A}(e),      \\bar{A}(e)=\\rm{AUC}(\\{P^{(-k)}_{i}(e)\\})', 'mlp-es'],
    ['p', 'where $P^{(-k)}_{i}(e)$ is the output after epoch $e$ of the network trained without fold $k$ for the samples of fold $k$; training stops when $\\bar{A}$ has not improved for the patience. The final networks are trained for $E^{*}$ epochs on all samples. With $R$ random samples and $N$ networks each,'],
    ['eq', 'S(\\bf{x})=\\frac{1}{RN}\\sum_{r,j}{P_{rj}(\\bf{x})},      s(\\bf{x})=\\sqrt{\\frac{1}{RN}\\sum_{r,j}{(P_{rj}(\\bf{x})-S(\\bf{x}))^{2}}}', 'mlp-ens'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Encoding and scaling** — feature matrix; z-scores with study-area statistics.',
      '**Training samples** — training deposits + $r·n_{d}$ non-deposit cells (distance from deposits, even spacing).',
      '**Early stopping** — k fold networks trained one epoch at a time in lockstep (`partial_fit`, Adam state kept); mean held-out AUC per epoch; $E^{*}$ [[mlp-es]]; held-out predictions at $E^{*}$ → CV AUC, confusion, permutation importance.',
      '**Ensemble** — $N$ networks with different initial weights trained for $E^{*}$ epochs on all samples.',
      '**Repeats and outputs** — for $R$ random samples; score and s.d. [[mlp-ens]]; tables; evaluation.',
    ]],
    ['table', {
      caption: 'Table 4.1. Implementation and verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3000, 6026],
      rows: [
        ['Network, back-propagation, Adam', 'scikit-learn 1.3.2 `MLPClassifier` (Pedregosa et al., 2011)'],
        ['Epoch-wise training, early stopping on AUC, ensemble', 'Own code; training loss decreases, learned signal, reproducible with a seed; E* = best held-out epoch (tests)'],
        ['Design defaults', 'eis_toolkit `train_MLP_classifier` (EIS Toolkit, 2024)'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Machine learning ▸ Neural Network (MLP)…*. Training data, evidence layers and training samples as for the other machine-learning methods.'],
    ['table', {
      caption: 'Table 5.1. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1500, 4926],
      rows: [
        ['Hidden layers (neurons)', '32, 16', 'Comma-separated neurons per hidden layer; keep the number of weights well below ~20 × the number of samples.'],
        ['Activation', 'ReLU', 'ReLU, tanh or sigmoid.'],
        ['Learning rate (Adam)', '0.001', '$η$ of [[mlp-adam]].'],
        ['L2 penalty α', '0.0001', 'Weight decay [[mlp-loss]]; larger = smoother.'],
        ['Batch size', '32', 'Samples per gradient step.'],
        ['Max. epochs / early stopping', '500 / 20', 'Upper limit and patience of [[mlp-es]].'],
        ['Networks per sample', '5', 'Ensemble size $N$ [[mlp-ens]].'],
        ['Repeats / cross-validation', '5 / 5-fold', 'Random barren samples $R$; folds of the early stopping.'],
        ['Outputs / prefix', 'score + s.d. / mlp', ''],
      ],
    }],
    ['p', 'Results dialog: samples, CV AUC, epochs, number of networks, ROC AUC of training and test deposits; tabs *Summary*, *Learning curve*, *Variable importance*, *Sample validation*, *Repeats*; *ROC / AUC…*, *Export tables (Excel)…*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin MLP.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (3 repeats × 5 networks, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}${RFJ && LRJ ? ` (random forest ${f(RFJ.robust_summary.test_auc_mean)}, logistic regression ${f(LRJ.robust_summary.test_auc_mean)})` : ''}. Early stopping after ${st(J).epochs} epochs keeps the training and test AUCs close.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — MLP prospectivity score (mean of 3 × 5 networks) with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) learning curve with the early-stopping epoch.'],
    ['table', { caption: `Table 6.${T()}. Carlin — permutation importance (CV AUC drop).`, head: ['Rank', 'Layer', 'Columns', 'CV AUC drop'], widths: [900, 3726, 1400, 3000], rows: impRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — model summary.`, head: ['Item', 'Value'], widths: [4526, 4500], rows: sumRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — validation of the training samples (sample 1).`, head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J) }],
    ['p', `The permutation importance ranks **${top(J).layer}** first. The learning curve (Figure 6.2d) shows the typical course: the held-out AUC rises with the training AUC during the first tens of epochs while the network learns the regional signature (host rock and Sb halo), then flattens while the training AUC keeps rising — from there on the network fits individual deposits, and training is stopped.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona MLP.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — MLP prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) learning curve.'],
    ['table', { caption: `Table 6.${T()}. Curnamona — permutation importance.`, head: ['Rank', 'Layer', 'Columns', 'CV AUC drop'], widths: [900, 3726, 1400, 3000], rows: impRows(K) }],
    ['p', `With seven occurrences the selected setting reaches a mean LODO AUC of **${f(lodo(K))}** and the importance ranks **${top(K).layer}** first. With 5–6 deposits per training fold, the early-stopping curve is noisy; the ensemble of networks is what keeps the score stable.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal MLP parameters for the demo projects and their meaning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component; z-scores put metres, log ppm and radiometric counts on one scale so that no proxy dominates the initial weights.'],
        ['Hidden layers', arch(J), arch(K), sizeReason(J)],
        ['Activation', P(J).activation, P(K).activation, actReason()],
        ['L2 penalty α', g(P(J).alpha, 2), g(P(K).alpha, 2), alphaReason(J)],
        ['Epochs (early stopping)', `${st(J).epochs}`, `${st(K).epochs}`, 'Chosen on held-out deposits: training stops when the network starts learning the particular deposits instead of the regional mineral-system signature.'],
        ['Networks per sample × repeats', '5 × 3', '5 × 3', 'Different random initial weights give different local optima with few deposits; the average represents the common, geologically meaningful signal and the spread shows where the networks disagree.'],
        ['Non-deposit ratio', `1 : ${f(P(J).neg_ratio, 0)}`, `1 : ${f(P(K).neg_ratio, 0)}`, ratioReason(J)],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'Width of the alteration / geochemical halo (Cline et al., 2005; Elburg et al., 2012).'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodo(K))}.`,
      '**Flexibility versus data.** The MLP can represent interactions and thresholds, but with tens of deposits its advantage over regularised linear models is small; early stopping and ensembling are essential.',
      '**Uncertainty.** The ensemble s.d. is high where the networks disagree — typically in evidence combinations without training samples.',
      '**Score, not probability.** The output depends on the deposit / barren ratio; it ranks cells.',
      '**eis_toolkit.** The design follows eis_toolkit’s MLP; the cross-validated early stopping on AUC corrects the unshuffled validation split of Keras for deposit-first sample tables.',
    ]],

    ['h1', '7 References'],
    ['refs', R('brown2000', 'carranza2008', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'eistoolkit', 'elburg2012',
      'fawcett2006', 'hansen1990', 'hofstra2000', 'hornik1989', 'kingma2015', 'muntean2011', 'neumann2000', 'pedregosa2011',
      'porwal2003', 'prechelt1998', 'raines2001', 'roberts2017', 'rumelhart1986', 'stevens2004', 'yousefi2015', 'zuo2020')],
  ],
};
