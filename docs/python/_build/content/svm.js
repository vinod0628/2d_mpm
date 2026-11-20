const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('svm_carlin');
const K = C.load('svm_curnamona');
const LRJ = C.load('logistic_regression_carlin');

const cfgOf = (X) => Object.assign({ kernel: 'rbf', tune: true, class_weight: 'none', neg_ratio: 1 }, X.selected);
const cj = cfgOf(J), ck = cfgOf(K);
const st = (X) => X.reference.tables.stats;
const impRows = (X) => X.reference.tables.importance.map((r) => [String(r.rank), r.layer, String(r.columns),
  pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const tuneRows = (X, k = 10) => [...X.reference.tables.tuning].sort((a, b) => b.cv_auc - a.cv_auc).slice(0, k)
  .map((r) => [(r.selected ? '**' : '') + g(r.C, 3) + (r.selected ? '**' : ''), g(r.gamma, 3), f(r.log2_C, 0),
    f(r.log2_gamma, 2), f(r.cv_auc, 4), r.selected ? 'selected' : '']);
const bounded = (X) => (X.reference.tables.sample_validation.find((r) => r.item.startsWith('Bounded')) || { value: '—' }).value.split(' of ')[0];
const top = (X) => X.reference.tables.importance[0];
const second = (X) => X.reference.tables.importance[1];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const kName = { rbf: 'RBF (Gaussian)', linear: 'linear', poly: 'polynomial', sigmoid: 'sigmoid' };
const ratio = (X) => X.reference.tables.params.neg_ratio;
const gf = (X) => st(X).gamma / st(X).gamma0;
let tn = 3;
const T = () => ++tn;

function gammaReason(X) {
  const q = gf(X);
  return `γ = ${g(st(X).gamma, 3)} = ${g(q, 3)} γ0. ${q <= 1
    ? 'A kernel width at or below the default means a smooth decision surface: the favourable domain is a broad region of evidence space, as expected for a mineral system whose footprint (alteration halo, geochemical dispersion, radiometric signature of a granite) changes gradually rather than at sharp, deposit-specific values.'
    : 'A kernel width above the default gives a more local decision surface: several distinct evidence signatures (for example different host-rock / anomaly combinations) are favourable, which is plausible where deposits of one system form in more than one setting.'}${q <= 0.1 && st(X).C >= 8 ? ' Together with the large C this lies on the ridge of equivalent models on which the RBF-SVM approaches a linear SVM (Keerthi and Lin, 2003): favourability rises smoothly and almost linearly along the evidence gradients (Sb enrichment, host-rock contact), consistent with the logistic-regression result.' : ''}`;
}
function cReason(X) {
  const c = st(X).C;
  return `C = ${g(c, 3)}. ${c <= 4
    ? 'A small to moderate penalty tolerates training errors: “non-deposit” cells are only unexplored ground and some lie in favourable domains (undiscovered deposits — a positive-unlabelled problem), so the margin must not be bent around them.'
    : 'A large penalty fits the training samples closely: the deposits share a sharply defined evidence signature (narrow anomaly and host combination), and few barren samples fall inside it, so a tight boundary costs little generality — the cross-validated AUC, not the training fit, confirmed this choice.'}`;
}
function ratioReason(X) {
  const d = X.dataset_info;
  const r = ratio(X);
  return `1 : ${f(r, 0)}. One deposit cell per ${int(d.study_km2 / d.deposit_cells)} km²: barren ground is far more varied than mineralised ground. ${r > 1
    ? 'Several barren samples per deposit describe that variety (all lithologies, unmineralised structures, background geochemistry) so that the boundary is not drawn too close to the barren side.'
    : 'An equal number keeps the classes balanced, so the margin is not dominated by the barren class; larger ratios did not improve the held-out ranking here.'}`;
}

module.exports = {
  file: 'svm.docx',
  code: 'SVM',
  short: 'Support vector machine',
  title: 'Support Vector Machine (SVM) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven machine-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'The support vector machine (Vapnik, 1995; Cortes and Vapnik, 1995) is a kernel-based classifier that separates two classes by the boundary with the largest margin. In mineral prospectivity mapping it learns the boundary between deposit cells and barren (non-deposit) cells in the space of the evidence layers and returns, for every cell, a signed distance to that boundary — the decision value — which ranks the prospectivity (Zuo and Carranza, 2011; Abedi et al., 2012; Rodriguez-Galiano et al., 2015). SVMs are effective with small training sets because the solution depends only on the training samples near the boundary (the support vectors), they model non-linear relations through the kernel, and they have a unique, globally optimal solution. MPM Studio (menu *Data-driven ▸ Machine learning ▸ Support Vector Machine*) adds cross-validated tuning of the two hyper-parameters, a Platt-calibrated score, averaging over several random barren samples and a cross-validated permutation importance.'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4000, 5026],
      rows: [
        ['`python/logic/data_driven/svm/model.py`', 'Scaling, tuning, cross-validation, Platt calibration, importance, prediction'],
        ['`python/logic/data_driven/svm/plugin.py`', 'Registration; run in the worker thread; outputs and tables'],
        ['`python/logic/data_driven/svm/dialogs.py`', 'Parameter and results dialogs (tuning heat map, importance chart)'],
        ['`python/logic/common/features.py`', 'Evidence encoding shared by the machine-learning methods (values, one-hot classes, missing data)'],
        ['`python/logic/common/training.py`', 'Train / test split of the deposits, non-deposit sampling, training / test points'],
        ['`python/tests/test_svm.py`', 'Platt vs direct minimisation, decision map vs scikit-learn, tuning, importance, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Maximum-margin classification'],
    ['p', 'For linearly separable classes many hyperplanes separate the training samples; statistical learning theory shows that the one with the largest margin — the largest distance to the nearest samples of either class — has the smallest bound on the generalisation error (Vapnik, 1995). The nearest samples lie on the margin and are called **support vectors**; all other samples could be removed without changing the solution. This sparsity is why SVMs work with few training samples, which is the normal situation in prospectivity mapping.'],
    ['h2', '2.2 Soft margin'],
    ['p', 'Deposit and barren cells overlap in evidence space: barren cells can lie in favourable ground (undiscovered deposits, or favourable rocks that were not mineralised), and deposits can have weak signatures. The soft-margin SVM (Cortes and Vapnik, 1995) allows samples inside the margin or on the wrong side at a cost $C$ per unit of violation. Small $C$ gives a wide, smooth margin that ignores individual awkward samples; large $C$ bends the boundary to classify the training samples correctly. Samples with the maximum weight $α_{i}=C$ are the margin violators (bounded support vectors).'],
    ['h2', '2.3 Kernels'],
    ['p', 'Non-linear boundaries are obtained by mapping the evidence vector $\\bf{x}$ into a high-dimensional feature space $φ(\\bf{x})$ in which a linear boundary is sought. Because the optimisation needs only inner products, the mapping is never computed explicitly: a kernel $K(\\bf{x},\\bf{x}\')=φ(\\bf{x})·φ(\\bf{x}\')$ replaces it (the “kernel trick”; Schölkopf and Smola, 2002). The Gaussian radial-basis-function (RBF) kernel measures the similarity of two cells by their distance in (scaled) evidence space; its width $γ$ sets how local the similarity is. It is the standard choice for MPM (Zuo and Carranza, 2011) and contains the linear kernel as a limiting case (Hsu et al., 2003).'],
    ['h2', '2.4 From decision values to a score'],
    ['p', 'The SVM output is the signed decision value $f(\\bf{x})$, not a probability. Platt (1999) fitted a sigmoid to the decision values of held-out samples to obtain calibrated scores; Lin et al. (2007) gave a numerically robust algorithm. The sigmoid is monotone, so the score ranks the cells exactly as $f(\\bf{x})$; it depends on the deposit / barren ratio of the training samples and is therefore a **favourability score**, not a probability per unit cell.'],
    ['h2', '2.5 SVM in mineral prospectivity mapping'],
    ['p', 'Zuo and Carranza (2011) used an RBF-SVM on deposits and randomly selected non-deposit locations to map Au prospectivity in China and found it superior to weights of evidence. Abedi et al. (2012) extended it to several prospectivity classes; Rodriguez-Galiano et al. (2015) compared SVM with random forest, neural networks and regression trees. Common practice — adopted here — is to scale the evidence, to draw non-deposit samples away from known deposits (Carranza et al., 2008) and to tune $C$ and $γ$ by cross-validation (Hsu et al., 2003). ArcSDM’s *SVM Train* tool (GTK) fits the same scikit-learn / LIBSVM classifier with z-scored evidence, $C=1$, $γ=1/p$ and the decision value as output (ArcSDM, 2025); it is the special case “tuning off” of the MPM Studio tool.'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Training data and scaling'],
    ['p', 'Training samples are the $n_{d}$ training-deposit cells ($y=+1$) and $n_{0}=r·n_{d}$ random non-deposit cells ($y=-1$) at least $d_{min}$ from every known deposit and at least the spacing $s$ apart (auto $s=\\frac{1}{2}\\sqrt{A/n_{0}}$, $A$ = area available). Continuous evidence keeps its values, a categorical layer with $K$ classes becomes $K$ one-hot columns. Every column $j$ is scaled with statistics of the whole study area (identical for every random sample):'],
    ['eq', 'z_{j}=\\frac{x_{j}-\\bar{x}_{j}}{s_{j}}  (z-score),      z_{j}=\\frac{x_{j}-\\rm{min}_{j}}{\\rm{max}_{j}-\\rm{min}_{j}}  (min–max)', 'svm-scale'],
    ['h2', '3.2 Primal and dual problems'],
    ['eq', '\\rm{min}_{\\bf{w},b,ξ}  \\frac{1}{2}‖\\bf{w}‖^{2}+\\sum_{i=1}^{n}{C_{i}ξ_{i}}    subject to  y_{i}(\\bf{w}·φ(\\bf{z}_{i})+b)≥1-ξ_{i},  ξ_{i}≥0', 'svm-primal'],
    ['p', 'The margin width is $2/‖\\bf{w}‖$; $ξ_{i}$ are slack variables. The Lagrange dual, solved by LIBSVM’s sequential minimal optimisation (Chang and Lin, 2011; Fan et al., 2005), is'],
    ['eq', '\\rm{max}_{α}  \\sum_{i}{α_{i}}-\\frac{1}{2}\\sum_{i}{\\sum_{j}{α_{i}α_{j}y_{i}y_{j}K(\\bf{z}_{i},\\bf{z}_{j})}}    subject to  0≤α_{i}≤C_{i},  \\sum_{i}{α_{i}y_{i}}=0', 'svm-dual'],
    ['eq', 'f(\\bf{z})=\\sum_{i∈SV}{α_{i}y_{i}K(\\bf{z}_{i},\\bf{z})}+b', 'svm-f'],
    ['p', 'Only support vectors ($α_{i}>0$) enter [[svm-f]]; $0<α_{i}<C_{i}$ lie exactly on the margin, $α_{i}=C_{i}$ inside it or misclassified. With class weights, $C_{i}=C·w_{y_{i}}$; *balanced* uses $w_{k}=n/(2n_{k})$.'],
    ['h2', '3.3 Kernels'],
    ['eq', 'K_{RBF}=\\exp(-γ‖\\bf{z}-\\bf{z}\'‖^{2}),  K_{lin}=\\bf{z}·\\bf{z}\',  K_{poly}=(γ\\bf{z}·\\bf{z}\'+c_{0})^{d},  K_{sig}=\\rm{tanh}(γ\\bf{z}·\\bf{z}\'+c_{0})', 'svm-kernels'],
    ['p', 'The default width is $γ_{0}=1/(p·\\bar{v})$ with $p$ the number of columns and $\\bar{v}$ their mean variance after scaling (= $1/p$ for z-scores, the value used by ArcSDM).'],
    ['h2', '3.4 Hyper-parameter tuning'],
    ['p', 'On the first random sample, every pair of the exponential grid (Hsu et al., 2003)'],
    ['eq', 'C∈\\{2^{-3},2^{-1},…,2^{9}\\},      γ∈\\{γ_{0}2^{-6},γ_{0}2^{-4},…,γ_{0}2^{4}\\}', 'svm-grid'],
    ['p', 'is scored by the AUC of the pooled held-out decision values of a stratified 5-fold cross-validation; the best pair is used (ties at four decimals → smaller $C$, then smaller $γ$). The linear kernel tunes $C$ only.'],
    ['h2', '3.5 Platt calibration'],
    ['p', 'With the held-out decision values $f_{i}$ of a $k$-fold cross-validation and Platt’s regularised targets $t_{+}=(N_{+}+1)/(N_{+}+2)$, $t_{-}=1/(N_{-}+2)$, the sigmoid parameters minimise the cross-entropy (Newton method with backtracking, Lin et al., 2007):'],
    ['eq', 'P(\\bf{z})=\\frac{1}{1+\\exp(Af(\\bf{z})+B)},      \\rm{min}_{A,B}-\\sum_{i}{[t_{i}\\ln P_{i}+(1-t_{i})\\ln(1-P_{i})]}', 'svm-platt'],
    ['p', 'A score that increases with $f$ requires $A<0$. With very few deposits the held-out decision values can be uninformative ($A≥0$); the sigmoid is then fitted to the training decision values so that the score never reverses the ranking of $f$ (a warning is shown).'],
    ['h2', '3.6 Repeats, uncertainty and importance'],
    ['eq', 'S(\\bf{x})=\\frac{1}{R}\\sum_{r=1}^{R}{P_{r}(\\bf{x})},      s(\\bf{x})=\\sqrt{\\frac{1}{R}\\sum_{r=1}^{R}{(P_{r}(\\bf{x})-S(\\bf{x}))^{2}}}', 'svm-score'],
    ['p', 'with one SVM per random non-deposit sample $r$ (Zuo and Wang, 2020). The importance of evidence layer $j$ is the drop of the cross-validated AUC when the values of its columns are permuted among the held-out samples of every fold (Breiman, 2001), averaged over five permutations:'],
    ['eq', 'I_{j}=\\rm{AUC}(f_{cv})-\\rm{AUC}(f_{cv}^{(π_{j})})', 'svm-imp'],
    ['p', 'For the linear kernel the weights $\\bf{w}=\\sum{α_{i}y_{i}\\bf{z}_{i}}$ of the scaled columns are also reported.'],

    ['h1', '4 Algorithm'],
    ['h2', '4.1 Steps'],
    ['steps', [
      '**Encoding** — resample the evidence to one grid; build the feature matrix of the study cells (`FeatureStack`); missing evidence excluded or imputed (median / most frequent class).',
      '**Scaling** — shift and scale of every column from the study area [[svm-scale]]; $γ_{0}$.',
      '**Training samples** — training deposits (after the spatially stratified hold-out and optional thinning) + $r·n_{d}$ non-deposit cells (distance from deposits, even spacing).',
      '**Tuning (sample 1)** — grid search [[svm-grid]] by 5-fold cross-validated AUC (parallel threads).',
      '**Cross-validation** — $k$-fold (default 5; leave-one-out ≤ 300 samples) held-out decision values → CV AUC, CV confusion matrix, Platt sigmoid [[svm-platt]], permutation importance [[svm-imp]] (sample 1).',
      '**Final model** — SVM on all samples of the draw; $f(\\bf{x})$ and $P(\\bf{x})$ for every study cell in chunks of 200 000 cells (parallel threads).',
      '**Repeats** — steps 3, 5 and 6 for $R$ random draws (same $C$, $γ$); score and s.d. [[svm-score]], mean decision value.',
      '**Outputs and evaluation** — `score.tif`, `score_std.tif`, optional `decision.tif`, training / test point shapefiles, tables; ROC, success-rate and P–A curves of the training and test deposits against all study cells.',
    ]],
    ['h2', '4.2 Library and own code'],
    ['table', {
      caption: 'Table 4.1. Components and verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3000, 6026],
      rows: [
        ['SVM training and decision values', 'scikit-learn 1.3.2 `SVC` (LIBSVM; Pedregosa et al., 2011; Chang and Lin, 2011); the decision map equals `SVC.decision_function` on the scaled matrix to 10⁻⁶ (test)'],
        ['Platt calibration', 'Own code (Lin et al., 2007, Algorithm 1); equals a direct minimisation of the regularised cross-entropy to 10⁻⁴; the score is monotone in f (test)'],
        ['Tuning, cross-validation, importance', 'Own code (stratified folds, pooled held-out AUC, permutation by layer)'],
        ['Non-deposit sampling, split, evaluation', 'Shared MPM Studio modules (also used by logistic regression and random forest)'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Machine learning ▸ Support Vector Machine…*. The training-data section (deposits, mask, hold-out 20 % spatially stratified, seed, thinning, absence points) is the same as for the other data-driven methods.'],
    ['table', {
      caption: 'Table 5.1. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2200, 6826],
      rows: [
        ['Use', 'Tick the evidence layers. Every column has equal weight in the kernel distance after scaling, so leave out layers without a geological rationale — they add noise to the distance.'],
        ['Variable type', '*Continuous* (values; use log-transformed geochemistry) or *Categorical (one-hot)* (one 0/1 column per class).'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Training samples.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1500, 4926],
      rows: [
        ['Non-deposit samples per deposit', '1', 'Ratio $r$; 1–5 are usual.'],
        ['Min. distance from deposits', '4 cells', '$d_{min}$; keep barren samples outside the alteration / geochemical halo of known deposits.'],
        ['Min. spacing between samples', 'auto', 'Even spread of the barren samples; 0 = none.'],
        ['Random samples (repeats)', '5', '$R$ SVMs, scores averaged [[svm-score]].'],
        ['Cross-validation', '5-fold', 'Folds for the Platt calibration, CV AUC and importance; 10-fold, 3-fold or leave-one-out (≤ 300 samples).'],
      ],
    }],
    ['table', {
      caption: 'Table 5.3. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1700, 4726],
      rows: [
        ['Kernel', 'RBF', 'RBF (non-linear, recommended), linear (interpretable weights), polynomial, sigmoid.'],
        ['Tune C and γ', 'on', 'Grid search [[svm-grid]]; off = use the C and γ entered below.'],
        ['C (penalty)', '1', 'Small = wide, smooth margin; large = close fit of the training samples.'],
        ['γ (kernel width)', 'auto = γ0', 'Small = smooth boundary; large = local boundary around individual samples.'],
        ['Degree / coef0', '3 / 0', 'Polynomial and sigmoid kernels only.'],
        ['Class weight', 'none', '*balanced* raises the penalty of deposit errors in proportion to the class sizes.'],
        ['Scaling', 'z-score', 'z-score, min–max [0, 1] or none (only for evidence already on a common scale).'],
        ['Missing evidence', 'exclude', 'Exclude cells or impute the median / most frequent class.'],
        ['Outputs / prefix', 'score + std / svm', 'Optional decision value f(x) (the ArcSDM output).'],
      ],
    }],
    ['p', 'The results dialog shows the training samples, the cross-validated AUC, the chosen C and γ and the ROC AUC of the training and test deposits, with the tabs *Summary*, *Parameter tuning* (table and CV-AUC heat map over log₂C × log₂γ), *Variable importance* (table and bar chart), *Sample validation* (training and CV confusion at f = 0, accuracy, AUC, support vectors, Platt A and B) and *Repeats*; *ROC / AUC…* and *Export tables (Excel)…* as for the other methods.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin SVM.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (3 repeats, 5-fold calibration, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${cj.label}**. Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}. The small gap between training and test AUC reflects the regularisation of the margin: unlike a random forest, the SVM does not memorise its training deposits.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — SVM prospectivity score (Platt-calibrated, mean of 3 samples) with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) cross-validated AUC of the C–γ grid (star = selected).'],
    ['table', {
      caption: `Table 6.${T()}. Carlin — the ten best pairs of the C–γ grid (reference run, 5-fold CV AUC).`,
      head: ['C', 'γ', 'log₂ C', 'log₂ γ', 'CV AUC', ''],
      widths: [1500, 1500, 1200, 1200, 1500, 2126],
      rows: tuneRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — model summary of the reference run.`,
      head: ['Item', 'Value'], widths: [4526, 4500], rows: sumRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — permutation importance (drop of the CV AUC, mean ± s.d. of five permutations).`,
      head: ['Rank', 'Evidence layer', 'Columns', 'CV AUC drop'],
      widths: [900, 3426, 1400, 3300],
      rows: impRows(J),
    }],
    ['table', {
      caption: `Table 6.${T()}. Carlin — validation of the training samples (sample 1).`,
      head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J),
    }],
    ['p', `The SVM uses ${st(J).support_vectors} support vectors, ${bounded(J)} of them bounded (α = C, Table 6.${tn}) — margin violators, which are expected because deposit and barren cells overlap in evidence space. The permutation importance ranks **${top(J).layer}** first and **${second(J).layer}** second. The favourable silty-carbonate host and the Sb halo together define the Carlin trap–footprint combination. ${gf(J) <= 0.1 && st(J).C >= 8 ? 'The tuned kernel is wide (small γ, large C), so the SVM behaves almost like a regularised linear classifier on the scaled evidence: the data support an additive, gradual combination of host rock and Sb halo — the structure logistic regression models directly, which explains why the SVM does not exceed it (mean test AUC ' + f(J.robust_summary.test_auc_mean) + ' vs ' + f(LRJ.robust_summary.test_auc_mean) + ' for logistic regression).' : 'The narrow kernel combines them non-linearly (an Sb anomaly counts most inside favourable host), which a linear model can only approximate.'}`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona SVM.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — SVM prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) cross-validated AUC of the C–γ grid.'],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — the ten best pairs of the C–γ grid (reference run).`,
      head: ['C', 'γ', 'log₂ C', 'log₂ γ', 'CV AUC', ''],
      widths: [1500, 1500, 1200, 1200, 1500, 2126],
      rows: tuneRows(K),
    }],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — permutation importance.`,
      head: ['Rank', 'Evidence layer', 'Columns', 'CV AUC drop'],
      widths: [900, 3426, 1400, 3300],
      rows: impRows(K),
    }],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — validation of the training samples (sample 1).`,
      head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(K),
    }],
    ['p', `With seven occurrences the selected setting reaches a mean LODO AUC of **${f(lodo(K))}**. The importance ranks **${top(K).layer}** and **${second(K).layer}** highest. With so few positives the cross-validated quantities (tuning surface, Platt sigmoid, importance) rest on 5–6 deposits per fold and are indicative only; the LODO test, which predicts each occurrence from the other six, is the measure to report.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal SVM parameters for the demo projects and their meaning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component (Carlin: pathway, trap, footprint; Curnamona: U–Th–REE granite source, HFSE enrichment, exposure). Geochemistry is log-transformed because trace elements are log-normally distributed and hydrothermal enrichment is multiplicative; after scaling every proxy carries equal weight in the kernel distance.'],
        ['Scaling', 'z-score', 'z-score', 'Puts metres (distance to faults), log ppm and radiometric counts on one scale; study-area statistics make the scaling independent of the random barren sample.'],
        ['Kernel', kName[cj.kernel], kName[ck.kernel], 'Ore formation is a conjunction of processes (fluid pathway **and** reactive trap **and** metal source): favourability is high only where several proxies are favourable together. The RBF kernel represents such interactions and thresholds; the linear kernel adds the proxies independently.'],
        ['C (penalty)', g(st(J).C, 3), g(st(K).C, 3), cReason(J)],
        ['γ (kernel width)', g(st(J).gamma, 3), g(st(K).gamma, 3), gammaReason(J)],
        ['Non-deposit ratio', `1 : ${f(ratio(J), 0)}`, `1 : ${f(ratio(K), 0)}`, ratioReason(J)],
        ['Class weight', cj.class_weight, ck.class_weight, 'Ranking (AUC) does not depend on the class weights much; balanced weights move the boundary towards the barren class, which matters for the f = 0 classification, not for the prospectivity ranking.'],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'About the width of the alteration / geochemical halo (hundreds of metres to ≈ 1 km for Carlin-type systems, Cline et al., 2005): cells inside the halo belong to the mineral system and must not be used as barren examples.'],
        ['Repeats', '3 (5 in the GUI)', '3 (5 in the GUI)', 'Averaging over several barren samples removes the dependence of the margin on one random draw and yields the score s.d. as an uncertainty map.'],
        ['Calibration folds', '5', '5', 'Five folds give each held-out set enough deposits for a stable sigmoid; leave-one-out is possible for small samples but costs one SVM per sample.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodo(K))}. The companion documents report weights of evidence, logistic regression and random forest with the same protocol.`,
      '**Generalisation.** The training and test AUCs are close because the margin is regularised (C, γ tuned on held-out data); a random forest fits its training deposits almost perfectly and must be judged by out-of-bag or test statistics.',
      '**Score, not probability.** The Platt score depends on the deposit / barren ratio of the samples; it ranks cells but is not a probability per unit cell. Use logistic regression where calibrated probabilities are needed.',
      '**Interpretability.** The RBF-SVM has no coefficients; the permutation importance and the linear-kernel weights are the interpretable outputs.',
      '**Positive-unlabelled data.** Barren samples are unexplored cells, some of which hold undiscovered deposits; a moderate C and several repeats reduce their influence on the boundary.',
    ]],

    ['h1', '7 References'],
    ['refs', R('abedi2012', 'breiman2001', 'carranza2008', 'chang2011', 'chung2003', 'cline2005', 'cortes1995', 'deepseam',
      'delong1988', 'elburg2012', 'fan2005', 'fawcett2006', 'hofstra2000', 'hsu2003', 'keerthi2003', 'lin2007', 'muntean2011',
      'neumann2000', 'pedregosa2011', 'platt1999', 'raines2001', 'roberts2017', 'rodriguez2015', 'scholkopf2002',
      'stevens2004', 'vapnik1995', 'yousefi2015', 'zuo2011', 'zuo2020', 'arcsdm')],
  ],
};
