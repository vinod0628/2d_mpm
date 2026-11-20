const { R } = require('../refs');
const C = require('../common');

const { f, g, int, pm } = C;
const J = C.load('rvm_carlin');
const K = C.load('rvm_curnamona');
const SVJ = C.has('svm_carlin') ? C.load('svm_carlin') : null;

const P = (X) => X.reference.tables.params;
const st = (X) => X.reference.tables.stats;
const imp = (X) => X.reference.tables.importance;
const top = (X) => imp(X)[0];
const lodo = (X) => X.search.find((s) => s.label === X.selected.label).mean_test_auc;
const impRows = (X) => imp(X).map((r) => [String(r.rank), r.layer, String(r.columns),
  pm(r.permutation_auc_drop, r.permutation_sd, 3), r.weights || '—']);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sumRows = (X) => X.reference.tables.summary.map((r) => [r.item, r.value]);
const tuneRows = (X) => (X.reference.tables.tuning || []).map((r) => [g(r.gamma, 3), g(r.gamma_factor, 3), f(r.cv_auc, 4),
  r.selected ? 'selected' : '']);
const kName = { rbf: 'RBF (Gaussian) on the training samples', linear: 'linear (relevance determination)' };

function relRows(X) {
  const rel = X.reference.tables.relevance || [];
  if (!rel.length) return [];
  if ('term' in rel[0]) return rel.map((r) => [r.term, f(r.weight, 3), f(r.sd, 3), g(r.alpha, 3), '', '']);
  const ev = Object.keys(rel[0]).filter((k) => !['row', 'col', 'cls', 'weight', 'alpha'].includes(k));
  return rel.map((r) => [`${r.row}, ${r.col}`, r.cls, f(r.weight, 3), g(r.alpha, 3),
    ev.map((k) => `${k.replace('geology_host', 'host').replace('_log_idw', '')} ${g(r[k], 3)}`).join('; '), '']);
}
function relTable(X, caption) {
  const rel = X.reference.tables.relevance || [];
  if (rel.length && 'term' in rel[0]) {
    return ['table', { caption, head: ['Term', 'Weight', 's.d.', 'α'], widths: [3026, 2000, 2000, 2000],
      rows: rel.map((r) => [r.term, f(r.weight, 3), f(r.sd, 3), g(r.alpha, 3)]) }];
  }
  return ['table', { caption, head: ['Cell (row, col)', 'Class', 'Weight', 'α', 'Evidence values'],
    widths: [1300, 1200, 1000, 1000, 4526], rows: relRows(X).map((r) => r.slice(0, 5)) }];
}
function relText(X, ds) {
  const rel = X.reference.tables.relevance || [];
  if (!rel.length) return '';
  if ('term' in rel[0]) {
    return `With the linear basis, relevance determination keeps ${rel.length - 1} of the evidence columns (Table 6.${tn}); the remaining columns were judged irrelevant by the evidence.`;
  }
  const dep = rel.filter((r) => r.cls === 'deposit');
  const bar = rel.filter((r) => r.cls !== 'deposit');
  const n = (k, one, many) => `${k} ${k === 1 ? one : many}`;
  const mean = (rows, k) => rows.reduce((a, r) => a + r[k], 0) / Math.max(rows.length, 1);
  let geo = '';
  if (ds === 'carlin' && dep.length && bar.length) {
    const host = (rows) => rows.filter((r) => r['geology_host[favourable host]'] === 1).length;
    geo = ` In Carlin, ${host(dep)} of ${n(dep.length, 'deposit prototype', 'deposit prototypes')} lie${dep.length === 1 ? 's' : ''} in the favourable silty-carbonate host with log Sb ${f(mean(dep, 'sb_log_idw'), 2)} (≈ ${f(10 ** mean(dep, 'sb_log_idw'), 1)} ppm), while ${host(bar)} of ${n(bar.length, 'barren prototype', 'barren prototypes')} lie${bar.length === 1 ? 's' : ''} in the host and ${bar.length === 1 ? 'its' : 'their'} log Sb is ${f(mean(bar, 'sb_log_idw'), 2)} (≈ ${f(10 ** mean(bar, 'sb_log_idw'), 1)} ppm): the trap–footprint combination of the Carlin system expressed by a handful of samples.`;
  } else if (ds === 'curnamona' && dep.length && bar.length) {
    const v = (k) => `${f(mean(dep, k), 1)} vs ${f(mean(bar, k), 1)}`;
    geo = ` In Curnamona the deposit prototype${dep.length === 1 ? '' : 's'} and the barren prototype${bar.length === 1 ? '' : 's'} differ as expected for REE derived from U–Th–REE-rich granite: Th ${v('rad_th')}, U ${v('rad_u')}, K ${v('rad_k')}, log Nb ${v('geochem_nb_log_idw')} and silica index ${v('aster_silica_index')} (deposit vs barren).`;
  }
  return `Only **${rel.length}** of the ${st(X).samples} training samples remain as relevance vectors (${n(dep.length, 'deposit', 'deposits')}, ${n(bar.length, 'non-deposit', 'non-deposits')}). They are “prototypes” of the mineral system: a positive weight on a deposit sample marks the evidence combination around which favourability is centred, a negative weight on a barren sample marks a typical barren combination (Table 6.${tn}; point layer on the map).${geo}`;
}
function gammaReason(X) {
  if (P(X).kernel !== 'rbf') return 'Linear basis: no kernel width.';
  const q = st(X).gamma / st(X).gamma0;
  return `γ = ${g(st(X).gamma, 3)} = ${g(q, 3)} γ0 (γ0 = 1/p). ${q <= 1 ? 'A wide kernel: each relevance vector influences a broad region of evidence space, so favourability changes gradually — as the footprint of a mineral system (alteration halo, geochemical dispersion, radiometric signature of a granite) does.' : 'A narrow kernel: favourability is concentrated around the evidence combinations of individual prototypes, appropriate where deposits occur in several distinct settings.'}`;
}
function ratioReason(X) {
  const d = X.dataset_info;
  const r = P(X).neg_ratio;
  return `1 : ${f(r, 0)}. One deposit cell per ${int(d.study_km2 / d.deposit_cells)} km²: barren ground is far more varied than mineralised ground. ${r > 1 ? 'Several barren samples per deposit let the evidence select barren prototypes for each distinct background (other lithologies, distal faults, background geochemistry).' : 'An equal number keeps the classes balanced; larger ratios did not improve the held-out ranking.'}`;
}
let tn = 3;
const T = () => ++tn;

module.exports = {
  file: 'rvm.docx',
  code: 'RVM',
  short: 'Relevance vector machine',
  title: 'Relevance Vector Machine (RVM) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven sparse Bayesian method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'The relevance vector machine (Tipping, 2001) is a sparse Bayesian kernel classifier. Like the support vector machine it builds a decision function from kernel functions centred on training samples, but it is a fully probabilistic model: every weight has its own Gaussian prior, the hyper-parameters are estimated by maximising the Bayesian evidence, and almost all weights are switched off. The few training samples that remain — the **relevance vectors** — define the model. For mineral prospectivity mapping this gives (i) a posterior probability with a predictive uncertainty for every cell, (ii) a very compact, interpretable model whose prototypes are real deposit and barren locations, (iii) no penalty parameter to tune, and (iv) with a linear basis, automatic relevance determination of the evidence layers. MPM Studio (menu *Data-driven ▸ Machine learning ▸ Relevance Vector Machine*) implements the model of Tipping (2001) with the fast marginal-likelihood algorithm of Tipping and Faul (2003).'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4000, 5026],
      rows: [
        ['`python/logic/data_driven/rvm/model.py`', 'Basis functions, Laplace posterior, fast evidence maximisation, prediction, importance'],
        ['`python/logic/data_driven/rvm/plugin.py`', 'Registration; run in the worker thread; outputs (score, predictive s.d., relevance-vector points)'],
        ['`python/logic/data_driven/rvm/dialogs.py`', 'Parameter and results dialogs (relevance vectors, kernel width, importance)'],
        ['`python/logic/common/features.py`, `training.py`', 'Evidence encoding, train / test split, non-deposit sampling'],
        ['`python/tests/test_rvm.py`', 'Fast algorithm vs re-estimation (same evidence maximum), sparsity, ARD, Carlin model'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Sparse Bayesian learning'],
    ['p', 'A generalised linear model on $M$ basis functions, $y(\\bf{x})=σ(\\sum_{m}{w_{m}φ_{m}(\\bf{x})})$, overfits if every weight is free. In sparse Bayesian learning each weight receives its own zero-mean Gaussian prior with precision (inverse variance) $α_{m}$. The $α_{m}$ are not fixed but estimated from the data by maximising the marginal likelihood (type-II maximum likelihood, the “evidence”; MacKay, 1992). Because the evidence penalises model complexity automatically (Occam’s razor), most $α_{m}$ go to infinity: the corresponding weights are exactly zero and their basis functions disappear. This automatic relevance determination (ARD) yields a sparse model without any regularisation constant to choose (Tipping, 2001; Bishop, 2006).'],
    ['h2', '2.2 Relevance vectors versus support vectors'],
    ['p', 'With one kernel function per training sample, the retained samples are the relevance vectors. Unlike support vectors, which lie on or inside the SVM margin (i.e. at the class boundary), relevance vectors tend to be *prototypical* samples away from the boundary; they are typically an order of magnitude fewer, the kernel need not satisfy Mercer’s condition, and the output is a probability rather than a distance to a hyperplane (Tipping, 2001). Their number is decided by the data, not by a penalty constant.'],
    ['h2', '2.3 Uncertainty'],
    ['p', 'Because the posterior over the weights is Gaussian (Laplace approximation), every prediction comes with a variance: the predictive uncertainty is low near the relevance vectors and high in parts of evidence space that the training data do not represent. The predictive probability is “moderated” towards ½ where the model is uncertain (MacKay, 1992). For exploration, the uncertainty map distinguishes *low prospectivity* (confidently barren) from *unknown* (evidence unlike any training sample).'],
    ['h2', '2.4 RVM in mineral prospectivity mapping'],
    ['p', 'Kernel methods have been used for MPM mainly as SVMs (Zuo and Carranza, 2011); the RVM keeps the advantages of kernels (non-linear boundaries, few samples) and adds probabilistic output, sparsity and automatic relevance determination, which address the small training sets and the need for interpretability typical of MPM. Training data, scaling and validation follow the same protocol as the other machine-learning methods of MPM Studio (Carranza et al., 2008; Zuo and Wang, 2020).'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Model and prior'],
    ['p', 'With $N$ training samples $\\bf{z}_{n}$ (scaled evidence) and labels $t_{n}∈\\{0,1\\}$, the basis matrix $Φ$ has rows $\\bf{φ}(\\bf{z})=[1,K(\\bf{z},\\bf{z}_{1}),…,K(\\bf{z},\\bf{z}_{N})]$ with the Gaussian kernel, or $[1,\\bf{z}]$ for the linear basis, and'],
    ['eq', 'p(\\bf{t}|\\bf{w})=\\prod_{n}{y_{n}^{t_{n}}(1-y_{n})^{1-t_{n}}},  y_{n}=σ(\\bf{φ}_{n}^{T}\\bf{w}),      p(\\bf{w}|α)=\\prod_{m}{\\cal{N}(w_{m}|0,α_{m}^{-1})}', 'rvm-model'],
    ['eq', 'K(\\bf{z},\\bf{z}\')=\\exp(-γ‖\\bf{z}-\\bf{z}\'‖^{2}),      γ_{0}=1/p', 'rvm-kernel'],
    ['h2', '3.2 Laplace approximation'],
    ['p', 'For fixed $α$, the posterior mode $\\bf{μ}$ maximises $\\ln p(\\bf{t}|\\bf{w})-\\frac{1}{2}\\bf{w}^{T}\\bf{A}\\bf{w}$ (Newton / IRLS); the posterior is approximated by a Gaussian with'],
    ['eq', 'Σ=(Φ^{T}\\bf{B}Φ+\\bf{A})^{-1},      \\bf{B}=\\rm{diag}(y_{n}(1-y_{n})),  \\bf{A}=\\rm{diag}(α)', 'rvm-laplace'],
    ['p', 'and the log evidence $\\ln p(\\bf{t}|α)≈\\ln p(\\bf{t}|\\bf{μ})-\\frac{1}{2}\\bf{μ}^{T}\\bf{A}\\bf{μ}+\\frac{1}{2}\\ln|\\bf{A}|-\\frac{1}{2}\\ln|Σ^{-1}|$. Tipping’s (2001) re-estimation is $α_{m}←γ_{m}/μ_{m}^{2}$ with $γ_{m}=1-α_{m}Σ_{mm}$.'],
    ['h2', '3.3 Fast marginal-likelihood maximisation'],
    ['p', 'Tipping and Faul (2003) showed that the evidence as a function of a single $α_{m}$ has a unique maximum in closed form. With the Gaussian approximation (pseudo-targets $\\hat{\\bf{t}}=Φ\\bf{μ}+\\bf{B}^{-1}(\\bf{t}-\\bf{y})$):'],
    ['eq', 'S_{m}=\\bf{φ}_{m}^{T}\\bf{B}\\bf{φ}_{m}-\\bf{φ}_{m}^{T}\\bf{B}ΦΣΦ^{T}\\bf{B}\\bf{φ}_{m},      Q_{m}=\\bf{φ}_{m}^{T}\\bf{B}\\hat{\\bf{t}}-\\bf{φ}_{m}^{T}\\bf{B}ΦΣΦ^{T}\\bf{B}\\hat{\\bf{t}}', 'rvm-sq'],
    ['eq', 's_{m}=\\frac{α_{m}S_{m}}{α_{m}-S_{m}},  q_{m}=\\frac{α_{m}Q_{m}}{α_{m}-S_{m}},      α_{m}^{*}=\\frac{s_{m}^{2}}{q_{m}^{2}-s_{m}} if q_{m}^{2}>s_{m}, else ∞', 'rvm-alpha'],
    ['p', '($s_{m}=S_{m}$, $q_{m}=Q_{m}$ for functions not in the model). The change of the log evidence for adding, re-estimating and deleting a function is'],
    ['eq', 'ΔL_{add}=\\frac{1}{2}[\\frac{Q_{m}^{2}-S_{m}}{S_{m}}+\\ln\\frac{S_{m}}{Q_{m}^{2}}],  ΔL_{re}=\\frac{1}{2}[\\frac{Q_{m}^{2}}{S_{m}+(1/α^{*}-1/α)^{-1}}-\\ln(1+S_{m}(1/α^{*}-1/α))],  ΔL_{del}=\\frac{1}{2}[\\frac{Q_{m}^{2}}{S_{m}-α_{m}}-\\ln(1-S_{m}/α_{m})]', 'rvm-dl'],
    ['p', 'Starting from the bias alone, the action with the largest $ΔL$ is applied, the Laplace posterior is updated, and the loop stops when no action increases the evidence by more than 10⁻⁶. Functions can re-enter after deletion.'],
    ['h2', '3.4 Prediction and uncertainty'],
    ['eq', 'a=\\bf{φ}(\\bf{z})^{T}\\bf{μ},  s^{2}=\\bf{φ}(\\bf{z})^{T}Σ\\bf{φ}(\\bf{z}),      P(\\bf{z})=σ(\\frac{a}{\\sqrt{1+πs^{2}/8}}),      s_{P}≈σ(a)(1-σ(a))s', 'rvm-pred'],
    ['p', 'With $R$ random non-deposit samples the score is the mean of $P$ and its s.d. over the samples is reported in addition to the mean predictive s.d. $s_{P}$. Importance: drop of the cross-validated AUC when a layer is permuted in the held-out folds.'],

    ['h1', '4 Algorithm'],
    ['steps', [
      '**Encoding and scaling** — feature matrix of the study cells; z-scores with study-area statistics.',
      '**Training samples** — training deposits + $r·n_{d}$ non-deposit cells (distance from deposits, even spacing).',
      '**Kernel width (sample 1, RBF)** — 5-fold cross-validated AUC for $γ=γ_{0}2^{k}$, $k=-6,-4,…,4$; best $γ$ (ties → smaller).',
      '**Cross-validation** — held-out probabilities → CV AUC, confusion matrix, permutation importance.',
      '**Evidence maximisation** — fast sequential algorithm [[rvm-sq]]–[[rvm-dl]] with the Laplace posterior [[rvm-laplace]].',
      '**Prediction** — $P$ and $s_{P}$ [[rvm-pred]] for every study cell in chunks (only the relevance vectors are evaluated).',
      '**Repeats and outputs** — mean score, s.d. over samples, mean predictive s.d.; relevance-vector point layer; tables; ROC / success-rate / P–A evaluation.',
    ]],
    ['table', {
      caption: 'Table 4.1. Implementation and verification.',
      head: ['Component', 'Implementation and verification'],
      widths: [3000, 6026],
      rows: [
        ['Laplace posterior', 'Own NumPy code (Newton with step halving, Cholesky)'],
        ['Fast evidence maximisation', 'Own code after Tipping and Faul (2003); converges to the same evidence, weights and retained functions as the original re-estimation $α←γ/μ^{2}$ (test, |Δ ln evidence| < 10⁻³), in a fraction of the time'],
        ['Moderated prediction', 'MacKay (1992); shrinks towards ½ (test)'],
        ['Sampling, split, evaluation', 'Shared MPM Studio modules'],
      ],
    }],

    ['h1', '5 How to use'],
    ['p', '*Data-driven ▸ Machine learning ▸ Relevance Vector Machine…*. Training data and evidence layers as for the SVM.'],
    ['table', {
      caption: 'Table 5.1. Training samples and model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2600, 1700, 4726],
      rows: [
        ['Non-deposit samples per deposit', '1', 'Ratio $r$; the RBF basis has one function per sample, so very large samples are slow.'],
        ['Min. distance / spacing', '4 cells / auto', 'Barren samples outside the halo of known deposits, evenly spread.'],
        ['Random samples (repeats)', '5', 'One RVM per draw; scores averaged.'],
        ['Cross-validation', '5-fold', 'Kernel-width tuning, CV AUC, importance.'],
        ['Basis functions', 'RBF', '*RBF on the training samples* (non-linear, relevance vectors = samples) or *linear* (relevance determination of the evidence columns).'],
        ['Tune the kernel width γ', 'on', 'Cross-validated choice of $γ$; off = the value below.'],
        ['γ (kernel width)', 'auto = 1/p', 'Small = smooth regional response; large = local response.'],
        ['Missing evidence', 'exclude', 'Exclude or impute.'],
        ['Outputs / prefix', 'score + relevance vectors + predictive s.d. + s.d. / rvm', ''],
      ],
    }],
    ['p', 'The results dialog shows the samples, the number of relevance vectors, the cross-validated AUC and the ROC AUC of the training and test deposits; tabs *Summary*, *Relevance vectors*, *Kernel width*, *Variable importance*, *Sample validation*, *Repeats*; *ROC / AUC…*, *Export tables (Excel)…*.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, `Table 6.${T()}. Evidence of the Carlin RVM.`),
    C.searchTable(J, `Table 6.${T()}. Carlin — candidate settings (3 repeats, spatial 70 / 30 splits).`),
    ['p', `The selected setting is **${J.selected.label}**. Over 10 independent splits it reaches a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}) with a training AUC of ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)}${SVJ ? ` (SVM: test ${f(SVJ.robust_summary.test_auc_mean)})` : ''}. The training and test AUCs are close: with only a few relevance vectors the model cannot memorise its training deposits.`],
    C.robustTable(J, `Table 6.${T()}. Carlin — robustness of the selected setting over 10 splits.`),
    C.curvesTable(J, `Table 6.${T()}. Carlin — reference run (seed 42, 30 % test deposits).`),
    ['img', J.figures.map, 520, 'Figure 6.1. Carlin — RVM prospectivity score (moderated probability, mean of 3 samples) with training (green) and test (blue) deposits.'],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC, (b) success / prediction rate, (c) P–A plot, (d) permutation importance.'],
    relTable(J, `Table 6.${T()}. Carlin — relevance vectors of the reference model (sample 1).`),
    ['p', relText(J, 'carlin')],
    ...(J.reference.tables.tuning && J.reference.tables.tuning.length ? [['table', {
      caption: `Table 6.${T()}. Carlin — kernel width (5-fold CV AUC, sample 1).`,
      head: ['γ', 'γ / γ0', 'CV AUC', ''], widths: [2000, 2000, 2500, 2526], rows: tuneRows(J),
    }]] : []),
    ['table', {
      caption: `Table 6.${T()}. Carlin — permutation importance.`,
      head: ['Rank', 'Layer', 'Columns', 'CV AUC drop', 'Linear weights'], widths: [800, 2226, 1000, 2000, 3000], rows: impRows(J),
    }],
    ['table', { caption: `Table 6.${T()}. Carlin — model summary.`, head: ['Item', 'Value'], widths: [4526, 4500], rows: sumRows(J) }],
    ['table', { caption: `Table 6.${T()}. Carlin — validation of the training samples (sample 1).`, head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J) }],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, `Table 6.${T()}. Evidence of the Curnamona RVM.`),
    C.searchTable(K, `Table 6.${T()}. Curnamona — candidate settings (leave-one-deposit-out).`),
    C.lodoTable(K, `Table 6.${T()}. Curnamona — leave-one-deposit-out results of the selected setting.`),
    C.curvesTable(K, `Table 6.${T()}. Curnamona — reference run with all seven occurrences.`),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — RVM prospectivity score with the seven REE occurrences.'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) permutation importance.'],
    relTable(K, `Table 6.${T()}. Curnamona — relevance vectors of the reference model.`),
    ['p', relText(K, 'curnamona')],
    ['table', {
      caption: `Table 6.${T()}. Curnamona — permutation importance.`,
      head: ['Rank', 'Layer', 'Columns', 'CV AUC drop', 'Linear weights'], widths: [800, 2226, 1000, 2000, 3000], rows: impRows(K),
    }],
    ['p', `With seven occurrences the selected setting reaches a mean LODO AUC of **${f(lodo(K))}**, and the importance ranks **${top(K).layer}** first. The Bayesian treatment is well suited to so few positives: the evidence prunes the model to the few samples the data can support, and the predictive s.d. shows where the occurrences give no information.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['table', {
      caption: `Table 6.${T()}. Ideal RVM parameters for the demo projects and their meaning.`,
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1600, 1400, 1400, 4626],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; silica index', 'One proxy per mineral-system component (pathway, trap, footprint; granite source, HFSE enrichment, exposure); z-scores give every proxy the same weight in the kernel distance.'],
        ['Basis functions', kName[P(J).kernel], kName[P(K).kernel], 'The RBF basis represents favourability as similarity to prototype deposits and barren cells — a natural expression of the analogue (“deposit-model”) reasoning of exploration: targets are cells whose combination of proxies resembles known deposits. The linear basis instead weighs each proxy independently and switches irrelevant ones off.'],
        ['Kernel width γ', P(J).kernel === 'rbf' ? g(st(J).gamma, 3) : '—', P(K).kernel === 'rbf' ? g(st(K).gamma, 3) : '—', gammaReason(J)],
        ['Non-deposit ratio', `1 : ${f(P(J).neg_ratio, 0)}`, `1 : ${f(P(K).neg_ratio, 0)}`, ratioReason(J)],
        ['Penalty', 'none (evidence)', 'none (evidence)', 'The Bayesian evidence decides how many prototypes the data support — no penalty constant has to be tuned on a few deposits, which removes one source of over-fitting compared with the SVM.'],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'Width of the alteration / geochemical halo: cells inside it belong to the mineral system and must not be barren prototypes (Cline et al., 2005; Elburg et al., 2012).'],
        ['Repeats', '3 (5 in the GUI)', '3 (5 in the GUI)', 'Different barren draws produce different barren prototypes; averaging represents the whole barren population.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** Mean test AUC on Carlin ${f(J.robust_summary.test_auc_mean)} (10 splits); mean LODO AUC on Curnamona ${f(lodo(K))}.`,
      `**Sparsity and interpretability.** ${st(J).relevance_vectors} relevance vectors (Carlin reference model) summarise the training data; each is a real location whose evidence can be inspected — prototypes that geologists can relate to the deposit model.`,
      '**Uncertainty.** The predictive s.d. map separates confidently barren ground from ground the training data do not represent — useful for planning where new data (geochemistry, mapping) would reduce uncertainty.',
      '**Score, not probability.** As for the other sample-based methods, the score depends on the deposit / barren ratio and ranks cells; it is not a probability per unit cell.',
      '**Cost.** The RBF basis has one candidate function per training sample; training is fast for hundreds of samples but grows with the cube of the active set.',
    ]],

    ['h1', '7 References'],
    ['refs', R('bishop2006', 'carranza2008', 'chung2003', 'cline2005', 'deepseam', 'delong1988', 'elburg2012', 'fawcett2006',
      'hofstra2000', 'mackay1992', 'muntean2011', 'neumann2000', 'raines2001', 'roberts2017', 'stevens2004',
      'tipping2001', 'tipping2003', 'yousefi2015', 'zuo2011', 'zuo2020', 'arcsdm')],
  ],
};
