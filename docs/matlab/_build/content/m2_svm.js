const { R, REF_NOTE } = require('../refs');
const { howto } = require('../howto');

module.exports = {
  file: 'M2_Support_Vector_Machine.docx',
  code: 'M2',
  short: 'Support vector machine',
  title: 'M2 Support Vector Machine (SVM) and Support Vector Regression (SVR)',
  subtitle: 'Supervised prospectivity mapping (classification) and grade estimation (regression)',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'The SVM was the first method built into the tool and served as the template for the plug-in contract (M0). Classification uses a soft-margin kernel SVM with Platt-calibrated probabilities; regression uses ε-insensitive support vector regression on a standardised response.'],
    ['table', {
      caption: 'Table 1. Files of the method.',
      head: ['File', 'Role'],
      widths: [3800, 5226],
      rows: [
        ['`classification/methods/clf_svm.m`', 'Classification plug-in (prospectivity probability)'],
        ['`regression/methods/reg_svm.m`', 'Regression plug-in (grade)'],
        ['`classification/run_svm_classification.m`\n`regression/run_svm_regression.m`', 'Run scripts'],
      ],
    }],

    ['h1', '2 Theory'],
    ['p', 'The support vector machine (Cortes and Vapnik, 1995; Vapnik, 1995) separates two classes by the hyperplane with the largest margin in a feature space induced by a kernel. Slack variables allow training errors at a cost $C$ (soft margin), and the solution depends only on the training samples on or inside the margin, the support vectors. Structural risk minimisation gives SVMs good generalisation from small training sets, which suits MPM, where only tens of deposits are known. Zuo and Carranza (2011) introduced the SVM as a tool for mapping mineral prospectivity, and it is one of the reference machine-learning methods in the comparison of Rodriguez-Galiano et al. (2015).'],
    ['p', 'The SVM output is a signed distance, not a probability. Platt (1999) fits a sigmoid to the outputs to obtain posterior probabilities, which are needed for a probability-scaled prospectivity map and for the Brier score. Support vector regression (Smola and Schölkopf, 2004) applies the same principle to real-valued targets with an ε-insensitive loss.'],

    ['h1', '3 Mathematical formulation'],
    ['h2', '3.1 Soft-margin classification'],
    ['p', 'With labels $y_{i}∈\\{-1,+1\\}$ (deposit = +1) and feature map $φ$, the primal problem is'],
    ['eq', '\\rm{min}_{\\bf{w},b,ξ}\\frac{1}{2}‖\\bf{w}‖^{2}+\\sum_{i=1}^{N}{C_{i}ξ_{i}}   \\txt{s.t.}   y_{i}(\\bf{w}^{\\rm{T}}φ(\\bf{x}_{i})+b)≥1-ξ_{i},  ξ_{i}≥0', 'svm-primal'],
    ['p', 'and its dual, solved in practice, is'],
    ['eq', '\\rm{max}_{α}\\sum_{i}{α_{i}}-\\frac{1}{2}\\sum_{i,j}{α_{i}α_{j}y_{i}y_{j}k(\\bf{x}_{i},\\bf{x}_{j})}   \\txt{s.t.}   0≤α_{i}≤C_{i},  \\sum_{i}{α_{i}y_{i}}=0', 'svm-dual'],
    ['p', 'giving the decision function'],
    ['eq', 'f(\\bf{x})=\\sum_{i∈\\rm{SV}}{α_{i}y_{i}k(\\bf{x}_{i},\\bf{x})}+b', 'svm-f'],
    ['p', 'The Gaussian (RBF) kernel is used by default (linear and polynomial kernels are available):'],
    ['eq', 'k(\\bf{x},\\bf{z})=\\rm{exp}(-‖\\bf{x}-\\bf{z}‖^{2}/s^{2}),      s=κ\\sqrt{P}', 'svm-k'],
    ['p', 'Because the features are z-scored (M0), the squared distance between two cells grows with the number of features $P$. Expressing the kernel width as a multiplier $κ$ of $\\sqrt{P}$ keeps the grid meaningful for any $P$; $κ=1$ is the common heuristic $γ=1/s^{2}=1/P$. With `BalanceClasses`, a uniform class prior rescales the per-sample box constraints $C_{i}$ so that both classes carry equal total weight.'],
    ['h2', '3.2 Platt probabilities'],
    ['p', 'The probability of the deposit class is a sigmoid of the SVM output (Platt, 1999):'],
    ['eq', 'P(y=1|f)=\\frac{1}{1+\\rm{exp}(Af+B)}', 'platt'],
    ['p', 'with $A,B$ fitted by maximum likelihood to cross-validated outputs (MATLAB `fitPosterior`, 10-fold by default), so the sigmoid is not fitted to over-confident training outputs. If the classes are perfectly separated, `fitPosterior` returns a step function instead.'],
    ['h2', '3.3 ε-insensitive support vector regression'],
    ['p', 'The response (model scale, e.g. log10 grade) is z-scored inside the method, $\\tilde{y}=(y-\\bar{y})/s_{y}$, so that $C$ and ε do not depend on the grade units. The primal problem is'],
    ['eq', '\\rm{min}\\frac{1}{2}‖\\bf{w}‖^{2}+C\\sum_{i}{(ξ_{i}+ξ_{i}^{*})}   \\txt{s.t.}   |\\tilde{y}_{i}-f(\\bf{x}_{i})|≤ε+ξ_{i}^{(*)}', 'svr'],
    ['p', 'with $f(\\bf{x})=\\sum_{i}{(α_{i}-α_{i}^{*})k(\\bf{x}_{i},\\bf{x})}+b$ (Smola and Schölkopf, 2004). The prediction is transformed back as $\\hat{y}=\\bar{y}+s_{y}f(\\bf{x})$; ε is therefore a fraction of the response standard deviation.'],

    ['h1', '4 Implementation'],
    ['h2', '4.1 MATLAB library or own code'],
    ['table', {
      caption: 'Table 2. Source of each component.',
      head: ['Component', 'Implementation'],
      widths: [3300, 5726],
      rows: [
        ['Dual QP [[svm-dual]], decision function [[svm-f]]', 'MATLAB `fitcsvm` (SMO solver with second-order working-set selection; Fan et al., 2005), `CacheSize = maximal`, `Standardize = false` (features already z-scored)'],
        ['Platt sigmoid [[platt]]', 'MATLAB `fitPosterior` (final model only)'],
        ['SVR [[svr]]', 'MATLAB `fitrsvm`'],
        ['Kernel-width convention [[svm-k]], response standardisation, class prior', 'Own code (`clf_svm.m`, `reg_svm.m`)'],
        ['Tuning, validation, maps', 'Own shared code (M0)'],
      ],
    }],
    ['p', 'During cross-validation the raw SVM output $f(\\bf{x})$ is used, because AUC only needs a ranking and calibration inside every fold would multiply the cost. The probability calibration [[platt]] is applied to the final model only.'],
    ['h2', '4.2 Hyper-parameters'],
    ['table', {
      caption: 'Table 3. Hyper-parameters.',
      head: ['Parameter', 'Default', 'CV grid', 'Meaning'],
      widths: [2100, 1200, 2400, 3326],
      rows: [
        ['`KernelFunction`', 'rbf', '—', 'rbf (gaussian), linear or polynomial'],
        ['`BoxConstraint`', '1', 'clf [0.1 1 10 100]\nreg [0.3 1 3 10]', '$C$'],
        ['`KernelScale`', '1', 'clf [0.25 0.5 1 2 4]\nreg [0.5 1 2 4]', 'Multiplier $κ$ in [[svm-k]]'],
        ['`Epsilon` (reg)', '0.1', '[0.05 0.2]', 'Tube half-width in response SD'],
        ['`BalanceClasses` (clf)', 'true', '—', 'Uniform class prior'],
        ['`Calibrate` (clf)', 'true', '—', 'Platt probabilities for the final model'],
      ],
    }],
    ['h2', '4.3 Departures from the original method'],
    ['bullets', [
      'None in the SVM itself. The kernel width is parameterised relative to $\\sqrt{P}$, and the regression response is standardised; both only change the scale of the hyper-parameters.',
    ]],

    ['h1', '5 Verification and demo results'],
    ['table', {
      caption: 'Table 4. Demo results.',
      head: ['Task / data', 'Result'],
      widths: [3000, 6026],
      rows: [
        ['Classification, Carlin Au, seed 42', 'Test AUC 0.950 (samples) and 0.947 (map). The P–A threshold captures 92 % of the test deposits in 13.6 % of the area.'],
        ['Classification, Carlin Au, 5 random splits (seeds 1–5)', 'Test AUC 0.906 ± 0.066 (samples), 0.889 ± 0.068 (map); CV AUC 0.922 ± 0.020; P–A target 6.4 ± 3.9 % of the area capturing 72 ± 15 % of the test deposits; 14 s per run.'],
        ['Regression, Carlin Th, seed 1', 'Test R² 0.27, RMSE 0.178 log10 units, cut-off AUC 0.79.'],
        ['Regression, Curnamona La, 3 splits', 'Test R² 0.145 ± 0.046, cut-off AUC 0.675 ± 0.055 (weak stand-in target).'],
      ],
    }],
    ['p', 'The 5-split comparison with gradient boosting is reported in M5.'],

    ...howto({ code: 'M2', tasks: ['clf', 'reg'], scripts: { clf: 'run_svm_classification.m', reg: 'run_svm_regression.m' }, key: { clf: 'clf_svm', reg: 'reg_svm' },
    params: ["% cfg.tune.grid = struct('BoxConstraint', [0.1 1 10 100], 'KernelScale', [0.25 0.5 1 2 4]);", "% cfg.params.KernelFunction = 'linear';"],
    notes: ['The kernel width is relative to the number of features ($κ\\sqrt{P}$); after adding or removing layers the same grid remains valid.', 'Highly correlated layers do not harm the SVM, but redundant layers dilute distances in the Gaussian kernel; prefer a compact evidence set.'],
    checks: ['The log reports the number of support vectors: close to the number of training samples indicates a too-narrow kernel or too small C.', 'Probabilities come from `fitPosterior`; if the classes are separable the probabilities become 0/1 (step function) — then judge the map by ranking (AUC, P–A), not by the 0.5 threshold.'] }),

    ['h1', 'References'],
    ['refs', R('cortes1995', 'vapnik1995', 'zuo2011', 'rodriguez2015', 'platt1999', 'smola2004', 'fan2005')],
    ['note', REF_NOTE],
  ],
};
