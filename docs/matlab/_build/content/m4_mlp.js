const { R, REF_NOTE } = require('../refs');
const { howto } = require('../howto');

module.exports = {
  file: 'M4_Multilayer_Perceptron.docx',
  code: 'M4',
  short: 'Neural network (MLP)',
  title: 'M4 Artificial Neural Network: Multilayer Perceptron (MLP) Ensemble',
  subtitle: 'Supervised prospectivity mapping (classification) and grade estimation (regression)',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'The neural-network method is a fully connected feed-forward network (multilayer perceptron) whose gradients come from back-propagation. Before implementing it, the classic MPM recipe (gradient-descent back-propagation with momentum and early stopping) was reviewed against small-data practice. The tool keeps the network but changes the training set-up: full-batch **L-BFGS** optimisation, **L2 weight decay** tuned by cross-validation instead of early stopping, and an **ensemble** of networks whose spread gives an uncertainty map.'],
    ['table', {
      caption: 'Table 1. Files of the method.',
      head: ['File', 'Role'],
      widths: [3800, 5226],
      rows: [
        ['`classification/methods/clf_mlp.m`', 'Classification plug-in (mean probability, ensemble SD)'],
        ['`regression/methods/reg_mlp.m`', 'Regression plug-in (mean grade, ensemble SD)'],
        ['`classification/run_mlp_classification.m`\n`regression/run_mlp_regression.m`', 'Run scripts'],
      ],
    }],

    ['h1', '2 Theory'],
    ['p', 'A multilayer perceptron maps the evidence vector through one or more hidden layers of nonlinear units to an output; all weights are learned by minimising a loss whose gradients are computed by back-propagation (Rumelhart et al., 1986). Feed-forward networks trained by back-propagation were among the first machine-learning methods in MPM: Singer and Kouda (1996) used one to search for Kuroko deposits, and Porwal et al. (2003) mapped mineral potential in the Aravalli Province, India. These studies trained by gradient descent with a learning rate and momentum and stopped training on a validation set.'],
    ['p', 'With a few tens of deposits, that recipe has two weaknesses. The learning rate and momentum must be tuned and the result varies between runs, and early stopping removes deposits from the training set to build the validation set. Three established alternatives are therefore used:'],
    ['bullets', [
      '**Quasi-Newton optimisation.** L-BFGS (Liu and Nocedal, 1989) uses curvature information from recent gradients, needs no learning rate or momentum, and converges reliably on full batches of a few hundred samples.',
      '**Weight decay.** An L2 penalty on the weights regularises the network; its strength is tuned by cross-validation together with the layer sizes, so every deposit stays in training.',
      '**Ensembles.** Averaging networks trained from different random initialisations removes most of the run-to-run variance (Hansen and Salamon, 1990). The spread of the members is a simple, well-performing estimate of model uncertainty (Lakshminarayanan et al., 2017).',
    ]],

    ['h1', '3 Mathematical formulation'],
    ['h2', '3.1 Network'],
    ['p', 'For hidden layers $l=1,…,L$ with activation $g$ (tanh by default; ReLU or logistic sigmoid are options):'],
    ['eq', '\\bf{h}^{(0)}=\\bf{x},      \\bf{h}^{(l)}=g(\\bf{W}^{(l)}\\bf{h}^{(l-1)}+\\bf{b}^{(l)})', 'mlp-fwd'],
    ['p', 'The output layer is a softmax over the two classes for classification, which for two classes equals a logistic sigmoid of the difference of the two output activations, and a single linear unit for regression.'],
    ['h2', '3.2 Loss'],
    ['p', 'Classification minimises the weighted cross-entropy plus an L2 penalty on the weights $\\bf{θ}$:'],
    ['eq', 'J(\\bf{θ})=-\\sum_{n}{c_{n}[y_{n}\\rm{log} p_{n}+(1-y_{n})\\rm{log}(1-p_{n})]}+\\frac{λ}{2}‖\\bf{θ}‖^{2}', 'mlp-ce'],
    ['p', 'with class-balancing weights $c_{n}$ (deposits and background count equally). Regression minimises the mean squared error of the z-scored response $\\tilde{y}=(y-\\bar{y})/s_{y}$ plus the same penalty. The relative scaling of the data term and the penalty follows MATLAB `fitcnet` / `fitrnet` (parameter `Lambda`).'],
    ['h2', '3.3 Optimisation'],
    ['p', 'Gradients $∇J$ are obtained by back-propagation. L-BFGS updates $\\bf{θ}_{t+1}=\\bf{θ}_{t}-η_{t}\\bf{H}_{t}∇J(\\bf{θ}_{t})$, where $\\bf{H}_{t}$ is a limited-memory approximation of the inverse Hessian built from recent pairs $(\\bf{θ}_{t+1}-\\bf{θ}_{t},∇J_{t+1}-∇J_{t})$ and $η_{t}$ comes from a line search (Liu and Nocedal, 1989). Training stops after at most 1000 iterations or when the gradient, loss or step tolerances are met.'],
    ['h2', '3.4 Ensemble prediction and uncertainty'],
    ['p', 'For $K$ networks trained from different random initialisations (optionally on bootstrap samples):'],
    ['eq', '\\bar{p}(\\bf{x})=\\frac{1}{K}\\sum_{k=1}^{K}{p_{k}(\\bf{x})},      u(\\bf{x})=\\sqrt{\\frac{1}{K-1}\\sum_{k=1}^{K}{{(p_{k}(\\bf{x})-\\bar{p}(\\bf{x}))}^{2}}}', 'mlp-ens'],
    ['p', 'For regression the member predictions are transformed back, $\\hat{y}_{k}=\\bar{y}+s_{y}f_{k}(\\bf{x})$, and the same mean and standard deviation are taken. $u(\\bf{x})$ is model (epistemic) uncertainty only; it does not include sample noise.'],

    ['h1', '4 Implementation'],
    ['h2', '4.1 MATLAB library or own code'],
    ['table', {
      caption: 'Table 2. Source of each component.',
      head: ['Component', 'Implementation'],
      widths: [3300, 5726],
      rows: [
        ['Network, back-propagation, loss [[mlp-ce]], L-BFGS', 'MATLAB `fitcnet` (classification) and `fitrnet` (regression), Statistics and Machine Learning Toolbox; Glorot-initialised weights (toolbox default), `Standardize = false`'],
        ['Class weights, response standardisation', 'Own code (`Weights` argument; z-scoring in `reg_mlp.m`)'],
        ['Ensemble, bagging option, uncertainty [[mlp-ens]]', 'Own code (`clf_mlp.m`, `reg_mlp.m`)'],
        ['Tuning, validation, maps', 'Own shared code (M0)'],
      ],
    }],
    ['p', 'Bayesian-regularised Levenberg–Marquardt (`trainbr`, Deep Learning Toolbox) was also considered. It suits small regressions but has no cross-entropy loss for classification, whereas the ensemble gives uncertainty for both tasks.'],
    ['h2', '4.2 Hyper-parameters'],
    ['table', {
      caption: 'Table 3. Hyper-parameters (classification / regression).',
      head: ['Parameter', 'Default', 'CV grid', 'Meaning'],
      widths: [2000, 1300, 2300, 3426],
      rows: [
        ['`LayerSizes`', '10', '{5, 10, 20}', 'Hidden-layer widths; a vector such as [20 10] gives a deeper net'],
        ['`Lambda`', '0.01', 'clf [1e-4 1e-3 1e-2 0.1 1]\nreg [0.01 0.1 1]', 'L2 strength $λ$'],
        ['`Activation`', 'tanh', '—', 'tanh, relu or sigmoid'],
        ['`NumNets`', '10', '—', 'Ensemble size $K$ of the final model'],
        ['`NumNetsTuning`', '3 / 2', '—', 'Ensemble size inside CV'],
        ['`Bootstrap`', 'false', '—', 'Train each member on a bootstrap sample (both classes kept)'],
        ['`BalanceClasses` (clf)', 'true', '—', 'Weights $c_{n}$ in [[mlp-ce]]'],
        ['`IterationLimit`', '1000', '—', 'Maximum L-BFGS iterations'],
      ],
    }],
    ['p', 'The regression grid starts at $λ=0.01$: smaller values need many more L-BFGS iterations and overfit the noisy grade data.'],
    ['h2', '4.3 Departures from the classic MPM recipe'],
    ['table', {
      caption: 'Table 4. Classic set-up and the set-up used here.',
      head: ['Aspect', 'Classic back-propagation MPM (Singer and Kouda, 1996; Porwal et al., 2003)', 'This tool'],
      widths: [1800, 3600, 3626],
      rows: [
        ['Optimiser', 'Gradient descent with learning rate and momentum', 'L-BFGS (no learning rate)'],
        ['Regularisation', 'Early stopping on a validation subset', 'L2 weight decay tuned by CV; all deposits used for training'],
        ['Variance between runs', 'Single network', 'Ensemble of $K$ networks'],
        ['Uncertainty', '—', 'Ensemble standard deviation [[mlp-ens]]'],
      ],
    }],

    ['h1', '5 Verification and demo results'],
    ['table', {
      caption: 'Table 5. Demo results.',
      head: ['Task / data', 'Result'],
      widths: [3000, 6026],
      rows: [
        ['Classification, Carlin Au, seed 42', 'Selected LayerSizes 20, λ = 0.1 (CV AUC 0.847); 10 networks converged in 33–51 L-BFGS iterations. Test AUC 0.940 (samples) and 0.940 (map). The P–A target covers 12.5 % of the area and captures 88 % of the training and 90 % of the test deposits; the C–A target covers 3.7 % and captures 50 % of the test deposits.'],
        ['Regression', 'Runs on Carlin Th and Curnamona La (including two hidden layers, [20 10]) completed during development; their metrics were not archived.'],
      ],
    }],

    ...howto({ code: 'M4', tasks: ['clf', 'reg'], scripts: { clf: 'run_mlp_classification.m', reg: 'run_mlp_regression.m' }, key: { clf: 'clf_mlp', reg: 'reg_mlp' },
    params: ["cfg.params.NumNets = 10;                         % ensemble size", "% cfg.tune.grid = struct('LayerSizes', {{10, [20 10]}}, 'Lambda', [1e-3 1e-2 1e-1]);"],
    notes: ['Inputs are z-scored by the pipeline; very skewed continuous layers (e.g. geochemistry) should be log-transformed during pre-processing.'],
    checks: ['The log reports the L-BFGS iterations per network; if they reach `IterationLimit`, raise it or increase `Lambda`.', 'The uncertainty map (ensemble SD) should be high in areas with evidence combinations unlike the training samples.'] }),

    ['h1', 'References'],
    ['refs', R('rumelhart1986', 'singer1996', 'porwal2003', 'liu1989', 'hansen1990', 'lakshmi2017', 'glorot2010')],
    ['note', REF_NOTE],
  ],
};
