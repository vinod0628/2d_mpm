const { R, REF_NOTE } = require('../refs');
const { howto } = require('../howto');

module.exports = {
  file: 'M1_Random_Forest.docx',
  code: 'M1',
  short: 'Random forest',
  title: 'M1 Random Forest (RF)',
  subtitle: 'Supervised prospectivity mapping (classification) and grade estimation (regression)',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Random forest is the first machine-learning method of the MATLAB MPM tool. It is used for both tasks of the tool: **target delineation** (classification of deposit vs. background cells, giving a prospectivity map) and **grade estimation** (regression of a grade variable). The shared pipeline (data loading, sampling, tuning, validation and outputs) is described in the M0 document; this record covers only the method.'],
    ['table', {
      caption: 'Table 1. Files of the method.',
      head: ['File', 'Role'],
      widths: [3800, 5226],
      rows: [
        ['`classification/methods/clf_rf.m`', 'Classification plug-in (prospectivity)'],
        ['`regression/methods/reg_rf.m`', 'Regression plug-in (grade) with uncertainty map'],
        ['`classification/run_rf_classification.m`\n`regression/run_rf_regression.m`', 'Run scripts (demo: Carlin Au / Carlin Th)'],
      ],
    }],

    ['h1', '2 Theory'],
    ['p', 'A random forest is an ensemble of classification or regression trees (Breiman et al., 1984) grown on bootstrap samples of the training data (bagging; Breiman, 1996). At every node, the best split is searched only among a random subset of $m$ of the $P$ predictors (Breiman, 2001). Bootstrap sampling and the random predictor subsets decorrelate the trees, so averaging them reduces variance without increasing bias much. Each tree is grown deep and is not pruned. The observations left out of a tree’s bootstrap sample (about 37 %, the out-of-bag or OOB samples) give an internal error estimate and a permutation measure of predictor importance (Breiman, 2001).'],
    ['p', 'In mineral prospectivity mapping, RF has been shown to perform well with few known deposits and with missing evidence values (Carranza and Laborte, 2015). It was the most accurate and stable of four machine-learning methods (neural networks, random forest, regression trees and support vector machines) in the comparison of Rodriguez-Galiano et al. (2015). It needs little tuning, handles mixed continuous and one-hot categorical evidence, and is insensitive to monotone transformations of the predictors.'],

    ['h1', '3 Mathematical formulation'],
    ['h2', '3.1 Ensemble prediction'],
    ['p', 'Let $D={\\{(\\bf{x}_{i},y_{i})\\}}_{i=1}^{N}$ be the training set with $\\bf{x}_{i}∈ℝ^{P}$. For $b=1,…,B$, a bootstrap sample $D_{b}$ of size $N$ is drawn with replacement from $D$ and a tree $T_{b}$ is grown on it. For classification ($y∈\\{0,1\\}$), the prospectivity score of a cell $\\bf{x}$ is the mean class-1 proportion of the leaves reached in the trees:'],
    ['eq', '\\hat{p}(\\bf{x})=\\frac{1}{B}\\sum_{b=1}^{B}{p_{b}(\\bf{x})}', 'rf-clf'],
    ['p', 'where $p_{b}(\\bf{x})$ is the (prior-weighted) fraction of deposit samples in the leaf of $T_{b}$ that contains $\\bf{x}$. With the default minimum leaf size of one, the leaves are pure and [[rf-clf]] equals the fraction of trees voting “deposit”. For regression, the prediction and its spread across the trees are'],
    ['eq', '\\hat{y}(\\bf{x})=\\frac{1}{B}\\sum_{b=1}^{B}{T_{b}(\\bf{x})},      s(\\bf{x})=\\sqrt{\\frac{1}{B-1}\\sum_{b=1}^{B}{{(T_{b}(\\bf{x})-\\hat{y}(\\bf{x}))}^{2}}}', 'rf-reg'],
    ['p', 'The spread $s(\\bf{x})$ is written as the uncertainty map of the regression. It measures the disagreement between trees (model uncertainty); it is not a calibrated prediction interval, for which quantile regression forests (Meinshausen, 2006) would be needed.'],
    ['h2', '3.2 Tree growing'],
    ['p', 'At each node $t$, $m$ predictors are drawn at random and the split $(j,c)$ that sends $x_{j}≤c$ to the left child $t_{L}$ and the rest to $t_{R}$ is chosen to maximise the impurity decrease'],
    ['eq', '\\Delta I(j,c)=I(t)-\\frac{N_{L}}{N_{t}}I(t_{L})-\\frac{N_{R}}{N_{t}}I(t_{R})', 'rf-split'],
    ['p', 'with the Gini index $I(t)=1-\\sum_{k}{p_{tk}^{2}}$ for classification and the mean squared error $I(t)=\\frac{1}{N_{t}}\\sum_{i∈t}{{(y_{i}-\\bar{y}_{t})}^{2}}$ for regression. Splitting stops when a node holds fewer than $2n_{\\rm{min}}$ samples or is pure, where $n_{\\rm{min}}$ is the minimum leaf size. The number of predictors sampled per split follows Breiman’s defaults, scaled by a tuned multiplier $k$:'],
    ['eq', 'm_{\\rm{clf}}=\\rm{round}(k\\sqrt{P}),      m_{\\rm{reg}}=\\rm{round}(kP/3),      1≤m≤P', 'rf-mtry'],
    ['h2', '3.3 Class balance'],
    ['p', 'Deposits and background cells are sampled 1:1 by default (M0), but buffered deposits or other ratios make the classes unequal. The option `BalanceClasses` sets a uniform class prior, $π_{0}=π_{1}=1/2$, so both classes carry the same total weight in the impurity and in the leaf proportions of [[rf-clf]].'],
    ['h2', '3.4 Out-of-bag permutation importance'],
    ['p', 'For tree $b$, let $e_{b}$ be its error on its OOB samples and $e_{b}^{(j)}$ the error after the values of predictor $j$ are permuted among those samples. The importance of predictor $j$ is the mean increase divided by its standard deviation over the trees (Breiman, 2001; MATLAB `OOBPermutedPredictorDeltaError`):'],
    ['eq', '\\rm{VI}_{j}=\\frac{\\bar{d}_{j}}{σ_{d_{j}}},      d_{bj}=e_{b}^{(j)}-e_{b}', 'rf-vi'],
    ['p', 'This native importance is reported per feature column (each one-hot geology class separately). The pipeline also computes a model-agnostic permutation importance per evidence layer on the test samples (M0).'],

    ['h1', '4 Implementation'],
    ['h2', '4.1 MATLAB library or own code'],
    ['table', {
      caption: 'Table 2. Source of each component.',
      head: ['Component', 'Implementation'],
      widths: [3300, 5726],
      rows: [
        ['Bootstrap, tree growing (CART, Gini / MSE), random predictor subsets, OOB', 'MATLAB library: `TreeBagger` (Statistics and Machine Learning Toolbox) — a direct implementation of Breiman (2001)'],
        ['Class prior', '`TreeBagger` option `Prior = uniform`'],
        ['Regression spread [[rf-reg]]', '`TreeBagger.predict`, second output (standard deviation over the trees)'],
        ['Native importance [[rf-vi]]', '`TreeBagger` with `OOBPredictorImportance = on`'],
        ['Predictor-count multiplier [[rf-mtry]]', 'Own code (`clf_rf.m`, `reg_rf.m`)'],
        ['Hyper-parameter tuning, validation, maps', 'Own shared code (`mpm_tune_cv`, pipeline; M0)'],
      ],
    }],
    ['p', 'No own algorithm was needed: `TreeBagger` follows the original algorithm (bootstrap samples, unpruned trees, a random predictor subset at every split, OOB error and importance). The plug-in only maps the tool’s parameters onto it. During tuning, the forest is grown with fewer trees and without OOB bookkeeping, then compacted to save memory.'],
    ['h2', '4.2 Hyper-parameters'],
    ['table', {
      caption: 'Table 3. Hyper-parameters (classification / regression).',
      head: ['Parameter', 'Default', 'CV grid', 'Meaning'],
      widths: [2300, 1300, 1800, 3626],
      rows: [
        ['`NumTrees`', '300', '—', 'Trees $B$ of the final model'],
        ['`NumTreesTuning`', '100', '—', 'Trees inside cross-validation'],
        ['`MinLeafSize`', '1 / 5', '[1 3 5] / [1 3 5 10]', 'Minimum samples per leaf $n_{\\rm{min}}$'],
        ['`NumPredictorsToSample`', '1', '[0.5 1 2]', 'Multiplier $k$ in [[rf-mtry]]'],
        ['`BalanceClasses`', 'true', '—', 'Uniform class prior (classification only)'],
        ['`UseParallel`', 'false', '—', 'Use an open parallel pool'],
      ],
    }],
    ['p', 'The grid is searched by 5-fold cross-validation, maximising AUC for classification and minimising RMSE for regression. Folds are stratified by class and grouped by deposit when buffer cells are used (M0).'],
    ['h2', '4.3 Departures from the original method'],
    ['bullets', [
      'None in the algorithm. The parameter $m$ is expressed as a multiplier of Breiman’s defaults ($\\sqrt{P}$ for classification, $P/3$ for regression), so the same grid works for any number of features.',
      'The regression uncertainty is the tree spread [[rf-reg]]. It is reported as model uncertainty, not as a predictive interval.',
    ]],

    ['h1', '5 Verification and demo results'],
    ['p', 'The data and protocol are described in M0. In the Carlin demo, 24 training and 10 test deposits are each matched by the same number of random background cells.'],
    ['table', {
      caption: 'Table 4. Demo results (single split, seed 42).',
      head: ['Task / data', 'Result'],
      widths: [3000, 6026],
      rows: [
        ['Classification, Carlin Au', 'Test AUC 0.960 (samples), 0.946 (map: test deposits vs. all cells); CV AUC 0.861. Prediction of the 0.91 M-cell map with 300 trees takes about 2.5–3 min.'],
        ['Regression, Curnamona La (early stand-in target)', 'Test R² 0.07–0.10 and cut-off AUC 0.67–0.68 in two runs, against a training R² of 0.84–0.91. The La samples are only weakly predictable from the layers (M0).'],
      ],
    }],
    ['p', 'A repeated-split comparison of RF with the other methods has not been run yet; the numbers above are single demo runs and should not be used to rank methods.'],

    ...howto({ code: 'M1', tasks: ['clf', 'reg'], scripts: { clf: 'run_rf_classification.m', reg: 'run_rf_regression.m' }, key: { clf: 'clf_rf', reg: 'reg_rf' },
    params: ["cfg.params.NumTrees = 500;                       % more trees (final model)", "% cfg.tune.grid = struct('MinLeafSize', [1 3 5], 'NumPredictorsToSample', [0.5 1 2]);"],
    notes: ['No feature scaling is needed by the trees, but the pipeline z-scores anyway (harmless).', 'With very few deposits (< 10) use `cfg.positives.buffer_cells = 1` and more negatives (`cfg.negatives.ratio = 2`), as in Carranza and Laborte (2015).'],
    checks: ['The importance table `importance_native.csv` (OOB permutation, per feature column) should agree broadly with the layer permutation importance.', 'Regression: the uncertainty map (tree spread) should be high where samples are sparse.', 'Map prediction with 300 trees takes a few minutes for ~1 M cells; this is normal.'] }),

    ['h1', 'References'],
    ['refs', R('breiman1984', 'breiman1996', 'breiman2001', 'carranza2015', 'rodriguez2015', 'meinshausen2006')],
    ['note', REF_NOTE],
  ],
};
