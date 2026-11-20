const { R } = require('../refs');
const C = require('../common');

const { f, int, pm } = C;
const J = C.load('random_forest_carlin');
const K = C.load('random_forest_curnamona');

const cfgOf = (X) => Object.assign({ n_trees: 300, max_features: 'sqrt', min_samples_leaf: 1, neg_ratio: 1, repeats: 3 }, X.selected);
const cj = cfgOf(J), ck = cfgOf(K);
const impRows = (X) => X.reference.tables.importance.map((r) => [String(r.rank), r.layer, f(r.impurity_pct, 1),
  pm(r.permutation_auc_drop, r.permutation_sd, 3)]);
const svRows = (X) => X.reference.tables.sample_validation.map((r) => [r.item, r.value]);
const sensRows = (X) => X.sensitivity.map((s) => [String(s.n_trees), f(s.oob_auc), f(s.train_auc), s.test_auc == null ? '—' : f(s.test_auc), f(s.seconds, 1)]);
const top = (X) => X.reference.tables.importance[0];
const mfName = { sqrt: '√p', log2: 'log₂ p', third: 'p / 3', all: 'all p (bagging)' };

function leafReason(v) {
  return v > 1
    ? `A minimum of ${v} samples per leaf forces every terminal "favourable domain" to contain several training samples, so a rule cannot be built around one deposit. Geologically this asks the forest to describe the district-scale mineral system (the combination of host, structure and geochemical halo shared by several deposits) instead of the local signature of a single occurrence.`
    : 'One sample per leaf (fully grown trees, Breiman’s default) was best: averaging hundreds of de-correlated trees already removes the variance of the individual deep trees, and the deposits of each project share a consistent multivariate signature, so the forest gains from resolving it in detail.';
}
function ratioReason(X, v) {
  const d = X.dataset_info;
  return `1 : ${f(v, 0)} (deposit : non-deposit). Ore-forming systems are rare — one deposit cell per ${int(d.study_km2 / d.deposit_cells)} km² — so the barren ground is far more varied than the mineralised ground (all lithologies, unmineralised faults, background geochemistry). ${v > 1 ? 'Several non-deposit samples per deposit let the forest learn this variety; the gain flattens beyond this ratio and larger samples start to outweigh the deposits in every bootstrap sample.' : 'With an equal number the classes are balanced in every bootstrap sample (Carranza and Laborte, 2015); larger ratios did not improve the held-out ranking here.'}`;
}

module.exports = {
  file: 'random_forest.docx',
  code: 'RF',
  short: 'Random forest',
  title: 'Random Forest (RF) for Mineral Prospectivity Mapping',
  subtitle: 'Data-driven machine-learning method of MPM Studio — theory, mathematics, algorithm, use and demo results',
  blocks: [
    ['h1', '1 Overview'],
    ['p', 'Random forest (Breiman, 2001) is the first machine-learning method of MPM Studio (menu *Data-driven ▸ Machine learning ▸ Random Forest*). It learns the multivariate signature of known deposits from evidence rasters, using the deposit cells as positive and randomly drawn non-deposit cells as negative training samples, and returns a prospectivity score map, a model-uncertainty map, out-of-bag (OOB) validation statistics and the importance of every evidence layer. It needs little tuning, accepts continuous and categorical evidence together, is insensitive to monotone transformations, captures interactions and threshold effects without specification, and is robust with few deposits (Carranza and Laborte, 2015; Rodriguez-Galiano et al., 2015; McKay and Harris, 2016).'],
    ['table', {
      caption: 'Table 1.1. Source files.',
      head: ['File', 'Role'],
      widths: [4000, 5026],
      rows: [
        ['`python/logic/data_driven/random_forest/model.py`', 'Algorithm (no GUI): training samples, forests, tree-vote maps, OOB, cross-validation, importance'],
        ['`python/logic/data_driven/random_forest/plugin.py`', 'Registration; run in the worker thread; outputs, tables, training / test points'],
        ['`python/logic/data_driven/random_forest/dialogs.py`', 'Parameter dialog and results dialog'],
        ['`python/logic/common/features.py`', 'Evidence → feature matrix (continuous values, one-hot classes, missing data) shared by the ML methods'],
        ['`python/logic/common/training.py`', 'Train / test split, thinning, non-deposit sampling, training / test point layers'],
        ['`python/evaluation/metrics.py`', 'ROC / AUC (DeLong CI), success-rate and P–A curves'],
        ['`python/tests/test_random_forest.py`', 'Unit tests (vote accumulation = scikit-learn, importance, OOB / CV, reproducibility, Carlin model)'],
      ],
    }],

    ['h1', '2 Fundamental theory'],
    ['h2', '2.1 Classification trees'],
    ['p', 'A classification tree (CART; Breiman et al., 1984) partitions the evidence space recursively into rectangular regions by binary splits of the form $x_{j}≤c$. At each node the split that makes the two child nodes as pure as possible in the deposit / non-deposit labels is chosen; the leaves store the class proportions of the training samples they contain. Trees capture non-linear responses and interactions (for example “anomalous Sb *and* favourable host”) without any functional form being specified, and they are invariant to monotone transformations of the evidence. Their weakness is high variance: small changes of the training data change the tree structure strongly.'],
    ['h2', '2.2 Bagging and the random subspace'],
    ['p', 'Bagging (Breiman, 1996) grows many trees on bootstrap samples of the training data and averages them. If $B$ identically distributed trees with variance $σ^{2}$ have pairwise correlation $ρ$, the variance of their average is $ρσ^{2}+(1-ρ)σ^{2}/B$ [[rf-var]]: averaging removes the second term, but the first term remains as long as the trees are correlated. Random forest (Breiman, 2001) de-correlates the trees further by trying only a random subset of $m$ of the $p$ evidence columns at every split, so strong proxies cannot dominate every tree and weaker but informative proxies also enter the model.'],
    ['h2', '2.3 Out-of-bag estimate'],
    ['p', 'Each bootstrap sample leaves out on average $(1-1/N)^{N}≈e^{-1}≈36.8$ % of the $N$ training samples. Predicting every sample only with the trees that did not see it gives the out-of-bag (OOB) prediction, an almost unbiased estimate of the generalisation error that costs no extra data (Breiman, 2001). With few deposits, as in mineral prospectivity mapping, this matters: every deposit can be used for training and still be validated.'],
    ['h2', '2.4 Random forest in mineral prospectivity mapping'],
    ['p', 'In mineral prospectivity mapping (MPM) the response is the presence of a known deposit in a cell. Because only deposit locations are known, non-deposit training samples are drawn at random from cells that are unlikely to be mineralised — away from known deposits (Carranza et al., 2008) — and the forest separates the two groups in evidence space. Rodriguez-Galiano et al. (2015) found RF the most accurate and stable of four machine-learning methods for epithermal gold; Carranza and Laborte (2015) showed that it works with few prospects and with missing evidence; McKay and Harris (2016) compared it favourably with a knowledge-driven model. The output score is the share of trees voting “deposit” — a ranking of prospectivity, not a probability per unit cell, because it depends on the deposit / non-deposit ratio of the sample (contrast with logistic regression with prior correction).'],

    ['h1', '3 Mathematical basis'],
    ['h2', '3.1 Training data'],
    ['p', 'Let $\\bf{x}∈ℝ^{p}$ be the evidence vector of a cell (continuous layers as values, a categorical layer with $K$ classes as $K$ one-hot columns — trees need no reference class). The training set is $D=\\{(\\bf{x}_{i},y_{i})\\}_{i=1}^{N}$ with $y_{i}=1$ for the $n_{d}$ training-deposit cells and $y_{i}=0$ for $n_{0}=r·n_{d}$ non-deposit cells drawn at random from the candidate set'],
    ['eq', '\\cal{C}=\\{c∈A : c∉\\cal{D}, \\rm{dist}(c,\\cal{D})≥d_{\\rm{min}}\\}', 'rf-cand'],
    ['p', 'where $A$ is the study area, $\\cal{D}$ the set of all known deposit cells and $d_{\\rm{min}}$ the minimum distance from deposits; the samples are further kept at least $s$ apart (dart throwing), with the automatic spacing'],
    ['eq', 's=\\frac{1}{2}\\sqrt{\\frac{|\\cal{C}|·a}{n_{0}}}', 'rf-spacing'],
    ['p', '($a$ = cell area), i.e. half the mean distance of $n_{0}$ points spread uniformly over the candidate area.'],
    ['h2', '3.2 Tree growing'],
    ['p', 'At node $t$ holding $N_{t}$ samples with class shares $p_{t0},p_{t1}$, the Gini impurity is'],
    ['eq', 'I(t)=1-\\sum_{k=0}^{1}{p_{tk}^{2}}=2p_{t1}(1-p_{t1})', 'rf-gini'],
    ['p', 'and the split $(j,c)$, chosen among $m$ randomly drawn columns, maximises the impurity decrease'],
    ['eq', 'ΔI(j,c)=I(t)-\\frac{N_{L}}{N_{t}}I(t_{L})-\\frac{N_{R}}{N_{t}}I(t_{R})', 'rf-split'],
    ['p', 'Nodes are split until they are pure or hold fewer than $2n_{\\rm{leaf}}$ samples ($n_{\\rm{leaf}}$ = minimum samples per leaf) or the maximum depth is reached. The number of columns per split is $m=\\sqrt{p}$ (default), $\\log_{2}p$, $p/3$ or $p$ (bagging).'],
    ['h2', '3.3 Ensemble score and uncertainty'],
    ['p', 'With $T$ trees and $p_{t}(\\bf{x})$ the deposit share of the leaf of tree $t$ reached by $\\bf{x}$, the score and its spread over the trees are'],
    ['eq', '\\hat{P}(\\bf{x})=\\frac{1}{T}\\sum_{t=1}^{T}{p_{t}(\\bf{x})}', 'rf-score'],
    ['eq', 's(\\bf{x})=\\sqrt{\\frac{1}{T}\\sum_{t=1}^{T}{p_{t}(\\bf{x})^{2}}-\\hat{P}(\\bf{x})^{2}}', 'rf-std'],
    ['p', 'Both are accumulated in one pass as $\\sum{p_{t}}$ and $\\sum{p_{t}^{2}}$. With $R$ random non-deposit samples (repeats; Zuo and Wang, 2020), one forest is trained per sample and all $R·T$ trees are pooled in [[rf-score]] and [[rf-std]]. The variance of an average of correlated trees,'],
    ['eq', '\\rm{Var}(\\hat{P})=ρσ^{2}+\\frac{1-ρ}{T}σ^{2}', 'rf-var'],
    ['p', 'shows why more trees help only until the second term is negligible, after which the score is stable; the remaining $ρσ^{2}$ is reduced only by de-correlation ($m<p$).'],
    ['h2', '3.4 Out-of-bag validation'],
    ['p', 'Let $O_{t}$ be the samples not in the bootstrap sample of tree $t$. The OOB score of sample $i$ is $\\hat{P}_{\\rm{oob}}(\\bf{x}_{i})=\\frac{1}{|T_{i}|}\\sum_{t∈T_{i}}{p_{t}(\\bf{x}_{i})}$ with $T_{i}=\\{t: i∈O_{t}\\}$; the OOB AUC is the Mann–Whitney AUC of these scores between deposits and non-deposits, and the OOB confusion matrix classifies $\\hat{P}_{\\rm{oob}}≥0.5$ as deposit.'],
    ['h2', '3.5 Variable importance'],
    ['p', 'The impurity importance (mean decrease of Gini) of column $j$ sums [[rf-split]] over all nodes that split on $j$, weighted by the share of samples reaching them, averaged over trees and normalised to 100 %. It is biased towards columns with many distinct values (Strobl et al., 2007). The permutation importance of an evidence layer $L$ (all its one-hot columns together) is the drop of the OOB AUC when its values are permuted among the samples (Breiman, 2001):'],
    ['eq', '\\rm{VI}_{L}=\\rm{AUC}_{\\rm{oob}}-\\rm{AUC}_{\\rm{oob}}^{(π_{L})}', 'rf-vi'],
    ['p', 'reported as mean ± s.d. of five permutations; a value ≤ 0 means the layer adds nothing that the others do not already provide.'],

    ['h1', '4 Algorithm'],
    ['h2', '4.1 Steps'],
    ['steps', [
      '**Evidence.** Resample every evidence raster to the grid of the first one if needed (nearest for categorical, bilinear for continuous). Define the study area (mask, and cells where every layer has data unless missing values are imputed). Encode the study cells as a float32 feature matrix: continuous values; one 0/1 column per class of a categorical layer (`FeatureStack`).',
      '**Deposits.** Map deposit points to cells (one per cell); optional thinning (a deposit is kept only if ≥ $d_{\\rm{thin}}$ from every kept one); spatially stratified or random train / test split (§6.3).',
      '**Candidates.** Cells of the study area without a known deposit and at least $d_{\\rm{min}}$ from every known (training *and* test) deposit, [[rf-cand]].',
      '**For each repeat r = 1 … R:** draw $n_{0}=r·n_{d}$ candidates in random order, accepting a cell only if it is ≥ $s$ from all accepted ones ([[rf-spacing]]); fit a forest of $T$ trees (`RandomForestClassifier`, bootstrap, $m$ columns per split, OOB enabled, all CPU cores); record the OOB AUC; predict every study cell tree by tree in chunks of 250 000 cells and accumulate $\\sum{p_{t}}$, $\\sum{p_{t}^{2}}$.',
      '**Maps.** Score [[rf-score]] and tree-vote standard deviation [[rf-std]] over all $R·T$ trees.',
      '**Validation.** OOB confusion matrix, accuracy and AUC of forest 1; optional stratified k-fold or leave-one-out cross-validation (a new forest per fold); ROC / success-rate / P–A curves of the training and test deposits against all background cells.',
      '**Importance.** Impurity importance (mean over repeats) per column and per layer; permutation importance [[rf-vi]] per layer on the OOB samples of forest 1.',
      '**Outputs.** `score.tif`, `score_std.tif`, `training_points.shp`, `test_points.shp` and CSV tables in `results/<run_id>/`; results dialog.',
    ]],
    ['h2', '4.2 Pseudocode'],
    ['code', [
      'X = FeatureStack(evidence, study_area)               # continuous + one-hot',
      'dep = deposit_cells(train);  cand = background_candidates(study, all_deposits, d_min)',
      'for r in 1..R:',
      '    neg = sample_positions(cand, n0 = ratio*len(dep), spacing = s, seed = seed+1000r)',
      '    rf  = RandomForestClassifier(T, max_features=m, min_samples_leaf, bootstrap, oob_score).fit(X[dep+neg], y)',
      '    for chunk in study_cells (250 000):  S += Σ_t p_t(X[chunk]);  Q += Σ_t p_t(X[chunk])²',
      'score = S / (R*T);   std = sqrt(Q / (R*T) - score²)',
      'oob = rf_1.oob_decision_function_;  importance = permutation(rf_1, OOB samples) + MDI',
    ]],
    ['h2', '4.3 Library and own code'],
    ['table', {
      caption: 'Table 4.1. Source of each component.',
      head: ['Component', 'Implementation'],
      widths: [3400, 5626],
      rows: [
        ['Trees, bootstrap, random column subsets, Gini, OOB predictions, MDI', 'scikit-learn 1.3 `RandomForestClassifier` (Pedregosa et al., 2011) — a direct implementation of Breiman (2001)'],
        ['Tree-vote mean and standard deviation [[rf-score]], [[rf-std]]', 'Own code: one pass over `estimators_` in thread-parallel chunks (verified equal to `predict_proba` and to the per-tree standard deviation)'],
        ['Pooling of repeats', 'Own code (Zuo and Wang, 2020)'],
        ['Permutation importance on OOB samples [[rf-vi]]', 'Own code (Breiman’s original definition; OOB indices from scikit-learn)'],
        ['Non-deposit sampling [[rf-cand]], [[rf-spacing]]', 'Own shared code (`logic/common/training.py`)'],
        ['Cross-validation, validation curves', 'Own code (`model.py`, `evaluation/metrics.py`)'],
      ],
    }],
    ['p', 'Computational cost: training is $O(T·m·N\\log N)$ for $N$ samples and is negligible here ($N$ ≤ a few hundred); prediction is $O(T·\\rm{depth})$ per cell — about 10 s for 900 000 cells and 500 trees on four cores. ArcSDM’s experimental *RF Train* tool uses the same scikit-learn estimator with 10 trees by default and leave-one-out cross-validation (ArcSDM, 2025); MPM Studio uses 500 trees, the OOB estimate and tree-vote uncertainty.'],

    ['h1', '5 How to use'],
    ['p', 'Open a project with evidence rasters and a deposit point layer, then *Data-driven ▸ Machine learning ▸ Random Forest…*. The dialog has four sections; *Run model* starts the computation in the background; outputs are written to `results/<prefix>_<MMDD>_<HHMMSS>/` and added to the map, and the results dialog opens.'],
    ['table', {
      caption: 'Table 5.1. Training data.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2300, 1600, 5126],
      rows: [
        ['Deposits (points)', 'layer named *deposit*', 'Known deposits / occurrences of the target type (training points).'],
        ['Study area mask', 'layer named *mask*', 'Cells outside are ignored; default = cells where evidence exists.'],
        ['Hold out for testing', '20 %', 'Share of deposits kept for an independent test AUC (0 % = all for training). Use 0 % with < ~15 deposits and rely on OOB.'],
        ['Split method', 'spatially stratified', 'Test deposits drawn evenly along a space-filling curve (in proportion to the deposit density); *random* = plain shuffle.'],
        ['Random seed', '42', 'Fixes every random choice (split, non-deposit samples, forests) — same seed, same result.'],
        ['Thin deposits (min. distance)', 'off', 'Keep a deposit only if it is at least this far from every kept one (declustering).'],
        ['Absence points (ROC)', 'all background cells', 'Optional known non-deposit points used as ROC negatives.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.2. Evidence layers.',
      head: ['Column', 'Meaning and guidance'],
      widths: [2300, 6726],
      rows: [
        ['Use', 'Tick the layers that represent the mineral-system proxies (source, pathway, trap, footprint).'],
        ['Values', 'Number of classes (class rasters) or *continuous*; the tooltip shows the value range.'],
        ['Variable type', '*Continuous* — the value is used directly (distances, geochemistry, geophysics); *Categorical (one-hot)* — one 0/1 column per class (lithology, host rock); at most 255 classes.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.3. Training samples.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2700, 1500, 4826],
      rows: [
        ['Non-deposit samples per deposit', '1', 'Ratio $r$ of non-deposit to deposit samples. 1 balances the classes; 3–10 describe the barren ground better (§6.7).'],
        ['Min. distance from deposits', '4 cells', '$d_{\\rm{min}}$ in [[rf-cand]]; set it to the width of the alteration / geochemical halo so that mineralised ground is not labelled barren.'],
        ['Min. spacing between samples', 'auto', '$s$ in [[rf-spacing]]; spreads the non-deposit samples evenly over all geological domains; 0 = no constraint.'],
        ['Random samples (repeats)', '5', 'Number $R$ of independent non-deposit samples; one forest each, all trees pooled.'],
        ['Cross-validation', 'none (OOB)', 'Optional 10-fold, 3-fold or leave-one-out (≤ 150 samples) cross-validation of the training samples; the OOB estimate is always reported.'],
      ],
    }],
    ['table', {
      caption: 'Table 5.4. Model options.',
      head: ['Setting', 'Default', 'Meaning and guidance'],
      widths: [2300, 1700, 5026],
      rows: [
        ['Number of trees', '500', '$T$ per forest. The score is stable from about 300 trees (Table 6.8); more trees cost time only.'],
        ['Features per split', '√p', '$m$; √p de-correlates the trees (default for classification); all p = bagging.'],
        ['Min. samples per leaf', '1', '$n_{\\rm{leaf}}$; larger values smooth the score and prevent rules built on single deposits.'],
        ['Max. tree depth', 'unlimited', 'Limits the number of splits along a path.'],
        ['Class weight', 'none', '*balanced* weights deposits up in proportion to the class sizes (useful with large ratios $r$).'],
        ['Missing evidence', 'exclude', 'Exclude cells with missing evidence from the study area, or set them to the median / most frequent class.'],
        ['Resample evidence', 'on', 'Resample to the first layer’s grid if grids differ.'],
        ['Outputs / prefix', 'score + std / rf', 'Score and tree-vote standard deviation rasters; run-folder prefix.'],
      ],
    }],
    ['p', 'The results dialog shows the training samples, the OOB AUC, the cross-validated AUC (if chosen) and the ROC AUC of the training and test deposits on the map, warnings, and the tabs *Summary*, *Variable importance* (table and bar chart), *Sample validation* (OOB confusion matrix, accuracy, AUC, CV) and *Repeats*. *ROC / AUC…* opens the shared validation dialog (ROC, success rate, P–A, summary); *Export tables (Excel)…* saves every table. Judge the model by the **test** and **OOB** AUC: the training-deposit AUC on the map is close to 1 because the forest has seen those deposits.'],

    ['h1', '6 Demo data: ideal parameters and results'],
    ...C.demoData(6),
    ...C.protocol(6),

    ['h2', '6.5 Carlin (Au)'],
    C.evidenceTable(J, 'Table 6.4. Evidence of the Carlin random forest.'),
    C.searchTable(J, 'Table 6.5. Carlin — candidate settings (300 trees, 3 repeats, spatial 70 / 30 splits).'),
    ['p', `The selected setting is **${cj.label}**. Over 10 independent splits it gives a test ROC AUC of **${pm(J.robust_summary.test_auc_mean, J.robust_summary.test_auc_sd)}** (range ${f(J.robust_summary.test_auc_min)}–${f(J.robust_summary.test_auc_max)}), while the training AUC is ${pm(J.robust_summary.train_auc_mean, J.robust_summary.train_auc_sd)} — the forest fits its training deposits almost perfectly, as expected for an ensemble of deep trees, so only the test and OOB AUCs measure prediction.`],
    C.robustTable(J, 'Table 6.6. Carlin — robustness of the selected setting over 10 splits.'),
    C.curvesTable(J, 'Table 6.7. Carlin — reference run (seed 42): training and test deposits against all background cells.'),
    ['img', J.figures.map, 520, `Figure 6.1. Carlin — RF prospectivity score (mean tree vote) with training (green) and test (blue) deposits.`],
    ['img', J.figures.curves, 600, 'Figure 6.2. Carlin — (a) ROC curves, (b) success / prediction-rate curves, (c) prediction–area plot, (d) permutation importance (OOB AUC drop).'],
    ['table', {
      caption: 'Table 6.8. Carlin — variable importance of the reference forest (permutation importance: mean ± s.d. of five shuffles).',
      head: ['Rank', 'Evidence layer', 'Impurity importance (%)', 'Permutation importance (OOB AUC drop)'],
      widths: [900, 2826, 2300, 3000],
      rows: impRows(J),
    }],
    ['table', {
      caption: 'Table 6.9. Carlin — out-of-bag validation of the training samples (forest 1).',
      head: ['Item', 'Value'], widths: [5526, 3500], rows: svRows(J),
    }],
    ['table', {
      caption: 'Table 6.10. Carlin — number of trees (selected setting, seed 42).',
      head: ['Trees', 'OOB AUC', 'Training AUC', 'Test AUC', 'Time (s)'], widths: [1426, 1900, 1900, 1900, 1900], rows: sensRows(J),
    }],
    ['p', `The permutation importance ranks **${top(J).layer}** first (OOB AUC drop ${f(top(J).permutation_auc_drop)}). The impurity importance distributes the credit more evenly and favours the continuous layers with many distinct values, as Strobl et al. (2007) predict; the permutation ranking is the one to interpret. Layers with a permutation importance close to zero add no information beyond the others — for example, the distance to faults is largely redundant with the host-rock and geochemical layers at this scale, because the deposits that lie near faults are also those in anomalous, favourable ground.`],

    ['h2', '6.6 Curnamona (REE)'],
    C.evidenceTable(K, 'Table 6.11. Evidence of the Curnamona random forest (chosen by the single-layer screening, Table 6.3).'),
    C.searchTable(K, 'Table 6.12. Curnamona — candidate settings (leave-one-deposit-out, 7 folds).'),
    C.lodoTable(K, 'Table 6.13. Curnamona — leave-one-deposit-out results of the selected setting.'),
    C.curvesTable(K, 'Table 6.14. Curnamona — reference run with all seven occurrences (training AUC only).'),
    ['img', K.figures.map, 520, 'Figure 6.3. Curnamona — RF prospectivity score with the seven REE occurrences (training).'],
    ['img', K.figures.curves, 600, 'Figure 6.4. Curnamona — (a) ROC, (b) success rate, (c) P–A plot, (d) permutation importance.'],
    ['table', {
      caption: 'Table 6.15. Curnamona — variable importance of the reference forest.',
      head: ['Rank', 'Evidence layer', 'Impurity importance (%)', 'Permutation importance (OOB AUC drop)'],
      widths: [900, 2826, 2300, 3000],
      rows: impRows(K),
    }],
    ['p', `With only seven occurrences the selected setting reaches a mean LODO AUC of **${f(K.search.find((s) => s.label === K.selected.label).mean_test_auc)}**: a held-out occurrence ranks on average above ${f(100 * K.search.find((s) => s.label === K.selected.label).mean_test_auc, 0)} % of the background cells. The permutation importance ranks **${top(K).layer}** first, followed by **${K.reference.tables.importance[1].layer}** and **${K.reference.tables.importance[2].layer}**. U is mobile and is concentrated in the hydrothermally overprinted granite and breccias that host the REE minerals, Nb is the HFSE companion of the REE in A-type systems, and Th (immobile, in monazite and allanite) marks the granite as a whole; K and the silica index add little once these are known.`],

    ['h2', '6.7 Ideal parameters and their geological meaning'],
    ['p', 'Table 6.16 lists the parameters recommended for the two demo projects. Each value was chosen by the held-out ranking of the deposits (§6.3), and each is explained in terms of the ore-forming system it represents.'],
    ['table', {
      caption: 'Table 6.16. Ideal random-forest parameters for the demo projects and their meaning.',
      head: ['Parameter', 'Carlin (Au)', 'Curnamona (REE)', 'Statistical and geological reasoning'],
      widths: [1700, 1300, 1300, 4726],
      rows: [
        ['Evidence', 'host rock (one-hot), distance to faults, log As, log Sb', 'radiometric Th, U, K; log Nb; ASTER silica index', 'Each layer is a proxy of a mineral-system component. Carlin: faults = fluid pathways; carbonate host = chemical trap; As–Sb = the hydrothermal footprint (Cline et al., 2005). Curnamona: Th–U–K = U–Th–REE-rich A-type granite and K-metasomatism (source / host); Nb = HFSE co-enrichment; silica index = quartz-rich cover that hides the basement (negative proxy).'],
        ['Variable type', 'host = categorical, rest continuous', 'all continuous', 'Lithology is nominal (one-hot); distances and concentrations are kept continuous so that the forest finds the geologically meaningful thresholds itself (fluid-flow reach from faults, anomaly threshold) instead of a fixed classification.'],
        ['Non-deposit ratio', `1 : ${f(cj.neg_ratio, 0)}`, `1 : ${f(ck.neg_ratio, 0)}`, ratioReason(J, cj.neg_ratio)],
        ['Min. distance from deposits', `${f(4 * J.dataset_info.cell_m, 0)} m`, `${f(4 * K.dataset_info.cell_m, 0)} m`, 'About the width of the alteration and geochemical halo: Carlin-type decarbonatisation, silicification and As–Sb anomalies extend hundreds of metres to about 1 km from ore (Cline et al., 2005); the REE-bearing breccias and metasomatic zones of the Mount Painter area are of similar or smaller width (Elburg et al., 2012). Cells inside the halo belong to the mineral system and must not be used as barren examples.'],
        ['Min. spacing of non-deposits', `auto (${int(J.reference.tables.stats.spacing)} m)`, `auto (${int(K.reference.tables.stats.spacing)} m)`, 'Spreads the barren examples over every geological domain (all lithologies, structural blocks, geochemical backgrounds) instead of letting random clusters over-represent one domain; geologically, “barren” ground is heterogeneous and must be sampled evenly.'],
        ['Repeats', '3 (5 in the GUI)', '3 (5 in the GUI)', 'Several random barren samples average out the luck of one draw (Zuo and Wang, 2020); the pooled score represents the whole barren population.'],
        ['Number of trees', '300–500', '300–500', 'Table 6.10: OOB and test AUC are stable from about 300 trees ([[rf-var]]); 500 gives a smooth score map.'],
        ['Features per split', mfName[cj.max_features], mfName[ck.max_features], 'The proxies of one mineral system are correlated (As with Sb from the same fluids; Th with U from the same granite). Random column subsets let every proxy contribute and make the model robust to one noisy layer.'],
        ['Min. samples per leaf', String(cj.min_samples_leaf), String(ck.min_samples_leaf), leafReason(cj.min_samples_leaf)],
        ['Hold-out / validation', '20–30 % spatially stratified', 'leave-one-deposit-out', 'With 34 deposits a hold-out leaves enough for training; with 7, every occurrence is needed for training and LODO measures how well an unseen occurrence of the same system is predicted.'],
      ],
    }],
    ['h2', '6.8 Discussion'],
    ['bullets', [
      `**Accuracy.** On Carlin the random forest reaches a mean test AUC of ${f(J.robust_summary.test_auc_mean)} over 10 splits; on Curnamona the mean LODO AUC is ${f(K.search.find((s) => s.label === K.selected.label).mean_test_auc)}. The companion documents report the weights-of-evidence and logistic-regression results with the same protocol.`,
      '**Score, not probability.** The score depends on the deposit / non-deposit ratio; it ranks cells correctly but is not a probability per unit cell. For probabilities use logistic regression with prior correction.',
      '**Uncertainty.** The tree-vote standard deviation is high on the boundary between favourable and unfavourable ground and low where all trees agree; it measures model disagreement, not the uncertainty of the input data.',
      '**Spatial autocorrelation.** Clustered deposits make random or stratified hold-outs optimistic (Roberts et al., 2017); a spatial block hold-out would give a more conservative estimate.',
      '**Few deposits.** With seven occurrences the importances and thresholds are indicative only; the LODO spread (Table 6.13) shows how much individual occurrences differ.',
    ]],

    ['h1', '7 References'],
    ['refs', R('breiman1984', 'breiman1996', 'breiman2001', 'carranza2008', 'carranza2015', 'chung2003', 'cline2005',
      'delong1988', 'deepseam', 'elburg2012', 'fawcett2006', 'hofstra2000', 'mckay2016', 'muntean2011', 'neumann2000',
      'pedregosa2011', 'probst2019', 'raines2001', 'roberts2017', 'rodriguez2015', 'stevens2004', 'strobl2007',
      'yousefi2015', 'zuo2020', 'arcsdm')],
  ],
};
