# 2D MPM — MATLAB machine-learning methods

MATLAB R2024a implementation of the data-driven ML methods of [`docs/methods_survey.md`](../docs/methods_survey.md).
There is no main GUI: **open a `run_*.m` script and run it (F5)**. When it finishes, a results window opens
with the training and test results.

| Folder | Purpose |
|---|---|
| `classification/` | **Target delineation** (prospectivity mapping). The model learns deposit cells vs. background cells and produces a prospectivity map (0–1), which is cut into exploration-target classes. |
| `regression/` | **Grade estimation** from known orebody / sample grades and the evidence layers. The model predicts a grade map. |
| `common/` | Code shared by every method: raster/point I/O, features, sampling, CV tuning, metrics, results GUI |
| `data/` | Point CSVs (deposits, samples) of the demo datasets |
| `results/` | One folder per run (`<dataset>/<task>_<method>_<timestamp>/`) |

Required toolboxes: Statistics and Machine Learning, Mapping (`readgeoraster`, `geotiffwrite`, `shaperead`),
and Image Processing (`bwdist`). All are installed. No Python is needed. Optional: Parallel Computing Toolbox +
an NVIDIA GPU (CUDA driver) to run the M9 CNN on the GPU; without them it runs on the CPU.

## Data: shared pre-processed database

MATLAB reads the co-registered GeoTIFF stack produced by the pre-processing (the Python tool or any GIS), e.g.
`demo_data/<dataset>/processed/` (older layout `demo_data/2d_mpm/<dataset>/processed/` also found) (`layers.csv` + `rasters/*.tif`, one common grid, `*.classes.csv` labels).

Point data (deposits, orebody / drillhole samples) is read from:
- **CSV** with `x`, `y` columns in the raster CRS plus any attributes (QGIS: *Save Features As → CSV,
  GEOMETRY = AS_XY*), or
- **point shapefile** (`.shp`).

GeoPackage is not read: MATLAB needs the Database Toolbox for that. The demo point CSVs are already in `data/`.
Your own data, e.g. drillhole composites, is any CSV or shapefile with coordinates and a grade column.

## Scripts

| Script | Method | Default demo |
|---|---|---|
| `classification/run_svm_classification.m` | SVM (RBF, Platt probabilities) | Carlin Au, 35 deposits |
| `classification/run_rf_classification.m` | Random forest | Carlin Au |
| `classification/run_rvm_classification.m` | Relevance vector machine (+ uncertainty map, relevance vectors) | Carlin Au |
| `classification/run_mlp_classification.m` | Neural network: MLP ensemble (+ uncertainty map) | Carlin Au |
| `classification/run_xgboost_classification.m` | XGBoost (own implementation of Chen & Guestrin 2016) | Carlin Au |
| `classification/run_lightgbm_classification.m` | LightGBM (own implementation of Ke et al. 2017) | Carlin Au |
| `classification/run_elm_classification.m` | Extreme learning machine: weighted regularised ELM ensemble (+ uncertainty map) | Carlin Au |
| `classification/run_iforest_classification.m` | Isolation forest, **unsupervised** anomaly detection (own implementation of Liu et al. 2008) | Carlin Au |
| `classification/run_ocsvm_classification.m` | One-class SVM, **unsupervised** anomaly detection (Schölkopf et al. 2001) | Carlin Au |
| `classification/run_autoencoder_classification.m` | Deep autoencoder, **unsupervised** anomaly detection (reconstruction error; Xiong & Zuo 2016) | Carlin Au |
| `classification/run_pubag_classification.m` | Positive-unlabelled (PU) bagging: deposits vs. unlabelled cells, biased-SVM ensemble (Mordelet & Vert 2014) (+ uncertainty map) | Carlin Au |
| `classification/run_cnn_classification.m` | CNN on image patches: w x w evidence windows, sliding-window + random-drop augmentation, CPU or NVIDIA GPU (+ uncertainty map) | Carlin Au |
| `classification/run_bn_classification.m` | Bayesian network classifiers: naive / tree-augmented / selective naive Bayes (+ parameter-uncertainty map) | Carlin Au |
| `classification/run_compare_classification.m` | All methods on the same split, side-by-side GUI | Carlin Au |
| `regression/run_svm_regression.m` | Support vector regression | Carlin Th in stream sediments (stand-in for grade) |
| `regression/run_rf_regression.m` | Random-forest regression (+ uncertainty map) | Carlin Th |
| `regression/run_rvm_regression.m` | Relevance vector regression (+ predictive-variance map, relevance vectors) | Carlin Th |
| `regression/run_mlp_regression.m` | Neural network: MLP ensemble regression (+ uncertainty map) | Carlin Th |
| `regression/run_xgboost_regression.m` | XGBoost regression | Carlin Th |
| `regression/run_lightgbm_regression.m` | LightGBM regression | Carlin Th |
| `regression/run_elm_regression.m` | Extreme learning machine regression ensemble (+ uncertainty map) | Carlin Th |
| `regression/run_compare_regression.m` | All methods on the same split | Carlin Th |

Set `dataset = 'carlin'` or `'curnamona'` at the top of a script. The dataset settings (layers, buffers,
negatives, target column) are in `classification/clf_demo_config.m` and `regression/reg_demo_config.m`.
Any `cfg` field can be overridden in the script. All options and their defaults are listed in
`common/util/mpm_default_config.m`.

To reopen a saved run later: `mpm_setup; mpm_results_viewer` (pick `result.mat`). To compare runs:
`mpm_compare_viewer` (pick several).

## What the pipelines do

**Classification** (`classification/mpm_classification_run.m`)
1. Load the evidence layers, one-hot encode categorical layers (or use them as numbers via
   `cfg.features.as_numeric`), and impute missing values.
2. Deposits → cells. Split into training and test sets: `random`, `block` (spatial), or `rule`
   (e.g. `@(T) T.RANDOM >= 700`).
3. Positives = training deposit cells, optionally plus a buffer (`cfg.positives.buffer_cells`, useful with few
   deposits). Negatives = random background cells at least `cfg.negatives.min_distance` m from any deposit.
   Train and test negatives are separate.
4. Z-score the features with training statistics. Tune hyper-parameters with k-fold CV grid search (AUC); for
   buffered deposits the folds are grouped by deposit. Fit the final model.
5. Predict the prospectivity map for every valid cell, plus an uncertainty map if the method provides one
   (RVM).
6. Validate:
   - sample ROC/AUC plus accuracy, precision, recall, F1, kappa and Brier score (train/test);
   - **map ROC/AUC**: deposit cells vs. every other cell, the same convention as `python/evaluation/metrics.py`;
   - success-rate and prediction-rate curves;
   - prediction–area (P-A) plot and its intersection.
7. Target classes (top 5 / 10 / 20 % of the area by default) with the deposit capture of each class.
   **Binary target grids** (1 = predicted deposit area, 0 = non-deposit) by P-A intersection and C-A fractal;
   see *Binary targets* below. Permutation importance is computed per evidence layer.

**Regression** (`regression/mpm_regression_run.m`)
1. Load the evidence layers as above. Grade samples → cells. Non-positive grades are dropped or replaced by
   half the minimum for the log transform. Samples sharing a cell are averaged.
2. Model scale = `log10(grade)` by default. Split into training and test sets (random / block / rule).
3. Tune by CV grid search (RMSE). Fit the final model. Predict the grade map (back-transformed, i.e. median
   grade under a log model) and an uncertainty map if the method provides one (RF: spread of the trees;
   RVM: predictive standard deviation).
4. Metrics on the model and grade scales for training, out-of-fold CV and test samples: R², RMSE, MAE, bias,
   Pearson and Spearman correlation.
5. **Cut-off ROC/AUC**: how well the predicted grade ranks samples above a cut-off grade (P75 by default).

Regression only estimates grade. Target delineation (P-A / C-A binary grids) belongs to classification.

Demo caveat: the demo datasets have no orebody grades. Surface geochemistry stands in for grade, and the
geochemical layers interpolated from those same samples are excluded to avoid leakage. The demo target was
chosen for its geological link to the evidence:

- **Carlin Th (ppm) in NURE stream sediments** (default). Airborne gamma-ray Th measures the same element
  (²⁰⁸Tl 2.61 MeV line, top ~30 cm), Th is immobile, and both data sets integrate surface material over an
  area. Spearman ρ of log Th with the airborne thorium layer at the samples = 0.52, against K 0.46, U 0.33
  (U is mobile), and the As / Sb pathfinders ≤ 0.31.
- **Curnamona La in rock samples** is still available but weak: no element exceeds ρ = 0.27 against any layer,
  because these are mineralisation-biased point samples.

## Outputs (per run)

`result.mat` (everything, including the model) · `prospectivity.tif` + `targets.tif` (classification) or
`predicted_grade.tif` (regression), plus `uncertainty.tif` when the method provides one (RF regression, RVM) · `metrics.csv` · `importance.csv` · `tuning.csv` ·
`targets.csv` / `deposits_scores.csv` / `samples.csv` · `log.txt` · classification also writes
`binary_pa.tif` + `binary_ca.tif` + `binary_targets.csv` + `ca_fractal_segments.csv`.

GeoTIFFs use the evidence grid and CRS, so they open in QGIS and in the Python GUI. Outside the study area,
float rasters are NaN and class rasters are 0.

## Results GUI (`common/gui/mpm_results_viewer.m`)

- **Classification tabs:** Summary (sample metrics, map metrics, target classes, settings and log) ·
  ROC / AUC (training and test; samples and whole map) · Success-rate & P-A · Prospectivity map
  (probability, target classes or uncertainty, with training/test deposits and relevance vectors) ·
  Confusion matrix · Importance · Tuning (CV).
- **Regression tabs:** Summary · Observed vs predicted (training / CV / test, model and grade scale) ·
  Residuals · Grade map (prediction or uncertainty, with sample and relevance-vector overlay) · Cut-off ROC ·
  Importance · Tuning.

Classification also has a **Binary targets (P-A / C-A)** tab: the C-A log-log plot with the fitted segments and
breaks, the binary map (P-A or C-A) with the deposits, and the summary table.

*Export figures…* writes one PNG per tab.

## Binary targets: P-A intersection and C-A fractal

These are common utilities (`common/eval/`) that turn a prospectivity map into a binary grid:
**1 = predicted deposit area**, **0 = non-deposit / hopeless area**, and NaN outside the study area
(GeoTIFF `single`).

| Rule | Function | Threshold | Needs deposits |
|---|---|---|---|
| **P-A intersection** (Yousefi & Carranza 2015) | `mpm_binary_targets(..., struct('methods', {{'pa'}}))` | the score where the prediction rate equals 100 − occupied area %: the map captures P % of the deposits in (100 − P) % of the area | yes |
| **C-A fractal** (Cheng et al. 1994) | `mpm_ca_fractal`, `mpm_binary_targets(..., {{'ca'}})` | a break between the straight-line segments of log10(area ≥ t) vs log10(t); `ca_break = 'highest'` = the most anomalous population | no |

- C-A breaks are found automatically: an exact least-squares segmentation (dynamic programming) into
  `cfg.binary.ca.segments` lines (default 3: background / intermediate / anomaly). The result is reproducible,
  with no manual picking of breaks.
- The classification pipeline runs both rules automatically (`cfg.binary.methods = {'pa'}` or `{'ca'}` to keep
  one; `cfg.binary.enable = false` to skip them). The regression pipeline does not use them.
- **Any GeoTIFF** works too, e.g. a Python WofE posterior or a knowledge-driven map:

```matlab
mpm_setup;
B = mpm_binary_from_geotiff('...\wofe_posterior.tif', 'matlab\data\nevada_carlin_au\deposits.csv');
% -> <name>_binary_pa.tif, <name>_binary_ca.tif, <name>_binary.csv, <name>_ca_segments.csv + viewer
B = mpm_binary_from_geotiff('...\map.tif');          % no deposits -> C-A only
```

The summary table reports the threshold, the area (% and km²), the training/test deposits captured, and the
efficiency (captured % / area %).

## Neural network (MLP)

`clf_mlp` / `reg_mlp`. The MLP is a feed-forward network, and its gradients come from backpropagation. The
classic MPM recipe (gradient-descent backprop with momentum and early stopping; Singer & Kouda 1996) is
replaced by a training set-up suited to small tabular data:

- **L-BFGS** (quasi-Newton, full batch; `fitcnet` / `fitrnet`). There is no learning rate or momentum to tune,
  and it converges reliably on a few hundred samples.
- **L2 weight decay `Lambda`** tuned by CV together with `LayerSizes`, instead of early stopping (which would
  hold back scarce deposits). Loss: cross-entropy for classification, MSE for regression.
- **Ensemble of `NumNets` networks** (different initialisations, optional `Bootstrap` bagging). Averaging
  removes the run-to-run variance of a single net. The spread across networks becomes the uncertainty map
  (model uncertainty only).
- Deeper nets: `cfg.params.LayerSizes = [20 10]`, or a grid such as
  `cfg.tune.grid = struct('LayerSizes', {{10, [20 10]}}, 'Lambda', [1e-3 1e-2])`.
- Also considered: Bayesian-regularised Levenberg–Marquardt (`trainbr`, Deep Learning Toolbox). It is good for
  small regressions but has no cross-entropy classification. The ensemble gives uncertainty for both tasks.

## Adding a method

A method is **one file** that returns a struct with `defaults`, `grid`, `train` and `predict`. Everything else is
shared.

1. Copy `classification/methods/clf_template.m` → `clf_<key>.m` (or `regression/methods/reg_template.m` →
   `reg_<key>.m`) and fill in `fit_model` / `predict_value`. The template already runs as written: it is a
   ridge logistic/linear regression.
2. Copy `run_svm_classification.m` → `run_<key>_classification.m` and change the single line `M = clf_svm();`.

Do not name local functions `predict` or `train`: they would shadow the toolbox functions inside the file.
Optional fields: `has_uncertainty` (predict returns `[score, uncertainty]` → uncertainty map) and
`relevant_samples` (`@(mdl)` → indices of the training samples a sparse model keeps → drawn on the maps).
`learning = 'pu'` switches the pipeline to positive-unlabelled training samples (see *PU bagging*).
Planned (survey): logistic regression.

## Independence from Python

The MATLAB tool is distributed separately from the Python tool. It shares only **data**: the preprocessed
rasters (`layers.csv` + GeoTIFFs) and point CSVs, which Python or any GIS can produce. It never calls Python.
Methods that MATLAB and its toolboxes do not provide (RVM, XGBoost, LightGBM, ELM, isolation forest, autoencoder, PU bagging, patch CNN, TAN / selective Bayesian networks) are implemented from the original
papers in `common/model/`.

## Gradient boosting: XGBoost and LightGBM

Both are written from scratch (`common/model/`):
- `mpm_xgboost_train.m`, `mpm_lightgbm_train.m`: the two algorithms;
- `mpm_gb_binning.m`, `mpm_gb_best_split.m`, `mpm_gb_loss.m`, `mpm_gb_predict.m`: shared tree utilities.

| | XGBoost (`clf_xgboost`, `reg_xgboost`) | LightGBM (`clf_lightgbm`, `reg_lightgbm`) |
|---|---|---|
| Paper | Chen & Guestrin 2016, KDD | Ke et al. 2017, NeurIPS |
| Objective | loss + γ·T + ½λ‖w‖² (+ α‖w‖₁); second-order gain (Eq. 7), leaf weight −G/(H+λ) | same second-order gain and leaf weights |
| Split finding | **exact greedy** over all distinct values (Alg. 1), or `TreeMethod='approx'` with quantile bins (Alg. 2) | **histogram** (≤ `MaxBin` bins) with **histogram subtraction** |
| Tree growth | **depth-wise** to `MaxDepth`, then **γ-pruning** bottom-up | **leaf-wise** (best-first) to `NumLeaves` |
| Sampling | row `Subsample`, `ColsampleByTree` | **GOSS** (`Boosting='goss'`), or bagging; `FeatureFraction` |
| Features | — | **EFB**: mutually exclusive features (one-hot geology) bundled into one histogram |
| Tuned by CV | `MaxDepth` × `NumRounds` (25–200) | `NumLeaves` × `NumRounds` (25–200) |

- **scikit-learn HistGradientBoosting** has no separate paper. It implements LightGBM's histogram + leaf-wise
  algorithm without GOSS or EFB, so it is not a separate method: `clf_lightgbm` with `Boosting='gbdt'` and
  `EFB=false` reproduces it.
- **Defaults** follow the libraries, except that tree size is reduced for MPM-size data (tens of deposits):
  XGBoost `MaxDepth` 3 and `Eta` 0.1; LightGBM `NumLeaves` 8 and `MinDataInLeaf` 5.
- **Importance:** native importance = total split gain per feature (GUI Importance tab, `importance_native.csv`).
- **Checks on synthetic data:**
  - Friedman #1 (noise floor 1.0): test RMSE 1.32 (XGBoost), 1.24 (LightGBM), 1.29 (MATLAB LSBoost reference);
  - γ-pruning shrinks trees from 7.6 to 1.7 leaves;
  - EFB bundles a 6-class one-hot block into one bundle with **identical** predictions to no bundling.

## Relevance vector machine (RVM)

No MATLAB toolbox provides an RVM, so it is implemented here: `common/model/mpm_rvm_train.m`,
`mpm_rvm_predict.m` and `mpm_rvm_basis.m` (Tipping 2001, sparse Bayesian learning). `clf_rvm` / `reg_rvm` are the
plug-ins.

- **Model:** a kernel expansion `f(x) = Σ w_i k(x, x_i) + b` with one ARD precision `α_i` per weight. The
  type-II maximum-likelihood re-estimation `α_i ← γ_i / μ_i²` drives most `α_i` to infinity. Those basis
  functions are pruned, and the few training samples that remain are the **relevance vectors**.
- **Regression:** Gaussian likelihood. The noise precision `β` is estimated from the data. Predictive
  SD = `sqrt(1/β + φᵀΣφ)` → `uncertainty.tif`.
- **Classification:** Bernoulli likelihood. A Laplace approximation (Newton / IRLS with step halving) runs at
  each `α` update. Probabilities are MacKay-moderated, `σ(m / sqrt(1 + π s²/8))`, so they move towards 0.5 away
  from the training data. The uncertainty map is ≈ the SD of the probability.
- **Tuning:** only `KernelScale` is tuned (multiplier of `sqrt(P)`, the same convention as SVM). There is no
  `C` or `ε`: the prior sets the sparsity.
- **Cost:** O(M³) per iteration with M = active basis functions. M shrinks quickly as functions are pruned.
  `MaxBasis` (default 2000) caps the candidate set for large sample sets.
- **Checked against Tipping (2001) on synthetic data:**
  - noisy sinc: 6 relevance vectors, noise SD estimated at 0.099 (true 0.1);
  - 2-D two-class problem: 4 relevance vectors vs. 70 SVM support vectors, at the same accuracy.

## Extreme learning machine (ELM)

MATLAB has no ELM, so it is implemented from the papers: `common/model/mpm_elm_train.m`, `mpm_elm_hidden.m` and
`mpm_elm_predict.m`, with Platt calibration in `mpm_platt_fit.m`. `clf_elm` / `reg_elm` are the plug-ins.

- **Network** (Huang, Zhu & Siew 2006, *Algorithm ELM*): one hidden layer of `NumHidden` random nodes
  `g(w·x + b)` and a linear output.
  1. Draw `w`, `b` at random. They are never trained.
  2. Compute the hidden-layer output matrix `H` (N × L).
  3. Solve the output weights analytically.

  There is no back-propagation and no iteration, so it is fast.
- **Output weights:** regularised ELM (Huang, Zhou, Ding & Zhang 2012), which minimises `½‖β‖² + C/2 Σ e²`:
  - N ≥ L: `β = (I/C + HᵀWH)⁻¹ HᵀWT`
  - N < L: `β = Hᵀ(I/C + WHHᵀ)⁻¹ WT`

  Both are solved in the equivalent symmetric (Cholesky) form. `C = Inf` gives the original
  `β = H⁺T` (Moore–Penrose pseudo-inverse).
- **Classification:**
  - Targets: deposit = +1, background = −1, one output node (Huang et al. 2012). As in Chen & Wu (2017), the
    ELM output ranks prospectivity.
  - `W` holds the weighted-ELM class weights (Zong, Huang & Chen 2013), rescaled to mean 1 so `C` keeps its
    unweighted meaning:
    - `Weighting = 'W1'`: `1/#class` (default);
    - `'W2'`: `0.618/#class` for the larger class;
    - `'none'`.
  - Probabilities: a Platt (1999) sigmoid is fitted to **out-of-fold** ensemble outputs. The folds are grouped
    so a deposit's buffer cells stay together, and the Newton solver follows Lin, Lin & Weng (2007).
- **Regression:** squared error on the model scale (log10 grade), with the target z-scored inside the method.
- **Ensemble:** `NumNets` (20) ELMs, each with its own random hidden layer, outputs averaged (Liu & Wang 2010;
  Cao et al. 2012). This removes the run-to-run variance due to the random nodes. The spread across members
  becomes `uncertainty.tif` (model uncertainty only, as for the MLP).
- **Random nodes:** Huang et al. (2006, Theorem 2.1) allow any interval and any continuous distribution. The
  inputs here are z-scored, so `w ~ U(−a, a)` with `a = InputScale·sqrt(3/P)` (pre-activation SD = `InputScale`)
  and `b ~ U(−1, 1)`.
  - `Activation`: `sigmoid` (paper default), `tanh`, `sine`, `hardlim`, `tribas` or `radbas`.
- **Tuning:** `NumHidden` × `C` (5 × 6 grid, `NumNetsTuning` = 5 members inside CV). Importance is permutation
  only.
- **Checked on synthetic data:**
  - The primal and dual solutions equal the explicit weighted-ELM formulas (difference ~1e−13), and `C = Inf`
    equals `pinv(H)·T` exactly.
  - SinC benchmark of Huang et al. (2006, §4.1.1; 5000 noisy training points, 20 sigmoid nodes): test RMSE
    0.0063 (paper: 0.0097).
  - The Platt fit agrees with `glmfit` logistic regression.
- **Demo runs (Carlin, seed 42; one split each, not a comparison):**

  | Task | Result | Time |
  |---|---|---|
  | Classification | test AUC 0.960 (samples) / 0.954 (map); P-A target 13.4 % of the area captures 90 % of test deposits | 49 s |
  | Th regression | test R² 0.282, cut-off AUC 0.772; top layer: airborne thorium | 113 s |

## One-class / anomaly detection (unsupervised): isolation forest, one-class SVM, autoencoder

These are three separate methods: `clf_iforest`, `clf_ocsvm` and `clf_autoencoder`, each with its own
`run_*_classification.m`. They are classification only: target delineation, not grade estimation.

**How they differ from the supervised methods.** The published MPM applications treat the evidence signature of
mineralised ground as a rare *multivariate anomaly* of the study area:
- isolation forest: Chen & Wu (2019);
- one-class SVM: Chen & Wu (2017);
- autoencoder: Xiong & Zuo (2016).

The pipeline therefore works as follows (method field `learning = 'unsupervised'`):

1. **Fit without labels** on `cfg.anomaly.n_fit` (10 000) random cells of the whole study area
   (`cfg.anomaly.fit_cells = 'all'`). Option `'background'` keeps only cells away from the training deposits
   (novelty detection).
2. **Deposits are not used in the fit.** Training deposits vs. training background only select the
   hyper-parameters (highest AUC; no CV folds are needed because the fit never sees labels). The threshold is the
   **Youden index** of the training ROC, as in Chen & Wu. Test deposits stay independent.
3. Everything after that is shared with the other methods: the score map ("anomaly score" / "reconstruction
   error"), ROC/AUC, success-rate, P-A, C-A, binary targets and importance. A numeric `cfg.threshold` is replaced
   by the Youden index, because anomaly scores are not probabilities.

| Method | Score (higher = more anomalous) | Implementation | Tuned |
|---|---|---|---|
| Isolation forest | `s = 2^(−E(h)/c(ψ))` ∈ (0, 1] | own, from Liu et al. (2008) | `SampleSize` ψ |
| One-class SVM | `1 − Σ αᵢ k(xᵢ, x)` (Σα = 1) | `fitcsvm` one-class QP, ρ recomputed | `Nu` × `KernelScale` |
| Autoencoder | reconstruction MSE `‖x − x̂‖²/P` | own back-propagation + Adam | `HiddenSizes` × `Bottleneck` |

**Isolation forest** (Liu, Ting & Zhou 2008):
- Algorithms 1–3 and Eqs. 1–2, with t = 100 trees and ψ points per tree drawn without replacement.
- Height limit `ceiling(log2 ψ)`; random attribute and random split between min and max.
- `c(n) = 2H(n−1) − 2(n−1)/n` (exact harmonic numbers); `E(h)` includes `c(Size)` at external nodes.
- MATLAB's `iforest` grows trees without the height limit, so it is used only as a cross-check.

**One-class SVM** (Schölkopf et al. 2001): `f(x) = Σ αᵢ k(xᵢ, x) − ρ`, with `0 ≤ αᵢ ≤ 1/(νl)` and `Σ αᵢ = 1`,
Gaussian kernel.
- The QP is solved by `fitcsvm` (one-class SMO).
- **Its `Bias` is not the paper's ρ:** with it, the ν-property fails (0.2 % outliers at ν = 0.2).
  ρ is therefore recomputed from the free support vectors (KKT conditions).
- `MaxSamples` (3000) caps the fit cells, because the QP is O(l²).

**Autoencoder** (Xiong & Zuo 2016; architecture of Hinton & Salakhutdinov 2006):
- Symmetric layers `P → HiddenSizes → code (linear) → mirror → P`, tanh hidden layers, MSE + L2 loss, Adam
  (Kingma & Ba 2015), Glorot initialisation.
- It is written in `common/model/mpm_autoencoder_*.m` because `trainnet` costs ~30 ms per iteration on such small
  networks: 52 s vs. 6.4 s for the same fit.
- No RBM pre-training: it isn't needed for shallow networks.

**Checks on synthetic data** (2000 normal points + 40 anomalies):

| Check | Result |
|---|---|
| Isolation forest AUC (own / MATLAB `iforest`) | 0.9999 / 0.9998; scores rank-correlated 0.97 |
| Isolation forest vectorised scoring vs. literal recursive Algorithm 3 | identical (difference 0) |
| One-class SVM ν-property | 5.2 % outside the support at ν = 0.05, 20.0 % at ν = 0.2 |
| Autoencoder gradients vs. finite differences | 2e−10 |
| Autoencoder vs. `trainnet` (same settings) | MSE 0.1568 / 0.1558, AUC 0.9998 / 0.9998, Spearman 0.997 |

**Carlin demo** (seed 42; test AUC on the map):

| Method | Test map AUC | Time |
|---|---|---|
| Isolation forest | 0.56 | 135 s |
| One-class SVM | 0.62 | 71 s |
| Autoencoder | 0.75 | 86 s |
| Supervised methods (for comparison) | 0.88–0.95 | |

Without the geology layer, all three drop to ≈ 0.45–0.5. Carlin-type deposits are therefore anomalous mainly in
the geology, not in the continuous geophysics. Unsupervised detectors cannot know which anomalies are
ore-related; use them where deposits are too few or absent for supervised training, or as an independent
second opinion.

## Positive-unlabelled (PU) bagging

`clf_pubag` (classification only). In MPM only the deposits are labelled; every other cell is either barren
or an undiscovered deposit. The supervised methods treat random cells beyond `cfg.negatives.min_distance` as
negatives, which assumes they are barren. PU learning drops that assumption (method field `learning = 'pu'`):

1. **Unlabelled set** instead of negatives: `cfg.pu.n_unlabelled` (10 000) random valid cells of the whole
   study area, with **no distance buffer**. Only the training deposits and the test background are excluded
   (test deposits are not: in a PU setting they are unknown). The usual training negatives are still drawn
   and discarded, so the test background is identical to that of the other methods.
2. **Bagging PU** (Mordelet & Vert 2014, Algorithm 1): `NumBags` (100) classifiers, each trained on all
   deposits vs. a bootstrap subsample of K = `SubsampleRatio` × N_P unlabelled cells (paper: K = N_P); the
   score is their average, the uncertainty map their standard deviation. Out-of-bag scores of the unlabelled
   training cells (Algorithm 2) are stored in the model (`oob_score`).
3. **Base learner:** the paper's **biased SVM** (`fitcsvm`, Gaussian kernel, uniform prior so that
   C+·N+ = C−·K) by default, or a classification tree (`BaseLearner = 'tree'`, as in MPM practice:
   Farahbakhsh et al. 2023). All SVM members share the kernel, so they are merged into one kernel expansion
   over the distinct support vectors (one kernel matrix per block of cells).
4. **Tuning:** CV AUC of deposits vs. unlabelled cells. With a fraction γ of hidden deposits in U,
   AUC_PU = γ/2 + (1 − γ)·AUC_PN, a linear increasing function of the true AUC, so it selects the same model.
   The score is not a probability → Youden threshold.

Engine: `common/model/mpm_pubag_train.m`, `mpm_pubag_predict.m` (own code; MATLAB has no PU learning).

**Checks on synthetic data** (10 features, 30 labelled positives, 3000 unlabelled with a share γ of hidden
positives; test AUC of positives vs. negatives, mean of 5 repetitions):

| γ | Oracle SVM (true negatives) | Biased SVM on all of U | PU bagging SVM (K = N_P) | PU bagging tree |
|---|---|---|---|---|
| 0.05 | 0.855 | 0.830 | 0.905 | 0.885 |
| 0.20 | 0.828 | 0.751 | 0.887 | 0.866 |
| 0.40 | 0.876 | 0.773 | 0.888 | 0.848 |

- The AUC identity holds (measured AUC_PU 0.892 / 0.821 / 0.736 vs. predicted 0.892 / 0.818 / 0.733).
- The merged kernel expansion equals the `fitcsvm` member scores (difference 1.6e−15).
- The out-of-bag scores rank hidden positives in U above true negatives (AUC 0.894 at γ = 0.2).

**Carlin demo** (seed 42; 24 training deposits, 10 000 unlabelled cells; same 10 + 10 test samples as SVM):
KernelScale 4, C 0.1, K = N_P (CV AUC_PU 0.911; the CV surface is flat, 0.90–0.91, for KernelScale ≥ 2).
Test AUC 0.950 (samples; SVM 0.950) and **0.962 (map; SVM 0.947)**. The P-A target covers 15.4 % of the
area and captures 83 % of the training and **all 10 test deposits**. Top layer: geology. 3 min.

## CNN on image patches (M9)

`clf_cnn` (classification only). Each cell is classified from the **w × w window of all evidence layers** around it
(method field `input = 'patch'`: the pipeline builds the z-scored evidence cube, `mpm_patch_source`, and passes
cell indices), so the network can use spatial context (contacts, structures, gradients near the cell).

- **Network** (own code, `common/model/mpm_cnn_*.m`): LeNet-type, conv 3×3 → ReLU → conv 3×3 → ReLU (valid) →
  global average pooling → dropout → sigmoid; weighted cross-entropy + L2; Adam. Written with im2col matrix
  products and exact back-propagation (checked against `dlgradient`, 2e−16), because `trainnet` was 270 ms and a
  `dlarray` loop 82 ms per iteration on this CPU.
- **CPU or NVIDIA GPU:** `cfg.params.ExecutionEnvironment = 'auto'` (default) uses the GPU when `canUseGPU` is true
  (Parallel Computing Toolbox + CUDA driver), otherwise the CPU; `'gpu'` / `'cpu'` force it. Only array operations
  are used, so the same code runs on `gpuArray`. (The GPU path could not be run on the development PC, which has no
  NVIDIA GPU.)
- **Samples:** sliding-window augmentation (every window containing a training deposit is a positive; background
  windows likewise, without deposits; Ding et al. 2023) and random-drop augmentation (5 % of input entries zeroed in
  every mini-batch; Li et al. 2021). Rotation/flip optional (off: geological orientation matters).
- **Map:** valid convolutions + global average pooling = fully convolutional network, so the whole map is scored by
  convolving the cube once (identical to a patch per cell, 2e−7; ~11 s per network for 0.91 M cells on the CPU).
- **Spatial leakage:** windows of nearby cells overlap. CV folds keep samples within the window reach
  (`M.spatial_reach`, Chebyshev 2h + shift) together; use a block split for the test set.
- **Ensemble:** `NumNets` (5) networks; uncertainty map = SD of their probabilities. Tuned: `PatchSize` [5 9 13].

**Checks:** synthetic map with deposits only on lithological contacts — test AUC CNN 1.000 vs per-cell RF 0.645 and
MLP 0.712. **Carlin** (seed 42): window 5×5 selected (CV 0.721 / 0.691 / 0.660 for 5 / 9 / 13); test map AUC 0.908
(random split; SVM 0.947). **Spatial block split** (20 km): CNN 0.897 vs SVM 0.993. On Carlin the signal is local
(host geology), so spatial context does not help; the CNN is meant for evidence whose spatial arrangement matters.

## Bayesian network classifiers (M10)

`clf_bn` (classification only), after Porwal, Carranza & Hale (2006), who compared naive, augmented naive and
selective naive Bayesian classifiers for mineral potential mapping. Own code (`common/model/mpm_bn_*.m`); MATLAB's
`fitcnb` covers naive Bayes only (used as a cross-check).

- **Nodes:** one per evidence layer. Categorical layers keep their classes (+ a missing state); the pipeline
  publishes which one-hot columns belong to which layer (`mpm_feature_info`). Continuous layers are discretised on
  the training data of every fit: MDL (Fayyad & Irani 1993) or equal-frequency classes.
- **Structures** (tuned): `naive` (= weights of evidence for binary evidence; verified to 4e−16), `tan`
  (tree-augmented naive Bayes, Friedman et al. 1997: maximum spanning tree of I(A_i; A_j | C)) and `selective`
  (only conditionally independent predictors, G-test on I(A_i; A_j | C)).
- **Parameters:** Dirichlet-smoothed tables (N0 = 5 towards the marginal, Friedman et al. 1997).
  **Uncertainty map:** SD of the posterior over draws from the Dirichlet posteriors of all tables.
- **Native importance:** mutual information of each layer with the class.

**Checks:** TAN recovers the dependent pairs of a synthetic network and beats naive Bayes where the dependence
matters (AUC 0.932 vs 0.701); naive Bayes equals `fitcnb` (Spearman 1.0000). **Carlin** (seed 42): naive Bayes with
equal-frequency classes selected (CV AUC 0.833; TAN 0.766 — too many parameters for 24 deposits); test map AUC
0.948, P-A target 10.7 % of the area capturing 80 % of the test deposits; 1 min.
