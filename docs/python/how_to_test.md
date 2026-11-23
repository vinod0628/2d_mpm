# How to test MPM Studio (Python)

This guide explains how to check the Python application at every level:
- the algorithms;
- the user interface, run headless;
- method performance on the demo data;
- the offline environment;
- the packaged executable.

All commands are for Windows `cmd`. Unless stated otherwise, they are run from the `python\` folder of the repository.

| Level | What it checks | Command | Time |
|---|---|---|---|
| Unit / model tests | formulas, algorithms, data tools, translations | `..\.venv\Scripts\python.exe -m pytest tests -q -k "not gui"` | ~1–2 min |
| GUI tests (offscreen) | dialogs, map canvas, full method runs as a user would do them | `..\.venv\Scripts\python.exe -m pytest tests\test_gui.py -q` | ~3 min |
| Full test suite | everything above + benchmark | `..\.venv\Scripts\python.exe -m pytest tests -q` | ~5 min |
| Method benchmark | training / test ROC, AUC per method on the demo projects | `..\.venv\Scripts\python.exe tools\benchmark_methods.py` | ~1 min |
| Environment | installed packages consistent, key libraries work together | `tools\pip_check.py`, `tools\smoke_test.py` | ~2 min |
| Release | the built exe starts and runs a model | `build_release.bat` (runs `--selftest`) | ~10 min |

## 1. Before you start

- **Python environment.** `.venv` in the repository root, Python 3.10, created offline with `python\install_env.bat`. See [offline_install.md](offline_install.md).
- **Demo data.** Tests that need real data use `demo_data\nevada_carlin_au\processed` and `demo_data\curnamona_ree\processed`. Without them, those tests are **skipped**, not failed. To create the data:
  ```bat
  cd python
  ..\.venv\Scripts\python.exe -m utils.prepare_demo_data
  ```
- **Isolation.** Tests never write into `demo_data`: they copy a demo project into a temporary folder. User settings (language, theme, recent projects) are redirected to a throw-away `.ini` file (`tests\conftest.py`). The GUI runs with `QT_QPA_PLATFORM=offscreen`, so no windows appear.

## 2. Running pytest

```bat
cd python
..\.venv\Scripts\python.exe -m pytest tests -q                                   :: everything
..\.venv\Scripts\python.exe -m pytest tests\test_logistic.py -q                  :: one file
..\.venv\Scripts\python.exe -m pytest tests\test_gui.py::test_logistic_regression_workflow -q   :: one test
..\.venv\Scripts\python.exe -m pytest tests -q -k "wofe or roc"                   :: by name
..\.venv\Scripts\python.exe -m pytest tests -q -k "not carlin and not gui"        :: quick, no large data
```

Useful options:

| Option | Use |
|---|---|
| `-x` | stop at the first failure |
| `-s` | show `print` output (e.g. the benchmark report) |
| `-v` | one line per test with its name |
| `--durations=10` | the 10 slowest tests |
| `-p no:cacheprovider` | do not create `.pytest_cache` (keeps the source tree clean) |
| `-W ignore::DeprecationWarning` | hide library deprecation warnings (rasterio, TensorFlow) |
| `-o faulthandler_timeout=300` | if a test hangs > 300 s, print the stack of every thread (shows *where* it waits) |

**Focused testing during development.** Run only the tests of the code you changed, for example `-k logistic`. Run the full suite before a release.

## 3. Test files

| File | Tests | What it covers | Demo data |
|---|---|---|---|
| `test_wofe.py` | 16 | WofE formulas (hand calculation), zero-deposit handling, cumulative generalisation, integer layers with many classes, missing-data variance against a literal port of ArcSDM's raster algebra, CI report, DeLong AUC against brute force, absence points, agreement with `eis_toolkit`, full Carlin model | partly |
| `test_logistic.py` | 13 | LR: ML vs statsmodels, ML vs ArcSDM `sdmlr.exe` output, Firth vs direct maximisation (+ modified score = 0), penalised LR test, ridge, unique conditions = per-cell fit, one binary layer = WofE contrast, prior correction, repeats + variable selection + cross-validation, log10 / missing data (ArcSDM rule), non-deposit sampler, Carlin model | partly |
| `test_random_forest.py` | 5 | RF: feature encoding (one-hot, imputation), tree-vote mean / std = scikit-learn, signal vs noise (importance, score, OOB, CV, OOB confusion), reproducibility + pooled repeats, Carlin model | partly |
| `test_svm.py` | 6 | SVM: Platt sigmoid = direct minimisation and monotone, decision map = scikit-learn `SVC` on the scaled matrix, z-score / min–max scaling, C–γ grid search, signal vs noise (importance, score, CV confusion), linear weights, reproducibility + repeats, score never reverses f(x) with few deposits, Carlin model | partly |
| `test_xgboost.py` | 6 | XGBoost: early-stopping rule, signal vs noise (SHAP / gain / permutation, effect direction), SHAP maps add up to the logit of the score, monotone constraint, reproducibility + repeats, missing values kept natively, CUDA request falls back to the CPU without a GPU, Carlin model | partly |
| `test_rvm.py` | 4 | RVM: fast sequential algorithm reaches the evidence maximum and weights of Tipping's re-estimation, moderated prediction, sparsity (few relevance vectors), signal vs noise importance, linear-basis relevance determination, reproducibility + repeats, Carlin model | partly |
| `test_mlp.py` | 4 | MLP: hidden-layer parsing, epoch-by-epoch Adam training learns and is reproducible, early stopping on the held-out AUC (E* = best epoch, patience), network ensemble, signal vs noise importance, Carlin model | partly |
| `test_elm.py` | 3 | ELM: output weights = ridge least squares (N ≥ L and N < L forms), RBF hidden layer, C tuning, ensemble, monotone calibration, importance, reproducibility, Carlin model | partly |
| `test_cnn.py` | 4 | CNN: D4 augmentation (8 distinct symmetries), exact patch windows / zero padding / outside-study cells, spatial signal learned, Carlin model (TensorFlow, slow) | partly |
| `test_bayesian_network.py` | 4 | Bayesian network: naive Bayes on all cells = WofE posterior, conditional mutual information, equal-area classes + missing class, TAN finds the dependent layer and corrects the double counting (predicted ≈ observed), sampled design, Carlin model | partly |
| `test_ebf.py` | 7 | evidential belief functions: class W_CD, W_CD̄, Bel, Dis, Unc, Pls by hand and ln W_CD = W⁺, ln W_CD̄ = W⁻ (WofE code), binary layer Unc = 0 and Dis_A = Bel_B (with and without the 0.5 correction), Bel = 1 certainty, Dempster's rule by hand / sums to 1 / commutative / associative / vacuous identity, learning with quantile and class layers (equal-area classes, single-layer AUC, Pls and mid scores), vacuous missing evidence (only the other layers decide) and exclusion, Bel = 1 warning and saturation removed by the 0.5 correction, tied quantiles, Carlin model, removed from the planned list | partly |
| `test_knowledge_overlays.py` | 4 | fuzzy membership functions (analytic values), operators (γ limits, = eis_toolkit), inference network, missing = 0.5, Boolean conditions, groups and area / deposit statistics, Carlin fuzzy and Boolean models | partly |
| `test_logistic_geometric.py` | 3 | logistic range → 0.01 / 0.99, negative evidence, geometric and weighted geometric mean, P–A weights (random layer → 0), Carlin model | partly |
| `test_index_overlay.py` | 5 | AHP: Saaty example (weights, λmax, CR), consistent matrix recovered with CR = 0, inconsistent judgements CR > 0.1; fuzzy AHP: δ = 0 = geometric-mean method, l ≤ m ≤ u, ranking kept; scoring rules, weighted mean, missing evidence (exclude / renormalise), CR warning, Carlin AHP and fuzzy-AHP models | partly |
| `test_topsis.py` | 5 | TOPSIS = textbook steps (vector / no normalisation, ideal and anti-ideal cells, weights change the ranking); VIKOR S, R, Q = definition for v = 0, 0.5, 1; Shannon-entropy weights by hand (constant layer → 0); missing evidence (renormalise / exclude), CR warning; Carlin TOPSIS-AHP and VIKOR-fuzzy-AHP models, TOPSIS removed from the planned list | partly |
| `test_ms_kprb.py` | 6 | MS-KPRB: analytic gradient = finite differences (missing + categorical proxies), expert network and residual boosting = the user's `kprb.py`, weights follow the data / held by a high κ, residual trees only for a control outside the expert model, Carlin model | partly |
| `test_hybrid_improved.py` | 9 | KP-RF pooling λ = κ/(κ + N) and limits λ = 0 / 1, knowledge-guided negatives; KW-SVM kernel weights; BS-WofE κ = 0 / ∞ limits, weight formula, Σ P = N, dependence deflation, direction flag; PT-Hurdle gate and tract; Carlin BS-WofE and KP-RF | partly |
| `test_dc_fuzzy.py` | 7 | data-calibrated fuzzy memberships: multi-class weights by hand (½ for empty counts), isotonic fit = scikit-learn (both directions), 1-D logistic regression = scikit-learn, σ(W) / stretched scaling, γ operator = fuzzy logic `combine` (missing layers left out), WofE and LR calibrations learn and stay monotone, noise layer not flagged, wrong expert direction flagged and neutralised (free fit follows the data), expert pooling limits (κ → ∞ = expert map, λ = ½), Carlin WofE and LR / γ models, removed from the planned list | partly |
| `test_neuro_fuzzy.py` | 6 | neuro-fuzzy (ANFIS-type): Takagi–Sugeno forward pass = definition (Gaussian MFs crossing at 0.5, grid-partition rules, normalised firing), premise gradient = finite differences (first and zero order), knowledge prior limits (κ → ∞ = expert rule outputs, κ = 0 = plain least squares), hybrid learning recovers an AND rule, learning rate 0 keeps the expert MFs, input choice / rule limit, AND-like expert prototypes, noise layer least important, reproducibility, Carlin model, ANFIS removed from the planned list | partly |
| `test_kt_rk.py` | 6 | KT-RK: strike rotation, block-wise kriging = scikit-learn GP, knowledge trend used when it explains the target, one-SE rule drops an uninformative trend, slope ≥ 0, per-cell compositing of samples | no |
| `test_fuzzy_wofe.py` | 6 | fuzzy weight limits W(1) = W⁺, W(0) = W⁻ and Poisson delta-method variances, grouped likelihood = logistic regression, β ≥ 0 bound with Σ P = N, classic and modified posteriors, duplicated layer deflated, data memberships 0.01 … 0.99, missing evidence W = 0, Carlin model | partly |
| `test_geo_env.py` | 4 | offline coordinate systems: PROJ_LIB / PROJ_DATA / GDAL_DATA of other GIS software are ignored (own ones kept), local EPSG lookup works, unrecognised CRSs (none, LOCAL_CS, "Undefined SRS") detected; such rasters / vectors load as stored in the project CRS, known CRSs are still reprojected | no |
| `test_training.py` | 5 | point selection: Hilbert curve, spatially stratified train/test split, ArcSDM thinning, non-deposit samples (inside study area, outside buffer, minimum spacing), training / test point layers keep the original points | no |
| `test_polygon_raster.py` | 6 | polygon to categorical raster: sequential / field codes (text and 1.5 refused, codes ≤ 0 → int32 with project NoData), class table with counts and the attributes constant per class, gap class or NoData, study-area mask, cell centre vs maximum area (narrow unit, sliver polygon reported as empty), skipped unlabeled features, grid snapped to the project grid lines, Curnamona map units on the study mask | partly |
| `test_statistics.py` | 4 | layer Statistics tab: count, nulls, mean, population standard deviation, skewness, Pearson kurtosis and quartiles by hand; normal sample (skewness 0, kurtosis 3), constant / empty values; histogram limits by number of intervals and by interval width (aligned, maximum covered, too many intervals refused); Sturges default, round widths, empirical cumulative distribution | no |
| `test_utils.py` | 9 | raster / vector tools, reclassification, natural breaks, symbology classification, vector reader, Shapefile writer for ArcMap, NoData type widening | partly |
| `test_splash.py` | 2 | start-up splash: built-in image 600 × 400, larger images scaled down with the aspect ratio kept (never enlarged), missing image → no splash; shown at least the minimum time (2 s; shortened in the test) after a fast start, closed at once after a slow one, then the follow-up action (open project) runs; a click does not close it | yes |
| `test_project.py` | 7 | project files, CRS and cell size, folders, run folders | yes |
| `test_targets.py` | 5 | target delineation (P-A / C-A thresholds) | partly |
| `test_measure.py` | 4 | distance / area measurement | no |
| `test_i18n.py` | 4 | translation catalogs complete with the same `{placeholders}`, Chinese in use, light/dark icon recolouring | no |
| `test_gui.py` | 42 | offscreen GUI workflows on both demo projects: WofE, LR, RF, SVM, XGBoost, RVM, MLP, ELM, CNN, Bayesian network, evidential belief functions, fuzzy / Boolean and logistic-geometric overlays, index overlay (AHP matrix editing), TOPSIS / VIKOR (method switch, entropy weights), hybrid MS-KPRB, KP-RF, KW-SVM, BS-WofE, PT-Hurdle, KT-RK, fuzzy WofE, neuro-fuzzy and data-calibrated fuzzy memberships from dialog to results dialog and run folder (incl. training / test point Shapefiles, ROC button), map drawing order = TOC order (rasters and vectors), identify, coordinates, pan, layer properties / statistics (histogram by intervals or width, cumulative curve, statistics box, PNG export) / symbology / labels, legend export, polygon to raster (reference-raster and study-polygon grids), lazy raster loading, project properties, new-project prompts, dialogs on a 1366 × 768 screen | yes |
| `test_benchmark.py` | 1 | performance report on Carlin (see §5) with minimum AUCs | yes |

Real-data tests are marked `@need_data` and are skipped when the demo data is missing.

## 4. GUI tests

GUI tests create a `MainWindow`, open a copy of a demo project, and drive dialogs exactly as a user would. Methods run in the worker thread (`win.run_task`), and `wait_tasks()` waits for them to finish. The tests then check:
- the output layers;
- the files in `results\<run_id>\`;
- the results dialog.

**Screenshots.** Set `MPM_SCREENSHOT_DIR` to save a PNG of every important window, which is useful for visual checks and documentation:
```bat
set MPM_SCREENSHOT_DIR=C:\temp\mpm_shots
..\.venv\Scripts\python.exe -m pytest tests\test_gui.py -q
```

**Rules for writing GUI tests:**
- **Never leave a modal dialog open.** `exec_()` blocks forever under the offscreen platform, and the test suite then hangs without an error. Patch the dialog first:
  ```python
  monkeypatch.setattr(QMessageBox, "exec_", lambda box: 0)
  monkeypatch.setattr(ProjectPropertiesDialog, "exec_", lambda dlg: 0)   # e.g. opened by New project
  ```
  If a run seems stuck, re-run that test with `-o faulthandler_timeout=60`. The stack trace shows which `exec_()` it is waiting in.
- **Rasters are read lazily** in background threads. Before checking pixels, load them (`layer.load()`) and rebuild the map (`canvas.redraw()`). Otherwise the map may still be blank.
- **Keep views deterministic.** For example, call `canvas.zoom_full()` before every check that compares screen regions.

## 5. Method performance benchmark (training / test AUC)

`tools\benchmark_methods.py` runs every implemented method on the demo projects through the same code path as its dialog. It holds out test deposits and prints a console report.

```bat
..\.venv\Scripts\python.exe tools\benchmark_methods.py                                   :: all methods, both demos
..\.venv\Scripts\python.exe tools\benchmark_methods.py --method logistic_regression --dataset carlin
..\.venv\Scripts\python.exe tools\benchmark_methods.py --test-pct 20 --split random --seed 7
..\.venv\Scripts\python.exe tools\benchmark_methods.py --csv bench.csv --roc-dir roc_curves
..\.venv\Scripts\python.exe -m pytest tests\test_benchmark.py -s                       :: as a test with thresholds
```

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | both | `carlin`, `curnamona` (repeatable) |
| `--method` | all | method key: `wofe`, `logistic_regression`, `random_forest`, `svm`, `xgboost`, `rvm`, `mlp`, `elm`, `cnn`, `bayesian_network` (repeatable) |
| `--test-pct` | 30 | % of deposits held out for testing |
| `--split` | spatial | `spatial` (stratified, even coverage) or `random` (ArcSDM Splitting) |
| `--seed` | 42 | random seed (split, non-deposit samples) |
| `--csv` | — | summary table as CSV |
| `--roc-dir` | — | every ROC curve (`fpr,tpr`) as CSV |

For the **training** and the **test** deposits, the report gives:
- the number of deposits and negatives;
- ROC AUC with the DeLong 95 % CI;
- success-rate AUC;
- the prediction–area intersection (% of deposits / % of area);
- ROC values: TPR at FPR 0.05 / 0.10 / 0.20 / 0.30 / 0.50.

It also repeats each method's warnings and ends with a summary table, for example:

```
=== Summary (ROC AUC)
  dataset    method                              training      test
  carlin     Weights of Evidence                    0.836     0.824
  carlin     Logistic Regression (all cells)        0.910     0.898
  carlin     Logistic Regression (sampled)          0.911     0.895
```

How to read it:
- **Same cells and deposits.** All methods are scored on the same cells (missing evidence excluded) and the same deposits, so the AUCs are directly comparable.
- **Test AUC.** This is the honest measure of predictive ability. With few test deposits its CI is wide; the report prints a note below 10 test deposits. Curnamona has only 7 deposits, so treat its test AUC as indicative only.
- **Training AUC much higher than test AUC** suggests overfitting: too many layers or classes for the number of deposits. Watch the events-per-variable warning.
- **TPR at a low FPR of 0** with a good AUC is normal for WofE. Its map has few distinct values, and the top class already covers more than that share of the area.

**New methods.** Add one entry to `CONFIGS` in `tools\benchmark_methods.py` (evidence and parameters per demo). Until then, the report says the method has no benchmark configuration.

## 6. Reference checks against other implementations

Several tests compare our numbers with independent implementations. Keep these when changing an algorithm.

| Our code | Reference | Test |
|---|---|---|
| WofE weights | `eis_toolkit` 1.1.6 (W+, W−, s(W+)) on Carlin | `test_wofe.py::test_matches_eis_toolkit` |
| WofE missing-data variance | literal port of ArcSDM `missingdatavar_func.py` | `test_wofe.py::test_missing_data_variance_matches_arcsdm` |
| LR maximum likelihood | statsmodels GLM (grouped binomial) | `test_logistic.py::test_ml_matches_statsmodels_grouped_binomial` |
| LR maximum likelihood | ArcSDM 5.03 `sdmlr.exe` output for the same case data | `test_logistic.py::test_ml_reproduces_arcsdm_sdmlr_exe` |
| Firth estimator | generic optimiser on ℓ + ½ ln\|I\| | `test_logistic.py::test_firth_maximises_penalised_likelihood_and_survives_separation` |
| ROC AUC + CI | brute-force Mann–Whitney / DeLong | `test_wofe.py::test_delong_auc_and_absence_points` |

The ArcSDM toolbox copies used as references are in `ArcSDM-master` (latest) and `ArcSDM-5.03.01` (classic logistic regression). They are read-only references: do not modify them. Where our logic intentionally differs from ArcSDM, the reason is documented in [mpm_studio.md](mpm_studio.md) (WofE and LR sections).

## 7. Translation checks

Every text shown in the GUI must be in each catalog (`python\i18n\*.json`). After adding or changing `tr("...")` / `N_("...")` texts:

```bat
..\.venv\Scripts\python.exe tools\i18n_update.py --prune          :: add new keys, remove unused ones
..\.venv\Scripts\python.exe tools\i18n_update.py --missing zh_CN  :: list untranslated texts (file:line)
..\.venv\Scripts\python.exe -m pytest tests\test_i18n.py -q       :: catalogs complete, same {placeholders}
```

## 8. Environment and release checks

From the **repository root**:

```bat
.venv\Scripts\python.exe python\tools\pip_check.py       :: dependency consistency of the installed packages
.venv\Scripts\python.exe python\tools\smoke_test.py      :: GDAL/PROJ, geospatial stack, eis_toolkit, ML libraries, demo runs
```

`install_env.bat` runs both automatically at its last step (`[4/4] pip check + smoke test`).

**Full check from scratch** (offline install → tests → release). Use this before a release, or after changing the wheel set in `python\whl\`:

1. Delete `.venv` (or run `python\install_env.bat /clean`). The install then takes about 15 minutes, followed by `pip check: ok` and `SMOKE TEST PASSED`.
2. Run the full test suite: `cd python && ..\.venv\Scripts\python.exe -m pytest tests -q -o faulthandler_timeout=300`.
3. Run `python\build_release.bat`. It builds the exe with PyInstaller, runs `"MPM Studio.exe" --selftest report.txt` headless (which must print `SELFTEST OK`), and writes `python\distribute\release\MPM_Studio_<version>_win64.zip`.

This chain takes 30–40 minutes. Run it in the background and redirect the output to a log file, for example `full_check.bat > full_check.log 2>&1`. Check the log at the end.

The self-test also works from source: `..\.venv\Scripts\python.exe main.py --selftest report.txt`.

## 9. Adding tests for a new method

Checklist for a new method in `python\logic\<category>\<method>\`:

1. **Model tests** (`tests\test_<method>.py`), with no Qt:
   - a hand-calculated or textbook example;
   - agreement with an independent implementation where one exists (statsmodels, scikit-learn, eis_toolkit, ArcSDM);
   - edge cases: too few deposits, missing data, classes without deposits, constant layers;
   - a real-data run on Carlin with a minimum training AUC, marked `@need_data`.
2. **GUI workflow** (`tests\test_gui.py`): open the dialog on a demo copy, set parameters, check `validate()`, run via `win.run_task(...)`, and check the output layers, run-folder files and results dialog. Patch every modal dialog (§4).
3. **Small screen**: add the dialog to the list in the small-display test (1366 × 768).
4. **Benchmark**: add the method to `CONFIGS` in `tools\benchmark_methods.py`.
5. **Translations**: run `i18n_update.py`, translate, then run `test_i18n.py`.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A test or the suite never finishes | A modal dialog is open (`exec_()`) under the offscreen platform. Re-run with `-o faulthandler_timeout=60` to see where, then patch that dialog's `exec_`. |
| Tests are *skipped* | Demo data is missing. Run `python -m utils.prepare_demo_data`. |
| A pixel comparison fails ("map did not move", blank map) | Rasters were not loaded yet (lazy loading) or the view changed. Call `layer.load()`, `canvas.redraw()` and `canvas.zoom_full()` first. |
| `pip check` reports a conflict | A wheel in `python\whl\` does not match the requirements. Fix the wheel set and run the clean install again. |
| `SELFTEST` fails in the built exe | A package is missing from the bundle. Check `report.txt` and the PyInstaller spec (`python\packaging\`). See [how_to_build_python_app.md](how_to_build_python_app.md). |
| Many `DeprecationWarning`s | Library notices, not errors. Hide them with `-W ignore::DeprecationWarning`. |
