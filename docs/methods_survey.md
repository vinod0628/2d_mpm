# 2D Mineral Prospectivity Mapping (MPM) — methods survey

Working catalogue of methods to implement in `python/` (Python 3.10, eis_toolkit-compatible)
and `matlab/`. Priority: **P1** = core / must-have, **P2** = common extension, **P3** = advanced / research.

Terminology: *data-driven* methods learn weights/parameters from known deposits (training points);
*knowledge-driven* methods take weights/memberships from expert judgement (no training points
needed, deposits used only for validation); *hybrid* methods mix the two.

## 1. Knowledge-driven

| # | Method | Idea | Key refs | Prio |
|---|--------|------|----------|------|
| K1 | Boolean logic | Binary evidence maps combined with AND/OR | Bonham-Carter 1994 | P1 |
| K2 | Index overlay (binary & multi-class) | Weighted sum of class scores, `S = Σ w_i s_ij / Σ w_i` | Bonham-Carter 1994; Carranza 2008 | P1 |
| K3 | Fuzzy logic overlay | Fuzzy memberships (linear, large/small sigmoid, Gaussian, near, categorical table) + operators AND, OR, algebraic product, algebraic sum, **gamma** | An et al. 1991; Bonham-Carter 1994; Porwal et al. 2003 | P1 |
| K4 | Logistic-function fuzzification + geometric average | Data-independent fuzzification of continuous evidence; integration by geometric mean | Yousefi & Carranza 2015 (Comput. Geosci.) | P1 |
| K5 | AHP (Analytic Hierarchy Process) | Pairwise-comparison matrix → eigenvector weights, consistency ratio check, then index overlay | Saaty 1980; Abedi et al. 2012 | P1 |
| K6 | Fuzzy AHP | Triangular fuzzy pairwise judgements (Chang's extent analysis / Buckley) | Buckley 1985; Chang 1996 | P2 |
| K7 | TOPSIS | Distance to ideal / anti-ideal solution | Pazand & Hezarkhani 2015; Abedi et al. 2013 | P2 |
| K8 | VIKOR / PROMETHEE / ELECTRE | Other MCDM outranking methods | Abedi et al. 2012–2013 | P3 |
| K9 | Knowledge-driven evidential belief (Dempster–Shafer) | Expert belief/disbelief/uncertainty per class, Dempster's rule | Moon 1990; An et al. 1994 | P2 |
| K10 | Wildcat / data-driven-weighted index | Weights from geochemical/geophysical anomaly strength with no deposits | Carranza 2008, 2010 | P3 |

## 2. Data-driven — statistical

| # | Method | Idea | Key refs | Prio |
|---|--------|------|----------|------|
| D1 | Weights of Evidence (WofE) | Bayesian log-linear. W+, W−, contrast C, studentized C, cumulative ascending/descending generalisation, categorical WofE, posterior probability + variance | Bonham-Carter et al. 1989; Agterberg 1992 | P1 |
| D1a | Conditional independence tests | Overall/omnibus test (N̂ vs N), Agterberg–Cheng CI test, pairwise χ² | Bonham-Carter 1994; Agterberg & Cheng 2002 | P1 |
| D2 | Logistic regression (LR) | `logit p = β0 + Σ βi xi`; also weighted LR to relax CI | Agterberg 1974; Agterberg et al. 1993 | P1 |
| D3 | Evidential Belief Functions (EBF, data-driven) | Bel/Dis/Unc/Pls from deposit densities per class, Dempster combination | Carranza & Hale 2003; Carranza et al. 2005 | P2 |
| D4 | Frequency ratio / information value | Simple density ratios per class | Lee & Pradhan 2007 | P2 |
| D5 | Fuzzy WofE | WofE with fuzzy memberships | Cheng & Agterberg 1999 | P3 |
| D6 | Maximum entropy (MaxEnt) | Presence-only density estimation | Phillips et al. 2006; Liu et al. 2018 | P3 |

## 3. Data-driven — machine learning

| # | Method | Idea | Key refs | Prio |
|---|--------|------|----------|------|
| M1 | Random Forest (RF) | Bagged decision trees; feature importance | Breiman 2001; Carranza & Laborte 2015; Rodriguez-Galiano et al. 2015 | P1 |
| M2 | Support Vector Machine (SVM) | RBF/linear kernel, probability by Platt scaling | Zuo & Carranza 2011 | P1 |
| M3 | Relevance Vector Machine (RVM) | Sparse Bayesian kernel classifier (Laplace approx., ARD priors) — no maintained sklearn implementation, so we will write our own | Tipping 2001 | P1 |
| M4 | Artificial neural network (MLP) | Feed-forward backprop | Singer & Kouda 1996; Porwal et al. 2003 | P1 |
| M5 | Gradient boosting (XGBoost/LightGBM/sklearn HGB) | Boosted trees | Chen & Guestrin 2016; Zhang et al. 2022 | P2 |
| M6 | Extreme learning machine (ELM) | Random hidden layer + least squares | Chen & Wu 2017 | P3 |
| M7 | One-class / anomaly detection (OC-SVM, Isolation Forest, autoencoder) | Learn background, flag anomalies | Chen & Wu 2017; Xiong & Zuo 2016 | P2 |
| M8 | Positive–unlabelled (PU) bagging | Treat non-deposit cells as unlabelled | Mordelet & Vert 2014; Farahbakhsh et al. 2023 | P2 |
| M9 | CNN on image patches | Spatial context learning | Sun et al. 2020; Li et al. 2020 | P3 |
| M10 | Bayesian network / naive Bayes | Probabilistic graphical classifier | Porwal et al. 2006 | P3 |

## 4. Hybrid

| # | Method | Key refs | Prio |
|---|--------|----------|------|
| H1 | Neuro-fuzzy (ANFIS-style) | Porwal et al. 2004 | P3 |
| H2 | Fuzzy memberships from WofE/LR contrast | Porwal et al. 2003 | P3 |

## 5. Shared pre-processing (needed by all methods)

- Raster stack on a common grid (reproject / resample / clip / mask), unit cell selection
- Vector → raster: categorical geology, **distance-to-feature** (faults, contacts, intrusions), feature density
- Geochemistry: interpolation (IDW, kriging), **compositional transforms** (alr/clr/ilr), robust PCA, factor analysis
- Anomaly thresholds: **concentration–area (C-A) fractal**, mean+2σ, percentiles
- Geophysics derivatives (already provided in the Curnamona set)
- Training data: buffer deposits, **negative-sample selection** (random, distance-based, low-favourability), class balancing (SMOTE, undersampling)
- Spatial cross-validation (block CV) to avoid optimistic accuracy

## 6. Validation / post-processing (needed by all methods)

- ROC / AUC; success-rate and prediction-rate curves (train vs held-out deposits)
- **Prediction–area (P-A) plot** and normalized density (Yousefi & Carranza 2015)
- Capture-efficiency curve; confusion matrix at chosen threshold
- Target delineation by C-A fractal or P-A intersection
- Uncertainty: WofE posterior variance, RF/bagging spread, bootstrap

## 7. References (abbreviated)

> Section 7 lists the core references. Short citations that appear only in the tables (e.g. Chen & Wu 2017,
> Abedi et al. 2012, Liu et al. 2018, Zhang et al. 2022) are pointers to check before citing in a
> publication; they were not each re-verified when this survey was written.

- Agterberg, F.P., Cheng, Q., 2002. Conditional independence test for weights-of-evidence modeling. Nat. Resour. Res. 11, 249–255.
- An, P., Moon, W.M., Rencz, A., 1991. Application of fuzzy set theory for integration of geological, geophysical and remote sensing data. Can. J. Explor. Geophys. 27, 1–11.
- Bonham-Carter, G.F., 1994. Geographic Information Systems for Geoscientists: Modelling with GIS. Pergamon.
- Bonham-Carter, G.F., Agterberg, F.P., Wright, D.F., 1989. Weights of evidence modelling: a new approach to mapping mineral potential. GSC Paper 89-9.
- Carranza, E.J.M., 2008. Geochemical Anomaly and Mineral Prospectivity Mapping in GIS. Elsevier.
- Carranza, E.J.M., Hale, M., 2003. Evidential belief functions for data-driven geologically constrained mapping of gold potential, Baguio district, Philippines. Ore Geol. Rev. 22, 117–132.
- Carranza, E.J.M., Laborte, A.G., 2015. Random forest predictive modeling of mineral prospectivity with small number of prospects and data with missing values in Abra (Philippines). Comput. Geosci. 74, 60–70.
- Farahbakhsh, E., Maughan, J., Müller, R.D., 2023. Prospectivity modelling of critical mineral deposits using a GAN with oversampling and positive-unlabelled bagging. Ore Geol. Rev. 162, 105665.
- Porwal, A., Carranza, E.J.M., Hale, M., 2003. Knowledge-driven and data-driven fuzzy models for predictive mineral potential mapping. Nat. Resour. Res. 12, 1–25.
- Raines, G.L., 2001 (rev. 2006). Resource materials for a GIS spatial analysis course. USGS OFR 01-221.
- Rodriguez-Galiano, V., et al., 2015. Machine learning predictive models for mineral prospectivity: an evaluation of neural networks, random forest, regression trees and support vector machines. Ore Geol. Rev. 71, 804–818.
- Tipping, M.E., 2001. Sparse Bayesian learning and the relevance vector machine. J. Mach. Learn. Res. 1, 211–244.
- Yousefi, M., Carranza, E.J.M., 2015. Fuzzification of continuous-value spatial evidence for mineral prospectivity mapping. Comput. Geosci. 74, 97–109.
- Yousefi, M., Carranza, E.J.M., 2015. Prediction–area (P–A) plot and C–A fractal analysis to classify and evaluate evidential maps for mineral prospectivity modeling. Comput. Geosci. 79, 69–81.
- Zuo, R., Carranza, E.J.M., 2011. Support vector machine: a tool for mapping mineral prospectivity. Comput. Geosci. 37, 1967–1975.
