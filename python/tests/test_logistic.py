"""Logistic regression: estimator checks against statsmodels / direct optimisation, design equivalences, sampling."""
import os
import warnings

import numpy as np
import pytest
from scipy import optimize
from scipy.special import expit

from conftest import CARLIN
from logic.common.training import background_candidates, sample_background
from logic.data_driven.logistic_regression import LogisticModel, Predictor, fit_binomial
from logic.data_driven.wofe import Evidence, WofEModel

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def _sim(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.normal(size=n), rng.integers(0, 2, n)]
    y = (rng.random(n) < expit(X @ np.array([-2.0, 1.0, -0.7]))).astype(float)
    return X, y


def test_ml_matches_statsmodels_grouped_binomial():
    import statsmodels.api as sm
    X, y = _sim()
    # grouped rows: successes / failures, as in weighted LR on unique conditions
    rng = np.random.default_rng(1)
    trials = rng.integers(1, 30, X.shape[0]).astype(float)
    a = np.floor(trials * expit(X @ np.array([-2.0, 1.0, -0.7])) + rng.random(X.shape[0]))
    b = trials - a
    mine = fit_binomial(X, a, b, "ml")
    ref = sm.GLM(np.c_[a, b], X, family=sm.families.Binomial()).fit()
    assert mine.converged
    assert np.allclose(mine.beta, ref.params, atol=1e-7)
    assert np.allclose(np.sqrt(np.diag(mine.cov)), ref.bse, rtol=1e-6)
    assert mine.loglik == pytest.approx(ref.llf - np.sum(_log_binom(a, b)), rel=1e-9)


def test_ml_reproduces_arcsdm_sdmlr_exe():
    """ArcSDM's classic Logistic Regression (sdmlr.exe, ArcSDM 5.03) on unique conditions: successes = training
    points, failures = area in unit cells. Reference numbers = sdmlr.exe output (logco.dat / logpol.tba) for this
    case.dat (ids 1-6: x1, x2, points, area)."""
    X = np.c_[np.ones(6), [1, 1, 1, 2, 2, 2], [1, 2, 3, 1, 2, 3]]
    pts = np.array([2, 3, 1, 4, 6, 5], float)
    area = np.array([500, 300, 200, 150, 100, 50], float)
    f = fit_binomial(X, pts, area, "ml")
    assert np.allclose(f.beta, [-8.224767, 2.141750, 0.518300], atol=5e-6)
    assert np.allclose(np.sqrt(np.diag(f.cov)), [1.041194, 0.488670, 0.277753], atol=1e-5)
    p = expit(X @ f.beta)
    sd = p * (1 - p) * np.sqrt(np.einsum("ij,jk,ik->i", X, f.cov, X))
    assert np.allclose(np.round(p, 5), [0.00382, 0.00639, 0.01069, 0.03159, 0.05192, 0.08422], atol=1e-5)
    assert np.allclose(np.round(sd, 5), [0.00190, 0.00260, 0.00517, 0.01155, 0.01308, 0.03016], atol=1e-5)
    assert np.allclose(np.round(p / sd, 5), [2.00480, 2.45574, 2.06544, 2.73357, 3.96816, 2.79254], atol=2e-3)


def _log_binom(a, b):
    from scipy.special import gammaln
    return gammaln(a + b + 1) - gammaln(a + 1) - gammaln(b + 1)


def test_firth_maximises_penalised_likelihood_and_survives_separation():
    rng = np.random.default_rng(3)
    n = 60
    x = rng.normal(size=n)
    X = np.c_[np.ones(n), x]
    y = (x > 0.3).astype(float)                       # complete separation: ML diverges
    ml = fit_binomial(X, y, 1 - y, "ml")
    assert ml.separation
    fi = fit_binomial(X, y, 1 - y, "firth")
    assert fi.converged and np.all(np.isfinite(fi.beta)) and abs(fi.beta[1]) < 50

    def neg_pen(beta):
        eta = X @ beta
        p = expit(eta)
        ll = -(y @ np.logaddexp(0, -eta)) - ((1 - y) @ np.logaddexp(0, eta))
        info = (X * (p * (1 - p))[:, None]).T @ X
        return -(ll + 0.5 * np.linalg.slogdet(info)[1])
    opt = optimize.minimize(neg_pen, np.zeros(2), method="BFGS", options=dict(gtol=1e-10))
    assert fi.objective >= -opt.fun - 1e-8                     # at least as good as a generic optimiser
    assert np.allclose(fi.beta, opt.x, atol=1e-3)
    # Firth's modified score is zero at the estimate
    p = expit(X @ fi.beta)
    w = p * (1 - p)
    h = w * ((X @ np.linalg.inv((X * w[:, None]).T @ X)) * X).sum(1)
    assert np.allclose(X.T @ (y - p + h * (0.5 - p)), 0, atol=1e-6)


def test_firth_penalised_lr_test_agrees_with_wald_in_large_samples():
    """Regression test: the restricted fit must keep the full model's penalty (Heinze & Schemper 2002)."""
    rng = np.random.default_rng(2)
    sig = np.ma.array(rng.normal(size=(150, 150)))
    noise = np.ma.array(rng.normal(size=(150, 150)))
    r, c = np.where(rng.random(sig.shape) < expit(-6 + 1.5 * sig))
    preds = [Predictor("sig", sig), Predictor("noise", noise)]
    fi = LogisticModel(estimator="firth").fit_predict(preds, r, c).variables.set_index("layer")
    ml = LogisticModel(estimator="ml").fit_predict(preds, r, c).variables.set_index("layer")
    assert fi.loc["noise", "lr_chi2"] == pytest.approx(fi.loc["noise", "wald_chi2"], rel=0.05)
    assert fi.loc["noise", "lr_chi2"] == pytest.approx(ml.loc["noise", "lr_chi2"], rel=0.02)


def test_ridge_shrinks_slopes_not_intercept_penalty():
    X, y = _sim()
    ml = fit_binomial(X, y, 1 - y, "ml")
    rd = fit_binomial(X, y, 1 - y, "ridge", ridge=500.0)
    assert np.all(np.abs(rd.beta[1:]) < np.abs(ml.beta[1:]))


def _grid_toy(seed=0):
    rng = np.random.default_rng(seed)
    geo = np.ma.array(rng.integers(1, 4, size=(80, 80)).astype(float))
    fav = np.ma.array((rng.random((80, 80)) < 0.3).astype(float))
    p = expit(-5 + 1.5 * (geo == 3) + 1.2 * fav)
    dep = rng.random((80, 80)) < p
    r, c = np.where(dep)
    return geo, fav, r, c


def test_unique_conditions_equal_cell_level_fit():
    geo, fav, r, c = _grid_toy()
    preds = [Predictor("geo", geo, "categorical"), Predictor("fav", fav, "categorical")]
    agg = LogisticModel(estimator="ml").fit_predict(preds, r, c)
    assert "unique conditions" in agg.fit_table.set_index("item").loc["Training design", "value"]
    # same model with the binary layer as a continuous 0/1 variable -> per-cell rows
    cell = LogisticModel(estimator="ml").fit_predict([preds[0], Predictor("fav", fav, "continuous")], r, c)
    pa, pc = agg.logit.compressed(), cell.logit.compressed()
    assert np.allclose(pa, pc, atol=1e-6)
    # ML with an intercept reproduces the observed number of deposits
    assert agg.stats["predicted"] == pytest.approx(len(r), rel=1e-6)


def test_single_binary_layer_coefficient_equals_wofe_contrast():
    """One binary pattern: LR is saturated, so B = C = W+ − W− and P = D_k / A_k (Agterberg et al. 1993)."""
    _, fav, r, c = _grid_toy(5)
    lr = LogisticModel(estimator="ml").fit_predict([Predictor("fav", fav, "continuous")], r, c)
    w = WofEModel(1, 1).fit_predict([Evidence("fav", fav.astype(int), "categorical")], r, c)
    t = w.layers[0].table.set_index("cls")
    contrast = t.loc[1, "W+"] - t.loc[0, "W+"]           # categorical W+ of class 1 minus that of class 0
    b = lr.coefficients.set_index("term").loc["fav", "B"]
    assert b == pytest.approx(contrast, abs=1e-6)
    assert np.allclose(lr.probability, w.posterior, atol=1e-9)


def test_prior_correction_recovers_intercept():
    rng = np.random.default_rng(11)
    x = np.ma.array(rng.normal(size=(300, 300)))
    true = np.array([-6.0, 1.0])
    dep = rng.random(x.shape) < expit(true[0] + true[1] * x)
    r, c = np.where(dep)
    full = LogisticModel(estimator="ml").fit_predict([Predictor("x", x)], r, c)
    samp = LogisticModel("sampled", estimator="ml", neg_ratio=20, repeats=1, seed=4).fit_predict(
        [Predictor("x", x)], r, c)
    raw = LogisticModel("sampled", estimator="ml", neg_ratio=20, prior_correction=False, seed=4).fit_predict(
        [Predictor("x", x)], r, c)
    b_full = full.coefficients.set_index("term")["B"]
    b_samp = samp.coefficients.set_index("term")["B"]
    b_raw = raw.coefficients.set_index("term")["B"]
    assert b_samp["(intercept)"] == pytest.approx(b_full["(intercept)"], abs=0.25)
    assert abs(b_raw["(intercept)"] - b_full["(intercept)"]) > 2.0          # uncorrected intercept is far off
    assert b_samp["x"] == pytest.approx(b_raw["x"], abs=1e-9)               # slopes are unaffected
    assert samp.stats["predicted"] == pytest.approx(len(r), rel=0.25)


def test_repeats_pool_and_selection_drops_noise():
    rng = np.random.default_rng(4)                            # noise unrelated to the deposits (full-data p ≈ 0.87)
    sig = np.ma.array(rng.normal(size=(150, 150)))
    noise = np.ma.array(rng.normal(size=(150, 150)))
    dep = rng.random(sig.shape) < expit(-6 + 1.5 * sig)
    r, c = np.where(dep)
    res = LogisticModel("sampled", neg_ratio=5, repeats=4, selection="backward_lr", alpha=0.05, seed=1).fit_predict(
        [Predictor("sig", sig), Predictor("noise", noise)], r, c)
    assert res.params["active"] == ["sig"]
    assert len(res.repeats) == 4 and res.repeats["auc_training"].min() > 0.8
    assert (res.probability_std.compressed() >= 0).all()
    assert set(res.variables["status"]) == {"in model", "removed"}
    # ArcSDM-style sample report: confusion matrix + leave-one-out cross-validation
    sv = res.sample_validation.set_index("item")["value"]
    assert sv["Cross-validation"].startswith("10-fold")         # > 300 samples: leave-one-out is capped
    assert float(sv["Cross-validated AUC (pooled held-out scores)"]) > 0.8
    tn, fp = map(int, sv["Confusion matrix: true non-deposit → predicted non-deposit / deposit"].split(" / "))
    fn, tp = map(int, sv["Confusion matrix: true deposit → predicted non-deposit / deposit"].split(" / "))
    assert tp + fn == len(r) and tn + fp == 5 * len(r)
    small = LogisticModel("sampled", neg_ratio=1, seed=1).fit_predict([Predictor("sig", sig)], r[:40], c[:40])
    sv = small.sample_validation.set_index("item")["value"]
    assert sv["Cross-validation"] == "leave-one-out" and float(sv["Cross-validated AUC (pooled held-out scores)"]) > 0.7


def test_log_transform_and_missing_handling():
    rng = np.random.default_rng(6)
    g = np.ma.array(np.exp(rng.normal(size=(60, 60))))
    g[:5, :] = np.ma.masked
    dep = rng.random(g.shape) < 0.03
    r, c = np.where(dep & ~np.ma.getmaskarray(g))
    ex = LogisticModel().fit_predict([Predictor("g", g, transform="log10")], r, c)
    im = LogisticModel(missing="impute").fit_predict([Predictor("g", g, transform="log10")], r, c,
                                                     study_mask=np.ones(g.shape, bool))
    assert ex.study_mask.sum() == 55 * 60 and im.study_mask.sum() == 60 * 60
    # ArcSDM missing-data rule: cells without data get the area-weighted mean -> the mean prediction of the layer
    b = im.coefficients.set_index("term")["B"]
    mean_logit = b["(intercept)"] + b["g"] * np.log10(np.ma.compressed(g)).mean()
    assert np.allclose(im.logit[:5, :], mean_logit, atol=1e-9)
    with pytest.raises(ValueError, match="log10"):
        LogisticModel().fit_predict([Predictor("g", np.ma.array(rng.normal(size=(20, 20))), transform="log10")],
                                    [1, 2], [1, 2])


def test_missing_categorical_gets_class_area_fractions():
    """ArcSDM CalcVals4Msng: a free (categorical) theme's dummies are set to the class area fractions."""
    rng = np.random.default_rng(8)
    geo = np.ma.array(rng.choice([1.0, 2.0, 3.0], size=(40, 40), p=[0.5, 0.3, 0.2]))
    geo[:4, :] = np.ma.masked
    r, c = np.where(rng.random(geo.shape) < 0.05)
    res = LogisticModel(missing="impute").fit_predict([Predictor("geo", geo, "categorical")], r, c,
                                                      study_mask=np.ones(geo.shape, bool))
    b = res.coefficients.set_index("term")["B"]
    vals = np.ma.compressed(geo)
    frac = {k: np.mean(vals == k) for k in (2.0, 3.0)}                 # class 1 = reference (largest area)
    expected = b["(intercept)"] + b["geo[2]"] * frac[2.0] + b["geo[3]"] * frac[3.0]
    assert np.allclose(res.logit[:4, :], expected, atol=1e-9)


def test_background_sampler_respects_distance():
    study = np.ones((50, 50), bool)
    cand = background_candidates(study, [25], [25], min_distance=10.0, res=(2.0, 2.0))   # 10 m = 5 cells
    rr, cc = np.nonzero(cand)
    assert not cand[25, 25] and np.min(np.hypot((rr - 25) * 2.0, (cc - 25) * 2.0)) >= 10.0
    r, c = sample_background(cand, 30, seed=1)
    assert len(r) == 30 and cand[r, c].all() and len(set(zip(r, c))) == 30
    with pytest.raises(ValueError):
        sample_background(cand, cand.sum() + 1)


@need_data
def test_carlin_logistic_regression_runs():
    import geopandas as gpd
    from evaluation.metrics import evaluate
    from utils.raster_tools import points_to_cells, read_raster

    def load(n):
        return read_raster(os.path.join(CARLIN, "rasters", n + ".tif"))
    geo, grid, _ = load("geology_host")
    preds = [Predictor("geology_host", geo, "categorical"), Predictor("dist_faults", load("dist_faults")[0]),
             Predictor("as_log_idw", load("as_log_idw")[0]), Predictor("sb_log_idw", load("sb_log_idw")[0])]
    mask = ~np.ma.getmaskarray(load("study_mask")[0])
    dep = gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg"))
    rows, cols = points_to_cells(dep, grid, valid_mask=mask)
    model = LogisticModel(unit_cell_km2=1.0, cell_area_km2=grid.cell_area / 1e6)
    # alone, distance to faults is significant: closer = more prospective
    one = model.fit_predict(preds[1:2], rows, cols, mask).coefficients.set_index("term")
    assert one.loc["dist_faults", "B"] < 0 and one.loc["dist_faults", "p_value"] < 0.05
    res = model.fit_predict(preds, rows, cols, mask)
    assert res.probability.max() <= 1 and res.probability.min() >= 0
    assert evaluate(res.probability, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.8
    coef = res.coefficients.set_index("term")
    assert coef.loc["geology_host[2]", "B"] > 0 and coef.loc["sb_log_idw", "B"] > 0
    assert coef.loc["sb_log_idw", "p_value"] < 0.001
    fit = res.fit_table.set_index("item")["value"]
    assert float(fit["Hosmer–Lemeshow p-value"]) > 0.05                 # adequate calibration
    assert res.stats["predicted"] == pytest.approx(res.n_deposits, rel=0.15)
