"""Support vector machine: Platt calibration, decision map = scikit-learn, tuning, importances, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature, FeatureStack
from logic.data_driven.svm import SVMModel, platt_fit, platt_prob

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def _toy(seed=3, n=100):
    rng = np.random.default_rng(seed)
    sig = np.ma.array(rng.normal(size=(n, n)))
    noise = np.ma.array(rng.normal(size=(n, n)))
    geo = np.ma.array(rng.integers(1, 4, size=(n, n)).astype(float))
    p = expit(-6.5 + 1.8 * sig + 1.5 * (geo == 2))
    r, c = np.where(rng.random(sig.shape) < p)
    return sig, noise, geo, r, c


def test_platt_equals_direct_minimisation():
    rng = np.random.default_rng(0)
    f = np.r_[rng.normal(1.0, 1.0, 40), rng.normal(-1.0, 1.0, 120)]
    y = np.r_[np.ones(40, int), np.zeros(120, int)]
    a, b = platt_fit(f, y)
    t = np.where(y == 1, 41 / 42, 1 / 122)                       # Platt's regularised targets

    def nll(v):
        p = np.clip(platt_prob(f, v[0], v[1]), 1e-15, 1 - 1e-15)
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))
    ref = minimize(nll, [0.0, 0.0], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=5000)).x
    assert a == pytest.approx(ref[0], abs=1e-4) and b == pytest.approx(ref[1], abs=1e-4)
    assert a < 0                                                  # larger f -> higher probability
    p = platt_prob(f, a, b)
    assert np.all(np.diff(p[np.argsort(f)]) >= 0)                 # monotone: ranking (AUC) of f is kept


def test_decision_map_equals_sklearn_on_scaled_matrix():
    sig, noise, geo, r, c = _toy(1, 40)
    feats = [Feature("sig", sig), Feature("geo", geo, "categorical")]
    st = FeatureStack(feats, np.ones(sig.shape, bool))
    m = SVMModel(C=2.0, gamma=0.3, tune=False)
    shift, scale = m.scaler(st, "standard")
    X = (st.matrix(np.arange(st.n_cells)) - shift) / scale
    assert np.allclose(X.mean(0), 0, atol=1e-5) and np.allclose(X.std(0), 1, atol=1e-4)
    y = np.zeros(st.n_cells, int)
    y[st.positions(r, c)] = 1
    ids = np.r_[np.flatnonzero(y == 1), np.flatnonzero(y == 0)[::7]]
    svc = m.svc(2.0, 0.3).fit(X[ids], y[ids])
    from sklearn.svm import SVC
    ref = SVC(C=2.0, gamma=0.3, kernel="rbf").fit(X[ids], y[ids]).decision_function(X)
    assert np.allclose(m._decision_map(svc, st, shift, scale), ref, atol=1e-6)
    lo, hi = m.scaler(st, "minmax")
    Xm = (st.matrix(np.arange(st.n_cells)) - lo) / hi
    assert Xm.min() == pytest.approx(0) and Xm.max() == pytest.approx(1)


def test_svm_finds_signal_tunes_and_ranks_importance():
    sig, noise, geo, r, c = _toy()
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = SVMModel(neg_ratio=3, min_spacing=0, seed=2).fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8
    tun = res.tuning
    assert len(tun) == 7 * 6 and tun.selected.sum() == 1
    best = tun[tun.selected].iloc[0]
    assert best.cv_auc == pytest.approx(tun.cv_auc.max(), abs=1e-4)
    assert res.stats["C"] == best.C and res.stats["gamma"] == best.gamma
    imp = res.importance.set_index("layer")
    assert imp.loc["sig", "permutation_auc_drop"] > imp.loc["geo", "permutation_auc_drop"] > \
        imp.loc["noise", "permutation_auc_drop"]
    assert res.score[sig > 1.5].mean() > res.score[sig < -1.5].mean() + 0.3
    assert 0 <= res.score.min() and res.score.max() <= 1 and res.score_std is None
    # score = Platt(decision): the same ranking
    s, d = np.ma.compressed(res.score), np.ma.compressed(res.decision)
    assert np.all(np.diff(s[np.argsort(d)]) >= -1e-12)
    sv = res.sample_validation.set_index("item")["value"]
    assert sv["Cross-validation"].startswith("5-fold")
    tn, fp = map(int, sv["CV confusion: true non-deposit → predicted non-deposit / deposit"].split(" / "))
    fn, tp = map(int, sv["CV confusion: true deposit → predicted non-deposit / deposit"].split(" / "))
    assert tp + fn == res.n_deposits and tn + fp == res.n_negatives


def test_svm_linear_kernel_reproducible_and_repeats():
    sig, noise, geo, r, c = _toy(5, 70)
    feats = [Feature("sig", sig), Feature("geo", geo, "categorical")]
    kw = dict(kernel="linear", tune=False, C=1.0, seed=7, cv_folds=3)
    a = SVMModel(**kw).fit_predict(feats, r, c)
    b = SVMModel(**kw).fit_predict(feats, r, c)
    assert np.array_equal(a.score, b.score) and len(a.tuning) == 0
    assert "linear_weight" in a.importance and a.importance.set_index("layer").loc["sig", "linear_weight"]
    assert float(a.importance.set_index("layer").loc["sig", "linear_weight"].split()[1]) > 0
    rep = SVMModel(**dict(kw, repeats=3)).fit_predict(feats, r, c)
    assert len(rep.repeats) == 3 and rep.score_std is not None and rep.score_std.max() > 0
    assert any("averaged" in m for m in rep.messages)
    with pytest.raises(ValueError):
        SVMModel(cv_folds=0)


@need_data
def test_carlin_svm():
    import geopandas as gpd
    from evaluation.metrics import evaluate
    from utils.raster_tools import points_to_cells, read_raster

    def load(n):
        return read_raster(os.path.join(CARLIN, "rasters", n + ".tif"))
    geo, grid, _ = load("geology_host")
    feats = [Feature("geology_host", geo, "categorical"), Feature("dist_faults", load("dist_faults")[0]),
             Feature("as_log_idw", load("as_log_idw")[0]), Feature("sb_log_idw", load("sb_log_idw")[0])]
    mask = ~np.ma.getmaskarray(load("study_mask")[0])
    rows, cols = points_to_cells(gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg")), grid, valid_mask=mask)
    res = SVMModel(neg_ratio=5, min_distance=1000, res=grid.res, seed=1).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.75
    assert res.importance.layer.iloc[0] in {"sb_log_idw", "as_log_idw", "geology_host"}


def test_score_never_reverses_the_decision_ranking():
    """Few deposits on pure noise: held-out decision values can be anti-informative (CV AUC < 0.5); the Platt
    sigmoid must still increase with f(x), otherwise the score would invert the SVM's ranking."""
    rng = np.random.default_rng(11)
    feats = [Feature("a", np.ma.array(rng.normal(size=(50, 50)))), Feature("b", np.ma.array(rng.normal(size=(50, 50))))]
    for seed in range(4):
        r, c = rng.integers(0, 50, 5), rng.integers(0, 50, 5)
        res = SVMModel(neg_ratio=5, seed=seed, tune=False, cv_folds=5, min_spacing=0).fit_predict(feats, r, c)
        assert res.stats["platt_A"] < 0
        s, d = np.ma.compressed(res.score), np.ma.compressed(res.decision)
        assert np.all(np.diff(s[np.argsort(d, kind="stable")]) >= -1e-12)
