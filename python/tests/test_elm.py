"""Extreme learning machine: least-squares output weights, regularisation, ensemble, importance, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature
from logic.data_driven.elm import ELMModel, random_hidden, solve_output

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def _toy(seed=3, n=90):
    rng = np.random.default_rng(seed)
    sig = np.ma.array(rng.normal(size=(n, n)))
    noise = np.ma.array(rng.normal(size=(n, n)))
    geo = np.ma.array(rng.integers(1, 4, size=(n, n)).astype(float))
    p = expit(-6.5 + 1.8 * sig + 1.5 * (geo == 2))
    r, c = np.where(rng.random(sig.shape) < p)
    return sig, noise, geo, r, c


def test_output_weights_are_ridge_least_squares():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(80, 3))
    t = (Z[:, 0] > 0).astype(float)
    hl = random_hidden(3, 30, "sigmoid", rng)
    H = hl(Z)
    for C in (0.1, 10.0):
        beta = solve_output(H, t, C)                              # N ≥ L branch
        ref = np.linalg.lstsq(np.r_[H, np.eye(30) / np.sqrt(C)], np.r_[t, np.zeros(30)], rcond=None)[0]
        assert np.allclose(beta, ref, atol=1e-8)
        wide = random_hidden(3, 150, "tanh", rng)                 # N < L branch = same ridge solution
        Hw = wide(Z)
        b1 = solve_output(Hw, t, C)
        b2 = np.linalg.solve(Hw.T @ Hw + np.eye(150) / C, Hw.T @ t)
        assert np.allclose(Hw @ b1, Hw @ b2, atol=1e-6)
    rbf = random_hidden(3, 20, "rbf", rng, Z)
    assert rbf(Z).shape == (80, 20) and np.all((rbf(Z) > 0) & (rbf(Z) <= 1))


def test_elm_signal_tuning_ensemble_and_reproducibility():
    sig, noise, geo, r, c = _toy()
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = ELMModel(neg_ratio=3, min_spacing=0, seed=2, repeats=2, networks=5).fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8 and res.stats["networks"] == 10 and res.stats["platt_A"] < 0
    tun = res.tuning
    assert tun.selected.sum() == 1 and tun[tun.selected].cv_auc.iloc[0] == pytest.approx(tun.cv_auc.max(), abs=1e-4)
    imp = res.importance.set_index("layer")
    assert imp.index[0] == "sig" and imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert res.score[sig > 1.5].mean() > res.score[sig < -1.5].mean() + 0.3
    assert 0 < res.score_std.max() < 0.5 and 0 <= res.score.min() and res.score.max() <= 1
    a = ELMModel(seed=4, tune=False, networks=3).fit_predict(feats[:1], r, c)
    b = ELMModel(seed=4, tune=False, networks=3).fit_predict(feats[:1], r, c)
    assert np.array_equal(a.score, b.score) and len(a.tuning) == 0
    with pytest.raises(ValueError):
        ELMModel(activation="softplus")


@need_data
def test_carlin_elm():
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
    res = ELMModel(neg_ratio=3, min_distance=1000, res=grid.res, seed=1).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.75
