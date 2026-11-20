"""Relevance vector machine: evidence maximisation (vs MacKay re-estimation), sparsity, ARD, prediction, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature
from logic.data_driven.rvm import RVMModel, fit_rvm, predict
from logic.data_driven.rvm.model import _posterior

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


def _mackay(Phi, t, iters=3000):
    """Reference: Tipping (2001) re-estimation α ← γ/μ² (slow but simple), pruning at α > 1e9."""
    keep = np.arange(Phi.shape[1])
    alpha, w = np.ones(keep.size), np.zeros(keep.size)
    for _ in range(iters):
        w, sigma, ev = _posterior(Phi[:, keep], t, alpha, w)
        gamma = 1 - alpha * np.diag(sigma)
        alpha = gamma / np.maximum(w * w, 1e-300)
        drop = (alpha > 1e9) & (keep != 0)
        keep, alpha, w = keep[~drop], alpha[~drop], w[~drop]
    w, sigma, ev = _posterior(Phi[:, keep], t, alpha, w)
    return keep, w, ev


def test_fast_sequential_algorithm_reaches_the_evidence_maximum():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 6))
    t = (rng.random(150) < expit(-0.5 + 1.6 * X[:, 0] - 1.0 * X[:, 3])).astype(float)
    Phi = np.c_[np.ones(150), X]                                    # linear basis: ARD logistic regression
    fit = fit_rvm(Phi, t)
    keep_ref, w_ref, ev_ref = _mackay(Phi, t)
    assert fit.converged and set(fit.keep) == set(keep_ref) and {1, 4} <= set(fit.keep)
    assert fit.log_evidence == pytest.approx(ev_ref, abs=1e-3)
    assert np.allclose(fit.mu, w_ref[np.argsort(keep_ref)], atol=2e-3)
    p, a, s = predict(fit, Phi[:, fit.keep])
    assert np.all((p > 0) & (p < 1)) and np.all(s > 0)
    assert np.all(np.abs(p - 0.5) <= np.abs(expit(a) - 0.5) + 1e-12)           # moderation shrinks towards ½


def test_rvm_is_sparse_finds_signal_and_ranks_importance():
    sig, noise, geo, r, c = _toy(3, 75)
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = RVMModel(neg_ratio=1, min_spacing=0, seed=2).fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8 and res.stats["relevance_vectors"] < 0.1 * res.stats["samples"]
    assert len(res.tuning) == 6 and res.tuning.selected.sum() == 1
    imp = res.importance.set_index("layer")
    assert imp.index[0] == "sig" and imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert res.score[sig > 1.5].mean() > res.score[sig < -1.5].mean() + 0.3
    assert 0 < res.pred_sd.max() < 0.5 and res.score_std is None
    rel = res.relevance
    assert len(rel) == res.stats["relevance_vectors"] and set(rel.cls) <= {"deposit", "non-deposit"}
    assert {"row", "col", "weight", "sig", "geo[2]"} <= set(rel.columns)


def test_linear_basis_relevance_determination_and_repeats():
    sig, noise, geo, r, c = _toy(5, 80)
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    a = RVMModel(kernel="linear", seed=7, cv_folds=3).fit_predict(feats, r, c)
    b = RVMModel(kernel="linear", seed=7, cv_folds=3).fit_predict(feats, r, c)
    assert np.array_equal(a.score, b.score) and len(a.tuning) == 0
    w = a.relevance.set_index("term")
    assert "sig" in w.index and w.loc["sig", "weight"] > 0                # the signal column is relevant
    assert "noise" not in w.index or abs(w.loc["noise", "weight"]) < 0.2 * w.loc["sig", "weight"]
    rep = RVMModel(kernel="linear", seed=7, cv_folds=3, repeats=3).fit_predict(feats, r, c)
    assert len(rep.repeats) == 3 and rep.score_std is not None and rep.score_std.max() > 0


@need_data
def test_carlin_rvm():
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
    res = RVMModel(neg_ratio=3, min_distance=1000, res=grid.res, seed=1).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.75 and res.stats["relevance_vectors"] < 40
