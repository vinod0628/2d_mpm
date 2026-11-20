"""XGBoost: early stopping, SHAP additivity, monotone constraints, native missing values, device fallback, data."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit, logit

from conftest import CARLIN
from logic.common.features import Feature, FeatureStack, study_area
from logic.data_driven.xgboost_mpm import XGBoostModel, best_round, cuda_available

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


def test_best_round_early_stopping():
    curve = np.array([0.6, 0.7, 0.75, 0.74, 0.74, 0.73, 0.9])
    assert best_round(curve, 3) == 2                       # stops 3 rounds after the best: the late 0.9 is not seen
    assert best_round(curve, 0) == 6 and best_round(curve, 10) == 6


def test_xgboost_signal_importance_and_shap_maps_add_up():
    sig, noise, geo, r, c = _toy()
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = XGBoostModel(neg_ratio=3, min_spacing=0, seed=2, shap=True, device="cpu").fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8 and 1 <= res.stats["trees"] <= 500 and res.stats["device"] == "cpu"
    imp = res.importance.set_index("layer")
    assert imp.index[0] == "sig" and imp.loc["sig", "direction"] > 0.8               # higher signal -> higher score
    assert imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert imp.shap_pct.sum() == pytest.approx(100) and imp.gain_pct.sum() == pytest.approx(100)
    assert res.score[sig > 1.5].mean() > res.score[sig < -1.5].mean() + 0.2 and res.score_std is None
    # TreeSHAP is additive: logit(score) = bias + Σ layer contributions (one booster)
    total = sum(np.ma.getdata(m) for m in res.shap_maps.values())
    bias = np.ma.getdata(logit(np.clip(res.score, 1e-7, 1 - 1e-7))) - total
    assert np.ptp(bias[res.study_mask]) < 1e-3
    c = res.curve
    assert len(c) >= res.stats["trees"] and c.cv_auc.iloc[res.stats["trees"] - 1] == pytest.approx(c.cv_auc.max())


def test_monotone_constraint_and_reproducibility():
    sig, noise, geo, r, c = _toy(5, 80)
    wiggly = np.ma.array(np.sin(3 * np.ma.getdata(sig)))
    feats = [Feature("sig", sig), Feature("wiggly", wiggly)]
    kw = dict(seed=4, device="cpu", early_stopping=0, n_rounds=60, max_depth=4, monotone={"sig": 1})
    a = XGBoostModel(**kw).fit_predict(feats, r, c)
    b = XGBoostModel(**kw).fit_predict(feats, r, c)
    assert np.array_equal(a.score, b.score)
    # with the other layer fixed, the score never decreases with sig
    from logic.common.features import FeatureStack as FS
    import xgboost as xgb
    m = XGBoostModel(**kw)
    st = FS(feats, np.ones(sig.shape, bool))
    X = st.matrix(np.arange(300))
    y = (X[:, 0] + np.random.default_rng(0).normal(size=300) > 0).astype(int)
    bst = xgb.train(m.booster_params(st, y, 1, "cpu"), xgb.DMatrix(X, label=y), 60)
    grid = np.c_[np.linspace(-3, 3, 200), np.full(200, 0.3)].astype(np.float32)
    assert np.all(np.diff(bst.predict(xgb.DMatrix(grid))) >= -1e-7)
    rep = XGBoostModel(**dict(kw, repeats=3)).fit_predict(feats, r, c)
    assert len(rep.repeats) == 3 and rep.score_std is not None and rep.score_std.max() > 0


def test_missing_values_kept_natively():
    sig, noise, geo, r, c = _toy(7, 60)
    holes = np.ma.array(np.ma.getdata(noise), mask=np.zeros(noise.shape, bool))
    holes.mask[:20, :] = True                                         # a third of the area without this layer
    feats = [Feature("sig", sig), Feature("holes", holes)]
    keep = XGBoostModel(missing="keep", device="cpu", seed=1).fit_predict(feats, r, c)
    excl = XGBoostModel(missing="exclude", device="cpu", seed=1).fit_predict(feats, r, c)
    assert keep.study_mask.sum() == sig.size and excl.study_mask.sum() == sig.size - 20 * 60
    assert np.isfinite(np.ma.getdata(keep.score)[:20]).all() and not keep.score.mask[:20].any()
    st = FeatureStack(feats, study_area(feats, None, "keep"), "keep")
    assert np.isnan(st.matrix(np.arange(5))[:, 1]).all()              # NaN reaches the booster


@pytest.mark.skipif(cuda_available(), reason="an NVIDIA GPU is present")
def test_device_auto_and_cuda_fall_back_to_cpu():
    sig, noise, geo, r, c = _toy(9, 50)
    res = XGBoostModel(device="cuda", seed=1, n_rounds=30).fit_predict([Feature("sig", sig)], r, c)
    assert res.stats["device"] == "cpu" and any("GPU" in m for m in res.messages)
    assert XGBoostModel(device="auto").resolved_device() == ("cpu", None)


@need_data
def test_carlin_xgboost():
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
    res = XGBoostModel(neg_ratio=5, min_distance=1000, res=grid.res, seed=1,
                       monotone={"dist_faults": -1, "sb_log_idw": 1}).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.75
    assert res.importance.layer.iloc[0] in {"sb_log_idw", "as_log_idw", "geology_host"}
