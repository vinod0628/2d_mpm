"""Neural network (MLP): early stopping on held-out AUC, ensemble, importance, reproducibility, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature
from logic.data_driven.mlp import MLPModel, parse_hidden

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


def test_parse_hidden():
    assert parse_hidden("32, 16") == (32, 16) and parse_hidden("8") == (8,) and parse_hidden([4, 2]) == (4, 2)
    with pytest.raises(ValueError):
        parse_hidden("0, 3")
    with pytest.raises(ValueError):
        parse_hidden("")


def test_epoch_by_epoch_training_learns_and_is_reproducible():
    """partial_fit epochs (Adam state kept between epochs): loss falls, the signal is learned, same seed = same net."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    m = MLPModel(hidden=(8,), batch_size=16, learning_rate=0.01)
    net = m.train(X, y, 30, seed=3)
    loss = np.array(net.loss_curve_)
    assert len(loss) == 30 and loss[-5:].mean() < 0.6 * loss[:3].mean()
    assert roc_auc_score(y, m._p1(net, X)) > 0.9
    assert np.array_equal(net.predict_proba(X), m.train(X, y, 30, seed=3).predict_proba(X))
    assert not np.array_equal(net.predict_proba(X), m.train(X, y, 30, seed=4).predict_proba(X))


def test_mlp_signal_importance_early_stopping_and_ensemble():
    sig, noise, geo, r, c = _toy()
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = MLPModel(neg_ratio=3, min_spacing=0, seed=2, repeats=2, networks=3).fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8 and res.stats["networks"] == 6
    cv = res.curve
    assert cv.cv_auc.iloc[res.stats["epochs"] - 1] == pytest.approx(cv.cv_auc.max())        # E* = best epoch
    assert len(cv) <= res.stats["epochs"] + 20                                              # patience 20
    imp = res.importance.set_index("layer")
    assert imp.index[0] == "sig" and imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert res.score[sig > 1.5].mean() > res.score[sig < -1.5].mean() + 0.3
    assert 0 < res.score_std.max() < 0.5
    a = MLPModel(seed=4, max_epochs=30, networks=2).fit_predict(feats[:1], r, c)
    b = MLPModel(seed=4, max_epochs=30, networks=2).fit_predict(feats[:1], r, c)
    assert np.array_equal(a.score, b.score)


@need_data
def test_carlin_mlp():
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
    res = MLPModel(neg_ratio=3, min_distance=1000, res=grid.res, seed=1).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.75
