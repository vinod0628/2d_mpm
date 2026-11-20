"""Random forest: tree-vote maps, importances, OOB / CV, feature encoding, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature, FeatureStack, study_area
from logic.data_driven.random_forest import RandomForestModel

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def _toy(seed=3, n=120):
    rng = np.random.default_rng(seed)
    sig = np.ma.array(rng.normal(size=(n, n)))
    noise = np.ma.array(rng.normal(size=(n, n)))
    geo = np.ma.array(rng.integers(1, 4, size=(n, n)).astype(float))
    p = expit(-6.5 + 1.8 * sig + 1.5 * (geo == 2))
    r, c = np.where(rng.random(sig.shape) < p)
    return sig, noise, geo, r, c


def test_feature_stack_one_hot_and_imputation():
    geo = np.ma.array([[1., 2.], [3., 2.]])
    val = np.ma.array([[0.5, 1.5], [2.5, 9.0]], mask=[[0, 0], [0, 1]])
    st = FeatureStack([Feature("geo", geo, "categorical", {2.0: "host"}), Feature("v", val)],
                      np.ones((2, 2), bool), missing="impute")
    assert st.names == ["geo[1]", "geo[host]", "geo[3]", "v"] and st.groups == {"geo": [0, 1, 2], "v": [3]}
    X = st.matrix(np.arange(4))
    assert X.dtype == np.float32 and X[:, :3].sum(1).tolist() == [1, 1, 1, 1]       # one class per cell
    assert X[3, 3] == pytest.approx(1.5)                                            # missing -> median
    assert study_area([Feature("v", val)], None, "exclude").sum() == 3
    assert st.positions([1, 0, 5], [1, 0, 0]).tolist() == [3, 0]                     # outside the grid dropped


def test_tree_vote_accumulation_equals_predict_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    m = RandomForestModel(n_trees=60, seed=1)
    rf = m._forest(1, 3).fit(X, y)
    s, q = m._tree_votes(rf, X, 2)
    mean = s / 60
    assert np.allclose(mean, rf.predict_proba(X)[:, 1])
    per_tree = np.array([t.predict_proba(X)[:, 1] for t in rf.estimators_])
    assert np.allclose(np.sqrt(np.maximum(q / 60 - mean ** 2, 0)), per_tree.std(0))


def test_random_forest_finds_signal_and_ranks_importance():
    sig, noise, geo, r, c = _toy()
    feats = [Feature("sig", sig), Feature("noise", noise), Feature("geo", geo, "categorical")]
    res = RandomForestModel(n_trees=150, neg_ratio=3, min_spacing=0, seed=2, cv_folds=5).fit_predict(feats, r, c)
    assert res.stats["oob_auc"] > 0.8
    imp = res.importance.set_index("layer")
    assert imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert imp.loc["sig", "impurity_pct"] > imp.loc["noise", "impurity_pct"]
    assert imp.impurity_pct.sum() == pytest.approx(100.0)
    hi = res.score[sig > 1.5].mean()
    lo = res.score[sig < -1.5].mean()
    assert hi > lo + 0.3                                                   # higher score where deposits are likely
    assert res.score.min() >= 0 and res.score.max() <= 1 and 0 <= res.score_std.max() <= 0.5
    sv = res.sample_validation.set_index("item")["value"]
    assert sv["Cross-validation"].startswith("5-fold") and float(sv["Cross-validated AUC"]) > 0.75
    tn, fp = map(int, sv["OOB confusion: true non-deposit → predicted non-deposit / deposit"].split(" / "))
    fn, tp = map(int, sv["OOB confusion: true deposit → predicted non-deposit / deposit"].split(" / "))
    assert tp + fn == res.n_deposits and tn + fp == res.n_negatives


def test_random_forest_is_reproducible_and_pools_repeats():
    sig, noise, geo, r, c = _toy(5, 80)
    feats = [Feature("sig", sig), Feature("geo", geo, "categorical")]
    a = RandomForestModel(n_trees=40, seed=7).fit_predict(feats, r, c)
    b = RandomForestModel(n_trees=40, seed=7).fit_predict(feats, r, c)
    d = RandomForestModel(n_trees=40, seed=8).fit_predict(feats, r, c)
    assert np.array_equal(a.score, b.score) and not np.array_equal(a.score, d.score)
    rep = RandomForestModel(n_trees=40, repeats=3, seed=7).fit_predict(feats, r, c)
    assert len(rep.repeats) == 3 and rep.stats["n_trees"] == 120
    assert any("pooled" in m for m in rep.messages)


@need_data
def test_carlin_random_forest():
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
    res = RandomForestModel(n_trees=200, neg_ratio=5, min_distance=1000, res=grid.res, seed=1).fit_predict(
        feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["oob_auc"] > 0.7
    assert res.importance.layer.iloc[0] in {"sb_log_idw", "as_log_idw", "geology_host"}
