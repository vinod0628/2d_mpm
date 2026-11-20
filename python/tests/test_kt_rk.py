"""M5 KT-RK — knowledge-trend regression-kriging (logic/hybrid/kt_rk)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.common.knowledge import ExpertLayer, KnowledgeNetwork, KnowledgeTable  # noqa: E402
from logic.hybrid.kt_rk.model import Kriging, _gp, kt_rk, rotate  # noqa: E402


def test_rotation_to_strike_and_ranges():
    xy = np.array([[1.0, 0.0], [0.0, 1.0]])
    u = rotate(xy, 90.0, 2.0, 0.5)                    # strike east: along = x / 2, across = ∓y / 0.5
    assert np.allclose(u[0], [0.5, 0.0], atol=1e-12) and np.allclose(np.abs(u[1]), [0.0, 2.0], atol=1e-12)
    u0 = rotate(xy, 0.0, 1.0, 1.0)                    # strike north: along = y
    assert np.allclose(u0[1], [1.0, 0.0], atol=1e-12)


def test_chunked_kriging_equals_sklearn():
    rng = np.random.default_rng(0)
    U = rng.uniform(0, 5, (200, 2))
    r = np.sin(U[:, 0]) + 0.3 * rng.normal(size=200)
    gp = _gp((0.3, 3), r.var()).fit(U, r)
    Ut = rng.uniform(0, 5, (4100, 2))                 # > 2 chunks
    m1, s1 = gp.predict(Ut, return_std=True)
    m2, s2 = Kriging(gp.kernel_, U, r).predict(Ut, return_std=True)
    assert np.allclose(m1, m2, atol=1e-8) and np.allclose(s1, s2, atol=1e-8)


def _world(n=40, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    A = np.cumsum(rng.normal(size=(n, n)), 1) / 5               # smooth expert proxy
    net = KnowledgeNetwork(KnowledgeTable([ExpertLayer("A", np.ma.array(A), "p", 1)]), np.ones((n, n), bool))
    K = net.expert()[2]
    obs = rng.choice(n * n, 150, replace=False)
    z = (3.0 * K[obs] if signal else 0.0) + 0.3 * rng.normal(size=obs.size)
    y = np.exp(z)
    rr, cc = np.divmod(np.arange(n * n), n)
    coords = np.column_stack([cc * 10.0, -rr * 10.0])
    return net, obs, y, coords


def test_knowledge_trend_is_used_when_it_explains_the_target():
    net, obs, y, coords = _world(signal=True)
    r = kt_rk(net, obs, y, coords, range_along=100, range_across=100, n_folds=5, n_trees=50)
    assert r.alpha > 0 and r.stats["b1"] > 0 and r.stats["corr_k_z"] > 0.5
    assert r.stats["c"] == pytest.approx(0.1 * np.median(y))
    assert np.all(np.isfinite(r.pred)) and r.pred.shape == (net.cells.size,)
    assert np.all(r.log_sd >= 0)
    sel = r.cv[r.cv.selected].iloc[0]
    assert sel.within_one_se and sel.alpha + sel.shrink == r.cv[r.cv.within_one_se].eval("alpha + shrink").min()


def test_one_se_rule_drops_an_uninformative_trend():
    net, obs, y, coords = _world(signal=False, seed=1)
    r = kt_rk(net, obs, y, coords, range_along=100, range_across=100, n_folds=5, n_trees=50)
    assert r.alpha + r.shrink <= 0.5


def test_sign_constraint_and_too_few_observations():
    net, obs, y, coords = _world(signal=True)
    r = kt_rk(net, obs, 1.0 / y, coords, range_along=100, range_across=100, n_folds=5, n_trees=30)
    assert r.stats["b1"] == 0.0                                 # negative relation -> no knowledge trend
    with pytest.raises(ValueError):
        kt_rk(net, obs[:5], y[:5], coords)


def test_sample_cells_average_per_cell():
    import geopandas as gpd
    from shapely.geometry import Point
    from logic.hybrid.kt_rk.plugin import sample_cells
    from utils.raster_tools import Grid
    from rasterio.transform import from_origin
    grid = Grid(from_origin(0, 100, 10, 10), None, 10, 10)
    gdf = gpd.GeoDataFrame(dict(v=[1.0, 3.0, 5.0, np.nan]),
                           geometry=[Point(5, 95), Point(6, 94), Point(55, 55), Point(25, 25)])
    study = np.ones((10, 10), bool)
    pos, vals, n = sample_cells(gdf, "v", grid, study, np.flatnonzero(study))
    assert n == 3 and list(pos) == [0, 45] and list(vals) == [2.0, 5.0]
