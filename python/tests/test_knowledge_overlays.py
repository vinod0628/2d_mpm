"""Knowledge-driven overlays: fuzzy memberships and operators, inference network, Boolean conditions, real data."""
import os

import numpy as np
import pytest

from conftest import CARLIN
from logic.knowledge_driven.boolean_logic import BoolLayer, boolean_overlay, condition, parse_values
from logic.knowledge_driven.fuzzy_logic import (FuzzyLayer, combine, default_params, fuzzy_overlay, membership,
                                                parse_params)

need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def test_membership_functions():
    x = np.linspace(0, 10, 101)
    lin = membership(x, "linear", (2, 8))
    assert lin[x <= 2].max() == 0 and lin[x >= 8].min() == 1 and np.isclose(lin[50], 0.5)
    dec = membership(x, "linear", (8, 2))                                   # a > b: decreasing
    assert np.allclose(dec, 1 - lin)
    s, z = membership(x, "s_shape", (2, 8)), membership(x, "z_shape", (2, 8))
    assert np.allclose(s + z, 1) and np.all(np.diff(s) >= 0)
    lg = membership(x, "logistic", (5, 2.0))
    assert np.isclose(lg[50], 0.5) and np.allclose(lg, 1 / (1 + np.exp(-2.0 * (x - 5))))   # Yousefi & Carranza
    g = membership(x, "gaussian", (5, 1.0))
    assert np.argmax(g) == 50 and np.isclose(g[60], np.exp(-0.5))
    lg_ = membership(x, "large", (5, 5))
    assert np.isclose(lg_[50], 0.5) and np.all(np.diff(lg_) >= 0)
    assert np.allclose(membership(x, "small", (5, 5))[1:], 1 - lg_[1:])
    assert np.allclose(membership(np.array([1., 2, 3]), "classes", parse_params("classes", "1=0.2; 2=0.9")),
                       [0.2, 0.9, 0.0])
    assert np.isnan(membership(np.array([np.nan]), "linear", (0, 1)))[0]
    i, sl = map(float, default_params("logistic", np.arange(1000.0)).split(","))
    assert membership(np.array([np.percentile(np.arange(1000.0), 2)]), "logistic", (i, sl))[0] == pytest.approx(0.01, abs=0.002)
    with pytest.raises(ValueError):
        parse_params("gaussian", "1, -2")


def test_fuzzy_operators_and_inference_network():
    m = np.array([[0.2, 0.8], [0.5, 0.5]])
    assert np.allclose(combine(m, "and"), [0.2, 0.5]) and np.allclose(combine(m, "or"), [0.5, 0.8])
    assert np.allclose(combine(m, "product"), [0.1, 0.4]) and np.allclose(combine(m, "sum"), [0.6, 0.9])
    assert np.allclose(combine(m, "gamma", 0.0), combine(m, "product"))
    assert np.allclose(combine(m, "gamma", 1.0), combine(m, "sum"))
    assert np.allclose(combine(m, "geomean"), np.sqrt([0.1, 0.4]))
    # eis_toolkit implements the same five operators
    from eis_toolkit.prediction.fuzzy_overlay import gamma_overlay, sum_overlay
    m3 = m[:, None, :]                                                   # bands × rows × cols
    assert np.allclose(combine(m, "gamma", 0.7), gamma_overlay(m3, 0.7)[0]) and np.allclose(combine(m, "sum"), sum_overlay(m3)[0])
    rng = np.random.default_rng(0)
    a = np.ma.array(rng.normal(size=(30, 30)), mask=np.zeros((30, 30), bool))
    a.mask[0, 0] = True
    b = np.ma.array(rng.integers(1, 4, size=(30, 30)).astype(float))
    layers = [FuzzyLayer("a", a, "linear", (-1, 1), "source"), FuzzyLayer("b", b, "classes", {2.0: 0.9}, "trap"),
              FuzzyLayer("c", a, "logistic", (0, 3), "source")]
    res = fuzzy_overlay(layers, {"source": ("or", 0.9)}, ("and", 0.9))
    mu = {k: np.ma.getdata(v) for k, v in res.memberships.items()}
    expect = np.minimum(np.maximum(mu["a"], mu["c"]), mu["b"])
    assert np.allclose(np.ma.getdata(res.favourability)[res.study_mask], expect[res.study_mask])
    assert not res.study_mask[0, 0] and list(res.network.node) == ["group source", "group trap", "favourability"]
    keep = fuzzy_overlay(layers, {"source": ("or", 0.9)}, ("and", 0.9), missing="value")
    assert keep.study_mask[0, 0] and np.isclose(keep.memberships["a"][0, 0], 0.5)


def test_boolean_conditions_groups_and_statistics():
    x = np.array([0.0, 1, 2, 3, 4])
    assert condition(x, "ge", 2).tolist() == [False, False, True, True, True]
    assert condition(x, "between", (1, 3)).tolist() == [False, True, True, True, False]
    assert condition(x, "outside", (1, 3)).tolist() == [True, False, False, False, True]
    assert condition(x, "in", (0.0, 4.0)).tolist() == [True, False, False, False, True]
    assert parse_values("between", "5, 2") == (2, 5)
    a = np.ma.array(np.tile(np.arange(10.0), (10, 1)))                  # columns 0 … 9
    b = np.ma.array(np.repeat(np.arange(10.0), 10).reshape(10, 10))     # rows 0 … 9
    dep = (np.array([8, 9, 1]), np.array([8, 9, 1]))
    res = boolean_overlay([BoolLayer("a", a, "ge", 5, "g1"), BoolLayer("b", b, "ge", 5, "g2")], {}, "and", None, dep)
    assert np.ma.getdata(res.favourable).sum() == 25 and np.ma.getdata(res.count).max() == 2
    comb = res.table.set_index("layer").loc["FAVOURABLE (combined)"]
    assert comb.area_pct == 25 and comb.deposits_inside == 2 and comb.ratio == pytest.approx((2 / 3) / 0.25)
    either = boolean_overlay([BoolLayer("a", a, "ge", 5, "g1"), BoolLayer("b", b, "ge", 5, "g2")], {}, "or")
    assert np.ma.getdata(either.favourable).sum() == 75


@need_data
def test_carlin_knowledge_overlays():
    import geopandas as gpd
    from evaluation.metrics import evaluate
    from utils.raster_tools import points_to_cells, read_raster

    def load(n):
        return read_raster(os.path.join(CARLIN, "rasters", n + ".tif"))
    geo, grid, _ = load("geology_host")
    sb, dist = load("sb_log_idw")[0], load("dist_faults")[0]
    mask = ~np.ma.getmaskarray(load("study_mask")[0])
    rows, cols = points_to_cells(gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg")), grid, valid_mask=mask)
    layers = [FuzzyLayer("host", geo, "classes", {2.0: 0.9, 1.0: 0.2}, "trap"),
              FuzzyLayer("faults", dist, "z_shape", (500, 5000), "pathway"),
              FuzzyLayer("sb", sb, "logistic", (0.6, 6.0), "footprint")]
    fz = fuzzy_overlay(layers, {}, ("gamma", 0.9), mask)
    assert evaluate(fz.favourability, rows, cols, valid_mask=fz.study_mask).curves["training"].auc > 0.7
    bl = boolean_overlay([BoolLayer("host", geo, "in", (2.0,)), BoolLayer("sb", sb, "ge", 0.6)], {}, "and", mask,
                         (rows, cols))
    comb = bl.table.set_index("layer").loc["FAVOURABLE (combined)"]
    assert comb.ratio > 2                                    # deposits strongly over-represented in the target area
