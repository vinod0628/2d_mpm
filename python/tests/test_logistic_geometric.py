"""Logistic-function fuzzification + geometric average (Yousefi & Carranza 2015)."""
import os

import numpy as np
import pytest

from conftest import CARLIN
from logic.knowledge_driven.logistic_geometric import (LGLayer, logistic, logistic_geometric, logistic_parameters,
                                                       parse_range, parse_scores)

need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def test_logistic_parameters_map_range_to_001_099():
    i, s = logistic_parameters(2.0, 12.0, "positive")
    assert i == 7.0 and s == pytest.approx(0.92)
    f = logistic([2.0, 7.0, 12.0], i, s)
    assert f[0] == pytest.approx(0.01, abs=1e-4) and f[1] == 0.5 and f[2] == pytest.approx(0.99, abs=1e-4)
    i, s = logistic_parameters(2.0, 12.0, "negative")
    assert logistic([2.0], i, s)[0] == pytest.approx(0.99, abs=1e-4)              # low values favourable
    with pytest.raises(ValueError):
        parse_range("5, 5")
    with pytest.raises(ValueError):
        parse_scores("1=0")
    assert parse_scores("1=0.2; 3=1") == {1.0: 0.2, 3.0: 1.0}


def test_geometric_average_and_pa_weights():
    rng = np.random.default_rng(0)
    a = np.ma.array(rng.normal(size=(40, 40)))
    b = np.ma.array(rng.integers(1, 3, size=(40, 40)).astype(float))
    layers = [LGLayer("a", a, "positive", (float(a.min()), float(a.max()))),
              LGLayer("b", b, "classes", {2.0: 0.8, 1.0: 0.3})]
    res = logistic_geometric(layers)
    fa, fb = (np.ma.getdata(res.fuzzified[k]) for k in ("a", "b"))
    assert np.allclose(np.ma.getdata(res.favourability), np.sqrt(fa * fb))       # plain geometric average
    w = logistic_geometric([LGLayer("a", a, "positive", (float(a.min()), float(a.max())), 3.0),
                            LGLayer("b", b, "classes", {2.0: 0.8, 1.0: 0.3}, 1.0)], "custom")
    assert np.allclose(np.ma.getdata(w.favourability), (fa ** 3 * fb) ** 0.25)  # weighted geometric mean
    r, c = np.where(np.ma.getdata(a) > 1.8)                                    # deposits only on high a
    pa = logistic_geometric(layers, "pa", deposits=(r, c))
    t = pa.table.set_index("layer")
    assert t.loc["a", "weight"] > 1 and t.loc["a", "normalised_density"] > 5 and t.loc["b", "weight"] == 0
    assert np.allclose(np.ma.getdata(pa.favourability), fa)                    # b weight 0 -> only a counts


@need_data
def test_carlin_logistic_geometric():
    import geopandas as gpd
    from evaluation.metrics import evaluate
    from utils.raster_tools import points_to_cells, read_raster

    def load(n):
        return read_raster(os.path.join(CARLIN, "rasters", n + ".tif"))
    geo, grid, _ = load("geology_host")
    sb, asn, dist = load("sb_log_idw")[0], load("as_log_idw")[0], load("dist_faults")[0]
    mask = ~np.ma.getmaskarray(load("study_mask")[0])
    rows, cols = points_to_cells(gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg")), grid, valid_mask=mask)
    rng_ = lambda x: (float(np.ma.min(x)), float(np.ma.max(x)))                # noqa: E731
    layers = [LGLayer("host", geo, "classes", {2.0: 0.9, 1.0: 0.1}), LGLayer("sb", sb, "positive", rng_(sb)),
              LGLayer("as", asn, "positive", rng_(asn)), LGLayer("faults", dist, "negative", rng_(dist))]
    res = logistic_geometric(layers, "equal", mask)
    ev = evaluate(res.favourability, rows, cols, valid_mask=res.study_mask)
    assert ev.curves["training"].auc > 0.8 and ev.curves["training"].pa_intersection["prediction"] > 60
