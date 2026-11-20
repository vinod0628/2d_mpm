"""Bayesian network classifiers: naive Bayes = weights of evidence, TAN structure and calibration, uncertainty."""
import os
import warnings

import numpy as np
import pytest
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature
from logic.data_driven.bayesian_network import BayesianNetworkModel, discretise, mutual_information

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def _data(seed=3, n=120):
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=(n, n))
    copy = sig + 0.3 * rng.normal(size=(n, n))
    noise = rng.normal(size=(n, n))
    geo = rng.integers(1, 4, size=(n, n)).astype(float)
    r, c = np.where(rng.random(sig.shape) < expit(-6.0 + 1.8 * sig + 1.5 * (geo == 2)))
    return sig, copy, noise, geo, r, c


def test_naive_bayes_on_all_cells_equals_weights_of_evidence():
    from logic.data_driven.wofe.model import Evidence, WofEModel
    rng = np.random.default_rng(1)
    a = rng.integers(1, 4, size=(80, 80)).astype(float)
    b = rng.integers(1, 3, size=(80, 80)).astype(float)
    r, c = np.where(rng.random(a.shape) < expit(-4 + 0.8 * (a == 2) + 0.6 * (b == 1)))
    nb = BayesianNetworkModel(structure="naive", alpha=1e-9, draws=0).fit_predict(
        [Feature("a", np.ma.array(a), "categorical"), Feature("b", np.ma.array(b), "categorical")], r, c)
    w = WofEModel(unit_cell_km2=1.0, cell_area_km2=1.0).fit_predict(
        [Evidence("a", np.ma.array(a), "categorical"), Evidence("b", np.ma.array(b), "categorical")], r, c)
    assert np.allclose(np.ma.getdata(nb.posterior), np.ma.getdata(w.posterior), rtol=1e-6, atol=1e-12)


def test_conditional_mutual_information_and_discretisation():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 2, 5000)
    a = rng.integers(0, 4, 5000)
    assert mutual_information(a, a, d, 4, 4) == pytest.approx(np.log(4), abs=0.02)        # I(A;A|D) = H(A)
    assert mutual_information(a, rng.integers(0, 4, 5000), d, 4, 4) < 0.01                # independent
    x = np.ma.array(np.arange(100, dtype=float).reshape(10, 10), mask=np.eye(10, dtype=bool))
    e = discretise(Feature("x", x), np.arange(100), 5, "class")
    assert e.k == 6 and e.labels[-1] == "missing" and np.all(np.bincount(e.codes)[:5] == 18)  # equal-area classes


def test_tan_learns_dependency_and_fixes_double_counting():
    sig, copy, noise, geo, r, c = _data()
    feats = [Feature("sig", np.ma.array(sig)), Feature("copy", np.ma.array(copy)),
             Feature("noise", np.ma.array(noise)), Feature("geo", np.ma.array(geo), "categorical")]
    nb = BayesianNetworkModel(structure="naive", draws=0).fit_predict(feats, r, c)
    tan = BayesianNetworkModel(structure="tan", draws=20).fit_predict(feats, r, c)
    e = tan.edges.set_index("child")
    assert e.loc["copy", "parent"] == "sig" or e.loc.get("sig", {}).get("parent") == "copy"      # strongest edge
    assert e.cond_mutual_information.max() > 0.5
    n = len(r)
    assert abs(tan.stats["predicted"] / n - 1) < 0.15 < abs(nb.stats["predicted"] / n - 1)       # NB double-counts
    comp = tan.comparison.set_index("structure")
    assert comp.loc["tan", "log_likelihood"] > comp.loc["naive", "log_likelihood"]
    assert 0 < tan.posterior_sd.max() < 0.5 and tan.unit_probability is not None
    assert set(tan.cpt_table.layer) == {"sig", "copy", "noise", "geo"}
    s = BayesianNetworkModel(design="sampled", neg_ratio=3, draws=0).fit_predict(feats, r, c)
    assert s.unit_probability is None and s.stats["training_auc"] > 0.8


@need_data
def test_carlin_bayesian_network():
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
    res = BayesianNetworkModel(structure="tan", cell_area_km2=0.0625, draws=10).fit_predict(feats, rows, cols, mask)
    assert evaluate(res.posterior, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert len(res.edges) == 3
