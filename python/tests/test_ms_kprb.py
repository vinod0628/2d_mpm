"""MS-KPRB and the shared hybrid knowledge network (logic/hybrid/common/knowledge.py)."""
import os
import sys

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.common.knowledge import (ExpertLayer, KnowledgeModel, KnowledgeNetwork,  # noqa: E402
                                           KnowledgeTable, boost_predict, boost_residual)
from logic.hybrid.ms_kprb.model import KPRBConfig, fit_kprb  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KPRB_REF = os.path.join(ROOT, "docs", "hybrid", "kprb")
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def _net(n=40, missing=False, categorical=False, seed=0, weights=(1, 2, 1, 1, 1)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n, 5))
    X[:, :, 4] = rng.integers(1, 4, (n, n))
    if missing:
        X[rng.random((n, n)) < 0.1, 1] = np.nan
    proc, dirs = ["a", "a", "b", "c", "c"], [1, -1, 1, 1, 1]
    layers = [ExpertLayer(f"L{j}", np.ma.masked_invalid(X[:, :, j]), proc[j], dirs[j], weights[j],
                          {1.0: 0.2, 2.0: 0.5, 3.0: 0.9} if (j == 4 and categorical) else None) for j in range(5)]
    return KnowledgeNetwork(KnowledgeTable(layers, {"a": 1, "b": 2, "c": 1}), np.ones((n, n), bool)), rng


@pytest.mark.parametrize("missing,categorical", [(False, False), (True, True)])
def test_analytic_gradient_matches_finite_differences(missing, categorical):
    net, rng = _net(missing=missing, categorical=categorical)
    y = (rng.random(net.X.shape[0]) < expit(-3 + 2 * np.nan_to_num(net.X[:, 0]))).astype(float)
    km = KnowledgeModel(net)
    U, obs, cm = km._prepare(net.X)
    sw = np.ones(y.size)
    th = km.theta0() + rng.normal(scale=0.3, size=km.theta0().size)
    f, g = km.value_grad(th, U, obs, cm, y, sw)
    assert f == pytest.approx(km.objective(th, net.X, y, sw), rel=1e-10)        # same J as the plain objective
    gn = approx_fprime(th, lambda t: km.value_grad(t, U, obs, cm, y, sw)[0], 1e-6)
    assert np.max(np.abs(g - gn)) < 1e-3 * max(1.0, np.max(np.abs(g)))


@pytest.mark.skipif(not os.path.exists(os.path.join(KPRB_REF, "kprb.py")), reason="user's reference code missing")
def test_expert_network_and_boosting_equal_reference_code():
    sys.path.insert(0, KPRB_REF)
    import kprb
    net, rng = _net(weights=(1, 2, 1, 1, 1))
    mu, F, K = net.expert()
    ref = kprb.KnowledgeModel([0, 0, 1, 2, 2], [1, -1, 1, 1, 1], w0=[1, 2, 1, 1, 1], v0=[1, 2, 1]).fit_scaler(net.X)
    mu2, F2, K2, _ = ref.parts(net.X)
    assert np.allclose(mu, mu2) and np.allclose(F, F2) and np.allclose(K, K2)
    y = (rng.random(net.X.shape[0]) < 0.1).astype(float)
    off = rng.normal(size=y.size)
    t1 = boost_residual(net.features(), y, off, n_trees=15)
    t2 = kprb.boost_residual(net.features(), y, off, n_trees=15, rng=0)
    assert np.allclose(boost_predict(t1, net.features(), off), kprb.boost_predict(t2, net.features(), off))


def test_calibration_moves_weights_with_the_data_and_confidence_holds_them():
    """Deposits follow proxy L0 only: the within-process weight of L0 rises above the expert 1/3; with a very
    large confidence κ the fitted weights stay at the expert values."""
    net, rng = _net(n=60, weights=(1, 2, 1, 1, 1))
    y = (rng.random(net.X.shape[0]) < expit(-4 + 2.5 * net.X[:, 0])).astype(float)
    w_free = KnowledgeModel(net, kappa=1.0).fit(net.X, y).unpack(KnowledgeModel(net).theta0())[2]
    km = KnowledgeModel(net, kappa=1.0).fit(net.X, y)
    w = km.unpack(km.theta)[2]
    assert net.w0[0] == pytest.approx(1 / 3) and w[0] > 0.6
    km_hi = KnowledgeModel(net, kappa=1e5).fit(net.X, y)
    assert np.allclose(km_hi.unpack(km_hi.theta)[2], net.w0, atol=0.02)
    assert np.allclose(w_free, net.w0)


def test_residual_trees_only_when_the_knowledge_misses_a_control():
    """Deposits follow the expert network: the held-out deviance is (near) minimal at T = 0. A control outside the
    expert model (interaction of two layers with ~0 expert weight) needs trees: T > 0 in every draw and a clearly
    lower held-out deviance than T = 0."""
    n = 80
    rng = np.random.default_rng(3)
    A, B, C = (rng.normal(size=(n, n)) for _ in range(3))
    study = np.ones((n, n), bool)
    for hidden in (False, True):
        eta = (-6 + 2.0 * A + 5.0 * (C > 0.5) * (B > 0.0)) if hidden else (-5 + 2.0 * A)
        dep = rng.random((n, n)) < expit(eta)
        layers = [ExpertLayer("A", np.ma.array(A), "p1", 1), ExpertLayer("B", np.ma.array(B), "p2", 1, 0.0001),
                  ExpertLayer("C", np.ma.array(C), "p2", 1, 1.0)]
        net = KnowledgeNetwork(KnowledgeTable(layers, {"p1": 1, "p2": 0.0001}), study)
        pos = rng.choice(np.flatnonzero(dep.ravel()), 60, replace=False)
        cand = np.setdiff1d(np.arange(n * n), pos)
        res = fit_kprb(net, pos, cand, KPRBConfig(draws=3, seed=1))
        dev = res.t_selection.set_index("trees").mean_heldout_deviance
        if hidden:
            assert (res.draws.trees > 0).all() and dev.min() < dev[0] - 10
        else:
            assert dev[0] < dev.min() + 3 and dev[0] < dev[50]
        assert res.prob.shape == (n * n,) and np.all((res.prob > 0) & (res.prob < 1))


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    from core.methods import REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    import geopandas as gpd
    load_builtin_methods()
    ev = []
    for n, proc, dirn, sc in [("geology_host", "trap", 1, "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1, None),
                              ("as_log_idw", "deposition", 1, None), ("sb_log_idw", "deposition", 1, None)]:
        data, grid, _ = read_raster(os.path.join(DEMO, "rasters", n + ".tif"))
        ev.append(dict(name=n, data=data, grid=grid, kind="categorical" if sc else "continuous", process=proc,
                       direction=dirn, scores=sc, weight=1.0))
    m, mg, _ = read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))
    p = dict(evidence=ev, process_weights={"deposition": 2.0}, mask=(m, mg),
             deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")), test_fraction=0.3, seed=42,
             split_method="spatial", draws=3, out_processes=True)
    res = REGISTRY["ms_kprb"].run(p, lambda *a: None)
    names = [o.name for o in res.outputs]
    assert names[:4] == ["prob", "prob_sd", "knowledge_prob", "delta"] and "expert" in names
    assert "limiting_process" in names and "process_deposition" in names
    assert res.evaluation.curves["training"].auc > 0.85 and res.evaluation.curves["test"].auc > 0.8
    pw = res.tables["process_weights"].set_index("process")
    assert pw.expert_weight["deposition"] == pytest.approx(0.5)
    assert set(res.tables["layer_weights"].layer) == {"geology_host", "dist_faults", "as_log_idw", "sb_log_idw"}
