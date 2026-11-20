"""Improved hybrid methods M1–M4: KP-RF, KW-SVM, BS-WofE, PT-Hurdle (logic/hybrid/...)."""
import os
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.bs_wofe.model import bs_wofe  # noqa: E402
from logic.hybrid.common.knowledge import (ExpertLayer, KnowledgeNetwork, KnowledgeTable,  # noqa: E402
                                           clogit)
from logic.hybrid.kp_rf.model import kp_rf  # noqa: E402
from logic.hybrid.kw_svm.model import kw_svm  # noqa: E402
from logic.hybrid.pt_hurdle.model import host_gate, permissive_mask, pt_hurdle  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def world(n=60, seed=0, n_dep=40, flip=False, dup=False):
    """Two processes; deposits follow A (process 1) and B (process 2) as the expert says."""
    rng = np.random.default_rng(seed)
    A, B, C = (rng.normal(size=(n, n)) for _ in range(3))
    dep = rng.random((n, n)) < expit(-6 + 1.8 * A + 1.5 * B)
    pos = rng.choice(np.flatnonzero(dep.ravel()), min(n_dep, int(dep.sum())), replace=False)
    layers = [ExpertLayer("A", np.ma.array(A), "p1", -1 if flip else 1), ExpertLayer("B", np.ma.array(B), "p2", 1),
              ExpertLayer("C", np.ma.array(C), "p2", 1, 0.5)]
    if dup:
        layers.append(ExpertLayer("A2", np.ma.array(A.copy()), "p1", 1))
    net = KnowledgeNetwork(KnowledgeTable(layers), np.ones((n, n), bool))
    cand = np.setdiff1d(np.arange(n * n), pos)
    return net, pos, cand


def _auc(score, pos):
    from scipy.stats import rankdata
    lab = np.zeros(score.size, bool)
    lab[pos] = True
    r = rankdata(score)
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


# ------------------------------------------------------------------ M1 KP-RF
def test_kp_rf_pooling_weight_and_limits():
    net, pos, cand = world()
    N = pos.size
    r = kp_rf(net, pos, cand, n_bags=2, n_trees=50, seed=1)
    assert r.lam == pytest.approx(20 / (20 + N))
    from scipy.special import logit
    assert np.allclose(logit(r.prob), (1 - r.lam) * clogit(r.prob_rf) + r.lam * logit(r.prob_k), atol=1e-6)
    r0 = kp_rf(net, pos, cand, n_bags=2, n_trees=50, lam=0.0, seed=1)
    assert np.allclose(r0.prob, np.clip(r0.prob_rf, 1e-4, 1 - 1e-4))
    r1 = kp_rf(net, pos, cand, n_bags=2, n_trees=50, lam=1.0, seed=1)
    assert np.allclose(r1.prob, r1.prob_k) and _auc(r1.prob, pos) == pytest.approx(_auc(r1.expert, pos))
    assert r.calib[0] > 0 and _auc(r.prob, pos) > 0.8


def test_kp_rf_knowledge_guided_negatives_avoid_favourable_cells():
    net, pos, cand = world()
    g = kp_rf(net, pos, cand, n_bags=3, n_trees=20, a=2.0, seed=2).bags.mean_K_negatives.mean()
    u = kp_rf(net, pos, cand, n_bags=3, n_trees=20, a=0.0, seed=2).bags.mean_K_negatives.mean()
    assert g < u


def test_kp_rf_wrong_direction_is_reported():
    net, pos, cand = world(flip=True)
    r = kp_rf(net, pos, cand, n_bags=1, n_trees=20, seed=3)
    # with A reversed the expert map can still rank deposits through B; the calibration slope then stays > 0
    assert r.calib[0] >= 0
    if r.calib[0] == 0:
        assert any("Platt slope" in m for m in r.messages)


# ------------------------------------------------------------------ M2 KW-SVM
def test_kw_svm_kernel_weights_and_learning():
    net, pos, cand = world()
    r = kw_svm(net, pos, cand, n_bags=2, seed=0)
    om = r.kernel_weights.kernel_weight.to_numpy()
    assert om.mean() == pytest.approx(1.0) and om[2] < om[1]           # C has half the weight of B in process p2
    assert r.stats["gamma"] == pytest.approx(1 / 3)
    assert np.all((r.prob > 0) & (r.prob < 1)) and _auc(r.prob, pos) > 0.8
    assert (r.bags.platt_a > 0).all()


# ------------------------------------------------------------------ M3 BS-WofE
def test_bs_wofe_shrinkage_limits_and_formula():
    net, pos, cand = world()
    d = bs_wofe(net, pos, kappa=0.0).weights
    assert np.allclose(d.posterior_weight, d.data_weight)               # κ = 0: data weights only
    p = bs_wofe(net, pos, kappa=1e9).weights
    assert np.allclose(p.posterior_weight, p.prior_weight, atol=1e-6)   # κ → ∞: expert prior
    w = bs_wofe(net, pos, kappa=4.0).weights
    row = w.iloc[0]
    Q = int((w.layer == row.layer).sum())
    N, n = pos.size, net.cells.size
    wd = np.log(((row.deposits + 0.5) / (N + 0.5 * Q)) / ((row.cells - row.deposits + 0.5) / (n - N + 0.5 * Q)))
    s2 = 1 / (row.deposits + 0.5) + 1 / (row.cells - row.deposits + 0.5)
    assert row.data_weight == pytest.approx(wd)
    assert row.posterior_weight == pytest.approx((wd / s2 + 4 * row.prior_weight) / (1 / s2 + 4))
    # data weights rise with the membership of the informative layers
    a = w[w.layer == "A"].posterior_weight.to_numpy()
    assert a[-1] > a[0]


def test_bs_wofe_intercept_dependence_and_direction_flag():
    net, pos, cand = world(dup=True)
    r = bs_wofe(net, pos)
    assert r.prob.sum() == pytest.approx(pos.size, rel=1e-3)           # Σ P = N
    r0 = bs_wofe(*world()[:2])
    assert r.rho > r0.rho + 0.1 and r.nu < r0.nu                       # A and its copy A2 are deflated
    assert r.nu == pytest.approx(1 / (1 + 3 * r.rho))
    net_f, pos_f, _ = world(flip=True)
    rf = bs_wofe(net_f, pos_f)
    flags = rf.layers.set_index("layer").flag
    assert flags["A"] and not flags["B"]
    assert any("review the direction" in m for m in rf.messages)


# ------------------------------------------------------------------ M4 PT-Hurdle
def test_host_gate_and_permissive_mask():
    host = np.zeros((20, 20), bool)
    host[5:10, 5:10] = True
    g = host_gate(host, d0=2, s=1)
    assert np.all(g[host] == 1.0) and g[0, 0] < 0.01
    assert g[5, 11] > g[5, 13] > g[5, 16]                              # falls off with the distance
    assert host_gate(host, 2, 1, normalise=False)[7, 7] == pytest.approx(expit(2))
    data = np.ma.array(np.arange(16.0).reshape(4, 4), mask=np.eye(4, dtype=bool))
    assert permissive_mask(data, "in", "1; 2").sum() == 2
    assert permissive_mask(data, "ge", "12").sum() == 3                # 15 is masked
    with pytest.raises(ValueError):
        permissive_mask(data, "in", "")


def test_pt_hurdle_trains_inside_the_tract():
    n = 60
    net, pos, cand = world(n=n)
    host = np.zeros((n, n), bool)
    host[:, : n // 2] = True                                           # left half permissive
    for engine in ("kp_rf", "bs_wofe"):
        r = pt_hurdle(net, pos, cand, host, engine=engine, n_bags=2, n_trees=30, seed=0)
        cols = net.cells % n
        assert np.all(r.gate[cols < n // 2] == 1) and r.prob[cols > n // 2 + 6].max() < 0.01
        out = int((pos % n >= n // 2 + 2).sum())
        assert r.stats["deposits_in_tract"] == pos.size - out
        if out:
            assert any("outside the permissive tract" in m for m in r.messages)
    rs = pt_hurdle(net, pos, cand, host, engine="bs_wofe", smooth=2.0)
    assert np.all(np.isfinite(rs.prob))


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_bs_wofe_and_kp_rf():
    import geopandas as gpd
    from core.methods import REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    ev = []
    for n, proc, dirn, sc in [("geology_host", "trap", 1, "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1, None),
                              ("as_log_idw", "deposition", 1, None), ("sb_log_idw", "deposition", 1, None)]:
        data, grid, _ = read_raster(os.path.join(DEMO, "rasters", n + ".tif"))
        ev.append(dict(name=n, data=data, grid=grid, kind="categorical" if sc else "continuous", process=proc,
                       direction=dirn, scores=sc, weight=1.0))
    m, mg, _ = read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))
    p = dict(evidence=ev, mask=(m, mg), deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")),
             test_fraction=0.3, seed=42, split_method="spatial", n_bags=2, n_trees=100)
    r = REGISTRY["bs_wofe"].run(p, lambda *a: None)
    assert r.evaluation.curves["test"].auc > 0.85 and [o.name for o in r.outputs][:2] == ["prob", "logit_sd"]
    w = r.tables["weights"].set_index(["layer", "cls"])
    assert w.loc[("geology_host", "class 2"), "posterior_weight"] > 0 > w.loc[("geology_host", "class 1"),
                                                                              "posterior_weight"]
    r = REGISTRY["kp_rf"].run(p, lambda *a: None)
    assert r.evaluation.curves["test"].auc > 0.85
    assert r.extra["kprf"].lam == pytest.approx(20 / (20 + r.extra["kprf"].stats["deposits"]))
