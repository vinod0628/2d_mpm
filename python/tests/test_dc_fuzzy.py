"""Data-calibrated fuzzy memberships (WofE / LR) on the expert inference network (logic/hybrid/dc_fuzzy)."""
import os
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.common.knowledge import ExpertLayer, KnowledgeNetwork, KnowledgeTable  # noqa: E402
from logic.hybrid.dc_fuzzy.model import (class_weights, dc_fuzzy, gamma_combine, isotonic,  # noqa: E402
                                         logistic_1d, to_membership)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def world(n=60, seed=0, n_dep=40, flip=False):
    """Two processes; deposits follow A (p1) and B (p2), C is noise; flip = wrong expert direction of A."""
    rng = np.random.default_rng(seed)
    A, B, C = (rng.normal(size=(n, n)) for _ in range(3))
    dep = rng.random((n, n)) < expit(-6 + 1.8 * A + 1.5 * B)
    pos = rng.choice(np.flatnonzero(dep.ravel()), min(n_dep, int(dep.sum())), replace=False)
    geo = (A > 0.3).astype(float) + 1                              # categorical: class 2 where A is high
    ls = [ExpertLayer("A", np.ma.array(A), "p1", -1 if flip else 1), ExpertLayer("B", np.ma.array(B), "p2", 1),
          ExpertLayer("C", np.ma.array(C), "p2", 1, 0.5),
          ExpertLayer("geo", np.ma.array(geo), "p1", 1, 1.0, {2.0: 0.8, 1.0: 0.2})]
    return KnowledgeNetwork(KnowledgeTable(ls), np.ones((n, n), bool)), pos


def _auc(score, pos):
    from scipy.stats import rankdata
    lab = np.zeros(score.size, bool)
    lab[pos] = True
    r = rankdata(score)
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


def test_class_weights_by_hand():
    cls = np.array([0] * 50 + [1] * 30 + [2] * 20)
    dep = np.zeros(100)
    dep[[0, 50, 51, 52, 80, 81]] = 1                               # 1, 3, 2 deposits; N = 6, n = 100
    cells, deps, W, sW = class_weights(cls, dep, 3)
    assert cells.tolist() == [50, 30, 20] and deps.tolist() == [1, 3, 2]
    assert W[1] == pytest.approx(np.log((3 / 6) / (27 / 94)))
    assert sW[1] == pytest.approx(np.sqrt(1 / 3 + 1 / 27))
    _, d0, W0, _ = class_weights(np.array([0, 0, 1, 1]), np.array([1.0, 0, 0, 0]), 2)
    assert d0[1] == 0 and np.isfinite(W0[1]) and W0[1] < 0       # empty count -> ½


def test_isotonic_and_logistic_match_references():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    y, w = rng.normal(size=12) + np.arange(12) * 0.2, rng.uniform(0.5, 3, 12)
    for inc in (True, False):
        ref = IsotonicRegression(increasing=inc).fit_transform(np.arange(12), y, sample_weight=w)
        assert np.allclose(isotonic(y, w, inc), ref)
    z = rng.normal(size=4000)
    t = (rng.random(4000) < expit(-3 + 1.2 * z)).astype(float)
    a, b, se = logistic_1d(z, t)
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(z.reshape(-1, 1), t)
    assert a == pytest.approx(lr.intercept_[0], abs=1e-4) and b == pytest.approx(lr.coef_[0, 0], abs=1e-4)
    assert 0 < se < 0.2 and abs(b - 1.2) < 3 * se


def test_membership_scaling_and_gamma():
    W = np.array([-2.0, 0.0, 1.0, 3.0])
    assert np.allclose(to_membership(W, "weight"), np.clip(expit(W), 0.01, 0.99))
    s = to_membership(W, "stretch")
    assert s[0] == pytest.approx(0.01) and s[-1] == pytest.approx(0.99) and np.all(np.diff(s) > 0)
    from logic.knowledge_driven.fuzzy_logic.model import combine
    mu = np.random.default_rng(5).uniform(0.05, 0.95, (30, 4))
    assert np.allclose(gamma_combine(mu, 0.8), combine(mu.T, "gamma", 0.8))
    mu[0, 2] = np.nan                                              # missing layer left out of the products
    assert gamma_combine(mu, 0.8)[0] == pytest.approx(combine(mu[0, [0, 1, 3]][:, None], "gamma", 0.8)[0])


def test_wofe_and_lr_calibration_learn_and_keep_directions():
    net, pos = world()
    for cal in ("wofe", "lr"):
        r = dc_fuzzy(net, pos, calibration=cal)
        assert _auc(r.prob, pos) > 0.85 and r.stats["against_expert"] == 0          # noise C is not flagged
        lay = r.layers.set_index("layer")
        assert lay.loc["A", "single_layer_auc"] > lay.loc["C", "single_layer_auc"]
        m = r.memberships[r.memberships.layer == "A"]
        assert np.all(np.diff(m.mu_data.values) >= -1e-12)         # monotone, increasing with A
    lr = dc_fuzzy(net, pos, calibration="lr").layers.set_index("layer")
    assert lr.loc["A", "slope"] > 3 * lr.loc["A", "slope_se"] and np.isnan(lr.loc["geo", "slope"])   # geo: classes
    g = dc_fuzzy(net, pos, combination="gamma", gamma=0.5)
    assert _auc(g.prob, pos) > 0.8 and not np.allclose(g.prob, dc_fuzzy(net, pos).prob)


def test_wrong_expert_direction_is_flagged_and_neutralised():
    net, pos = world(flip=True)
    for cal in ("wofe", "lr"):
        r = dc_fuzzy(net, pos, calibration=cal)
        assert r.stats["against_expert"] == 1 and any("A:" in m for m in r.messages)
        assert np.nanstd(r.mu_data[:, 0]) < 0.05                   # flattened: the data cannot use A backwards
        free = dc_fuzzy(net, pos, calibration=cal, monotone=False)
        assert np.nanstd(free.mu_data[:, 0]) > 0.05                # without the constraint the data take over


def test_expert_pooling_limits():
    net, pos = world()
    big = dc_fuzzy(net, pos, kappa=1e9)
    mu_exp = net.expert()[0]
    assert big.stats["lam"] < 1e-6 and np.allclose(big.mu, mu_exp, atol=1e-6)
    assert np.allclose(big.prob, net.expert()[2], atol=1e-6)      # κ → ∞: the expert map
    half = dc_fuzzy(net, pos, kappa=pos.size)
    assert half.stats["lam"] == pytest.approx(0.5)
    assert np.allclose(half.mu, 0.5 * half.mu_data + 0.5 * mu_exp, atol=1e-9)


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    import geopandas as gpd
    from core.methods import PLANNED, REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    assert "dc_fuzzy" in REGISTRY and not any("Data-calibrated" in p[2] for p in PLANNED)
    ev = []
    for n, proc, dirn, sc in [("geology_host", "trap", 1, "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1, None),
                              ("as_log_idw", "deposition", 1, None), ("sb_log_idw", "deposition", 1, None)]:
        data, grid, _ = read_raster(os.path.join(DEMO, "rasters", n + ".tif"))
        ev.append(dict(name=n, data=data, grid=grid, kind="categorical" if sc else "continuous", process=proc,
                       direction=dirn, scores=sc, weight=1.0))
    m, mg, _ = read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))
    p = dict(evidence=ev, mask=(m, mg), deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")),
             test_fraction=0.3, seed=42, split_method="spatial", out_memberships=True)
    r = REGISTRY["dc_fuzzy"].run(p, lambda *a: None)
    assert r.evaluation.curves["test"].auc > 0.8 and r.outputs[0].name == "fuzzy"
    assert {o.name for o in r.outputs} >= {"expert", "mu_geology_host", "mu_sb_log_idw"}
    lay = r.tables["layers"].set_index("layer")
    assert lay.agrees_with_expert.all() and lay.loc["sb_log_idw", "single_layer_auc"] > 0.7
    p.update(calibration="lr", combination="gamma")
    assert REGISTRY["dc_fuzzy"].run(p, lambda *a: None).evaluation.curves["test"].auc > 0.8
