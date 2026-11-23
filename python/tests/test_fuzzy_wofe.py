"""Fuzzy weights of evidence — classic and modified (logic/hybrid/fuzzy_wofe)."""
import os
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.common.knowledge import ExpertLayer, KnowledgeNetwork, KnowledgeTable  # noqa: E402
from logic.hybrid.fuzzy_wofe.model import fit_modified, fuzzy_weight, fuzzy_weight_var, fuzzy_wofe  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def test_fuzzy_weight_limits_and_variance():
    a, b, c, d = 12.0, 8.0, 3000.0, 17000.0                # deposits / non-deposits in B and B̄
    pBD, pnBD, pBnD, pnBnD = a / (a + b), b / (a + b), c / (c + d), d / (c + d)
    wp, wm = np.log(pBD / pBnD), np.log(pnBD / pnBnD)
    assert fuzzy_weight(1.0, pBD, pnBD, pBnD, pnBnD) == pytest.approx(wp)
    assert fuzzy_weight(0.0, pBD, pnBD, pBnD, pnBnD) == pytest.approx(wm)
    mu = np.linspace(0, 1, 11)
    w = fuzzy_weight(mu, pBD, pnBD, pBnD, pnBnD)
    assert np.all(np.diff(w) > 0)                           # positive contrast -> W rises with μ
    assert fuzzy_weight_var(1.0, a, b, c, d) == pytest.approx(1 / a + 1 / c)
    assert fuzzy_weight_var(0.0, a, b, c, d) == pytest.approx(1 / b + 1 / d)


def test_grouped_fit_equals_logistic_regression_and_constraint():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(4)
    W = np.column_stack([rng.integers(0, 6, 20000) * 0.4 - 1.0, rng.integers(0, 4, 20000) * 0.5 - 0.7])
    y = rng.random(20000) < expit(-3 + 1.2 * W[:, 0] - 0.8 * W[:, 1])     # second layer works against it
    b0, b, se = fit_modified(W, y, nonnegative=False)
    lr = LogisticRegression(C=1e9, max_iter=5000).fit(W, y)
    assert b0 == pytest.approx(lr.intercept_[0], abs=1e-3) and np.allclose(b, lr.coef_[0], atol=1e-3)
    assert np.all(np.isfinite(se)) and np.all(se > 0)
    c0, c, cse = fit_modified(W, y, nonnegative=True)
    assert c[1] == 0.0 and c[0] > 0 and np.isnan(cse[2])                   # bound active, s.e. undefined
    assert expit(c0 + W @ c).sum() == pytest.approx(y.sum(), rel=1e-3)     # Σ P = N also with the bound


def _world(n=70, seed=0, dup=False, missing=False):
    rng = np.random.default_rng(seed)
    A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    dep = rng.random((n, n)) < expit(-6 + 1.5 * A + 1.2 * B)
    pos = np.flatnonzero(dep.ravel())
    Am = np.ma.array(A, mask=(rng.random((n, n)) < 0.1) if missing else False)
    layers = [ExpertLayer("A", Am, "p1", 1), ExpertLayer("B", np.ma.array(B), "p2", 1)]
    if dup:
        layers.append(ExpertLayer("A2", np.ma.array(A + 0.05 * rng.normal(size=(n, n))), "p1", 1))
    return KnowledgeNetwork(KnowledgeTable(layers), np.ones((n, n), bool)), pos


def _auc(score, pos):
    from scipy.stats import rankdata
    lab = np.zeros(score.size, bool)
    lab[pos] = True
    r = rankdata(score)
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


def test_classic_and_modified_posteriors():
    net, pos = _world()
    r = fuzzy_wofe(net, pos, model="classic")
    assert _auc(r.prob, pos) > 0.8 and np.array_equal(r.prob, r.prob_classic)
    lay = r.layers.set_index("layer")
    assert (lay.contrast > 0).all() and (lay.pattern_deposits > 0).all()
    w = r.weights[r.weights.layer == "A"]
    assert np.all(np.diff(w.sort_values("membership").fuzzy_weight.to_numpy()) >= -1e-12)
    m = fuzzy_wofe(net, pos, model="modified")
    assert m.prob.sum() == pytest.approx(pos.size, rel=1e-3)     # logistic regression: Σ P = N
    assert (m.layers.beta > 0).all()                              # evidence used in its expert direction
    ll = lambda p: np.sum(np.log(np.where(np.isin(np.arange(p.size), pos), p, 1 - p)))  # noqa: E731
    assert ll(m.prob_modified) >= ll(m.prob_classic)              # the fitted coefficients can only improve the fit


def test_modified_model_deflates_dependent_layers():
    net0, pos0 = _world()
    r0 = fuzzy_wofe(net0, pos0)
    net, pos = _world(dup=True)
    r = fuzzy_wofe(net, pos)
    assert np.array_equal(pos, pos0)
    # (N/S of the classic model is not a reliable dependence check for fuzzy weights: compressed weights shift the
    # favourable cells up and the background down, so S barely changes; the β coefficients measure it directly)
    b0, b = r0.layers.set_index("layer").beta, r.layers.set_index("layer").beta
    assert b["A"] < b0["A"] and b["A2"] < b0["A"]                # each copy gets less than A alone ...
    assert b["A"] + b["A2"] < 1.3 * b0["A"]                       # ... together about A's weight
    assert b["B"] == pytest.approx(b0["B"], rel=0.25)             # the independent layer is not affected
    assert r.prob_modified.sum() == pytest.approx(pos.size, rel=1e-3)


def test_data_memberships_patterns_and_missing():
    net, pos = _world(missing=True)
    r = fuzzy_wofe(net, pos, membership="data", n_classes=8)
    for _, w in r.weights.groupby("layer"):                     # contrast range -> 0.01 … 0.99
        assert w.membership.min() == pytest.approx(0.01, abs=1e-9)
        assert w.membership.max() == pytest.approx(0.99, abs=1e-9)
    # a cell with missing A gets W_A = 0: its classic logit = prior logit + W_B of its class
    from scipy.special import logit
    c = fuzzy_wofe(net, pos, model="classic")
    miss = np.flatnonzero(~np.isfinite(net.X[:, 0]))[0]
    wb = c.weights[c.weights.layer == "B"].fuzzy_weight.to_numpy()
    assert np.isclose(logit(c.prob_classic[miss]) - logit(pos.size / net.cells.size), wb).any()
    h = fuzzy_wofe(net, pos, pattern="half")
    w = h.weights
    assert (w[w.in_pattern].membership >= 0.5).all()
    assert np.isfinite(r.prob).all() and np.isfinite(r.logit_sd).all()
    with pytest.raises(ValueError):
        fuzzy_wofe(net, pos[:2])


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
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
             test_fraction=0.3, seed=42, split_method="spatial")
    r = REGISTRY["fuzzy_wofe"].run(p, lambda *a: None)
    assert [o.name for o in r.outputs][:3] == ["post", "post_classic", "logit_sd"]
    assert r.evaluation.curves["training"].auc > 0.85 and r.evaluation.curves["test"].auc > 0.8
    lay = r.tables["layers"].set_index("layer")
    assert lay.loc["geology_host", "w_plus"] > 0 > lay.loc["geology_host", "w_minus"]
    assert r.extra["fwofe"].prob.sum() == pytest.approx(r.extra["fwofe"].stats["deposits"], rel=1e-2)
