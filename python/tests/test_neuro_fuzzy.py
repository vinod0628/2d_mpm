"""Neuro-fuzzy inference (ANFIS-type) on the knowledge network (logic/hybrid/neuro_fuzzy)."""
import os
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.hybrid.common.knowledge import ExpertLayer, KnowledgeNetwork, KnowledgeTable  # noqa: E402
from logic.hybrid.neuro_fuzzy.model import (TSKSystem, choose_inputs, expert_at,  # noqa: E402
                                            neuro_fuzzy, train_tsk)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def world(n=60, seed=0, n_dep=40, layers=3):
    """Two processes; deposits follow A (p1) and B (p2) as the expert says, C is noise."""
    rng = np.random.default_rng(seed)
    A, B, C = (rng.normal(size=(n, n)) for _ in range(3))
    dep = rng.random((n, n)) < expit(-6 + 1.8 * A + 1.5 * B)
    pos = rng.choice(np.flatnonzero(dep.ravel()), min(n_dep, int(dep.sum())), replace=False)
    ls = [ExpertLayer("A", np.ma.array(A), "p1", 1), ExpertLayer("B", np.ma.array(B), "p2", 1),
          ExpertLayer("C", np.ma.array(C), "p2", 1, 0.5)][:layers]
    net = KnowledgeNetwork(KnowledgeTable(ls), np.ones((n, n), bool))
    return net, pos, np.setdiff1d(np.arange(n * n), pos)


def _auc(score, pos):
    from scipy.stats import rankdata
    lab = np.zeros(score.size, bool)
    lab[pos] = True
    r = rankdata(score)
    return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())


def test_tsk_forward_matches_definition():
    s = TSKSystem(2, 2, "linear")
    assert s.R == 4 and s.rule_idx.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert np.exp(-(0.5 - 0) ** 2 / (2 * s.sigma[0, 0] ** 2)) == pytest.approx(0.5)     # MFs cross at 0.5
    rng = np.random.default_rng(1)
    s.theta = rng.normal(size=s.theta.size)
    X = rng.random((7, 2))
    A = np.exp(-(X[:, :, None] - s.c) ** 2 / (2 * s.sigma ** 2))                        # n × in × M
    w = np.stack([A[:, 0, a] * A[:, 1, b] for a, b in s.rule_idx], 1)
    P = s.theta.reshape(4, 3)
    f = X @ P[:, :2].T + P[:, 2]
    assert np.allclose(s.predict(X), (w / w.sum(1, keepdims=True) * f).sum(1))
    z = TSKSystem(3, 3, "constant")
    assert z.R == 27 and z.theta.size == 27 and np.allclose(z.prototypes()[5], [0, 0.5, 1.0])


def test_premise_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    X, y, sw = rng.random((40, 2)), (rng.random(40) < 0.3).astype(float), rng.uniform(0.5, 2, 40)
    for cons in ("linear", "constant"):
        s = TSKSystem(2, 3, cons)
        s.c += rng.normal(scale=0.05, size=s.c.shape)
        s.theta = rng.normal(size=s.theta.size)
        _, _, gc, gs = s.error_and_grad(X, y, sw)
        h = 1e-6
        for (i, m) in [(0, 0), (1, 2), (0, 1)]:
            for which, g in (("c", gc), ("s", gs)):
                st = s.state()
                if which == "c":
                    s.c[i, m] += h
                else:
                    s.sigma[i, m] *= np.exp(h)
                e1 = s.error_and_grad(X, y, sw)[0]
                s.set_state(st)
                if which == "c":
                    s.c[i, m] -= h
                else:
                    s.sigma[i, m] *= np.exp(-h)
                e0 = s.error_and_grad(X, y, sw)[0]
                s.set_state(st)
                assert g[i, m] == pytest.approx((e1 - e0) / (2 * h), rel=1e-4, abs=1e-8)


def test_knowledge_prior_limits_and_learning():
    rng = np.random.default_rng(3)
    X = rng.random((300, 2))
    y = ((X[:, 0] > 0.6) & (X[:, 1] > 0.5)).astype(float)                              # rule: both high
    q0 = np.array([0.1, 0.2, 0.3, 0.9])
    s = TSKSystem(2, 2, "constant")
    s.solve_consequents(X, y, np.ones(300), q0, kappa=1e9)
    assert np.allclose(s.theta, q0, atol=1e-6)                                           # κ → ∞: expert rules
    s.solve_consequents(X, y, np.ones(300), q0, kappa=0.0)
    wbar = s.normalised(X)
    assert np.allclose(s.theta, np.linalg.lstsq(wbar, y, rcond=None)[0], atol=1e-6)      # κ = 0: plain LSE
    lin = TSKSystem(2, 2, "linear")
    h = train_tsk(lin, X, y, np.ones(300), q0, kappa=0.1, epochs=150, lr=0.05)
    assert h[-1][0] < h[0][0]                                                            # premises learn
    assert lin.predict(np.array([[0.95, 0.95]]))[0] > 0.6 > lin.predict(np.array([[0.1, 0.1]]))[0]
    keep = TSKSystem(2, 2, "linear")
    train_tsk(keep, X, y, np.ones(300), q0, kappa=0.1, epochs=20, lr=0.0)
    assert np.allclose(keep.c, TSKSystem(2, 2).c)                                        # lr = 0: expert MFs


def test_inputs_rules_and_expert_prototypes():
    net, pos, cand = world()
    assert choose_inputs(net, "auto", 2) == ("layers", ["A", "B", "C"])                  # 2^3 = 8 ≤ 64
    assert choose_inputs(net, "processes", 3) == ("processes", ["p1", "p2"])
    with pytest.raises(ValueError):
        choose_inputs(net, "bad", 2)
    K = expert_at(net, "processes", np.array([[0.99, 0.99], [0.01, 0.99], [0.01, 0.01]]))
    assert K[0] > K[1] > K[2]                                                            # AND-like
    big = KnowledgeNetwork(KnowledgeTable([ExpertLayer(f"L{i}", np.ma.array(np.random.default_rng(i).normal(
        size=(20, 20))), "p1" if i < 5 else "p2") for i in range(9)]), np.ones((20, 20), bool))
    assert choose_inputs(big, "auto", 2)[0] == "processes"
    with pytest.raises(ValueError):
        choose_inputs(big, "layers", 2)                                                  # 2^9 = 512 > MAX_RULES


def test_neuro_fuzzy_learns_and_reports():
    net, pos, cand = world()
    r = neuro_fuzzy(net, pos, cand, n_bags=3, epochs=60, seed=1)
    assert r.input_mode == "layers" and r.stats["rules"] == 8 and len(r.rules) == 8
    assert _auc(r.prob, pos) > 0.85 and np.all((r.prob > 0) & (r.prob < 1))
    imp = r.importance.set_index("input").auc_drop
    assert imp["A"] > imp["C"] and imp["B"] > imp["C"]                                   # noise layer least
    top = r.rules.iloc[0]
    assert "A is high" in top.antecedent and "B is high" in top.antecedent
    assert top.lift > 1 and r.rules.firing_area_pct.sum() == pytest.approx(100, abs=1e-6)
    assert set(r.mf.columns) >= {"input", "mf", "centre_expert", "centre", "sigma"} and len(r.mf) == 6
    p = neuro_fuzzy(net, pos, cand, input_mode="processes", n_mf=3, consequent="constant", n_bags=2,
                    epochs=40, check_fraction=0, seed=1)
    assert p.stats["rules"] == 9 and p.bags.checking_samples.eq(0).all() and _auc(p.prob, pos) > 0.85
    a = neuro_fuzzy(net, pos, cand, n_bags=2, epochs=30, seed=7)
    b = neuro_fuzzy(net, pos, cand, n_bags=2, epochs=30, seed=7)
    assert np.array_equal(a.prob, b.prob)                                                # reproducible


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    import geopandas as gpd
    from core.methods import PLANNED, REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    assert "neuro_fuzzy" in REGISTRY and not any("ANFIS" in p[2] for p in PLANNED)
    ev = []
    for n, proc, dirn, sc in [("geology_host", "trap", 1, "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1, None),
                              ("as_log_idw", "deposition", 1, None), ("sb_log_idw", "deposition", 1, None)]:
        data, grid, _ = read_raster(os.path.join(DEMO, "rasters", n + ".tif"))
        ev.append(dict(name=n, data=data, grid=grid, kind="categorical" if sc else "continuous", process=proc,
                       direction=dirn, scores=sc, weight=1.0))
    m, mg, _ = read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))
    p = dict(evidence=ev, mask=(m, mg), deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")),
             test_fraction=0.3, seed=42, split_method="spatial", n_bags=3, epochs=60)
    r = REGISTRY["neuro_fuzzy"].run(p, lambda *a: None)
    assert r.evaluation.curves["test"].auc > 0.85 and [o.name for o in r.outputs][:2] == ["prob", "prob_sd"]
    assert {"summary", "rules", "membership", "importance", "bags"} <= set(r.tables)
    assert r.extra["neurofuzzy"].input_mode == "layers" and len(r.tables["rules"]) == 16
