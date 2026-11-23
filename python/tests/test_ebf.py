"""Evidential belief functions — Dempster–Shafer (logic/data_driven/ebf)."""
import os
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.data_driven.ebf.model import EBFEvidence, EBFModel, auc, class_ebf, dempster  # noqa: E402
from logic.data_driven.wofe.model import binary_weights  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def world(n=80, seed=0, n_dep=50):
    """Deposits follow A (continuous) and the class layer G; C is noise."""
    rng = np.random.default_rng(seed)
    A, C = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    G = rng.integers(1, 4, (n, n))
    dep = rng.random((n, n)) < expit(-6.5 + 1.8 * A + 1.2 * (G == 3))
    idx = rng.choice(np.flatnonzero(dep.ravel()), min(n_dep, int(dep.sum())), replace=False)
    return dict(A=np.ma.array(A), C=np.ma.array(C), G=np.ma.array(G)), np.unravel_index(idx, (n, n))


def test_class_beliefs_by_hand_and_wofe_link():
    n_c, n_cd = np.array([50.0, 30, 20]), np.array([1.0, 3, 2])        # N(T) = 100, N(D) = 6
    w_d, w_nd, bel, dis, unc, pls, over = class_ebf(n_cd, n_c, 6, 100)
    assert w_d[1] == pytest.approx((3 / 6) / (27 / 94))
    assert w_nd[1] == pytest.approx((3 / 6) / ((94 - 27) / 94))
    assert bel == pytest.approx(w_d / w_d.sum()) and dis == pytest.approx(w_nd / w_nd.sum())
    assert np.allclose(bel + dis + unc, 1) and np.allclose(pls, 1 - dis) and over == 0
    wp, _, wm, _, _, _ = binary_weights(n_cd, n_c, 6, 100)               # W_CD = exp W+, W_CD̄ = exp W−
    assert np.allclose(np.log(w_d), wp) and np.allclose(np.log(w_nd), wm)


def test_binary_layer_has_no_uncertainty_and_zero_rule():
    for zero in ("zero", "haldane"):
        _, _, bel, dis, unc, _, _ = class_ebf(np.array([8.0, 2]), np.array([300.0, 700]), 10, 1000, zero)
        assert np.allclose(unc, 0, atol=1e-12) and dis[0] == pytest.approx(bel[1])
    _, _, bel, dis, _, _, _ = class_ebf(np.array([0.0, 10]), np.array([300.0, 700]), 10, 1000, "zero")
    assert bel.tolist() == [0.0, 1.0] and dis[1] == 0                  # certainty: Bel = 1
    _, _, bel, dis, unc, _, _ = class_ebf(np.array([0.0, 10]), np.array([300.0, 700]), 10, 1000, "haldane")
    assert 0 < bel[0] < 0.15 and 0 < dis[1] < 0.15 and dis[1] == pytest.approx(bel[0])     # 0.5 of 10


def test_dempster_rule():
    rng = np.random.default_rng(1)

    def bpa():
        m = rng.dirichlet(np.ones(3), 200)
        return m[:, 0], m[:, 1], m[:, 2]
    x, y, z = bpa(), bpa(), bpa()
    xy = dempster(*x, *y)[:3]
    assert np.allclose(sum(xy), 1)
    assert np.allclose(xy, dempster(*y, *x)[:3])                              # commutative
    assert np.allclose(dempster(*xy, *z)[:3], dempster(*x, *dempster(*y, *z)[:3])[:3])   # associative
    one, zero = np.ones(200), np.zeros(200)
    assert np.allclose(dempster(*x, zero, zero, one)[:3], x)                    # vacuous = identity
    b, d, u, k = dempster(np.array([0.6]), np.array([0.1]), np.array([0.3]), np.array([0.5]), np.array([0.2]),
                          np.array([0.3]))
    beta = 1 - 0.6 * 0.2 - 0.1 * 0.5
    assert k[0] == pytest.approx(1 - beta) and b[0] == pytest.approx((0.30 + 0.18 + 0.15) / beta)
    assert d[0] == pytest.approx((0.02 + 0.03 + 0.06) / beta) and u[0] == pytest.approx(0.09 / beta)


def test_model_learns_and_reports():
    lay, (r, c) = world()
    ev = [EBFEvidence("A", lay["A"], "quantile", 8), EBFEvidence("G", lay["G"], "values", labels={3.0: "host"}),
          EBFEvidence("C", lay["C"], "quantile", 8)]
    res = EBFModel().fit_predict(ev, r, c)
    lab = np.zeros(lay["A"].shape, bool)
    lab[r, c] = True
    assert auc(res.bel.ravel(), lab.ravel()) > 0.8
    assert np.allclose(res.bel + res.dis + res.unc, 1) and np.allclose(res.pls, res.bel + res.unc)
    t = res.layer_table.set_index("layer")
    assert t.loc["A", "single_layer_auc"] > t.loc["C", "single_layer_auc"] + 0.1
    assert t.loc["G", "best_class"] == "host" and np.isnan(t.loc["A", "conflict_when_added"])
    qa = res.layers[0].table
    assert len(qa) == 8 and np.allclose(qa.area_pct, 12.5, atol=0.1)                     # equal-area classes
    assert np.all(np.diff(qa.Bel.values[:4]) > -0.05) and qa.Bel.values[-1] > qa.Bel.values[0]
    assert len(res.pairwise) == 3 and res.stats["auc_bel"] == pytest.approx(auc(res.bel.ravel(), lab.ravel()))
    for s in ("pls", "mid"):
        alt = EBFModel(score=s).fit_predict(ev, r, c)
        assert np.allclose(alt.score, alt.pls if s == "pls" else 0.5 * (alt.bel + alt.pls))


def test_missing_evidence_vacuous_or_excluded():
    lay, (r, c) = world()
    A = lay["A"].copy()
    A[:20, :] = np.ma.masked                                         # no A evidence in the top rows
    ev = [EBFEvidence("A", A, "quantile", 8), EBFEvidence("G", lay["G"], "values")]
    res = EBFModel().fit_predict(ev, r, c)
    g = res.layer_maps["G"]
    top = np.s_[:20, :]
    assert np.allclose(res.bel[top], g["bel"][top]) and np.allclose(res.unc[top], g["unc"][top])   # only G decides
    assert res.layer_table.set_index("layer").loc["A", "missing_cells"] == 20 * 80
    ex = EBFModel(missing="exclude").fit_predict(ev, r, c)
    assert ex.study_mask.sum() == 60 * 80 and np.ma.getmaskarray(ex.bel)[top].all()


def test_certainty_warning_and_quantile_ties():
    lay, (r, c) = world()
    host = np.ma.array(np.where(lay["G"].filled(0) == 3, 2, 1))          # every deposit on the host class
    host[r, c] = 2
    ev = [EBFEvidence("host", host, "values"), EBFEvidence("A", lay["A"], "quantile", 5)]
    res = EBFModel().fit_predict(ev, r, c)
    assert any("Bel = 1" in m for m in res.layers[0].messages)
    on = host.filled(0) == 2
    assert np.allclose(res.bel[on], 1)                                # saturated: A cannot rank host cells
    hal = EBFModel(zero_deposit="haldane").fit_predict(ev, r, c)
    assert not any("Bel = 1" in m for m in hal.layers[0].messages)
    from scipy.stats import spearmanr
    assert spearmanr(hal.bel[on], lay["A"][on]).correlation > 0.8          # A ranks the host cells again
    ties = np.ma.array(np.repeat([0.0, 0.0, 0.0, 1.0, 2.0], 80 * 16).reshape(80, 80))   # 60 % one value
    t = EBFModel().fit_predict([EBFEvidence("t", ties, "quantile", 10)], r, c).layers[0].table
    assert 2 <= len(t) <= 3 and t.cells.sum() == 80 * 80 and (t.cells > 0).all()


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    import geopandas as gpd
    from core.methods import PLANNED, REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    assert "ebf" in REGISTRY and not any("belief" in p[2] for p in PLANNED)
    ev = []
    for n, cl in [("geology_host", "values"), ("dist_faults_class", "values"), ("as_log_idw", "quantile"),
                  ("sb_log_idw", "quantile")]:
        data, grid, _ = read_raster(os.path.join(DEMO, "rasters", n + ".tif"))
        ev.append(dict(name=n, data=data, grid=grid, classing=cl, n_classes=8))
    m, mg, _ = read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))
    p = dict(evidence=ev, mask=(m, mg), deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")),
             test_fraction=0.3, seed=42, split_method="spatial", zero_deposit="haldane", out_layers=True)
    r = REGISTRY["ebf"].run(p, lambda *a: None)
    assert r.evaluation.curves["test"].auc > 0.8 and r.outputs[0].name == "bel"
    assert {o.name for o in r.outputs} >= {"dis", "unc", "pls", "bel_geology_host", "bel_sb_log_idw"}
    lay = r.tables["layers"].set_index("layer")
    assert lay.loc["sb_log_idw", "single_layer_auc"] > 0.75 and lay.loc["geology_host", "missing_cells"] > 0
    assert {"summary", "layers", "pairwise", "classes_geology_host"} <= set(r.tables)
