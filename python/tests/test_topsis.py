"""TOPSIS / VIKOR with expert, AHP, fuzzy-AHP or entropy weights (logic/knowledge_driven/topsis)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.knowledge_driven.index_overlay.model import IOLayer, reciprocal_matrix  # noqa: E402
from logic.knowledge_driven.topsis.model import entropy_weights, mcdm  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")

# 5 alternatives (cells) × 3 benefit criteria, already scored 0 … 1
X = np.array([[0.9, 0.2, 0.6],
              [0.4, 0.8, 0.5],
              [0.1, 0.1, 0.1],
              [1.0, 1.0, 1.0],
              [0.6, 0.5, 0.9]])
W = np.array([0.5, 0.3, 0.2])


def _layers(x=X, weights=W):
    return [IOLayer(f"c{j}", np.ma.array(x[:, j].reshape(1, -1)), "positive", (0.0, 1.0), w)
            for j, w in enumerate(weights)]


def _topsis_reference(x, w, vector=True):
    """Textbook TOPSIS (Hwang & Yoon 1981), written out step by step."""
    r = x / np.sqrt((x ** 2).sum(0)) if vector else x
    v = r * w
    ap, am = v.max(0), v.min(0)
    dp = np.sqrt(((v - ap) ** 2).sum(1))
    dm = np.sqrt(((v - am) ** 2).sum(1))
    return dm / (dp + dm)


def _vikor_reference(x, w, v):
    fs, fm = x.max(0), x.min(0)
    g = w * (fs - x) / (fs - fm)
    S, R = g.sum(1), g.max(1)
    return v * (S - S.min()) / (S.max() - S.min()) + (1 - v) * (R - R.min()) / (R.max() - R.min()), S, R


def test_topsis_matches_textbook_steps():
    for norm in ("vector", "linear"):
        r = mcdm(_layers(), "topsis", "custom", normalisation=norm)
        c = np.ma.getdata(r.index).ravel()
        assert np.allclose(c, _topsis_reference(X, W, norm == "vector"))
        assert c[3] == pytest.approx(1.0) and c[2] == pytest.approx(0.0)    # ideal / anti-ideal cells
        assert {"d_plus", "d_minus"} == set(r.parts)
    # weights matter: raising the weight of criterion 1 moves alternative 1 above alternative 0
    lo = np.ma.getdata(mcdm(_layers(weights=[0.6, 0.2, 0.2]), "topsis", "custom").index).ravel()
    hi = np.ma.getdata(mcdm(_layers(weights=[0.2, 0.6, 0.2]), "topsis", "custom").index).ravel()
    assert lo[0] > lo[1] and hi[1] > hi[0]


def test_vikor_matches_definition_and_strategy_weight():
    for v in (0.0, 0.5, 1.0):
        r = mcdm(_layers(), "vikor", "custom", v=v)
        q, S, R = _vikor_reference(X, W, v)
        assert np.allclose(np.ma.getdata(r.parts["Q"]).ravel(), q)
        assert np.allclose(np.ma.getdata(r.parts["S"]).ravel(), S)
        assert np.allclose(np.ma.getdata(r.parts["R"]).ravel(), R)
        assert np.allclose(np.ma.getdata(r.index).ravel(), 1 - q)
    # v = 1: ranking by group utility only; v = 0: by the largest weighted shortfall only
    s1 = np.ma.getdata(mcdm(_layers(), "vikor", "custom", v=1.0).index).ravel()
    assert np.array_equal(np.argsort(-s1), np.argsort(_vikor_reference(X, W, 1.0)[1]))
    with pytest.raises(ValueError):
        mcdm(_layers(), "vikor", v=1.5)


def test_entropy_weights():
    S = np.array([[0.5] * 6,                              # constant: no information -> weight 0
                  [1, 0, 0, 0, 0, 0],                     # concentrated -> large weight
                  [0.2, 0.4, 0.6, 0.8, 1.0, 0.5]], float)
    w = entropy_weights(S, np.zeros_like(S, bool))
    assert w[0] == pytest.approx(0.0, abs=1e-12) and w[1] > w[2] > 0 and w.sum() == pytest.approx(1.0)
    # by hand for row 2: E = -Σ p ln p / ln m
    p = S[2] / S[2].sum()
    e2 = -(p * np.log(p)).sum() / np.log(6)
    assert w[2] / w[1] == pytest.approx((1 - e2) / 1.0)            # row 1: E = 0
    r = mcdm(_layers(), "topsis", "entropy")
    assert r.weights.sum() == pytest.approx(1.0) and r.table.weight.tolist() == pytest.approx(r.weights.tolist())


def test_missing_renormalise_and_exclude():
    a = np.ma.array([[0.0, 0.5, 1.0, 0.25]], mask=[[False, False, False, True]])
    b = np.ma.array([[0.0, 1.0, 0.5, 0.75]])
    la, lb = IOLayer("a", a, "positive", (0.0, 1.0)), IOLayer("b", b, "positive", (0.0, 1.0))
    rn = mcdm([la, lb], "topsis", normalisation="linear")
    assert rn.study_mask.sum() == 4
    assert rn.index[0, 3] == pytest.approx(0.75)        # only b: (0.75 − 0) / (1 − 0)
    ex = mcdm([la, lb], "topsis", normalisation="linear", missing="exclude")
    assert ex.study_mask.sum() == 3 and any("excluded" in m for m in ex.messages)
    vk = mcdm([la, lb], "vikor")
    assert np.isfinite(np.ma.getdata(vk.index)).all() and vk.index.max() == pytest.approx(1.0)
    bad = reciprocal_matrix({(0, 1): 5, (1, 2): 5, (0, 2): 1 / 5}, 3)
    r = mcdm(_layers(), "topsis", "ahp", pairwise=bad)
    assert any("CR" in m for m in r.messages) and r.ahp_table is not None


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    import geopandas as gpd
    from core.methods import PLANNED, REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    assert "topsis" in REGISTRY and not any(p[2] == "TOPSIS" for p in PLANNED)
    rd = lambda n: read_raster(os.path.join(DEMO, "rasters", n + ".tif"))[:2]  # noqa: E731
    setup = [("geology_host", "classes", "2=1; 1=0"), ("dist_faults", "negative", "0, 6000"),
             ("as_log_idw", "positive", "0.6, 1.9"), ("sb_log_idw", "positive", "0, 1.3")]
    ev = [dict(name=n, data=rd(n)[0], grid=rd(n)[1], rule=r, spec=s, weight=1.0) for n, r, s in setup]
    A = reciprocal_matrix({(0, 1): 3, (0, 2): 2, (0, 3): 1, (1, 2): 1 / 2, (1, 3): 1 / 3, (2, 3): 1 / 2}, 4)
    p = dict(evidence=ev, method="topsis", weighting="ahp", pairwise=A.tolist(),
             mask=read_raster(os.path.join(DEMO, "rasters", "study_mask.tif"))[:2],
             deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")), out_parts=True)
    r = REGISTRY["topsis"].run(p, lambda *a: None)
    assert r.evaluation.curves["training"].auc > 0.8 and "ahp" in r.tables
    assert {o.name for o in r.outputs} >= {"topsis", "d_plus", "d_minus"}
    p.update(method="vikor", weighting="fuzzy_ahp", v=0.5)
    rv = REGISTRY["topsis"].run(p, lambda *a: None)
    assert rv.evaluation.curves["training"].auc > 0.8 and {o.name for o in rv.outputs} >= {"vikor", "S", "R", "Q"}
