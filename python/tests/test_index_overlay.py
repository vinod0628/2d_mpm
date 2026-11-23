"""Index overlay with AHP / fuzzy-AHP weights (logic/knowledge_driven/index_overlay)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from logic.knowledge_driven.index_overlay.model import (IOLayer, ahp_weights, fuzzy_ahp_weights,  # noqa: E402
                                                        index_overlay, layer_scores, reciprocal_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO = os.path.join(ROOT, "demo_data", "nevada_carlin_au", "processed")


def test_ahp_textbook_example_and_consistency():
    # Saaty-type 3 × 3 example: weights ≈ (0.637, 0.258, 0.105), λmax ≈ 3.039, CR ≈ 0.033
    A = reciprocal_matrix({(0, 1): 3, (0, 2): 5, (1, 2): 3}, 3)
    r = ahp_weights(A)
    assert np.allclose(r.weights, [0.637, 0.258, 0.105], atol=2e-3)
    assert r.lambda_max == pytest.approx(3.039, abs=2e-3) and r.cr == pytest.approx(0.033, abs=3e-3)
    # perfectly consistent matrix a_ij = w_i / w_j: weights recovered, CR = 0
    w = np.array([0.5, 0.3, 0.15, 0.05])
    r = ahp_weights(np.outer(w, 1 / w))
    assert np.allclose(r.weights, w) and r.cr == pytest.approx(0.0, abs=1e-10)
    # inconsistent judgements (A > B > C but C > A) -> CR ≥ 0.1
    assert ahp_weights(reciprocal_matrix({(0, 1): 5, (1, 2): 5, (0, 2): 1 / 5}, 3)).cr > 0.1


def test_fuzzy_ahp_limits():
    A = reciprocal_matrix({(0, 1): 3, (0, 2): 5, (1, 2): 3, (0, 3): 7, (1, 3): 5, (2, 3): 3}, 4)
    f0 = fuzzy_ahp_weights(A, delta=0.0)                       # δ = 0: Buckley = crisp geometric-mean method
    gm = np.prod(A, axis=1) ** (1 / 4)
    assert np.allclose(f0.weights, gm / gm.sum()) and np.allclose(f0.fuzzy[:, 0], f0.fuzzy[:, 2])
    f1 = fuzzy_ahp_weights(A, delta=1.0)
    assert np.all(f1.fuzzy[:, 0] <= f1.fuzzy[:, 1]) and np.all(f1.fuzzy[:, 1] <= f1.fuzzy[:, 2])
    assert f1.weights.sum() == pytest.approx(1.0) and np.all(np.diff(f1.weights) < 0)   # ranking kept
    assert f1.weights[0] < f0.weights[0]                       # fuzziness pulls the weights together


def test_scores_and_weighted_mean():
    x = np.ma.array(np.arange(10.0).reshape(2, 5))
    assert np.allclose(layer_scores(IOLayer("a", x, "positive", (2.0, 6.0))).ravel()[[0, 4, 9]], [0, 0.5, 1])
    assert np.allclose(layer_scores(IOLayer("a", x, "negative", (2.0, 6.0))).ravel()[[0, 4, 9]], [1, 0.5, 0])
    assert layer_scores(IOLayer("a", x, "ge", 7.0)).sum() == 3
    c = layer_scores(IOLayer("c", x, "classes", {1.0: 0.9, 2.0: 0.2}))
    assert c.count() == 2 and c.ravel()[1] == pytest.approx(0.9)     # classes without a score: no data
    a = IOLayer("a", x, "positive", (0.0, 9.0), 3.0)
    b = IOLayer("b", np.ma.array(np.ones((2, 5))), "ge", 0.5, 1.0)
    r = index_overlay([a, b], "custom")
    assert np.allclose(r.index, 0.75 * np.arange(10).reshape(2, 5) / 9 + 0.25)
    assert np.allclose(r.weights, [0.75, 0.25])


def test_missing_rules_and_cr_warning():
    x = np.ma.array(np.arange(9.0).reshape(3, 3), mask=np.eye(3, dtype=bool))
    y = np.ma.array(np.full((3, 3), 5.0))
    la, lb = IOLayer("a", x, "positive", (0.0, 8.0)), IOLayer("b", y, "ge", 1.0)
    ex = index_overlay([la, lb], "equal", missing="exclude")
    assert ex.study_mask.sum() == 6 and any("excluded" in m for m in ex.messages)
    rn = index_overlay([la, lb], "equal", missing="renormalise")
    assert rn.study_mask.sum() == 9 and rn.index[0, 0] == pytest.approx(1.0)    # only layer b at the masked cells
    bad = reciprocal_matrix({(0, 1): 5, (1, 2): 5, (0, 2): 1 / 5}, 3)
    lc = IOLayer("c", y, "le", 9.0)
    r = index_overlay([la, lb, lc], "ahp", pairwise=bad)
    assert any("CR" in m for m in r.messages) and r.ahp_table is not None
    with pytest.raises(ValueError):
        index_overlay([la, lb], "ahp", pairwise=np.ones((3, 3)))


@pytest.mark.skipif(not os.path.isdir(DEMO), reason="demo data missing")
def test_carlin_model():
    import geopandas as gpd
    from core.methods import REGISTRY, load_builtin_methods
    from utils.raster_tools import read_raster
    load_builtin_methods()
    rd = lambda n: read_raster(os.path.join(DEMO, "rasters", n + ".tif"))[:2]  # noqa: E731
    setup = [("geology_host", "classes", "2=1; 1=0"), ("dist_faults", "negative", "0, 6000"),
             ("as_log_idw", "positive", "0.6, 1.9"), ("sb_log_idw", "positive", "0, 1.3")]
    ev = [dict(name=n, data=rd(n)[0], grid=rd(n)[1], rule=r, spec=s, weight=1.0) for n, r, s in setup]
    A = reciprocal_matrix({(0, 1): 3, (0, 2): 2, (0, 3): 1, (1, 2): 1 / 2, (1, 3): 1 / 3, (2, 3): 1 / 2}, 4)
    p = dict(evidence=ev, weighting="ahp", pairwise=A.tolist(), mask=read_raster(os.path.join(DEMO, "rasters",
             "study_mask.tif"))[:2], deposits=gpd.read_file(os.path.join(DEMO, "vectors", "deposits.gpkg")))
    r = REGISTRY["index_overlay"].run(p, lambda *a: None)
    assert r.evaluation.curves["training"].auc > 0.8
    assert r.extra["io"].ahp.cr < 0.1 and "ahp" in r.tables
    w = r.tables["evidence"].set_index("layer").weight
    assert w["geology_host"] > w["dist_faults"]
    p.update(weighting="fuzzy_ahp", delta=1.0)
    rf = REGISTRY["index_overlay"].run(p, lambda *a: None)
    assert rf.evaluation.curves["training"].auc > 0.8 and "fuzzy_l" in rf.tables["ahp"]
