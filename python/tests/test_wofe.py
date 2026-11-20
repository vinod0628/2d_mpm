import os
import warnings

import numpy as np
import pytest

from conftest import CARLIN
from logic.common.training import split_deposits
from logic.data_driven.wofe import Evidence, WofEModel, binary_weights
from evaluation.metrics import evaluate

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def test_binary_weights_hand_calculation():
    # Bonham-Carter style toy: N=100 unit cells, D=10, pattern 20 cells containing 6 deposits
    wp, swp, wm, swm, c, sc = binary_weights(6, 20, 10, 100)
    assert wp == pytest.approx(np.log((6 / 10) / (14 / 90)))
    assert wm == pytest.approx(np.log((4 / 10) / (76 / 90)))
    assert swp == pytest.approx(np.sqrt(1 / 6 + 1 / 14))
    assert swm == pytest.approx(np.sqrt(1 / 4 + 1 / 76))
    assert c == pytest.approx(wp - wm)
    assert sc == pytest.approx(np.sqrt(swp ** 2 + swm ** 2))


def test_zero_deposits_are_undefined_or_haldane():
    wp, *_ = binary_weights(0, 20, 10, 100)
    assert np.isnan(wp)
    wp_h, *_ = binary_weights(0, 20, 10, 100, zero_deposit="haldane")
    assert wp_h == pytest.approx(np.log((0.5 / 10) / (20 / 90)))


def _toy():
    rng = np.random.default_rng(1)
    ev = np.ma.array(rng.integers(1, 4, size=(60, 60)))
    dep = np.zeros(ev.shape, bool)
    r, c = np.where((ev == 3) & (rng.random(ev.shape) < 0.05))
    dep[r, c] = True
    return ev, r, c


def test_posterior_is_prior_when_evidence_uninformative():
    ev = np.ma.array(np.tile([1, 2], (50, 25)))         # half/half, deposits evenly spread
    rows, cols = np.array([0, 0, 10, 10]), np.array([0, 1, 4, 5])
    res = WofEModel(unit_cell_km2=1, cell_area_km2=1).fit_predict([Evidence("e", ev)], rows, cols)
    assert res.layers[0].class_weight[1.0] == pytest.approx(res.layers[0].class_weight[2.0], abs=1e-9)
    assert np.allclose(res.posterior, res.prior, atol=1e-6)


def test_posterior_ranks_favourable_class_highest():
    ev, r, c = _toy()
    res = WofEModel(1, 1).fit_predict([Evidence("e", ev, "categorical")], r, c)
    post = res.posterior
    assert post[ev == 3].mean() > post[ev == 1].mean()
    ev_res = evaluate(post, r, c)
    assert ev_res.curves["training"].auc > 0.8
    assert res.ci["Observed deposits N(D)"] == len(r)


def test_cumulative_selection_and_generalisation():
    ev, r, c = _toy()
    lw = WofEModel(1, 1).fit_predict([Evidence("e", ev, "descending", min_studentized=0)], r, c).layers[0]
    assert lw.threshold == 3.0                          # class 3 alone is the best descending pattern
    assert set(lw.class_gen.values()) == {1, 2}
    assert lw.class_weight[3.0] > 0 > lw.class_weight[1.0]


def _arcsdm_md_variance(w, post, base):
    """Literal port of ArcSDM-master missingdatavar_func.create_missing_data_variance_raster (one layer)."""
    r1 = np.where(base & (w == 0.0), post, 0.0)                       # CON(W == 0.0, PPRB, 0.0)
    r2 = r1 / (1.0 - r1)                                              # R1 / (1 - R1)
    var = np.zeros(w.shape)
    vals, counts = np.unique(w[base], return_counts=True)             # FloatRasterSearchcursor
    total = float(base.sum())
    for v, n in zip(vals, counts):
        if v == 0.0:
            continue
        with np.errstate(divide="ignore"):
            e1 = np.where(r2 == 0.0, 0.0, np.exp(np.log(r2) + v))    # CON(R2 == 0, 0, EXP(LN(R2) + w))
        e2 = e1 / (1 + e1)
        var += (e2 - r1) ** 2 * (n / total)                           # SQR(Exp2 - R1) * (A_j / A)
    return np.where(base, var, 0.0)


def test_missing_data_variance_matches_arcsdm():
    ev, r, c = _toy()
    ev[:10, :] = np.ma.masked                                        # missing evidence in the study area
    other = np.ma.array(np.tile([1, 2], (60, 30)))
    study = np.ones(ev.shape, bool)
    # ArcSDM finds missing cells by "weight == 0", which would also catch known classes whose weight is 0
    # (e.g. classes without deposits under the zero rule); we use the true missing mask. The 0.5-deposit
    # correction leaves no known class at weight 0, so both definitions coincide here.
    res = WofEModel(1, 1, zero_deposit="haldane").fit_predict([Evidence("e", ev), Evidence("o", other)], r, c, study)
    assert res.md_variance is not None
    w = res.weight_rasters["e"].filled(0.0)
    expected = _arcsdm_md_variance(w, res.posterior.filled(0), study)
    assert np.allclose(res.md_variance.filled(0), expected, rtol=1e-10, atol=1e-15)
    assert res.md_variance[:10, :].min() > 0 and np.all(res.md_variance[10:, :] == 0)
    assert np.allclose(res.total_std, np.sqrt(res.posterior_std ** 2 + res.md_variance))
    assert np.allclose(res.confidence, res.posterior / res.total_std)
    # zero rule: classes without deposits have weight 0 but are known -> no missing-data variance there
    zr = WofEModel(1, 1).fit_predict([Evidence("e", ev), Evidence("o", other)], r, c, study)
    assert np.all(zr.md_variance[10:, :] == 0) and zr.md_variance[:10, :].min() > 0
    # no missing data -> no variance term, total std = std
    full = WofEModel(1, 1).fit_predict([Evidence("o", other)], [0, 0, 10], [0, 1, 4], study)
    assert full.md_variance is None and np.allclose(full.total_std, full.posterior_std)


def test_ci_report_as_arcsdm():
    ev, r, c = _toy()
    ci = WofEModel(1, 1).fit_predict([Evidence("e", ev)], r, c).ci
    p = ci["Probability of conditional dependence (%)"]
    assert ci["Overall CI (%)"] == pytest.approx(2 * min(p, 100 - p))
    assert ci["Agterberg-Cheng p (one-tailed, T > N(D))"] == pytest.approx(1 - p / 100)


def test_integer_evidence_with_many_classes_cumulative():
    """No class limit for cumulative types on integer rasters: every class value is a candidate threshold."""
    rng = np.random.default_rng(4)
    dist = np.ma.array(rng.integers(0, 5000, size=(120, 120)))          # ~4,900 integer values (distance in m)
    r, c = np.where((dist < 600) & (rng.random(dist.shape) < 0.05))     # deposits close to the feature
    lw = WofEModel(1, 1).fit_predict([Evidence("dist", dist, "ascending")], r, c).layers[0]
    assert len(lw.table) == np.unique(dist).size > 255                   # one threshold per value
    assert lw.threshold is not None and 300 < lw.threshold < 1500        # threshold near the true 600 m limit
    assert lw.studentized >= 1.96
    from logic.data_driven.wofe import MAX_CATEGORICAL
    with pytest.raises(ValueError, match="cumulative"):                  # categorical: one weight per value -> guard
        WofEModel(1, 1).fit_predict([Evidence("dist", dist, "categorical")], r, c)
    assert MAX_CATEGORICAL >= 255


@need_data
def test_carlin_integer_layers_with_many_classes():
    """Integer evidence with thousands of values (distance in whole metres, Sb ×1000) as cumulative types."""
    import time
    import geopandas as gpd
    from utils.raster_tools import points_to_cells, read_raster
    mask = ~np.ma.getmaskarray(read_raster(os.path.join(CARLIN, "rasters", "study_mask.tif"))[0])
    dist, grid, _ = read_raster(os.path.join(CARLIN, "rasters", "dist_faults.tif"))
    sb, _, _ = read_raster(os.path.join(CARLIN, "rasters", "sb_surface.tif"))
    dist = np.ma.round(dist).astype(np.int32)                             # integer metres
    sb = np.ma.round(sb * 1000).astype(np.int32)
    dep = gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg"))
    rows, cols = points_to_cells(dep, grid, valid_mask=mask)
    t0 = time.time()
    ev = [Evidence("dist_faults", dist, "ascending", criterion="max_studentized"),
          Evidence("sb", sb, "descending", criterion="max_studentized")]   # dialog default for many values
    res = WofEModel(1.0, grid.cell_area / 1e6).fit_predict(ev, rows, cols, mask)
    assert time.time() - t0 < 60
    assert all(len(lw.table) > 255 for lw in res.layers)
    for lw in res.layers:
        sel = lw.table[lw.table.selected].iloc[0]
        assert sel.cum_deposits >= 5 and sel.cum_area > 0.005 * lw.table.area.sum()   # not a tiny pattern
    auc_cont = evaluate(res.posterior, rows, cols, valid_mask=res.study_mask).curves["training"].auc
    # same two layers after reclassification (dist_faults_class, sb_q8): the direct use is as good
    cls = [Evidence("d", read_raster(os.path.join(CARLIN, "rasters", "dist_faults_class.tif"))[0], "ascending",
                    criterion="max_studentized"),
           Evidence("s", read_raster(os.path.join(CARLIN, "rasters", "sb_q8.tif"))[0], "descending",
                    criterion="max_studentized")]
    rc = WofEModel(1.0, grid.cell_area / 1e6).fit_predict(cls, rows, cols, mask)
    auc_cls = evaluate(rc.posterior, rows, cols, valid_mask=rc.study_mask).curves["training"].auc
    assert auc_cont > 0.7 and auc_cont >= auc_cls - 0.03, (auc_cont, auc_cls)
    # a pattern without deposits is never selected, whatever the criterion
    lw = WofEModel(1.0, grid.cell_area / 1e6).fit_predict([Evidence("sb", sb, "descending")], rows, cols, mask).layers[0]
    assert lw.table[lw.table.selected].iloc[0].cum_deposits > 0


def test_split_deposits():
    r, c = np.arange(20), np.arange(20)
    tr_r, tr_c, te_r, te_c = split_deposits(r, c, 0.25, seed=3)
    assert len(te_r) == 5 and len(tr_r) == 15
    assert not set(te_r) & set(tr_r)


def test_evaluation_perfect_and_random():
    rng = np.random.default_rng(0)
    score = np.ma.array(rng.random((100, 100)))
    r, c = rng.integers(0, 100, 30), rng.integers(0, 100, 30)
    perfect = score.copy()
    perfect[r, c] = 2.0
    assert evaluate(perfect, r, c).curves["training"].auc == pytest.approx(1.0)
    assert 0.35 < evaluate(score, r, c).curves["training"].auc < 0.65
    e = evaluate(perfect, r, c).curves["training"]
    assert e.area_frac[0] == 0 and e.capture_frac[-1] == 1


def test_delong_auc_and_absence_points():
    from evaluation.metrics import delong_auc
    rng = np.random.default_rng(5)
    pos = np.round(rng.normal(1, 1, 40), 1)                       # rounded -> ties
    neg = np.round(rng.normal(0, 1, 70), 1)
    auc, se, (lo, hi) = delong_auc(pos, neg)
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])   # brute-force U statistic
    assert auc == pytest.approx(psi.mean())
    v10, v01 = psi.mean(1), psi.mean(0)
    assert se == pytest.approx(np.sqrt(v10.var(ddof=1) / 40 + v01.var(ddof=1) / 70))
    assert lo < auc < hi
    # absence points replace the background as ROC negatives (ArcSDM ROC tool "absence locations")
    score = np.ma.array(rng.random((50, 50)))
    r, c = np.array([1, 2, 3]), np.array([1, 2, 3])
    score[r, c] = 2.0
    nr, nc = np.array([10, 11]), np.array([10, 11])
    score[nr, nc] = 3.0                                            # absences score highest -> AUC 0
    ev = evaluate(score, r, c, negatives=(nr, nc)).curves["training"]
    assert ev.n_neg == 2 and ev.negatives == "absence" and ev.auc == pytest.approx(0.0)
    assert evaluate(score, r, c).curves["training"].auc > 0.99


# ------------------------------------------------------------------ against eis_toolkit (real data)
@need_data
@pytest.mark.parametrize("raster,wtype", [("aeromag", "categorical"), ("dist_faults_class", "ascending"),
                                          ("as_q8", "descending")])
def test_matches_eis_toolkit(raster, wtype):
    import geopandas as gpd
    import rasterio
    from eis_toolkit.prediction.weights_of_evidence import weights_of_evidence_calculate_weights
    from utils.raster_tools import Grid, points_to_cells, read_raster

    path = os.path.join(CARLIN, "rasters", raster + ".tif")
    dep = gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg"))
    arr, grid, _ = read_raster(path)
    with rasterio.open(path) as src:
        eis = weights_of_evidence_calculate_weights(
            src, dep, raster_nodata=src.nodata,
            weights_type="unique" if wtype == "categorical" else wtype)[0]
    rows, cols = points_to_cells(dep, grid, valid_mask=~np.ma.getmaskarray(arr))
    model = WofEModel(unit_cell_km2=grid.cell_area / 1e6, cell_area_km2=grid.cell_area / 1e6)
    res = model.fit_predict([Evidence(raster, arr, wtype)], rows, cols, study_mask=~np.ma.getmaskarray(arr))
    t = res.layers[0].table
    in_pattern = t["cum_deposits"] if "cum_deposits" in t else t["deposits"]
    n_dep = res.layers[0].n_deposits
    mine = dict(zip(t["cls"].astype(float), zip(t["W+"], t["W-"], t["s(W+)"], in_pattern)))
    checked = 0
    for _, row in eis.iterrows():
        cls = float(row["Class"])
        wp, wm, swp, n_in = mine[cls]
        # where a deposit count is 0 the weights are undefined; eis patches with 0/0.99, we use 0.5 or NaN
        if n_in in (0, n_dep) or not np.isfinite(wp) or not np.isfinite(wm):
            continue
        assert wp == pytest.approx(row["W+"], abs=1.5e-4), cls
        assert wm == pytest.approx(row["W-"], abs=1.5e-4), cls
        assert swp == pytest.approx(row["S_W+"], abs=1.5e-4), cls
        checked += 1
    assert checked >= 2


@need_data
def test_carlin_full_model_runs():
    import geopandas as gpd
    from utils.raster_tools import points_to_cells, read_class_table, read_raster

    names = [("geology_host", "categorical"), ("dist_faults_class", "ascending"), ("as_q8", "descending"),
             ("sb_q8", "descending")]
    evid = []
    for n, t in names:
        a, grid, _ = read_raster(os.path.join(CARLIN, "rasters", n + ".tif"))
        tab = read_class_table(os.path.join(CARLIN, "rasters", n + ".tif"))
        evid.append(Evidence(n, a, t, labels=dict(zip(tab.value.astype(float), tab.label)) if tab is not None else None))
    mask = ~np.ma.getmaskarray(read_raster(os.path.join(CARLIN, "rasters", "study_mask.tif"))[0])
    dep = gpd.read_file(os.path.join(CARLIN, "vectors", "deposits.gpkg"))
    rows, cols = points_to_cells(dep, grid, valid_mask=mask)
    res = WofEModel(unit_cell_km2=1.0, cell_area_km2=grid.cell_area / 1e6).fit_predict(evid, rows, cols, mask)
    assert 0 < res.prior < 0.01
    assert res.posterior.max() <= 1 and res.posterior.min() >= 0
    assert res.layers[1].threshold is not None
    assert evaluate(res.posterior, rows, cols, valid_mask=mask).curves["training"].auc > 0.75
    assert len(res.pairwise) == 6
