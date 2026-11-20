"""Convolutional neural network: patch extraction, D4 augmentation, training on spatial context, real data."""
import os
import warnings

import numpy as np
import pytest
from scipy import ndimage
from scipy.special import expit

from conftest import CARLIN
from logic.common.features import Feature, FeatureStack, study_area
from logic.data_driven.cnn import CNNModel, PatchGrid, dihedral, parse_filters

warnings.filterwarnings("ignore")
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


def test_dihedral_augmentation_is_the_d4_group():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(1, 5, 5, 2))
    aug = dihedral(p)
    assert aug.shape == (8, 5, 5, 2)
    flat = {a.tobytes() for a in aug}
    assert len(flat) == 8                                            # all 8 symmetries distinct for a random patch
    assert np.allclose(aug[0], p[0]) and np.allclose(aug[2], np.rot90(p[0], 2))
    assert np.allclose(np.sort(aug.ravel()), np.sort(np.tile(p.ravel(), 8)))       # values only rearranged


def test_patch_grid_windows_and_padding():
    a = np.ma.array(np.arange(30, dtype=float).reshape(5, 6))
    mask = np.ones((5, 6), bool)
    mask[0, 0] = False                                               # one cell outside the study area
    st = FeatureStack([Feature("a", a)], study_area([Feature("a", a)], mask, "exclude"))
    pg = PatchGrid(st, (5, 6), 3)
    col = st.column(0).astype(float)
    z = (a - col.mean()) / col.std()
    pos = st.positions([2, 0], [3, 5])
    P = pg.patches(pos)
    assert P.shape == (2, 3, 3, 1)
    assert np.allclose(P[0, :, :, 0], z[1:4, 2:5])                   # interior window
    assert np.allclose(P[1, 0, :, 0], 0) and np.allclose(P[1, :, 2, 0], 0)      # zero padding at the border
    assert np.isclose(P[1, 1, 1, 0], z[0, 5])
    corner = pg.patches(st.positions([1], [1]))[0, :, :, 0]
    assert corner[0, 0] == 0                                         # cell outside the study area -> 0 (mean)
    assert parse_filters("16, 32") == (16, 32)
    with pytest.raises(ValueError):
        CNNModel(patch=8)


def test_cnn_learns_spatial_signal():
    rng = np.random.default_rng(3)
    n = 70
    f = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3)
    f = (f - f.mean()) / f.std()
    noise = ndimage.gaussian_filter(rng.normal(size=(n, n)), 3)
    r, c = np.where(rng.random((n, n)) < expit(-5.5 + 2.0 * f))
    feats = [Feature("sig", np.ma.array(f)), Feature("noise", np.ma.array(noise))]
    res = CNNModel(patch=5, neg_ratio=2, min_spacing=0, seed=2, networks=1, max_epochs=40, patience=10,
                   filters=(8,), dense=16).fit_predict(feats, r, c)
    assert res.stats["cv_auc"] > 0.8 and res.stats["device"] in ("CPU", "GPU")
    imp = res.importance.set_index("layer")
    assert imp.index[0] == "sig" and imp.loc["sig", "permutation_auc_drop"] > imp.loc["noise", "permutation_auc_drop"]
    assert res.score[f > 1.5].mean() > res.score[f < -1.5].mean() + 0.3
    assert 0 <= res.score.min() and res.score.max() <= 1


@need_data
def test_carlin_cnn():
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
    res = CNNModel(neg_ratio=3, min_distance=1000, res=grid.res, seed=1, networks=1, max_epochs=60).fit_predict(
        feats, rows, cols, mask)
    assert evaluate(res.score, rows, cols, valid_mask=res.study_mask).curves["training"].auc > 0.85
    assert res.stats["cv_auc"] > 0.7
