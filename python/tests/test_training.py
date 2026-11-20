"""Point selection shared by the data-driven methods: train/test split, thinning, non-deposit sampling."""
import numpy as np
import pytest

from logic.common.training import (auto_spacing, background_candidates, hilbert_index, sample_background,
                                   split_deposits, thin_points)


def test_hilbert_curve_visits_every_cell_through_neighbours():
    r, c = np.divmod(np.arange(64), 8)
    d = hilbert_index(r, c, order=3)
    assert sorted(d) == list(range(64))                         # bijection
    order = np.argsort(d)
    steps = np.abs(np.diff(r[order])) + np.abs(np.diff(c[order]))
    assert np.all(steps == 1)                                   # consecutive positions are 4-neighbours


def test_spatial_split_covers_the_area_evenly():
    rng = np.random.default_rng(0)
    west = np.c_[rng.integers(0, 100, 70), rng.integers(0, 50, 70)]      # 70 deposits in the west half
    east = np.c_[rng.integers(0, 100, 30), rng.integers(150, 200, 30)]   # 30 in the east
    pts = np.r_[west, east]
    for seed in range(20):
        tr_r, tr_c, te_r, te_c = split_deposits(pts[:, 0], pts[:, 1], 0.2, seed=seed, method="spatial")
        assert len(te_r) == 20 and len(tr_r) == 80
        n_east = int((te_c >= 150).sum())
        assert 5 <= n_east <= 7                                  # 30 % of the test set (6) — every seed
    random_east = [int((split_deposits(pts[:, 0], pts[:, 1], 0.2, seed=s, method="random")[3] >= 150).sum())
                   for s in range(200)]
    assert max(random_east) - min(random_east) >= 6              # a plain random split is much less even


def test_training_and_test_point_layers_keep_the_original_points():
    import geopandas as gpd
    from rasterio.transform import from_origin
    from shapely.geometry import Point
    from utils.raster_tools import Grid, points_to_cells
    grid = Grid(from_origin(0, 1000, 10, 10), None, 100, 100)
    rng = np.random.default_rng(1)
    xy = rng.uniform(5, 995, size=(40, 2))
    gdf = gpd.GeoDataFrame({"name": [f"d{i}" for i in range(40)]}, geometry=[Point(x, y) for x, y in xy])
    gdf.loc[40] = ["twin", Point(xy[0, 0] + 0.5, xy[0, 1] + 0.5)]     # same cell as d0
    gdf.loc[41] = ["outside", Point(5000, 5000)]                         # outside the grid: in no set
    rows, cols = points_to_cells(gdf, grid)
    tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, 0.25, seed=2)
    from logic.common.training import split_point_layers
    train, test = split_point_layers(gdf, grid, tr_r, tr_c, te_r, te_c)
    assert len(train) + len(test) == 41 and not set(train.name) & set(test.name)   # every point inside, once
    assert "outside" not in set(train.name) | set(test.name)
    assert ("twin" in set(train.name)) == ("d0" in set(train.name))     # points of one cell share the set
    assert set(train.split) == {"training"} and set(test.split) == {"test"}
    assert list(train.columns[:1]) == ["name"]                           # original attributes kept
    assert len(test) >= len(te_r)


def test_thinning_as_arcsdm():
    rows = np.array([0, 0, 0, 10, 10])
    cols = np.array([0, 1, 5, 0, 1])
    keep = thin_points(rows, cols, 3.0)                          # cell units
    assert keep.tolist() == [True, False, True, True, False]     # greedy in input order
    kr, kc = rows[keep], cols[keep]
    d = np.hypot(kr[:, None] - kr[None], kc[:, None] - kc[None])
    assert d[np.triu_indices(len(kr), 1)].min() >= 3.0
    assert thin_points(rows, cols, 0).all()


def test_background_samples_are_spread_and_inside_the_study_area():
    study = np.zeros((200, 200), bool)
    study[20:180, 10:190] = True
    cand = background_candidates(study, [100], [100], min_distance=20.0)
    n = 100
    sp = auto_spacing(int(cand.sum()), n)
    r, c = sample_background(cand, n, seed=3, min_spacing=sp)
    assert len(r) == n and cand[r, c].all()                     # inside the study area, outside the buffer
    d = np.hypot(r[:, None] - r[None], c[:, None] - c[None])
    assert d[np.triu_indices(n, 1)].min() >= sp                 # evenly spread (no two closer than the spacing)
    r0, c0 = sample_background(cand, n, seed=3)                  # unconstrained: clumps occur
    d0 = np.hypot(r0[:, None] - r0[None], c0[:, None] - c0[None])
    assert d0[np.triu_indices(n, 1)].min() < sp
    with pytest.raises(ValueError, match="spacing"):
        sample_background(cand, n, seed=3, min_spacing=60.0)    # cannot fit 100 points 60 cells apart
