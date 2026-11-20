import numpy as np
import pytest

from evaluation.targets import (binary_target, ca_curve, fit_segments, pa_curve, target_class_table, target_stats)


def test_fit_segments_recovers_breakpoints():
    x = np.linspace(0, 3, 90)
    y = np.where(x < 1, 5 - 0.5 * x, np.where(x < 2, 4.5 - 2.0 * (x - 1), 2.5 - 6.0 * (x - 2)))
    segs = fit_segments(x, y, 3)
    starts = [x[s[0]] for s in segs]
    assert starts[1] == pytest.approx(1.0, abs=0.05) and starts[2] == pytest.approx(2.0, abs=0.05)
    assert [round(s[2], 2) for s in segs] == [-0.5, -2.0, -6.0]


def test_ca_curve_separates_anomaly_population():
    rng = np.random.default_rng(0)
    bg = rng.lognormal(0.0, 0.3, 9000)                 # background population
    anom = rng.lognormal(2.0, 0.3, 1000)               # anomaly population (~e^2)
    ca = ca_curve(np.ma.array(np.r_[bg, anom].reshape(100, 100)), n_segments=3)
    assert len(ca.breakpoints) == 2 and ca.shift == 0
    assert 2.0 < ca.target_threshold < 8.0             # between the populations' typical values
    assert np.all(np.diff(ca.log_a) <= 0)              # A(>=v) decreases with v


def test_ca_shift_for_non_positive_values():
    a = np.ma.array(np.linspace(-5, 5, 400).reshape(20, 20))
    ca = ca_curve(a, n_segments=2)
    assert ca.shift > 5 and np.all(np.isfinite(ca.log_v))


def test_pa_curve_intersection_and_binary():
    rng = np.random.default_rng(1)
    score = np.ma.array(rng.random((100, 100)))
    r, c = rng.integers(0, 100, 40), rng.integers(0, 100, 40)
    score[r, c] = 0.9 + 0.1 * rng.random(40)           # deposits sit on high scores
    pa = pa_curve(score, r, c)
    t = pa.intersection["threshold"]
    assert pa.intersection["prediction"] == pytest.approx(100 - pa.intersection["area"], abs=1.0)
    st = target_stats(score, t, cell_area_km2=0.25, dep_rows=r, dep_cols=c)
    assert st["deposits_in_target_pct"] >= 90 and st["target_area_pct"] < 20
    b = binary_target(score, t)
    assert set(np.unique(b.compressed())) == {0, 1} and int(b.sum()) == st["target_cells"]


def test_binary_keeps_mask_and_class_colours():
    a = np.ma.array([[1.0, 5.0], [9.0, np.nan]], mask=[[0, 0], [1, 0]])
    b = binary_target(a, 5.0)
    assert b.tolist() == [[0, 1], [None, None]]
    t = target_class_table()
    assert t.loc[t.value == 1, "color"].item() == "#e53935" and t.loc[t.value == 0, "color"].item().endswith("00")
