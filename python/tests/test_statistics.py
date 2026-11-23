"""Descriptive statistics and histogram intervals of the layer Statistics tab (utils/statistics.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.statistics import describe, ecdf, histogram_edges, nice_width, sturges_bins  # noqa: E402


def test_describe_by_hand():
    x = [2, 4, 4, 4, 5, 5, 7, 9, np.nan]
    st = describe(x, nulls=2)
    v = np.array(x[:-1], float)
    assert st["count"] == 8 and st["nulls"] == 3                   # NaN counted with the given nulls
    assert (st["min"], st["max"], st["range"], st["sum"]) == (2, 9, 7, 40)
    assert st["mean"] == 5 and st["std"] == pytest.approx(2.0)     # population standard deviation
    assert st["cv"] == pytest.approx(0.4)
    d = v - 5
    assert st["skewness"] == pytest.approx(np.mean(d ** 3) / 8)    # m3 / m2^1.5, m2 = 4
    assert st["kurtosis"] == pytest.approx(np.mean(d ** 4) / 16)   # Pearson: m4 / m2²
    assert (st["q1"], st["median"], st["q3"]) == (4, 4.5, 5.5)


def test_describe_normal_sample_and_edge_cases():
    z = np.random.default_rng(0).normal(10, 2, 200000)
    st = describe(z)
    assert st["skewness"] == pytest.approx(0, abs=0.02) and st["kurtosis"] == pytest.approx(3, abs=0.05)
    lognormal = describe(np.exp(z / 4))
    assert lognormal["skewness"] > 0.5 and lognormal["mean"] > lognormal["median"]   # right-skewed (grades)
    c = describe([3.0, 3.0, 3.0])
    assert c["std"] == 0 and c["skewness"] is None and c["kurtosis"] is None
    e = describe([np.nan])
    assert e["count"] == 0 and e["nulls"] == 1 and e["mean"] is None


def test_histogram_edges():
    e = histogram_edges(0.0, 10.0, count=4)
    assert np.allclose(e, [0, 2.5, 5, 7.5, 10])
    w = histogram_edges(0.3, 2.1, width=0.5)                       # aligned to multiples of the width
    assert np.allclose(w, [0, 0.5, 1.0, 1.5, 2.0, 2.5])
    w = histogram_edges(-3.0, 3.0, width=1.0)                      # the maximum lies on a limit: still covered
    assert w[0] == -3 and w[-1] >= 3 and len(w) == 7
    counts, _ = np.histogram([-3, 0, 3], w)
    assert counts.sum() == 3
    assert len(histogram_edges(5.0, 5.0, count=10)) == 2           # a single value
    with pytest.raises(ValueError):
        histogram_edges(0, 1e6, width=1e-3)                        # too many intervals
    with pytest.raises(ValueError):
        histogram_edges(0, 1, width=0)


def test_defaults_and_ecdf():
    assert sturges_bins(34) == 7 and sturges_bins(10 ** 6) == 21 and sturges_bins(1) == 5
    assert nice_width(7.0, 7) == 1 and nice_width(9.0, 4) == 2.5 and nice_width(0.9, 10) == pytest.approx(0.1)
    x, F = ecdf([3, 1, 2, np.nan])
    assert list(x) == [1, 2, 3] and np.allclose(F, [1 / 3, 2 / 3, 1])
    x, F = ecdf(np.arange(100000), max_points=500)
    assert len(x) <= 500 and F[-1] == 1 and np.all(np.diff(F) > 0)
