"""Performance benchmark of the methods on the Carlin demo (console report).

    ..\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark.py -s     # -s shows the report

The full report for every method / demo: python/tools/benchmark_methods.py.
"""
import os
import sys

import pytest

from conftest import CARLIN

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
need_data = pytest.mark.skipif(not os.path.isdir(CARLIN), reason="run utils.prepare_demo_data first")


@need_data
def test_carlin_benchmark_training_and_test_auc():
    import benchmark_methods as bm
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    rows, failures = bm.run_benchmark(["carlin"], test_pct=30, seed=42, split="spatial", out=out)
    assert failures == 0
    by = {(r["method"], r["set"]): r for r in rows}
    methods = {r["method"] for r in rows}
    assert {"Weights of Evidence", "Logistic Regression (all cells)", "Logistic Regression (sampled)"} <= methods
    for m in methods:
        tr, te = by[(m, "training")], by[(m, "test")]
        assert tr["deposits"] == by[("Weights of Evidence", "training")]["deposits"]   # same cells for every method
        assert te["deposits"] == by[("Weights of Evidence", "test")]["deposits"]
        assert tr["roc_auc"] > 0.80, (m, tr)
        assert te["roc_auc"] > 0.70, (m, te)
        assert tr["auc_ci_low"] < tr["roc_auc"] < tr["auc_ci_high"]
        assert 0 <= tr["tpr_at_fpr_0.10"] <= tr["tpr_at_fpr_0.50"] <= 1
    assert any("Summary (ROC AUC)" in s for s in lines)
