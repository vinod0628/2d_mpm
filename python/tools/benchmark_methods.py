"""Console benchmark of the MPM methods on the demo projects (training / test ROC, AUC, success rate, P-A).

Run from the project root::

    .venv\\Scripts\\python.exe python\\tools\\benchmark_methods.py                       # every method, both demos
    .venv\\Scripts\\python.exe python\\tools\\benchmark_methods.py --method wofe --dataset carlin
    .venv\\Scripts\\python.exe python\\tools\\benchmark_methods.py --test-pct 30 --split random --seed 7
    .venv\\Scripts\\python.exe python\\tools\\benchmark_methods.py --csv bench.csv --roc-dir roc_curves

Each method runs exactly as from its dialog (``method.run(params)`` in the same code path), with fixed evidence per
demo project (``CONFIGS`` below — add an entry when a new method is implemented). Cells with missing evidence are
excluded for every method, so all methods are scored on the same cells and deposits. For the training and the held-out
test deposits it prints: number of deposits / negatives, ROC AUC with the DeLong 95 % CI, success-rate AUC, the
prediction–area intersection and ROC values (TPR at fixed false-positive rates). ``--roc-dir`` writes every ROC curve
(FPR, TPR) as CSV. Exit code 0 = all runs succeeded.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_PYGEOS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)
DEMO = os.path.join(os.path.dirname(PY), "demo_data")

import numpy as np  # noqa: E402

FPR_POINTS = (0.05, 0.10, 0.20, 0.30, 0.50)
DATASETS = {
    "carlin": ("Carlin-type Au, Nevada", "nevada_carlin_au"),
    "curnamona": ("REE, Curnamona", "curnamona_ree"),
}


# ============================================================================== data
class Demo:
    """Rasters (with class labels) and deposits of one processed demo project."""

    def __init__(self, folder: str):
        import geopandas as gpd
        self.folder = folder
        self.deposits = gpd.read_file(os.path.join(folder, "vectors", "deposits.gpkg"))
        self._cache = {}

    def raster(self, name: str):
        if name not in self._cache:
            from utils.raster_tools import read_class_table, read_raster
            path = os.path.join(self.folder, "rasters", name + ".tif")
            data, grid, _ = read_raster(path)
            tab = read_class_table(path)
            labels = dict(zip(tab.value.astype(float), tab.label.astype(str))) if tab is not None and "label" in tab else {}
            self._cache[name] = (data, grid, labels)
        return self._cache[name]

    def evidence(self, name: str, **kw) -> dict:
        data, grid, labels = self.raster(name)
        return dict(name=name, data=data, grid=grid, labels=labels, **kw)

    def mask(self):
        data, grid, _ = self.raster("study_mask")
        return data, grid


def _wofe(layers):
    def make(d: Demo) -> dict:
        # missing="exclude" like the LR runs: every method is scored on the same cells and deposits
        return dict(evidence=[d.evidence(n, type=t, criterion="max_contrast", min_stud=1.96) for n, t in layers],
                    unit_cell_km2=1.0, missing="exclude", zero_deposit="zero", prefix="wofe")
    return make


def _lr(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k, transform=t) for n, k, t in layers], unit_cell_km2=1.0,
                 estimator="firth", selection="none", missing="exclude", prefix="lr")
        p.update(opts)
        return p
    return make


def _rf(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], n_trees=300, max_features="sqrt",
                 min_samples_leaf=1, max_depth=0, class_weight="none", min_spacing=-1, repeats=3, cv_folds=0,
                 missing="exclude", prefix="rf")
        p.update(opts)
        return p
    return make


def _svm(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], kernel="rbf", tune=True, C=1.0, gamma=0.0,
                 degree=3, coef0=0.0, class_weight="none", scaling="standard", min_spacing=-1, repeats=3,
                 cv_folds=5, missing="exclude", prefix="svm")
        p.update(opts)
        return p
    return make


def _xgb(layers, monotone=None, **opts):
    def make(d: Demo) -> dict:
        ev = [d.evidence(n, kind=k) for n, k, _ in layers]
        for e in ev:
            e["monotone"] = (monotone or {}).get(e["name"], 0)
        p = dict(evidence=ev, n_rounds=500, learning_rate=0.05, max_depth=3, min_child_weight=1.0, subsample=0.8,
                 colsample=0.8, reg_lambda=1.0, reg_alpha=0.0, min_split_loss=0.0, early_stopping=50,
                 class_weight="none", min_spacing=-1, repeats=3, cv_folds=5, missing="exclude", device="auto",
                 prefix="xgb")
        p.update(opts)
        return p
    return make


def _rvm(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], kernel="rbf", tune=True, gamma=0.0,
                 min_spacing=-1, repeats=3, cv_folds=5, missing="exclude", prefix="rvm")
        p.update(opts)
        return p
    return make


def _mlp(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], hidden=(32, 16), activation="relu",
                 learning_rate=0.001, alpha=1e-4, batch_size=32, max_epochs=500, patience=20, networks=5,
                 min_spacing=-1, repeats=3, cv_folds=5, missing="exclude", prefix="mlp")
        p.update(opts)
        return p
    return make


def _elm(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], hidden=100, activation="sigmoid", C=1.0,
                 tune=True, networks=10, min_spacing=-1, repeats=3, cv_folds=5, missing="exclude", prefix="elm")
        p.update(opts)
        return p
    return make


def _cnn(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], patch=9, filters=(8, 16), dense=16,
                 dropout=0.5, learning_rate=3e-4, batch_size=32, max_epochs=200, patience=20, augment=True,
                 networks=1, min_spacing=-1, repeats=2, cv_folds=3, missing="exclude", prefix="cnn")
        p.update(opts)
        return p
    return make


def _bn(layers, **opts):
    def make(d: Demo) -> dict:
        p = dict(evidence=[d.evidence(n, kind=k) for n, k, _ in layers], structure="tan", n_classes=5, alpha=1.0,
                 design="all", missing="class", draws=20, unit_cell_km2=1.0, prefix="bn")
        p.update(opts)
        return p
    return make


def _hybrid(table, prefix, **opts):
    """Hybrid methods: knowledge table [(layer, process, direction or 'cls=score; …')]."""
    def make(d: Demo) -> dict:
        ev = []
        for n, proc, dirn in table:
            cat = isinstance(dirn, str)
            ev.append(d.evidence(n, kind="categorical" if cat else "continuous", process=proc,
                                 direction=1 if cat else dirn, scores=dirn if cat else None, weight=1.0))
        p = dict(evidence=ev, process_weights={}, prefix=prefix)
        p.update(opts)
        return p
    return make


CARLIN_KNOW = [("geology_host", "trap", "2=0.9; 1=0.1"), ("dist_faults", "pathway", -1),
               ("as_log_idw", "deposition", 1), ("sb_log_idw", "deposition", 1)]
CURNAMONA_KNOW = [("rad_th", "source", 1), ("rad_u", "source", 1), ("geochem_nb_log_idw", "deposition", 1),
                  ("aster_silica_index", "footprint", -1)]

CARLIN_LR = [("geology_host", "categorical", "none"), ("dist_faults", "continuous", "none"),
             ("as_log_idw", "continuous", "none"), ("sb_log_idw", "continuous", "none")]
CURNAMONA_LR = [("geology_units", "categorical", "none"), ("dist_faults", "continuous", "none"),
                ("dist_granite", "continuous", "none"), ("geochem_la_log_idw", "continuous", "none")]

# dataset -> {variant label: (method key, params builder)}
CONFIGS = {
    "carlin": {
        "Weights of Evidence": ("wofe", _wofe([("geology_host", "categorical"), ("dist_faults_class", "ascending"),
                                               ("as_q8", "descending"), ("sb_q8", "descending")])),
        "Logistic Regression (all cells)": ("logistic_regression", _lr(CARLIN_LR, design="all")),
        "Logistic Regression (sampled)": ("logistic_regression",
                                          _lr(CARLIN_LR, design="sampled", neg_ratio=5, min_distance=1000.0,
                                              min_spacing=-1, repeats=5, cv_folds=-1)),
        "Random Forest": ("random_forest", _rf(CARLIN_LR, neg_ratio=5, min_distance=1000.0)),
        "Support Vector Machine": ("svm", _svm(CARLIN_LR, neg_ratio=5, min_distance=1000.0)),
        "XGBoost": ("xgboost", _xgb(CARLIN_LR, {"dist_faults": -1}, neg_ratio=5, min_distance=1000.0)),
        "Relevance Vector Machine": ("rvm", _rvm(CARLIN_LR, neg_ratio=3, min_distance=1000.0)),
        "Neural Network (MLP)": ("mlp", _mlp(CARLIN_LR, neg_ratio=3, min_distance=1000.0)),
        "Extreme Learning Machine": ("elm", _elm(CARLIN_LR, neg_ratio=3, min_distance=1000.0)),
        "Convolutional Neural Network": ("cnn", _cnn(CARLIN_LR, neg_ratio=3, min_distance=1000.0)),
        "Bayesian Network (TAN)": ("bayesian_network", _bn(CARLIN_LR)),
        "Fuzzy WofE (modified)": ("fuzzy_wofe", _hybrid(CARLIN_KNOW, "fwofe", model="modified", n_classes=10)),
    },
    "curnamona": {
        "Weights of Evidence": ("wofe", _wofe([("geology_units", "categorical"), ("dist_faults_class", "ascending"),
                                               ("geochem_la_q8", "descending")])),
        "Logistic Regression (all cells)": ("logistic_regression", _lr(CURNAMONA_LR, design="all")),
        "Logistic Regression (sampled)": ("logistic_regression",
                                          _lr(CURNAMONA_LR, design="sampled", neg_ratio=10, min_distance=800.0,
                                              min_spacing=-1, repeats=5, cv_folds=-1)),
        "Random Forest": ("random_forest", _rf(CURNAMONA_LR, neg_ratio=10, min_distance=800.0)),
        "Support Vector Machine": ("svm", _svm(CURNAMONA_LR, neg_ratio=10, min_distance=800.0)),
        "XGBoost": ("xgboost", _xgb(CURNAMONA_LR, {"dist_faults": -1, "dist_granite": -1}, neg_ratio=10,
                                    min_distance=800.0)),
        "Relevance Vector Machine": ("rvm", _rvm(CURNAMONA_LR, neg_ratio=5, min_distance=800.0)),
        "Neural Network (MLP)": ("mlp", _mlp(CURNAMONA_LR, neg_ratio=5, min_distance=800.0)),
        "Extreme Learning Machine": ("elm", _elm(CURNAMONA_LR, neg_ratio=5, min_distance=800.0)),
        "Convolutional Neural Network": ("cnn", _cnn(CURNAMONA_LR, neg_ratio=5, min_distance=800.0)),
        "Bayesian Network (TAN)": ("bayesian_network", _bn(CURNAMONA_LR, n_classes=3)),
        "Fuzzy WofE (classic)": ("fuzzy_wofe", _hybrid(CURNAMONA_KNOW, "fwofe", model="classic", n_classes=10)),
    },
}


# ============================================================================== reporting
def tpr_at(curve, fpr: float) -> float:
    """ROC operating point: highest TPR reachable with a false-positive rate ≤ fpr."""
    ok = curve.fpr <= fpr + 1e-12
    return float(curve.tpr[ok].max()) if ok.any() else 0.0


def _fmt_row(name, c):
    pa = c.pa_intersection
    pa_txt = f"{pa['prediction']:5.1f} / {pa['area']:5.1f}" if pa else "    —      "
    roc = " ".join(f"{tpr_at(c, f):4.2f}" for f in FPR_POINTS)
    return (f"  {name:<9}{c.n_pos:>9}{c.n_neg:>12,}   {c.auc:6.3f}  {c.auc_ci[0]:5.3f}–{c.auc_ci[1]:5.3f}"
            f"   {c.sr_auc:6.3f}   {pa_txt}    {roc}")


HEADER = ("  set       deposits   negatives   ROC AUC  95% CI (DeLong)  SR-AUC   P-A dep% / area%   TPR at FPR "
          + " ".join(f"{f:.2f}" for f in FPR_POINTS))


def run_benchmark(datasets=None, methods=None, test_pct: float = 30.0, seed: int = 42, split: str = "spatial",
                  roc_dir: str = None, out=print):
    """Run the configured methods; returns one summary dict per (dataset, variant, set)."""
    from core.methods import REGISTRY, load_builtin_methods
    load_builtin_methods()
    rows, failures = [], 0
    for ds in datasets or list(DATASETS):
        title, sub = DATASETS[ds]
        folder = os.path.join(DEMO, sub, "processed")
        if not os.path.isdir(folder):
            out(f"\n=== {title}: demo data not found ({folder}) — run utils.prepare_demo_data")
            continue
        demo = Demo(folder)
        out(f"\n=== {title} ({sub}) — {len(demo.deposits)} deposit points, hold-out {test_pct:g} % "
            f"({split} split, seed {seed})")
        configured = {REGISTRY[k].key for k, _ in CONFIGS.get(ds, {}).values() if k in REGISTRY}
        for key in REGISTRY:
            if (methods is None or key in methods) and key not in configured:
                out(f"--- {REGISTRY[key].name}: no benchmark configuration (add one to CONFIGS in {__file__})")
        for label, (key, make) in CONFIGS.get(ds, {}).items():
            if methods is not None and key not in methods:
                continue
            if key not in REGISTRY:
                out(f"--- {label}: method '{key}' is not registered")
                continue
            params = make(demo)
            params.update(deposits=demo.deposits, deposits_name="deposits", mask=demo.mask(),
                          test_fraction=test_pct / 100.0, seed=seed, split_method=split, auto_align=True,
                          out_std=False, out_conf=False, out_t=False)
            t0 = time.time()
            try:
                result = REGISTRY[key].run(params, lambda p, m: None)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                out(f"--- {label}: FAILED — {type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            ev = result.evaluation
            out(f"--- {label}  [{dt:.1f} s]  layers: {', '.join(e['name'] for e in params['evidence'])}")
            out(HEADER)
            for name, c in ev.curves.items():
                out(_fmt_row(name, c))
                rows.append(dict(dataset=ds, method=label, set=name, deposits=c.n_pos, negatives=c.n_neg,
                                 roc_auc=round(c.auc, 4), auc_ci_low=round(c.auc_ci[0], 4),
                                 auc_ci_high=round(c.auc_ci[1], 4), success_rate_auc=round(c.sr_auc, 4),
                                 pa_prediction=round(c.pa_intersection.get("prediction", np.nan), 2),
                                 pa_area=round(c.pa_intersection.get("area", np.nan), 2),
                                 **{f"tpr_at_fpr_{f:.2f}": round(tpr_at(c, f), 4) for f in FPR_POINTS},
                                 seconds=round(dt, 1)))
                if roc_dir:
                    os.makedirs(roc_dir, exist_ok=True)
                    fn = os.path.join(roc_dir, f"{ds}_{key}_{label.split('(')[-1].strip(')').replace(' ', '_')}"
                                               f"_{name}.csv".replace("Logistic_Regression", "lr"))
                    np.savetxt(fn, np.c_[c.fpr, c.tpr], delimiter=",", header="fpr,tpr", comments="", fmt="%.6f")
            if "test" in ev.curves and ev.curves["test"].n_pos < 10:
                out(f"  note: only {ev.curves['test'].n_pos} test deposits — the test AUC is very uncertain")
            for m in result.messages:
                out(f"  ⚠ {m}")
    if rows:
        out("\n=== Summary (ROC AUC)")
        out(f"  {'dataset':<11}{'method':<34}{'training':>10}{'test':>10}")
        seen = []
        for r in rows:
            k = (r["dataset"], r["method"])
            if k in seen:
                continue
            seen.append(k)
            tr = next((x["roc_auc"] for x in rows if (x["dataset"], x["method"]) == k and x["set"] == "training"), None)
            te = next((x["roc_auc"] for x in rows if (x["dataset"], x["method"]) == k and x["set"] == "test"), None)
            out(f"  {k[0]:<11}{k[1]:<34}{'—' if tr is None else f'{tr:.3f}':>10}{'—' if te is None else f'{te:.3f}':>10}")
    return rows, failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=list(DATASETS), action="append", help="demo project (repeatable)")
    ap.add_argument("--method", action="append", help="method key, e.g. wofe, logistic_regression (repeatable)")
    ap.add_argument("--test-pct", type=float, default=30.0, help="deposits held out for testing (default 30)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["spatial", "random"], default="spatial")
    ap.add_argument("--csv", help="write the summary table to this CSV file")
    ap.add_argument("--roc-dir", help="write every ROC curve (fpr,tpr) as CSV into this folder")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    rows, failures = run_benchmark(a.dataset, a.method, a.test_pct, a.seed, a.split, a.roc_dir)
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsummary written to {a.csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
