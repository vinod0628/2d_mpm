"""Demo-data experiments for the Python method documents (docs/python/*.docx).

Runs every method exactly as the GUI does (``REGISTRY[key].run(params)``) on the two demo projects and writes
``results/<method>_<dataset>.json`` and ``figures/*.png`` that the document sources (content/*.js) read.

Protocol
  Carlin (34 deposit cells): candidate settings are compared by the mean test ROC AUC over 5 spatially stratified
  70/30 splits (seeds 100–104); the selected setting is re-evaluated on 10 independent splits (seeds 0–9,
  mean ± s.d.) and run once with seed 42 for the figures and tables.
  Curnamona (7 deposits): leave-one-deposit-out (LODO) — the model is trained on 6 deposits and the held-out deposit
  is ranked against all background cells (AUC of one positive = its percentile); candidates are compared by the
  mean LODO AUC; the reference run uses all 7 deposits.

    .venv\\Scripts\\python.exe docs\\python\\_build\\experiments.py                 # everything (~30 min)
    .venv\\Scripts\\python.exe docs\\python\\_build\\experiments.py --method wofe --dataset carlin --quick
"""
from __future__ import annotations

import argparse

AUC_TIE = 0.005      # candidate selection: differences below this are treated as ties
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_PYGEOS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = os.path.join(ROOT, "python")
sys.path.insert(0, PY)
DEMO = os.path.join(ROOT, "demo_data")
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import geopandas as gpd  # noqa: E402

from core.methods import REGISTRY, load_builtin_methods  # noqa: E402
from evaluation.metrics import delong_auc  # noqa: E402
from utils.raster_tools import points_to_cells, read_class_table, read_raster  # noqa: E402

load_builtin_methods()

DATASETS = {
    "carlin": dict(folder="nevada_carlin_au", title="Carlin-type Au, north-central Nevada (USA)", protocol="split"),
    "curnamona": dict(folder="curnamona_ree", title="REE, northern Curnamona Province (South Australia)",
                      protocol="lodo"),
}
EVIDENCE = {
    "carlin": dict(
        wofe=[("geology_host", "categorical"), ("dist_faults_class", "ascending"), ("as_q8", "descending"),
              ("sb_q8", "descending")],
        cont=[("geology_host", "categorical"), ("dist_faults", "continuous"), ("as_log_idw", "continuous"),
              ("sb_log_idw", "continuous")],
        ebf=[("geology_host", "values"), ("dist_faults_class", "values"), ("as_log_idw", "quantile"),
             ("sb_log_idw", "quantile")]),
    # Curnamona REE: chosen from the single-layer AUCs (univariate.py) — radiometric U / Th / K highs, Nb, low silica;
    # "<layer>@q8" = the layer reclassified into 8 quantile classes (Data ▸ Reclassify raster), as WofE needs classes
    "curnamona": dict(
        wofe=[("rad_th@q8", "descending"), ("rad_u@q8", "descending"), ("geochem_nb_log_idw@q8", "descending"),
              ("aster_silica_index@q8", "ascending")],
        cont=[("rad_th", "continuous"), ("rad_u", "continuous"), ("rad_k", "continuous"),
              ("geochem_nb_log_idw", "continuous"), ("aster_silica_index", "continuous")],
        ebf=[("rad_th", "quantile"), ("rad_u", "quantile"), ("geochem_nb_log_idw", "quantile"),
             ("aster_silica_index", "quantile")]),
}


# ============================================================================== data
class Demo:
    def __init__(self, key):
        self.key = key
        self.folder = os.path.join(DEMO, DATASETS[key]["folder"], "processed")
        self.deposits = gpd.read_file(os.path.join(self.folder, "vectors", "deposits.gpkg"))
        self._cache = {}
        self.layers = pd.read_csv(os.path.join(self.folder, "layers.csv"))

    def raster(self, name):
        if name not in self._cache and name.endswith("@q8"):           # 8 quantile classes (Reclassify raster)
            from utils.reclass import compute_breaks, reclass_by_breaks
            data, grid, _ = self.raster(name[:-3])
            breaks = compute_breaks(np.ma.compressed(data), "quantile", 8)
            cls, table = reclass_by_breaks(data, breaks)
            cls = np.ma.array(np.ma.filled(cls, 0).astype(np.uint8), mask=np.ma.getmaskarray(cls))
            labels = dict(zip(table.value.astype(float), table.label.astype(str))) if "label" in table else {}
            self._cache[name] = (cls, grid, labels)
        if name not in self._cache:
            path = os.path.join(self.folder, "rasters", name + ".tif")
            data, grid, _ = read_raster(path)
            tab = read_class_table(path)
            labels = dict(zip(tab.value.astype(float), tab.label.astype(str))) if tab is not None and "label" in tab else {}
            self._cache[name] = (data, grid, labels)
        return self._cache[name]

    def mask(self):
        data, grid, _ = self.raster("study_mask")
        return data, grid

    def describe(self):
        m, g = self.mask()
        valid = int((~np.ma.getmaskarray(m)).sum())
        rows, cols = points_to_cells(self.deposits, g, valid_mask=~np.ma.getmaskarray(m))
        crs = g.crs.to_epsg() if g.crs is not None else None
        return dict(title=DATASETS[self.key]["title"], crs=f"EPSG:{crs}" if crs else (g.crs.to_wkt()[:80] if g.crs else ""),
                    cell_m=float(g.res[0]), width=int(g.width), height=int(g.height), study_cells=valid,
                    study_km2=valid * g.cell_area / 1e6, deposit_points=int(len(self.deposits)),
                    deposit_cells=int(len(rows)))

    def layer_info(self, names):
        out = []
        for n, kind in names:
            row = self.layers[self.layers.name == n.replace("@q8", "")]
            data, grid, labels = self.raster(n)
            v = np.ma.compressed(data)
            desc = str(row.description.iloc[0]) if len(row) else ""
            if n.endswith("@q8"):
                desc += " — reclassified into 8 quantile classes"
            out.append(dict(name=n, kind=kind, description=desc,
                            n_values=int(np.unique(v[:: max(1, v.size // 200000)]).size),
                            min=float(v.min()), max=float(v.max()), dtype=str(data.dtype)))
        return out


# ============================================================================== params (as the dialogs build them)
def base_params(d: Demo, test_fraction, seed, deposits=None):
    return dict(deposits=deposits if deposits is not None else d.deposits, deposits_name="deposits", mask=d.mask(),
                test_fraction=test_fraction, seed=seed, split_method="spatial", thin_distance=0.0, negatives=None,
                auto_align=True, unit_cell_km2=1.0)


def wofe_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], type=t,
               criterion=cfg["criterion"], min_stud=cfg["min_stud"]) for n, t in EVIDENCE[d.key]["wofe"]]
    return dict(evidence=ev, missing=cfg.get("missing", "zero"), zero_deposit=cfg["zero_deposit"], out_std=True,
                out_conf=True, out_weights=False, prefix="wofe")


def ebf_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], classing=c,
               n_classes=cfg.get("n_classes", 10)) for n, c in EVIDENCE[d.key]["ebf"]]
    return dict(evidence=ev, missing=cfg.get("missing", "vacuous"), zero_deposit=cfg.get("zero_deposit", "zero"),
                score=cfg.get("score", "bel"), out_maps=True, out_layers=False, prefix="ebf")


def lr_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k, transform="none")
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, design="all", estimator="firth", ridge=1.0, selection="none", alpha=0.10, missing="exclude",
             neg_ratio=5.0, min_distance=4 * d.mask()[1].res[0], min_spacing=-1.0, repeats=5, prior_correction=True,
             cv_folds=0, out_std=True, out_t=True, out_logit=False, prefix="lr")
    p.update(cfg)
    return p


def rf_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, n_trees=300, max_features="sqrt", min_samples_leaf=1, max_depth=0, class_weight="none",
             neg_ratio=1.0, min_distance=4 * d.mask()[1].res[0], min_spacing=-1.0, repeats=3, cv_folds=0,
             missing="exclude", out_std=True, prefix="rf")
    p.update(cfg)
    return p


def svm_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, kernel="rbf", tune=True, C=1.0, gamma=0.0, degree=3, coef0=0.0, class_weight="none",
             scaling="standard", neg_ratio=1.0, min_distance=4 * d.mask()[1].res[0], min_spacing=-1.0, repeats=3,
             cv_folds=5, missing="exclude", out_std=True, out_decision=True, prefix="svm")
    p.update(cfg)
    return p


MONO = {"carlin": {"dist_faults": -1, "as_log_idw": 1, "sb_log_idw": 1},          # geological expectation
        "curnamona": {"rad_th": 1, "rad_u": 1, "geochem_nb_log_idw": 1, "aster_silica_index": -1}}


def xgb_params(d, cfg):
    cfg = dict(cfg)
    mono = MONO[d.key] if cfg.pop("mono", False) else {}
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k,
               monotone=mono.get(n, 0)) for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, n_rounds=500, learning_rate=0.05, max_depth=3, min_child_weight=1.0, subsample=0.8,
             colsample=0.8, reg_lambda=1.0, reg_alpha=0.0, min_split_loss=0.0, early_stopping=50, class_weight="none",
             neg_ratio=1.0, min_distance=4 * d.mask()[1].res[0], min_spacing=-1.0, repeats=3, cv_folds=5,
             missing="exclude", device="auto", out_std=True, out_shap=False, prefix="xgb")
    p.update(cfg)
    return p


def rvm_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, kernel="rbf", tune=True, gamma=0.0, neg_ratio=1.0, min_distance=4 * d.mask()[1].res[0],
             min_spacing=-1.0, repeats=3, cv_folds=5, missing="exclude", out_std=True, out_pred_sd=True, prefix="rvm")
    p.update(cfg)
    return p


def mlp_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, hidden=(32, 16), activation="relu", learning_rate=0.001, alpha=1e-4, batch_size=32,
             max_epochs=500, patience=20, networks=5, neg_ratio=1.0, min_distance=4 * d.mask()[1].res[0],
             min_spacing=-1.0, repeats=3, cv_folds=5, missing="exclude", out_std=True, prefix="mlp")
    p.update(cfg)
    return p


def elm_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, hidden=100, activation="sigmoid", C=1.0, tune=True, networks=10, neg_ratio=1.0,
             min_distance=4 * d.mask()[1].res[0], min_spacing=-1.0, repeats=3, cv_folds=5, missing="exclude",
             out_std=True, prefix="elm")
    p.update(cfg)
    return p


def cnn_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, patch=9, filters=(8, 16), dense=16, dropout=0.5, learning_rate=3e-4, batch_size=32,
             max_epochs=200, patience=20, augment=True, networks=1, neg_ratio=3.0, min_distance=4 * d.mask()[1].res[0],
             min_spacing=-1.0, repeats=2, cv_folds=3, missing="exclude", out_std=True, prefix="cnn")
    p.update(cfg)
    return p


def bn_params(d, cfg):
    ev = [dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], labels=d.raster(n)[2], kind=k)
          for n, k in EVIDENCE[d.key]["cont"]]
    p = dict(evidence=ev, structure="tan", n_classes=5, alpha=1.0, design="all", missing="class", draws=30,
             unit_cell_km2=1.0, neg_ratio=5.0, min_distance=4 * d.mask()[1].res[0], out_unit=True, out_sd=True,
             prefix="bn")
    p.update(cfg)
    return p


# ------------------------------------------------------------------ knowledge-driven methods (no training)
KNOWLEDGE = ("fuzzy_logic", "boolean_logic", "logistic_geometric", "index_overlay", "topsis")
# deposit-model layers: (layer, mineral-system group)
KD_LAYERS = {
    "carlin": [("geology_host", "trap"), ("dist_faults", "pathway"), ("as_log_idw", "footprint"),
               ("sb_log_idw", "footprint")],
    "curnamona": [("rad_th", "source"), ("rad_u", "source"), ("geochem_nb_log_idw", "enrichment"),
                  ("aster_silica_index", "exposure")],
}
KD_NEGATIVE = {"dist_faults", "aster_silica_index"}                 # low values favourable


def _vals(d, n):
    return np.ma.compressed(d.raster(n)[0][d.mask()[0].mask == False]).astype(float)   # noqa: E712


def fuzzy_params(d, cfg):
    from logic.knowledge_driven.fuzzy_logic import default_params
    ev = []
    for n, g in KD_LAYERS[d.key]:
        v = _vals(d, n)
        if n == "geology_host":
            f, p = "classes", "2=0.9; 1=0.1"
        elif n == "dist_faults":
            f, p = "z_shape", cfg.get("faults", "500, 5000")
        elif n in KD_NEGATIVE:
            lo, hi = np.percentile(v, [2, 98])
            f, p = "logistic", f"{(lo + hi) / 2:.6g}, {-9.2 / (hi - lo):.6g}"
        else:
            f, p = "logistic", default_params("logistic", v)
        ev.append(dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], function=f, params=p, group=g))
    groups = sorted({g for _, g in KD_LAYERS[d.key]})
    return dict(evidence=ev, group_ops={g: (cfg.get("group_op", "or"), 0.9) for g in groups},
                final_op=(cfg.get("final_op", "gamma"), cfg.get("gamma", 0.9)), missing="exclude",
                out_groups=True, out_memberships=False, prefix="fuzzy")


def boolean_params(d, cfg):
    q = cfg.get("q", 75)
    ev = []
    for n, g in KD_LAYERS[d.key]:
        v = _vals(d, n)
        if n == "geology_host":
            c, val = "in", "2"
        elif n == "dist_faults":
            c, val = "le", str(cfg.get("faults_m", 3000))
        elif n in KD_NEGATIVE:
            c, val = "le", f"{np.percentile(v, 100 - q):.6g}"
        else:
            c, val = "ge", f"{np.percentile(v, q):.6g}"
        ev.append(dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], cond=c, values=val, group=g))
    groups = sorted({g for _, g in KD_LAYERS[d.key]})
    return dict(evidence=ev, group_ops={g: cfg.get("group_op", "or") for g in groups},
                final_op=cfg.get("final_op", "and"), out_conditions=False, prefix="boolean")


def lg_params(d, cfg):
    ev = []
    for n, g in KD_LAYERS[d.key]:
        if n in cfg.get("drop", ()):
            continue
        v = _vals(d, n)
        if n == "geology_host":
            dirn, spec = "classes", "2=0.9; 1=0.1"
        else:
            lo, hi = (np.percentile(v, [2, 98]) if cfg.get("robust") else (v.min(), v.max()))
            dirn, spec = ("negative" if n in KD_NEGATIVE else "positive"), f"{lo:.6g}, {hi:.6g}"
        ev.append(dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], direction=dirn, spec=spec, weight=1.0))
    return dict(evidence=ev, weighting=cfg.get("weighting", "equal"), out_fuzzy=False, prefix="lgfa")


# index overlay: AHP judgements (upper triangle, KD_LAYERS order) from the importance of each mineral-system proxy
AHP_JUDGEMENTS = {
    # host (trap) ≥ Sb > As > faults: reactive host and the Sb halo are the strongest Carlin criteria
    "carlin": {(0, 1): 3, (0, 2): 2, (0, 3): 1, (1, 2): 1 / 2, (1, 3): 1 / 3, (2, 3): 1 / 2},
    # Th = U (U–Th–REE granite source) > Nb (co-enrichment) > silica (cover)
    "curnamona": {(0, 1): 1, (0, 2): 2, (0, 3): 3, (1, 2): 2, (1, 3): 3, (2, 3): 2},
}


def ahp_info(key):
    """AHP and fuzzy-AHP weights of the demo judgement matrix (for the document)."""
    from logic.knowledge_driven.index_overlay.model import ahp_weights, fuzzy_ahp_weights, reciprocal_matrix
    names = [n for n, _ in KD_LAYERS[key]]
    A = reciprocal_matrix(AHP_JUDGEMENTS[key], len(names))
    a, f1, f2 = ahp_weights(A), fuzzy_ahp_weights(A, 1.0), fuzzy_ahp_weights(A, 2.0)
    return dict(layers=names, matrix=A.tolist(), ahp=a.weights.tolist(), lambda_max=a.lambda_max, ci=a.ci, cr=a.cr,
                fuzzy1=f1.weights.tolist(), fuzzy1_tfn=f1.fuzzy.tolist(), fuzzy2=f2.weights.tolist())


def io_params(d, cfg):
    from logic.knowledge_driven.index_overlay.model import reciprocal_matrix
    ev = []
    for n, g in KD_LAYERS[d.key]:
        v = _vals(d, n)
        if n == "geology_host":
            rule, spec = "classes", "2=1; 1=0"
        elif cfg.get("binary"):
            q = np.percentile(v, 25 if n in KD_NEGATIVE else 75)
            rule, spec = ("le" if n in KD_NEGATIVE else "ge"), f"{q:.6g}"
        else:
            lo, hi = (np.percentile(v, [2, 98]) if cfg.get("robust", True) else (v.min(), v.max()))
            rule, spec = ("negative" if n in KD_NEGATIVE else "positive"), f"{lo:.6g}, {hi:.6g}"
        ev.append(dict(name=n, data=d.raster(n)[0], grid=d.raster(n)[1], rule=rule, spec=spec, weight=1.0))
    w = cfg.get("weighting", "equal")
    P = reciprocal_matrix(AHP_JUDGEMENTS[d.key], len(ev)).tolist() if w in ("ahp", "fuzzy_ahp") else None
    return dict(evidence=ev, weighting=w, pairwise=P, delta=cfg.get("delta", 1.0),
                missing=cfg.get("missing", "renormalise"), out_scores=False, prefix="index")


def topsis_params(d, cfg):
    p = io_params(d, cfg)
    method = cfg.get("method", "topsis")
    p.update(method=method, normalisation=cfg.get("normalisation", "vector"), v=cfg.get("v", 0.5),
             out_parts=False, prefix=method)
    return p


def topsis_vector_weights(key):
    """Effective layer weights of TOPSIS vector normalisation, w_j / sqrt(sum x_j^2), normalised (for the document)."""
    from logic.knowledge_driven.index_overlay.model import IOLayer, layer_scores
    from logic.knowledge_driven.index_overlay.plugin import parse_spec
    d = Demo(key)
    study = ~np.ma.getmaskarray(d.mask()[0])
    out = []
    for label, cfg in (("equal weights, 2–98 % ranges", {}), ("equal weights, min–max ranges", dict(robust=False)),
                       ("AHP weights, min–max ranges", dict(robust=False, weighting="ahp"))):
        p = io_params(d, cfg)
        w = (np.array(ahp_info(key)["ahp"]) if cfg.get("weighting") == "ahp" else np.ones(len(p["evidence"])))
        w = w / w.sum()
        norms = []
        for e in p["evidence"]:
            s = layer_scores(IOLayer(e["name"], e["data"], e["rule"], parse_spec(e["rule"], e["spec"])))
            v = np.ma.getdata(s)[study & ~np.ma.getmaskarray(s)]
            norms.append(np.sqrt((v ** 2).sum()))
        eff = w / np.array(norms)
        out.append(dict(label=label, layers=[e["name"] for e in p["evidence"]], weights=w.tolist(),
                        effective=(eff / eff.sum()).tolist()))
    return out


# hybrid methods: mineral-system knowledge table per project: (layer, process, direction or class scores, weight)
HYBRID_KNOWLEDGE = {
    "carlin": [("geology_host", "trap", "2=0.9; 1=0.1", 1.0), ("dist_faults", "pathway", -1, 1.0),
               ("as_log_idw", "deposition", 1, 1.0), ("sb_log_idw", "deposition", 1, 1.0)],
    "curnamona": [("rad_th", "source", 1, 1.0), ("rad_u", "source", 1, 1.0),
                  ("geochem_nb_log_idw", "deposition", 1, 1.0), ("aster_silica_index", "footprint", -1, 1.0)],
}


def hybrid_evidence(d):
    ev = []
    for n, proc, dirn, w in HYBRID_KNOWLEDGE[d.key]:
        data, grid, labels = d.raster(n)
        cat = isinstance(dirn, str)
        ev.append(dict(name=n, data=data, grid=grid, labels=labels, kind="categorical" if cat else "continuous",
                       process=proc, direction=1 if cat else dirn, scores=dirn if cat else None, weight=w))
    return ev


def fwofe_params(d, cfg):
    return dict(evidence=hybrid_evidence(d), process_weights={}, model=cfg.get("model", "modified"),
                membership=cfg.get("membership", "expert"), pattern=cfg.get("pattern", "max_contrast"),
                n_classes=cfg.get("n_classes", 10), nonnegative=cfg.get("nonnegative", True), min_distance=3 * d.raster(HYBRID_KNOWLEDGE[d.key][0][0])[1].res[0],
                out_processes=False, prefix="fwofe")



def nf_params(d, cfg):
    return dict(evidence=hybrid_evidence(d), process_weights={}, inputs=cfg.get("inputs", "auto"),
                n_mf=cfg.get("n_mf", 2), consequent=cfg.get("consequent", "linear"), kappa=cfg.get("kappa", 1.0),
                epochs=cfg.get("epochs", 100), lr=cfg.get("lr", 0.02), check_pct=cfg.get("check_pct", 25.0),
                neg_ratio=cfg.get("neg_ratio", 3.0), n_bags=cfg.get("n_bags", 10),
                min_distance=3 * d.raster(HYBRID_KNOWLEDGE[d.key][0][0])[1].res[0], out_processes=False,
                prefix="neurofuzzy")



def dcf_params(d, cfg):
    return dict(evidence=hybrid_evidence(d), process_weights={}, calibration=cfg.get("calibration", "wofe"),
                scaling=cfg.get("scaling", "weight"), monotone=cfg.get("monotone", True),
                n_classes=cfg.get("n_classes", 10), kappa=cfg.get("kappa", 0.0),
                combination=cfg.get("combination", "network"), gamma=cfg.get("gamma", 0.9),
                min_distance=3 * d.raster(HYBRID_KNOWLEDGE[d.key][0][0])[1].res[0], out_processes=False,
                out_memberships=False, prefix="dcfuzzy")


CANDIDATES = {
    "ebf": [
        dict(label="10 quantile classes, Bel = 0 without deposits (original EBF), score Bel (default)"),
        dict(label="10 quantile classes, 0.5-deposit correction, score Bel", zero_deposit="haldane"),
        dict(label="5 quantile classes, Bel = 0 without deposits", n_classes=5),
        dict(label="5 quantile classes, 0.5-deposit correction", n_classes=5, zero_deposit="haldane"),
        dict(label="8 quantile classes, 0.5-deposit correction", n_classes=8, zero_deposit="haldane"),
        dict(label="20 quantile classes, 0.5-deposit correction", n_classes=20, zero_deposit="haldane"),
        dict(label="10 quantile classes, 0.5-deposit correction, score Pls", zero_deposit="haldane", score="pls"),
        dict(label="10 quantile classes, 0.5-deposit correction, score (Bel + Pls) / 2", zero_deposit="haldane",
             score="mid"),
    ],
    "dc_fuzzy": [
        dict(label="WofE, μ = σ(W), monotone, 10 classes, inference network (default)"),
        dict(label="WofE, stretched 0.01–0.99, monotone, 10 classes", scaling="stretch"),
        dict(label="WofE, μ = σ(W), not monotone, 10 classes", monotone=False),
        dict(label="WofE, μ = σ(W), monotone, 5 classes", n_classes=5),
        dict(label="WofE, μ = σ(W), monotone, 20 classes", n_classes=20),
        dict(label="LR, μ = σ(W), monotone", calibration="lr"),
        dict(label="LR, stretched 0.01–0.99, monotone", calibration="lr", scaling="stretch"),
        dict(label="WofE, μ = σ(W), monotone, pooled with the expert (κ = 20)", kappa=20.0),
        dict(label="WofE, μ = σ(W), monotone, fuzzy γ = 0.9", combination="gamma", gamma=0.9),
        dict(label="LR, μ = σ(W), monotone, fuzzy γ = 0.9", calibration="lr", combination="gamma", gamma=0.9),
    ],
    "neuro_fuzzy": [
        dict(label="one input per layer, 2 MFs, first order, κ = 1, checking 25 % (default)"),
        dict(label="one input per layer, 2 MFs, zero order (constant rules), κ = 1", consequent="constant"),
        dict(label="one input per layer, 2 MFs, first order, κ = 0 (data only)", kappa=0.0),
        dict(label="one input per layer, 2 MFs, first order, κ = 10 (strong expert prior)", kappa=10.0),
        dict(label="one input per layer, expert MFs kept (learning rate 0), κ = 1", lr=0.0),
        dict(label="one input per layer, no checking set (100 epochs)", check_pct=0.0),
        dict(label="one input per layer, 3 MFs (81 rules), first order, κ = 1", inputs="layers", n_mf=3),
        dict(label="one input per process, 2 MFs, first order, κ = 1", inputs="processes"),
        dict(label="one input per process, 3 MFs, first order, κ = 1", inputs="processes", n_mf=3),
    ],
    "topsis": [
        dict(label="TOPSIS, vector normalisation, equal weights, 2–98 % ranges (default)"),
        dict(label="TOPSIS, no normalisation, equal weights", normalisation="linear"),
        dict(label="TOPSIS, vector normalisation, AHP weights", weighting="ahp"),
        dict(label="TOPSIS, vector normalisation, fuzzy AHP weights (δ = 1)", weighting="fuzzy_ahp", delta=1.0),
        dict(label="TOPSIS, vector normalisation, entropy weights", weighting="entropy"),
        dict(label="TOPSIS, vector normalisation, equal weights, min–max ranges", robust=False),
        dict(label="TOPSIS, vector normalisation, AHP weights, min–max ranges", weighting="ahp", robust=False),
        dict(label="TOPSIS, vector normalisation, entropy weights, min–max ranges", weighting="entropy",
             robust=False),
        dict(label="VIKOR, v = 0.5, equal weights", method="vikor", v=0.5),
        dict(label="VIKOR, v = 1 (group utility only), equal weights", method="vikor", v=1.0),
        dict(label="VIKOR, v = 0 (regret only), equal weights", method="vikor", v=0.0),
        dict(label="VIKOR, v = 0.5, AHP weights", method="vikor", v=0.5, weighting="ahp"),
        dict(label="VIKOR, v = 0.5, entropy weights", method="vikor", v=0.5, weighting="entropy"),
        dict(label="VIKOR, v = 0.5, equal weights, min–max ranges", method="vikor", v=0.5, robust=False),
    ],
    "index_overlay": [
        dict(label="multi-class, equal weights, 2–98 % ranges (default)"),
        dict(label="multi-class, AHP weights", weighting="ahp"),
        dict(label="multi-class, fuzzy AHP weights (δ = 1)", weighting="fuzzy_ahp", delta=1.0),
        dict(label="multi-class, fuzzy AHP weights (δ = 2)", weighting="fuzzy_ahp", delta=2.0),
        dict(label="multi-class, equal weights, min–max ranges", robust=False),
        dict(label="binary (upper / lower quartile), equal weights", binary=True),
        dict(label="binary (upper / lower quartile), AHP weights", binary=True, weighting="ahp"),
    ],
    "fuzzy_wofe": [
        dict(label="modified, expert memberships, max. contrast pattern, 10 classes (default)"),
        dict(label="classic, expert memberships, max. contrast pattern, 10 classes", model="classic"),
        dict(label="modified, data memberships, 10 classes", membership="data"),
        dict(label="modified, expert memberships, 5 classes", n_classes=5),
        dict(label="modified, expert memberships, 20 classes", n_classes=20),
        dict(label="modified, expert memberships, μ ≥ 0.5 pattern", pattern="half"),
        dict(label="modified, unconstrained β", nonnegative=False),
    ],
    "wofe": [
        dict(label="max C, Stud(C) ≥ 1.0", criterion="max_contrast", min_stud=1.0, zero_deposit="zero"),
        dict(label="max C, Stud(C) ≥ 1.5", criterion="max_contrast", min_stud=1.5, zero_deposit="zero"),
        dict(label="max C, Stud(C) ≥ 1.96 (default)", criterion="max_contrast", min_stud=1.96, zero_deposit="zero"),
        dict(label="max C, Stud(C) ≥ 2.5", criterion="max_contrast", min_stud=2.5, zero_deposit="zero"),
        dict(label="max Stud(C)", criterion="max_studentized", min_stud=1.96, zero_deposit="zero"),
        dict(label="max C, Stud(C) ≥ 1.96, Haldane 0.5", criterion="max_contrast", min_stud=1.96,
             zero_deposit="haldane"),
    ],
    "logistic_regression": [
        dict(label="all cells, Firth (default)", design="all", estimator="firth"),
        dict(label="all cells, ML", design="all", estimator="ml"),
        dict(label="all cells, Firth, backward LR test (α 0.10)", design="all", estimator="firth",
             selection="backward_lr"),
        dict(label="sampled 1:5, Firth, 5 repeats", design="sampled", estimator="firth", neg_ratio=5.0),
        dict(label="sampled 1:10, Firth, 5 repeats", design="sampled", estimator="firth", neg_ratio=10.0),
    ],
    "random_forest": [
        dict(label="1:1 non-deposits, √p, leaf 1", neg_ratio=1.0),
        dict(label="1:3 non-deposits, √p, leaf 1", neg_ratio=3.0),
        dict(label="1:5 non-deposits, √p, leaf 1", neg_ratio=5.0),
        dict(label="1:10 non-deposits, √p, leaf 1", neg_ratio=10.0),
        dict(label="1:5 non-deposits, √p, leaf 5", neg_ratio=5.0, min_samples_leaf=5),
        dict(label="1:5 non-deposits, all p (bagging), leaf 1", neg_ratio=5.0, max_features="all"),
    ],
    "svm": [
        dict(label="RBF, tuned C and γ, 1:1 non-deposits", neg_ratio=1.0),
        dict(label="RBF, tuned C and γ, 1:3 non-deposits", neg_ratio=3.0),
        dict(label="RBF, tuned C and γ, 1:5 non-deposits", neg_ratio=5.0),
        dict(label="RBF, tuned C and γ, 1:10 non-deposits", neg_ratio=10.0),
        dict(label="RBF, tuned, 1:5, balanced class weight", neg_ratio=5.0, class_weight="balanced"),
        dict(label="linear, tuned C, 1:5", neg_ratio=5.0, kernel="linear"),
        dict(label="RBF, C = 1, γ = 1/p (ArcSDM default), 1:5", neg_ratio=5.0, tune=False),
    ],
    "xgboost": [
        dict(label="depth 3, η 0.05, 1:1 non-deposits (default)", neg_ratio=1.0),
        dict(label="depth 3, η 0.05, 1:3 non-deposits", neg_ratio=3.0),
        dict(label="depth 3, η 0.05, 1:5 non-deposits", neg_ratio=5.0),
        dict(label="depth 2, η 0.05, 1:5 non-deposits", neg_ratio=5.0, max_depth=2),
        dict(label="depth 5, η 0.05, 1:5 non-deposits", neg_ratio=5.0, max_depth=5),
        dict(label="depth 3, 1:5, geological monotone constraints", neg_ratio=5.0, mono=True),
        dict(label="depth 3, 1:1, geological monotone constraints", neg_ratio=1.0, mono=True),
    ],
    "rvm": [
        dict(label="RBF, tuned γ, 1:1 non-deposits (default)", neg_ratio=1.0),
        dict(label="RBF, tuned γ, 1:3 non-deposits", neg_ratio=3.0),
        dict(label="RBF, tuned γ, 1:5 non-deposits", neg_ratio=5.0),
        dict(label="linear (relevance determination), 1:1", neg_ratio=1.0, kernel="linear"),
        dict(label="linear (relevance determination), 1:3", neg_ratio=3.0, kernel="linear"),
        dict(label="RBF, γ = 1/p (not tuned), 1:3", neg_ratio=3.0, tune=False),
    ],
    "mlp": [
        dict(label="32-16 ReLU, α 1e-4, 1:1 non-deposits (default)", neg_ratio=1.0),
        dict(label="32-16 ReLU, α 1e-4, 1:3 non-deposits", neg_ratio=3.0),
        dict(label="32-16 ReLU, α 1e-4, 1:5 non-deposits", neg_ratio=5.0),
        dict(label="8 ReLU (one small layer), 1:3", neg_ratio=3.0, hidden=(8,)),
        dict(label="64-32 ReLU, 1:3", neg_ratio=3.0, hidden=(64, 32)),
        dict(label="32-16 ReLU, α 0.01 (strong L2), 1:3", neg_ratio=3.0, alpha=0.01),
        dict(label="32-16 tanh, 1:3", neg_ratio=3.0, activation="tanh"),
    ],
    "elm": [
        dict(label="100 sigmoid, tuned C, 1:1 non-deposits (default)", neg_ratio=1.0),
        dict(label="100 sigmoid, tuned C, 1:3 non-deposits", neg_ratio=3.0),
        dict(label="100 sigmoid, tuned C, 1:5 non-deposits", neg_ratio=5.0),
        dict(label="30 sigmoid, tuned C, 1:3", neg_ratio=3.0, hidden=30),
        dict(label="300 sigmoid, tuned C, 1:3", neg_ratio=3.0, hidden=300),
        dict(label="100 Gaussian RBF, tuned C, 1:3", neg_ratio=3.0, activation="rbf"),
        dict(label="100 sigmoid, C = 1 (not tuned), 1:3", neg_ratio=3.0, tune=False),
    ],
    "cnn": [
        dict(label="patch 9, 8-16 filters, augmented, 1:3 (default)", neg_ratio=3.0),
        dict(label="patch 5, 8-16 filters, augmented, 1:3", neg_ratio=3.0, patch=5),
        dict(label="patch 15, 8-16 filters, augmented, 1:3", neg_ratio=3.0, patch=15),
        dict(label="patch 9, no augmentation, 1:3", neg_ratio=3.0, augment=False),
        dict(label="patch 9, augmented, 1:1", neg_ratio=1.0),
    ],
    "bayesian_network": [
        dict(label="TAN, 5 classes, all cells (default)", structure="tan"),
        dict(label="naive Bayes, 5 classes, all cells", structure="naive"),
        dict(label="TAN, 3 classes, all cells", structure="tan", n_classes=3),
        dict(label="TAN, 8 classes, all cells", structure="tan", n_classes=8),
        dict(label="TAN, 5 classes, α = 10", structure="tan", alpha=10.0),
        dict(label="TAN, 5 classes, deposits + 1:5 random cells", structure="tan", design="sampled"),
    ],
    "fuzzy_logic": [
        dict(label="OR within groups, gamma 0.9 between groups (default)"),
        dict(label="OR within groups, gamma 0.7", gamma=0.7),
        dict(label="OR within groups, gamma 0.95", gamma=0.95),
        dict(label="OR within groups, fuzzy AND between groups", final_op="and"),
        dict(label="OR within groups, algebraic product between groups", final_op="product"),
        dict(label="OR within groups, geometric average between groups", final_op="geomean"),
        dict(label="gamma 0.9 within and between groups", group_op="gamma"),
    ],
    "boolean_logic": [
        dict(label="upper quartile, OR within groups, AND between (default)"),
        dict(label="above median, OR within, AND between", q=50),
        dict(label="upper decile, OR within, AND between", q=90),
        dict(label="upper quartile, AND within, AND between", group_op="and"),
        dict(label="upper quartile, OR within, OR between", final_op="or"),
        dict(label="upper quartile, faults ≤ 1500 m", faults_m=1500),
    ],
    "logistic_geometric": [
        dict(label="all layers, data range, equal weights (default)"),
        dict(label="all layers, 2–98 % range, equal weights", robust=True),
        dict(label="all layers, data range, P–A weights", weighting="pa"),
        dict(label="all layers, 2–98 % range, P–A weights", robust=True, weighting="pa"),
    ],
}
BUILD = {"wofe": wofe_params, "logistic_regression": lr_params, "random_forest": rf_params, "svm": svm_params,
         "xgboost": xgb_params, "rvm": rvm_params, "mlp": mlp_params, "elm": elm_params, "cnn": cnn_params,
         "bayesian_network": bn_params, "fuzzy_logic": fuzzy_params, "boolean_logic": boolean_params,
         "logistic_geometric": lg_params, "fuzzy_wofe": fwofe_params, "index_overlay": io_params,
         "topsis": topsis_params, "neuro_fuzzy": nf_params, "dc_fuzzy": dcf_params,
         "ebf": ebf_params}
HYBRID = ("fuzzy_wofe", "neuro_fuzzy", "dc_fuzzy")


def run(method, d, cfg, test_fraction, seed, deposits=None):
    p = BUILD[method](d, {k: v for k, v in cfg.items() if k != "label"})
    p.update(base_params(d, test_fraction, seed, deposits))
    t0 = time.time()
    res = REGISTRY[method].run(p, lambda *a: None)
    return res, time.time() - t0


def curve_stats(c):
    pa = c.pa_intersection or {}
    return dict(deposits=c.n_pos, negatives=c.n_neg, auc=c.auc, auc_lo=c.auc_ci[0], auc_hi=c.auc_ci[1],
                sr_auc=c.sr_auc, pa_prediction=pa.get("prediction"), pa_area=pa.get("area"),
                tpr_at={str(f): float(c.tpr[c.fpr <= f + 1e-12].max()) for f in (0.05, 0.1, 0.2, 0.3)})


def split_eval(method, d, cfg, seeds):
    rows = []
    for s in seeds:
        res, dt = run(method, d, cfg, 0.3, s)
        tr, te = res.evaluation.curves["training"], res.evaluation.curves["test"]
        rows.append(dict(seed=s, train_auc=tr.auc, test_auc=te.auc, test_sr=te.sr_auc,
                         test_pa=(te.pa_intersection or {}).get("prediction"), seconds=dt))
    return pd.DataFrame(rows)


def lodo_eval(method, d, cfg):
    m, g = d.mask()
    valid = ~np.ma.getmaskarray(m)
    rows, cols = points_to_cells(d.deposits, g, valid_mask=valid)
    out = []
    for i in range(len(d.deposits)):
        held = d.deposits.iloc[[i]]
        r, c = points_to_cells(held, g, valid_mask=valid)
        if not len(r):
            continue
        res, dt = run(method, d, cfg, 0.0, 0, deposits=d.deposits.drop(d.deposits.index[i]))
        score = np.ma.asarray(res.outputs[0].data)
        sv = ~np.ma.getmaskarray(score) & valid
        if not sv[r[0], c[0]]:
            continue
        others = np.zeros(score.shape, bool)
        others[rows, cols] = True
        bg = np.ma.getdata(score)[sv & ~others]
        auc, _, _ = delong_auc(np.array([float(score[r[0], c[0]])]), bg)
        out.append(dict(deposit=i, row=int(r[0]), col=int(c[0]), lodo_auc=auc,
                        train_auc=res.evaluation.curves["training"].auc, seconds=dt))
    return pd.DataFrame(out)


# ============================================================================== figures
def fig_map(d, res, method, path, title):
    out = res.outputs[0]
    score = np.ma.asarray(out.data)
    g = out.grid
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=200)
    im = ax.imshow(score, extent=g.extent, cmap="inferno", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(out.name, fontsize=8)
    for v in res.vectors:
        pts = v.gdf.to_crs(g.crs) if (v.gdf.crs is not None and g.crs is not None) else v.gdf
        if not len(pts):
            continue
        xs, ys = [p.x for p in pts.geometry], [p.y for p in pts.geometry]
        if v.name == "training_points":
            ax.scatter(xs, ys, s=22, c="#43a047", edgecolors="white", linewidths=0.6, label=f"training ({len(xs)})")
        else:
            ax.scatter(xs, ys, s=34, marker="^", c="#29b6f6", edgecolors="#102027", linewidths=0.6,
                       label=f"test ({len(xs)})")
    if not any(len(v.gdf) for v in res.vectors):           # nothing trained (knowledge-driven): show all deposits
        pts = d.deposits.to_crs(g.crs) if (d.deposits.crs is not None and g.crs is not None) else d.deposits
        ax.scatter([p.x for p in pts.geometry], [p.y for p in pts.geometry], s=30, marker="^", c="#29b6f6",
                   edgecolors="#102027", linewidths=0.6, label=f"deposits ({len(pts)})")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_curves(res, method, path, extra):
    ev = res.evaluation
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4), dpi=200)
    colors = {"training": "#1f77b4", "test": "#d62728"}
    ax = axs[0, 0]
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, c="grey")
    for k, c in ev.curves.items():
        ax.plot(c.fpr, c.tpr, c=colors.get(k), lw=1.6, label=f"{k}: AUC {c.auc:.3f}")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC curve")
    ax.legend(fontsize=7, loc="lower right")
    ax = axs[0, 1]
    ax.plot([0, 100], [0, 100], ls="--", lw=0.8, c="grey")
    for k, c in ev.curves.items():
        ax.plot(c.area_frac * 100, c.capture_frac * 100, c=colors.get(k), lw=1.6, label=f"{k}: {c.sr_auc:.3f}")
    ax.set(xlabel="Study area, highest scores first (%)", ylabel="Deposits captured (%)",
           title="(b) Success / prediction rate")
    ax.legend(fontsize=7, loc="lower right")
    ax = axs[1, 0]
    c = ev.curves["training"]
    ax.plot(c.pa_threshold, c.pa_prediction, c="#1f77b4", lw=1.6, label="prediction rate")
    ax2 = ax.twinx()
    ax2.plot(c.pa_threshold, c.pa_area, c="#ff7f0e", lw=1.6, label="occupied area")
    ax2.set_ylim(100, 0)
    ax.set(xlabel="Score threshold", ylabel="Deposits (%)", title="(c) Prediction–area plot (training)")
    ax2.set_ylabel("Study area (%)", fontsize=8)
    ax2.tick_params(labelsize=7)
    if c.pa_intersection:
        ax.axvline(c.pa_intersection["threshold"], ls=":", c="grey")
    extra(axs[1, 1])
    for a in axs.ravel():
        a.tick_params(labelsize=7)
        a.title.set_fontsize(9)
        a.xaxis.label.set_fontsize(8)
        a.yaxis.label.set_fontsize(8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def extra_wofe(res):
    w = res.extra["wofe"]

    def draw(ax):
        cum = [l for l in w.layers if l.weight_type != "categorical"]
        for l in cum:
            t = l.table
            ax.plot(np.arange(len(t)), t["C"], marker="o", ms=3, lw=1.2, label=f"{l.name} ({l.weight_type[:4]}.)")
            sel = int(np.flatnonzero(t["selected"].values)[0])
            ax.plot(sel, t["C"].iloc[sel], marker="*", ms=11, c="k")
        ax.set(xlabel="Cumulative class (ordered)", ylabel="Contrast C", title="(d) Contrast of cumulative patterns (★ = selected)")
        ax.legend(fontsize=6)
    return draw


def extra_lr(res):
    co = res.extra["lr"].coefficients
    co = co[(co.term != "(intercept)") & co.s_B.notna()]

    def draw(ax):
        y = np.arange(len(co))
        z = co.z.values
        ax.barh(y, z, color=["#1f77b4" if v > 0 else "#d62728" for v in z])
        for x in (-1.96, 1.96):
            ax.axvline(x, c="grey", lw=0.8, ls="--")
        ax.axvline(0, c="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(co.term, fontsize=6)
        ax.set(xlabel="Wald z = B / s(B)   (|z| > 1.96: p < 0.05)", title="(d) LR coefficients (significance)")
    return draw


def extra_rf(res):
    imp = res.extra["rf"].importance.iloc[::-1]

    def draw(ax):
        y = np.arange(len(imp))
        ax.barh(y, imp.permutation_auc_drop, xerr=imp.permutation_sd, color="#1f77b4", ecolor="grey")
        ax.set_yticks(y)
        ax.set_yticklabels(imp.layer, fontsize=7)
        ax.axvline(0, c="grey", lw=0.8)
        ax.set(xlabel="OOB AUC drop when shuffled", title="(d) Permutation importance")
    return draw


def extra_svm(res):
    r = res.extra["svm"]
    tab = r.tuning

    def draw(ax):
        if len(tab) and tab.log2_gamma.nunique() > 1:
            cs, gs = np.sort(tab.log2_C.unique()), np.sort(tab.log2_gamma.unique())
            z = tab.pivot(index="log2_gamma", columns="log2_C", values="cv_auc").reindex(index=gs, columns=cs)
            im = ax.imshow(z.values, origin="lower", aspect="auto", cmap="viridis",
                           extent=(cs[0] - 1, cs[-1] + 1, gs[0] - 1, gs[-1] + 1))
            ax.figure.colorbar(im, ax=ax, label="CV AUC")
            ax.plot(np.log2(r.stats["C"]), np.log2(r.stats["gamma"]), "r*", ms=12)
            ax.set(xlabel="log2 C", ylabel="log2 γ", title="(d) Grid search (5-fold CV AUC)")
        else:
            imp = r.importance.iloc[::-1]
            y = np.arange(len(imp))
            ax.barh(y, imp.permutation_auc_drop, xerr=imp.permutation_sd, color="#1f77b4", ecolor="grey")
            ax.set_yticks(y)
            ax.set_yticklabels(imp.layer, fontsize=7)
            ax.axvline(0, c="grey", lw=0.8)
            ax.set(xlabel="CV AUC drop when shuffled", title="(d) Permutation importance")
    return draw


def extra_xgb(res):
    imp = res.extra["xgb"].importance.iloc[::-1]

    def draw(ax):
        y = np.arange(len(imp))
        col = ["#d62728" if (dv is not None and dv == dv and dv < 0) else "#1f77b4" for dv in imp.direction]
        ax.barh(y, imp.shap_pct, color=col)
        ax.set_yticks(y)
        ax.set_yticklabels(imp.layer, fontsize=7)
        ax.set(xlabel="mean |SHAP| share (%)  (red: score falls with the value)", title="(d) SHAP importance")
    return draw


def extra_rvm(res):
    imp = res.extra["rvm"].importance.iloc[::-1]

    def draw(ax):
        y = np.arange(len(imp))
        ax.barh(y, imp.permutation_auc_drop, xerr=imp.permutation_sd, color="#1f77b4", ecolor="grey")
        ax.set_yticks(y)
        ax.set_yticklabels(imp.layer, fontsize=7)
        ax.axvline(0, c="grey", lw=0.8)
        ax.set(xlabel="CV AUC drop when shuffled", title="(d) Permutation importance")
    return draw


def extra_mlp(res):
    c = res.extra["mlp"].curve
    e = res.extra["mlp"].stats["epochs"]

    def draw(ax):
        ax.plot(c["epoch"], c.train_auc, label="training folds")
        ax.plot(c["epoch"], c.cv_auc, label="held-out folds")
        ax.axvline(e, c="r", lw=1, ls="--")
        ax.legend(fontsize=7)
        ax.set(xlabel="epoch", ylabel="AUC", title="(d) Learning curve (early stopping)")
    return draw


def extra_elm(res):
    t = res.extra["elm"].tuning
    C = res.extra["elm"].stats["C"]

    def draw(ax):
        if len(t):
            ax.plot(t.log2_C, t.cv_auc, "o-")
            ax.axvline(np.log2(C), c="r", lw=1, ls="--")
        ax.set(xlabel="log2 C", ylabel="CV AUC", title="(d) Regularisation (5-fold CV AUC)")
    return draw


def extra_cnn(res):
    c = res.extra["cnn"].curve
    e = res.extra["cnn"].stats["epochs"]

    def draw(ax):
        ax.plot(c["epoch"], c.train_auc, label="training folds")
        ax.plot(c["epoch"], c.cv_auc, label="held-out folds")
        ax.axvline(e, c="r", lw=1, ls="--")
        ax.legend(fontsize=7)
        ax.set(xlabel="epoch", ylabel="AUC", title="(d) Learning curve")
    return draw


def extra_bn(res):
    lay = res.extra["bn"].layers.iloc[::-1]

    def draw(ax):
        y = np.arange(len(lay))
        ax.barh(y, lay.auc_drop_without_layer, color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(lay.layer, fontsize=7)
        ax.axvline(0, c="grey", lw=0.8)
        ax.set(xlabel="training AUC drop without the layer", title="(d) Evidence layers")
    return draw


def extra_kd(res):
    tab = (res.tables.get("memberships") if "memberships" in res.tables else res.tables.get("conditions")
           if "conditions" in res.tables else res.tables.get("evidence"))

    def draw(ax):
        if res.method == "boolean_logic":
            t = tab[tab.layer != "FAVOURABLE (combined)"]
            y = np.arange(len(t))
            ax.barh(y, t.ratio.astype(float), color="#1f77b4")
            ax.set_yticks(y)
            ax.set_yticklabels(t.layer, fontsize=7)
            ax.axvline(1, c="grey", lw=0.8, ls="--")
            ax.set(xlabel="deposit share / area share", title="(d) Conditions")
        else:
            col = "weight" if "weight" in tab else "mean"
            y = np.arange(len(tab))
            ax.barh(y, tab[col].astype(float), color="#1f77b4")
            ax.set_yticks(y)
            ax.set_yticklabels(tab.layer, fontsize=7)
            ax.set(xlabel="weight" if col == "weight" else "mean membership", title="(d) Evidence")
    return draw


def extra_fwofe(res):
    w = res.tables["weights"]

    def draw(ax):
        for i, (lay, t) in enumerate(w.groupby("layer", sort=False)):
            t = t.sort_values("membership")
            ax.plot(t.membership, t.fuzzy_weight, "o-", ms=3, lw=1.2, label=lay)
        ax.axhline(0, c="grey", lw=0.8)
        ax.set(xlabel="class membership μ", ylabel="fuzzy weight W(μ)", title="(d) Fuzzy weights")
        ax.legend(fontsize=6)
    return draw


def extra_nf(res):
    mf = res.tables["membership"]

    def draw(ax):
        x = np.linspace(0, 1, 201)
        for i, (inp, t) in enumerate(mf.groupby("input", sort=False)):
            col = f"C{i}"
            for _, r in t.iterrows():
                ax.plot(x, np.exp(-(x - r.centre) ** 2 / (2 * r.sigma ** 2)), c=col, lw=1.3,
                        label=inp if r.mf == t.mf.iloc[0] else None)
        e = mf.iloc[: mf.mf.nunique()]
        for _, r in e.iterrows():
            ax.plot(x, np.exp(-(x - r.centre_expert) ** 2 / (2 * r.sigma_expert ** 2)), c="grey", ls="--", lw=1.0)
        ax.set(xlabel="input (expert membership / process score)", ylabel="membership A(x)", ylim=(0, 1.05),
               title="(d) Learnt MFs (dashed: expert start)")
        ax.legend(fontsize=6)
    return draw


def extra_dcf(res):
    mem = res.tables["memberships"]

    def draw(ax):
        for i, (lay, t) in enumerate(mem.groupby("layer", sort=False)):
            k = np.arange(len(t))
            ax.plot(k, t.mu_data.values, "o-", ms=3, lw=1.3, c=f"C{i}", label=lay)
            ax.plot(k, t.mu_expert.values, ls="--", lw=0.9, c=f"C{i}")
        ax.axhline(0.5, c="grey", lw=0.8)
        ax.set(xlabel="class (ascending value / category)", ylabel="membership μ", ylim=(0, 1),
               title="(d) Calibrated μ (dashed: expert)")
        ax.legend(fontsize=6)
    return draw


def extra_ebf(res):
    r = res.extra["ebf"]

    def draw(ax):
        for i, l in enumerate(r.layers):
            t = l.table
            k = np.arange(len(t)) / max(len(t) - 1, 1)
            ax.plot(k, t.Bel.values, "o-", ms=3, lw=1.3, c=f"C{i}", label=l.name)
            ax.plot(k, t.Pls.values, ls="--", lw=0.9, c=f"C{i}")
        ax.set(xlabel="class (ascending value / category), scaled 0–1", ylabel="Bel (solid), Pls (dashed)",
               ylim=(0, 1.02), title="(d) Class belief functions")
        ax.legend(fontsize=6)
    return draw


def fig_ebf_maps(d, res, path):
    """Uncertainty and plausibility of the reference run (the other two EBF maps)."""
    outs = {o.name: o for o in res.outputs}
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    for ax, (k, title, cmap) in zip(axs, (("unc", "(a) Uncertainty Unc", "cividis"),
                                          ("pls" if "pls" in outs else "bel", "(b) Plausibility Pls" if "pls" in outs
                                           else "(b) Belief Bel", "magma"))):
        o = outs[k]
        im = ax.imshow(np.ma.asarray(o.data), extent=o.grid.extent, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        pts = d.deposits.to_crs(o.grid.crs) if (d.deposits.crs is not None and o.grid.crs is not None) else d.deposits
        ax.scatter([p.x for p in pts.geometry], [p.y for p in pts.geometry], s=16, marker="^", c="#29b6f6",
                   edgecolors="#102027", linewidths=0.5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.ticklabel_format(style="plain", useOffset=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


EXTRA = {"ebf": extra_ebf, "dc_fuzzy": extra_dcf, "neuro_fuzzy": extra_nf, "fuzzy_wofe": extra_fwofe, "index_overlay": extra_kd, "topsis": extra_kd, "fuzzy_logic": extra_kd, "boolean_logic": extra_kd, "logistic_geometric": extra_kd, "wofe": extra_wofe, "logistic_regression": extra_lr, "random_forest": extra_rf, "svm": extra_svm,
         "xgboost": extra_xgb, "rvm": extra_rvm, "mlp": extra_mlp, "elm": extra_elm, "cnn": extra_cnn,
         "bayesian_network": extra_bn}


# ============================================================================== method tables
def method_tables(method, res):
    if method == "wofe":
        w = res.extra["wofe"]
        layers = []
        for l in w.layers:
            cols = [c for c in ["cls", "label", "cells", "area", "deposits", "cum_area", "cum_deposits", "W+", "s(W+)",
                                "W-", "s(W-)", "C", "s(C)", "Stud(C)", "selected", "gen_class", "weight", "s_weight"]
                    if c in l.table.columns]
            layers.append(dict(name=l.name, type=l.weight_type, threshold=l.threshold, contrast=l.contrast,
                               studentized=l.studentized, table=l.table[cols].to_dict("records"),
                               gen=l.gen_table.to_dict("records")))
        return dict(prior=w.prior, n_area=w.n_area, ci={k: (float(v) if v == v else None) for k, v in w.ci.items()},
                    pairwise=w.pairwise.to_dict("records"), layers=layers)
    if method == "logistic_regression":
        r = res.extra["lr"]
        return dict(coefficients=r.coefficients.to_dict("records"), variables=r.variables.to_dict("records"),
                    fit=r.fit_table.to_dict("records"), hl=r.hosmer_lemeshow.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), stats=r.stats)
    if method == "ebf":
        out = {k: v.to_dict("records") for k, v in res.tables.items()}
        out["stats"] = res.extra["ebf"].stats
        out["messages"] = res.messages
        return out
    if method in KNOWLEDGE or method in HYBRID:
        out = {k: v.to_dict("records") for k, v in res.tables.items()}
        if method == "fuzzy_wofe":
            out["stats"] = res.extra["fwofe"].stats
        if method == "dc_fuzzy":
            out["stats"] = res.extra["dcfuzzy"].stats
            out["messages"] = res.extra["dcfuzzy"].messages
        if method == "neuro_fuzzy":
            out["stats"] = res.extra["neurofuzzy"].stats
            out["input_mode"] = res.extra["neurofuzzy"].input_mode
        return out
    if method == "cnn":
        r = res.extra["cnn"]
        return dict(importance=r.importance.to_dict("records"), curve=r.curve.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                    repeats=r.repeats.to_dict("records"), stats=r.stats, params=r.params)
    if method == "bayesian_network":
        r = res.extra["bn"]
        return dict(edges=r.edges.to_dict("records"), cpt=r.cpt_table.to_dict("records"),
                    layers=r.layers.to_dict("records"), comparison=r.comparison.to_dict("records"),
                    summary=r.summary.to_dict("records"), stats=r.stats, params=r.params)
    if method == "elm":
        r = res.extra["elm"]
        return dict(tuning=r.tuning.to_dict("records"), importance=r.importance.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                    repeats=r.repeats.to_dict("records"), stats=r.stats, params=r.params)
    if method == "mlp":
        r = res.extra["mlp"]
        return dict(importance=r.importance.to_dict("records"), curve=r.curve.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                    repeats=r.repeats.to_dict("records"), stats=r.stats, params=r.params)
    if method == "rvm":
        r = res.extra["rvm"]
        return dict(relevance=r.relevance.to_dict("records"), tuning=r.tuning.to_dict("records"),
                    importance=r.importance.to_dict("records"), sample_validation=r.sample_validation.to_dict("records"),
                    summary=r.summary.to_dict("records"), repeats=r.repeats.to_dict("records"), stats=r.stats,
                    params=r.params)
    if method == "xgboost":
        r = res.extra["xgb"]
        return dict(importance=r.importance.to_dict("records"), curve=r.curve.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                    repeats=r.repeats.to_dict("records"), stats=r.stats, params=r.params)
    if method == "svm":
        r = res.extra["svm"]
        return dict(tuning=r.tuning.to_dict("records"), importance=r.importance.to_dict("records"),
                    sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                    repeats=r.repeats.to_dict("records"), stats=r.stats, params=r.params)
    r = res.extra["rf"]
    return dict(importance=r.importance.to_dict("records"), importance_columns=r.importance_columns.to_dict("records"),
                sample_validation=r.sample_validation.to_dict("records"), summary=r.summary.to_dict("records"),
                repeats=r.repeats.to_dict("records"), stats=r.stats)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ============================================================================== main
def experiment(method, key, quick=False, log=print):
    d = Demo(key)
    cands = CANDIDATES[method][:2] if quick else CANDIDATES[method]
    protocol = "expert" if method in KNOWLEDGE else DATASETS[key]["protocol"]
    t0 = time.time()
    search = []
    for cfg in cands:
        log(f"  [{method}/{key}] candidate: {cfg['label']}")
        if protocol == "expert":                         # nothing is trained: every deposit validates
            r_, dt_ = run(method, d, cfg, 0.0, 42)
            c_ = r_.evaluation.curves["training"]
            row = dict(label=cfg["label"], cfg=cfg, mean_test_auc=c_.auc, sd_test_auc=None, mean_train_auc=c_.auc,
                       seconds=dt_, pa_prediction=(c_.pa_intersection or {}).get("prediction"),
                       pa_area=(c_.pa_intersection or {}).get("area"))
            if "evidence" in r_.tables and "weight" in r_.tables["evidence"]:
                row["weights"] = dict(zip(r_.tables["evidence"].layer, r_.tables["evidence"].weight.astype(float)))
            if method == "boolean_logic":
                comb = r_.tables["conditions"].set_index("layer").loc["FAVOURABLE (combined)"]
                row.update(fav_area_pct=float(comb.area_pct), fav_deposits_pct=float(comb.deposits_pct),
                           fav_ratio=float(comb.ratio) if comb.ratio == comb.ratio else None,
                           # a binary target map is judged by deposits captured minus area used (Youden's J)
                           select_score=(float(comb.deposits_pct) - float(comb.area_pct)) / 100.0)
            search.append(row)
        elif protocol == "split":
            df = split_eval(method, d, cfg, [100] if quick else [100, 101, 102, 103, 104])
            search.append(dict(label=cfg["label"], cfg=cfg, mean_test_auc=df.test_auc.mean(),
                               sd_test_auc=df.test_auc.std(ddof=1) if len(df) > 1 else None,
                               mean_train_auc=df.train_auc.mean(), seconds=df.seconds.mean()))
        else:
            df = lodo_eval(method, d, cfg)
            search.append(dict(label=cfg["label"], cfg=cfg, mean_test_auc=df.lodo_auc.mean(),
                               sd_test_auc=df.lodo_auc.std(ddof=1), mean_train_auc=df.train_auc.mean(),
                               seconds=df.seconds.mean(), lodo=df.to_dict("records")))
        log(f"      {'mean test' if protocol == 'split' else 'mean LODO' if protocol == 'lodo' else 'all-deposit'} "
            f"AUC {search[-1]['mean_test_auc']:.4f}")
    # settings within AUC_TIE of the best are equivalent -> the earliest candidate (default / simplest) is chosen
    key_ = "select_score" if all("select_score" in s for s in search) else "mean_test_auc"
    top = max(s[key_] for s in search)
    best = next(s for s in search if s[key_] >= top - AUC_TIE)
    cfg = best["cfg"]
    log(f"  [{method}/{key}] selected: {cfg['label']}")
    robust = None
    if protocol == "expert":
        ref, dt = run(method, d, cfg, 0.0, 42)
    elif protocol == "split":
        robust = split_eval(method, d, cfg, [0] if quick else list(range(10)))
        ref, dt = run(method, d, cfg, 0.3, 42)
    else:
        ref, dt = run(method, d, cfg, 0.0, 42)
    sens = []
    if method == "random_forest" and not quick:           # number of trees: stability of the score
        for n in (50, 100, 300, 500, 1000):
            c2 = dict(cfg, n_trees=n)
            r2, dt2 = run(method, d, c2, 0.3 if protocol == "split" else 0.0, 42)
            sens.append(dict(n_trees=n, oob_auc=r2.extra["rf"].stats["oob_auc"],
                             train_auc=r2.evaluation.curves["training"].auc,
                             test_auc=r2.evaluation.curves["test"].auc if "test" in r2.evaluation.curves else None,
                             seconds=dt2))
    if method == "xgboost" and not quick:                 # learning rate: number of trees and accuracy
        for eta in (0.01, 0.05, 0.1, 0.3):
            c2 = dict(cfg, learning_rate=eta)
            r2, dt2 = run(method, d, c2, 0.3 if protocol == "split" else 0.0, 42)
            sens.append(dict(learning_rate=eta, trees=r2.extra["xgb"].stats["trees"],
                             cv_auc=r2.extra["xgb"].stats["cv_auc"], train_auc=r2.evaluation.curves["training"].auc,
                             test_auc=r2.evaluation.curves["test"].auc if "test" in r2.evaluation.curves else None,
                             seconds=dt2))
    tag = f"{method}_{key}"
    title = {"wofe": "WofE posterior probability", "logistic_regression": "LR probability",
             "random_forest": "RF prospectivity score", "svm": "SVM prospectivity score",
             "xgboost": "XGBoost prospectivity score", "rvm": "RVM prospectivity score",
             "mlp": "MLP prospectivity score", "elm": "ELM prospectivity score",
             "cnn": "CNN prospectivity score", "bayesian_network": "Bayesian network posterior probability",
             "fuzzy_logic": "Fuzzy favourability", "boolean_logic": "Boolean favourable area",
             "logistic_geometric": "Geometric average of logistic memberships",
             "fuzzy_wofe": "Fuzzy WofE posterior probability", "index_overlay": "Index overlay",
             "topsis": "TOPSIS / VIKOR prospectivity", "neuro_fuzzy": "Neuro-fuzzy calibrated score",
             "dc_fuzzy": "Data-calibrated fuzzy favourability", "ebf": "EBF integrated score"}[method]
    fig_map(d, ref, method, os.path.join(FIG, f"{tag}_map.png"), f"{title} — {DATASETS[key]['title']}")
    fig_curves(ref, method, os.path.join(FIG, f"{tag}_curves.png"), EXTRA[method](ref))
    if method == "ebf":
        fig_ebf_maps(d, ref, os.path.join(FIG, f"{tag}_unc.png"))
    out = dict(method=method, dataset=key, dataset_info=d.describe(),
               evidence=d.layer_info([(n, "categorical" if isinstance(x, str) else "continuous")
                                      for n, _, x, _ in HYBRID_KNOWLEDGE[key]] if method in HYBRID
                                     else EVIDENCE[key][method if method in ("wofe", "ebf") else "cont"]),
               knowledge=[dict(layer=n, process=pr, direction=x if isinstance(x, str) else ("+" if x > 0 else "−"),
                               weight=w) for n, pr, x, w in HYBRID_KNOWLEDGE[key]] if method in HYBRID else None,
               protocol=protocol, search=search, selected=cfg,
               robust=robust.to_dict("records") if robust is not None else None,
               robust_summary=None if robust is None else dict(
                   train_auc_mean=robust.train_auc.mean(), train_auc_sd=robust.train_auc.std(ddof=1) if len(robust) > 1 else None,
                   test_auc_mean=robust.test_auc.mean(), test_auc_sd=robust.test_auc.std(ddof=1) if len(robust) > 1 else None,
                   test_auc_min=robust.test_auc.min(), test_auc_max=robust.test_auc.max(),
                   test_sr_mean=robust.test_sr.mean(), test_pa_mean=robust.test_pa.mean(), seconds_mean=robust.seconds.mean()),
               reference=dict(seed=42, seconds=dt, curves={k: curve_stats(c) for k, c in ref.evaluation.curves.items()},
                              messages=ref.messages, tables=method_tables(method, ref)),
               sensitivity=sens, figures=dict(map=f"{tag}_map.png", curves=f"{tag}_curves.png"),
               ahp=ahp_info(key) if method in ("index_overlay", "topsis") else None,
               vector_weights=topsis_vector_weights(key) if method == "topsis" else None,
               runtime_s=time.time() - t0)
    with open(os.path.join(OUT, f"{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(out), fh, ensure_ascii=False, indent=1)
    log(f"  [{method}/{key}] done in {time.time() - t0:.0f} s")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", choices=list(CANDIDATES))
    ap.add_argument("--dataset", action="append", choices=list(DATASETS))
    ap.add_argument("--quick", action="store_true", help="one seed, two candidates (smoke test)")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    for key in a.dataset or list(DATASETS):
        for method in a.method or list(CANDIDATES):
            print(f"=== {method} / {key}", flush=True)
            experiment(method, key, a.quick, log=lambda s: print(s, flush=True))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
