"""Evidential belief functions (EBF) — data-driven Dempster–Shafer model (An et al. 1994; Carranza & Hale 2003;
Carranza et al. 2005; Carranza 2008).

Pure numpy implementation (no GUI). Typical use::

    ev = [EBFEvidence("geology", geol_array, "values", labels=labels),
          EBFEvidence("as_ppm", as_array, "quantile", n_classes=10)]
    res = EBFModel().fit_predict(ev, deposit_rows, deposit_cols, study_mask)
    res.bel, res.dis, res.unc, res.pls, res.layers[0].table

Belief functions of class C_j of an evidence layer (cells counted on known evidence; N(T) cells, N(D) deposit
cells, N(C_j) cells of the class, N(C_j∩D) deposit cells in the class)::

    W_CjD    = [N(Cj∩D) / N(D)] / [(N(Cj) − N(Cj∩D)) / (N(T) − N(D))]          (= exp W+ of WofE)
    W_CjD̄    = [(N(D) − N(Cj∩D)) / N(D)] / [(N(T) − N(D) − N(Cj) + N(Cj∩D)) / (N(T) − N(D))]   (= exp W−)
    Bel_j = W_CjD / Σ_j W_CjD,   Dis_j = W_CjD̄ / Σ_j W_CjD̄,   Unc_j = 1 − Bel_j − Dis_j,   Pls_j = Bel_j + Unc_j

Two layers are combined with Dempster's rule of combination (An et al. 1994, eqs. as in Carranza & Hale 2003)::

    β   = 1 − Bel1·Dis2 − Dis1·Bel2                          (1 − β = conflicting mass)
    Bel = (Bel1·Bel2 + Bel1·Unc2 + Bel2·Unc1) / β
    Dis = (Dis1·Dis2 + Dis1·Unc2 + Dis2·Unc1) / β
    Unc = Unc1·Unc2 / β

The rule is commutative and associative, so the layer order does not matter. A cell without evidence of a layer
gets the vacuous belief function (Bel 0, Dis 0, Unc 1), the identity of the rule: the layer then neither supports
nor refutes the cell and only the other layers decide (or the cell is excluded, ``missing="exclude"``).
A binary layer (two classes) has Unc = 0 in both classes (Dis of one class = Bel of the other).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from i18n import N_

CLASSINGS = ("values", "quantile")
CLASSING_LABELS = {"values": N_("class values"), "quantile": N_("quantile classes")}
MISSING = ("vacuous", "exclude")
ZERO_RULES = ("zero", "haldane")
SCORES = ("bel", "pls", "mid")
SCORE_LABELS = {"bel": N_("belief Bel (lower bound)"), "pls": N_("plausibility Pls (upper bound)"),
                "mid": N_("mid-point (Bel + Pls) / 2")}
MAX_VALUES = 1000           # "values" classing: one class per distinct value
MAX_QUANTILES = 50


# ============================================================================== core formulas
def class_ebf(n_cd, n_c, n_d, n_t, zero: str = "zero"):
    """Vectorised EBF of the classes of one layer.

    n_cd: deposit cells per class; n_c: cells per class; n_d: deposit cells; n_t: cells (both on known evidence).
    zero: "zero" -> classes without deposits get W_CD = 0 (Bel = 0, the original EBF); a class holding every
          deposit gets W_CD̄ = 0 (Dis = 0);
          "haldane" -> empty deposit counts N(C∩D) = 0 and N(D) − N(C∩D) = 0 are set to 0.5 (small positive Bel /
          Dis instead of 0; prevents Bel = 1 certainty, e.g. in a binary layer with a class without deposits).
    Empty non-deposit counts in a denominator (a class of deposit cells only) are set to 0.5.
    Returns W_CD, W_CDbar, Bel, Dis, Unc, Pls (arrays) and the number of classes whose Bel + Dis > 1 (rescaled).
    """
    n_cd, n_c = np.asarray(n_cd, float), np.asarray(n_c, float)
    a, b = n_cd, n_d - n_cd                                   # deposits inside / outside the class
    if zero == "haldane":
        a, b = np.where(a == 0, 0.5, a), np.where(b == 0, 0.5, b)
    n_nd = n_t - n_d
    c_out = np.maximum(n_c - n_cd, 0.5)                       # non-deposit cells in the class
    d_out = np.maximum(n_nd - n_c + n_cd, 0.5)                # non-deposit cells outside the class
    w_d = (a / n_d) / (c_out / n_nd)
    w_nd = (b / n_d) / (d_out / n_nd)
    s_d, s_nd = w_d.sum(), w_nd.sum()
    bel = w_d / s_d if s_d > 0 else np.zeros_like(w_d)
    dis = w_nd / s_nd if s_nd > 0 else np.zeros_like(w_nd)
    tot = bel + dis
    over = tot > 1.0
    if over.any():                                            # rare (tiny classes): keep Bel : Dis, Unc = 0
        bel = np.where(over, bel / tot, bel)
        dis = np.where(over, dis / tot, dis)
    unc = np.clip(1.0 - bel - dis, 0.0, 1.0)
    return w_d, w_nd, bel, dis, unc, bel + unc, int(over.sum())


def dempster(b1, d1, u1, b2, d2, u2):
    """Dempster's rule for two belief functions on {D, D̄}. Returns Bel, Dis, Unc and the conflict 1 − β."""
    k = b1 * d2 + d1 * b2
    beta = np.maximum(1.0 - k, 1e-12)
    return ((b1 * b2 + b1 * u2 + b2 * u1) / beta, (d1 * d2 + d1 * u2 + d2 * u1) / beta, u1 * u2 / beta, k)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann–Whitney ROC AUC (ties ½)."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fmt(x: float) -> str:
    return f"{x:.4g}"


# ============================================================================== data classes
@dataclass
class EBFEvidence:
    """One evidence layer. ``classing``: "values" = one class per value (class / integer raster),
    "quantile" = ``n_classes`` equal-area classes of the values inside the study area (any numeric raster)."""

    name: str
    array: np.ma.MaskedArray
    classing: str = "values"
    n_classes: int = 10
    labels: Optional[Dict[float, str]] = None

    def __post_init__(self):
        if self.classing not in CLASSINGS:
            raise ValueError(f"{self.name}: classing must be one of {CLASSINGS}")
        if self.classing == "quantile" and not 2 <= int(self.n_classes) <= MAX_QUANTILES:
            raise ValueError(f"{self.name}: 2 … {MAX_QUANTILES} quantile classes")


@dataclass
class LayerEBF:
    name: str
    classing: str
    table: pd.DataFrame                    # per class: cells, deposits, W_CD, W_CDbar, W+, Bel, Dis, Unc, Pls
    n_deposits: int
    n_cells: int
    messages: List[str] = field(default_factory=list)


@dataclass
class EBFResult:
    layers: List[LayerEBF]
    bel: np.ma.MaskedArray
    dis: np.ma.MaskedArray
    unc: np.ma.MaskedArray
    pls: np.ma.MaskedArray
    score: np.ma.MaskedArray               # Bel, Pls or (Bel + Pls) / 2
    layer_maps: Dict[str, Dict[str, np.ma.MaskedArray]]     # layer -> {"bel", "dis", "unc"}
    summary: pd.DataFrame
    layer_table: pd.DataFrame
    pairwise: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


# ============================================================================== model
class EBFModel:
    """Evidential belief functions with Dempster's rule.

    missing: "vacuous" -> cells without evidence of a layer get Bel 0, Dis 0, Unc 1 for that layer (area kept);
             "exclude" -> cells with any missing evidence are removed from the study area.
    zero_deposit: "zero" (Bel = 0, the original EBF) or "haldane" (0.5 deposits) for classes without deposits.
    score: the prospectivity score — "bel" (belief, the usual EBF map), "pls" (plausibility) or "mid".
    """

    def __init__(self, missing: str = "vacuous", zero_deposit: str = "zero", score: str = "bel"):
        if missing not in MISSING or zero_deposit not in ZERO_RULES or score not in SCORES:
            raise ValueError("invalid EBF option")
        self.missing, self.zero_deposit, self.score = missing, zero_deposit, score

    # -------------------------------------------------------------- one layer
    def classify(self, ev: EBFEvidence, known: np.ndarray):
        """Class index (0 … k−1) of the known cells, class codes and labels."""
        vals = np.ma.getdata(np.ma.asarray(ev.array))[known].astype(float)
        if ev.classing == "values":
            classes, idx = np.unique(vals, return_inverse=True)
            if classes.size > MAX_VALUES:
                raise ValueError(f"Evidence '{ev.name}' has {classes.size:,} distinct values — too many for one class "
                                 f"per value (at most {MAX_VALUES}). Use quantile classes.")
            labels = ev.labels or {}
            lab = [labels.get(float(c), f"{c:g}") for c in classes]
            return idx.ravel(), classes, lab
        from utils.reclass import compute_breaks
        breaks = np.asarray(compute_breaks(vals, "quantile", int(ev.n_classes)), float)
        idx = np.digitize(vals, breaks, right=True)
        edges = np.r_[vals.min(), breaks, vals.max()]
        used = np.unique(idx)                                  # interpolated quantiles can leave a class empty
        remap = np.full(breaks.size + 1, -1)
        remap[used] = np.arange(used.size)
        lab = [f"{_fmt(edges[i])} – {_fmt(edges[i + 1])}" for i in used]
        return remap[idx], (used + 1).astype(float), lab

    def layer_ebf(self, ev: EBFEvidence, dep: np.ndarray, base: np.ndarray):
        """Belief functions of the classes of one layer and the class index of every known cell."""
        known = base & ~np.ma.getmaskarray(np.ma.asarray(ev.array))
        idx, classes, lab = self.classify(ev, known)
        if classes.size < 2:
            raise ValueError(f"Evidence '{ev.name}' has fewer than 2 classes inside the study area.")
        k = classes.size
        n_c = np.bincount(idx, minlength=k).astype(float)
        n_cd = np.bincount(idx, weights=dep[known].astype(float), minlength=k)
        n_t, n_d = float(known.sum()), float(n_cd.sum())
        if n_d < 1:
            raise ValueError(f"Evidence '{ev.name}': no training deposits fall on known evidence.")
        w_d, w_nd, bel, dis, unc, pls, n_over = class_ebf(n_cd, n_c, n_d, n_t, self.zero_deposit)
        msgs = []
        empty = int((n_cd == 0).sum())
        if empty:
            rule = "Bel = 0" if self.zero_deposit == "zero" else "Bel from 0.5 deposits"
            msgs.append(f"{empty} of {k} classes without training deposits ({rule})")
        sure = np.flatnonzero(bel >= 1.0 - 1e-9)
        if sure.size:
            msgs.append(f"class {lab[sure[0]]} has Bel = 1 (no deposits in the other classes): it overrides every other "
                        "layer in Dempster's rule — use the 0.5-deposit correction or more classes")
        if n_over:
            msgs.append(f"{n_over} class(es) with Bel + Dis > 1 rescaled to Unc = 0 (very small classes)")
        wplus = np.where(w_d > 0, np.log(np.where(w_d > 0, w_d, 1.0)), np.nan)     # undefined without deposits
        codes = classes.astype(int) if np.all(classes == np.round(classes)) else classes
        table = pd.DataFrame(dict(cls=codes, label=lab, cells=n_c.astype(int), deposits=n_cd.astype(int),
                                  area_pct=100 * n_c / n_t, deposits_pct=100 * n_cd / n_d, W_CD=w_d, W_CDbar=w_nd,
                                  **{"W+": wplus}, Bel=bel, Dis=dis, Unc=unc, Pls=pls, favourable=w_d > 1.0))
        return LayerEBF(ev.name, ev.classing, table, int(n_d), int(n_t), msgs), known, idx

    # -------------------------------------------------------------- full model
    def fit_predict(self, evidence: Sequence[EBFEvidence], dep_rows, dep_cols,
                    study_mask: Optional[np.ndarray] = None) -> EBFResult:
        if not evidence:
            raise ValueError("At least one evidence layer is required.")
        shape = np.ma.asarray(evidence[0].array).shape
        for ev in evidence:
            if np.ma.asarray(ev.array).shape != shape:
                raise ValueError(f"Evidence '{ev.name}' is not on the same grid as '{evidence[0].name}'.")
        known_all = [~np.ma.getmaskarray(np.ma.asarray(ev.array)) for ev in evidence]
        base = np.logical_or.reduce(known_all) if study_mask is None else np.asarray(study_mask, bool).copy()
        if self.missing == "exclude":
            base &= np.logical_and.reduce(known_all)
        dep = np.zeros(shape, bool)
        dep[np.asarray(dep_rows, int), np.asarray(dep_cols, int)] = True
        dep &= base
        n_d = int(dep.sum())
        if n_d < 2:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 2.")
        msgs = []
        if n_d < 10:
            msgs.append(f"only {n_d} training deposits: class beliefs are uncertain (use few classes)")

        cells = np.flatnonzero(base.ravel())
        label = dep.ravel()[cells]
        bel = np.zeros(cells.size)
        dis = np.zeros(cells.size)
        unc = np.ones(cells.size)                             # vacuous start = identity of Dempster's rule
        layers, maps, rows, codes = [], {}, [], []
        for i, ev in enumerate(evidence):
            lw, known, idx = self.layer_ebf(ev, dep, base)
            t = lw.table
            b, d, u = np.zeros(shape), np.zeros(shape), np.ones(shape)
            b[known], d[known], u[known] = t.Bel.values[idx], t.Dis.values[idx], t.Unc.values[idx]
            g = np.full(shape, -1, np.int16)
            g[known] = np.where(t.favourable.values[idx], 2, 1)
            codes.append(g)
            bl, dl, ul = b.ravel()[cells], d.ravel()[cells], u.ravel()[cells]
            bel, dis, unc, conflict = dempster(bel, dis, unc, bl, dl, ul)
            miss = int((base & ~known).sum())
            if miss:
                lw.messages.append(f"{miss:,} study-area cells without evidence (vacuous: Unc = 1)")
            fav = t.loc[t.Bel.idxmax()]
            rows.append(dict(layer=ev.name, classing=ev.classing, classes=len(t), deposits=lw.n_deposits,
                             missing_cells=miss, bel_max=float(t.Bel.max()), bel_contrast=float(t.Bel.max() - t.Bel.min()),
                             best_class=str(fav.label), best_class_area_pct=float(fav.area_pct),
                             mean_unc=float(ul.mean()), single_layer_auc=auc(bl, label),
                             conflict_when_added=float(conflict.mean()) if i else np.nan))
            m = ~known
            maps[ev.name] = dict(bel=np.ma.array(b, mask=m), dis=np.ma.array(d, mask=m), unc=np.ma.array(u, mask=m))
            layers.append(lw)
        pls = bel + unc
        score = {"bel": bel, "pls": pls, "mid": 0.5 * (bel + pls)}[self.score]

        def raster(v):
            out = np.zeros(shape)
            out.ravel()[cells] = v
            return np.ma.array(out, mask=~base)

        layer_table = pd.DataFrame(rows)
        from logic.data_driven.wofe.model import WofEModel
        pairwise = WofEModel._pairwise(np.stack(codes), dep, [ev.name for ev in evidence])
        dependent = pairwise[pairwise.p_value < 0.05] if len(pairwise) else pairwise
        if len(dependent):
            msgs.append("Dempster's rule assumes independent evidence; dependent pairs (χ², p < 0.05): " +
                        ", ".join(f"{a} – {b}" for a, b in zip(dependent.layer_1, dependent.layer_2)))
        stats = dict(layers=len(evidence), deposits=n_d, cells=int(cells.size), mean_bel=float(bel.mean()),
                     mean_dis=float(dis.mean()), mean_unc=float(unc.mean()), mean_pls=float(pls.mean()),
                     deposit_mean_bel=float(bel[label].mean()), deposit_mean_unc=float(unc[label].mean()),
                     auc_bel=auc(bel, label), auc_pls=auc(pls, label), auc_mid=auc(bel + pls, label),
                     dependent_pairs=int(len(dependent)))
        summary = pd.DataFrame([
            ("Evidence layers", ", ".join(ev.name for ev in evidence)),
            ("Training deposit cells", n_d), ("Study-area cells", int(cells.size)),
            ("Combination", "Dempster's rule (An et al. 1994)"),
            ("Missing evidence", self.missing), ("Classes without deposits", self.zero_deposit),
            ("Prospectivity score", self.score),
            ("Mean Bel / Dis / Unc / Pls", f"{stats['mean_bel']:.4f} / {stats['mean_dis']:.4f} / "
                                            f"{stats['mean_unc']:.4f} / {stats['mean_pls']:.4f}"),
            ("Mean Bel / Unc at training deposits", f"{stats['deposit_mean_bel']:.4f} / {stats['deposit_mean_unc']:.4f}"),
            ("Training AUC of Bel / Pls / mid-point", f"{stats['auc_bel']:.4f} / {stats['auc_pls']:.4f} / "
                                                      f"{stats['auc_mid']:.4f}")], columns=["item", "value"])
        return EBFResult(layers=layers, bel=raster(bel), dis=raster(dis), unc=raster(unc), pls=raster(pls),
                         score=raster(score), layer_maps=maps, summary=summary, layer_table=layer_table,
                         pairwise=pairwise, study_mask=base, n_deposits=n_d, stats=stats,
                         params=dict(missing=self.missing, zero_deposit=self.zero_deposit, score=self.score),
                         messages=msgs)
