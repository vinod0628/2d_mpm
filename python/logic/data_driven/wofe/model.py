"""Weights of Evidence (WofE) model — Bonham-Carter, Agterberg & Wright (1989); Bonham-Carter (1994).

Pure numpy implementation (no GUI). Typical use::

    ev = [Evidence("geology", geol_array, "categorical", labels),
          Evidence("dist_faults", dist_cls, "ascending")]
    model = WofEModel(unit_cell_km2=1.0, cell_area_km2=grid.cell_area / 1e6)
    res = model.fit_predict(ev, deposit_rows, deposit_cols, study_mask)
    res.posterior, res.posterior_std, res.total_std, res.layers[0].table, res.ci

Definitions (areas in unit cells, D = deposits, B = binary pattern)::

    W+ = ln[ P(B|D) / P(B|~D) ]        s²(W+) = 1/n(B∩D)  + 1/n(B∩~D)
    W- = ln[ P(~B|D) / P(~B|~D) ]      s²(W-) = 1/n(~B∩D) + 1/n(~B∩~D)
    C  = W+ - W-,  s(C) = sqrt(s²(W+) + s²(W-)),  studentized C = C / s(C)
    posterior logit = ln[D/(N-D)] + Σ W_k(cell)
    s²(P_post) ≈ P_post² · [1/D + Σ s²(W_k)]

Compared with ArcSDM (GTK, github.com/gtkfi/ArcSDM, master) — same formulas; deliberate differences:
  * one training point per cell (ArcSDM counts every point, i.e. duplicates in a cell count twice);
  * zero / all-deposit classes: weight 0 or the 0.5-deposit (Haldane) correction instead of ArcSDM's 0.99 / 0.001
    substitutions; cumulative types use the 0.5 correction;
  * cumulative threshold: largest C with Stud(C) ≥ minimum; if none qualifies the largest Stud(C) is used with a
    warning (ArcSDM leaves the table incomplete);
  * one evidence layer: s(P) uses the same formula (ArcSDM outputs W_STD itself);
  * Agterberg–Cheng s(T) is summed over unique conditions (as in the paper; ArcSDM groups by distinct s(P) value).
Adopted from ArcSDM ``missingdatavar_func.py`` / ``calculateresponse.py`` / ``agterbergchengci.py``: the
missing-data variance (Bonham-Carter 1994; Agterberg et al. 1990), total std = √(s²(P) + Var_MD), the
conditional-independence report (n/T, probability of conditional dependence, overall CI %).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from i18n import N_

WEIGHT_TYPES = ("categorical", "ascending", "descending")
WEIGHT_TYPE_LABELS = {
    "categorical": N_("Categorical (unique classes)"),
    "ascending": N_("Cumulative ascending (≤ threshold)"),
    "descending": N_("Cumulative descending (≥ threshold)"),
}
MAX_CATEGORICAL = 1000      # categorical: one weight per value; cumulative types have no limit (value = threshold)
CRITERIA = ("max_contrast", "max_studentized")
CRITERION_LABELS = {"max_contrast": N_("Maximum contrast C"), "max_studentized": N_("Maximum studentized contrast")}


# ============================================================================== core formulas
def binary_weights(n_bd, n_b, n_d, n_t, zero_deposit: str = "nan"):
    """Vectorised W+, s(W+), W-, s(W-), C, s(C) for binary patterns.

    n_bd: deposits in pattern; n_b: pattern area; n_d: total deposits; n_t: total area (unit cells).
    zero_deposit: "nan" -> undefined weights where a deposit count is 0;
                  "haldane" -> replace zero deposit counts by 0.5.
    """
    n_bd, n_b = np.asarray(n_bd, float), np.asarray(n_b, float)
    n_bd_ = n_bd.copy()
    n_nbd = n_d - n_bd
    if zero_deposit == "haldane":
        n_bd_ = np.where(n_bd_ == 0, 0.5, n_bd_)
        n_nbd = np.where(n_nbd == 0, 0.5, n_nbd)
    b_nd = n_b - n_bd                       # pattern area without deposits
    nb_nd = (n_t - n_b) - (n_d - n_bd)      # outside area without deposits
    nd_t = n_t - n_d
    with np.errstate(divide="ignore", invalid="ignore"):
        wp = np.log((n_bd_ / n_d) / (b_nd / nd_t))
        wm = np.log((n_nbd / n_d) / (nb_nd / nd_t))
        vwp = 1.0 / n_bd_ + 1.0 / b_nd
        vwm = 1.0 / n_nbd + 1.0 / nb_nd
    bad_p = ~((n_bd_ > 0) & (b_nd > 0))
    bad_m = ~((n_nbd > 0) & (nb_nd > 0))
    wp, vwp = np.where(bad_p, np.nan, wp), np.where(bad_p, np.nan, vwp)
    wm, vwm = np.where(bad_m, np.nan, wm), np.where(bad_m, np.nan, vwm)
    c = wp - wm
    sc = np.sqrt(vwp + vwm)
    return wp, np.sqrt(vwp), wm, np.sqrt(vwm), c, sc


def missing_data_variance(weights: np.ndarray, post: np.ndarray, missing: np.ndarray,
                          study: np.ndarray) -> np.ndarray:
    """Variance of the posterior probability due to missing evidence of one layer (Bonham-Carter 1994, ch. 9;
    ArcSDM ``missingdatavar_func``): for every class j of the layer, the posterior the cell would have if the
    evidence were known to be class j, weighted by the class's share of the layer's area::

        Var_MD = Σ_j (P_j − P)² · A_j / A,   O_j = O · exp(W_j)  (O = posterior odds with weight 0 for the layer)
    """
    var = np.zeros(post.shape)
    cells = missing & study
    if not cells.any():
        return var
    p = post[cells]
    odds = p / (1.0 - p)
    vals, counts = np.unique(weights[study & ~missing], return_counts=True)
    total = float(study.sum())                       # the layer's study area incl. its missing cells (ArcSDM)
    v = np.zeros(p.shape)
    for w, n in zip(vals, counts):
        o = odds * np.exp(w)
        v += (o / (1.0 + o) - p) ** 2 * (n / total)
    var[cells] = v
    return var


# ============================================================================== data classes
@dataclass
class Evidence:
    """One evidential layer. ``array`` holds class codes (masked = missing data)."""

    name: str
    array: np.ma.MaskedArray
    weight_type: str = "categorical"
    labels: Optional[Dict[float, str]] = None
    criterion: str = "max_contrast"        # cumulative types only
    min_studentized: float = 1.96          # cumulative types only (filter for max_contrast)

    def __post_init__(self):
        if self.weight_type not in WEIGHT_TYPES:
            raise ValueError(f"{self.name}: weight_type must be one of {WEIGHT_TYPES}")
        if self.criterion not in CRITERIA:
            raise ValueError(f"{self.name}: criterion must be one of {CRITERIA}")


@dataclass
class LayerWeights:
    name: str
    weight_type: str
    table: pd.DataFrame                    # per-class statistics
    class_weight: Dict[float, float]       # class value -> generalised weight
    class_std: Dict[float, float]          # class value -> s(weight)
    class_gen: Dict[float, int]            # class value -> generalised class code
    gen_table: pd.DataFrame                # generalised classes summary
    threshold: Optional[float] = None      # cumulative: selected class value
    contrast: float = np.nan
    studentized: float = np.nan
    n_deposits: int = 0
    area: float = 0.0
    messages: List[str] = field(default_factory=list)


@dataclass
class WofEResult:
    layers: List[LayerWeights]
    prior: float
    n_area: float                          # study area, unit cells
    n_deposits: int
    posterior: np.ma.MaskedArray
    posterior_std: np.ma.MaskedArray       # due to the weights
    confidence: np.ma.MaskedArray          # posterior / total std
    weight_rasters: Dict[str, np.ma.MaskedArray]
    ci: Dict[str, float]
    pairwise: pd.DataFrame
    study_mask: np.ndarray
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)
    md_variance: Optional[np.ma.MaskedArray] = None   # due to missing evidence (None: no missing evidence)
    total_std: Optional[np.ma.MaskedArray] = None     # √(posterior_std² + md_variance)

    def summary_table(self) -> pd.DataFrame:
        rows = [("Study area (unit cells)", f"{self.n_area:,.2f}"),
                ("Unit cell area (km²)", f"{self.params['unit_cell_km2']:g}"),
                ("Training deposits (cells)", self.n_deposits),
                ("Prior probability", f"{self.prior:.6f}"),
                ("Prior logit", f"{np.log(self.prior / (1 - self.prior)):.4f}")]
        for k, v in self.ci.items():
            rows.append((k, f"{v:.4f}" if isinstance(v, float) else v))
        return pd.DataFrame(rows, columns=["item", "value"])


# ============================================================================== model
class WofEModel:
    """Weights-of-evidence model.

    unit_cell_km2: unit cell area (ArcSDM default 1 km²); cell_area_km2: raster cell area.
    missing: "zero" -> missing evidence gets weight 0 (area kept) and adds the missing-data variance;
             "exclude" -> cells with any missing evidence are removed from the study area.
    zero_deposit: "zero" -> classes without deposits get weight 0;
                  "haldane" -> weights computed with 0.5 deposits (strongly negative instead of undefined).
    """

    def __init__(self, unit_cell_km2: float = 1.0, cell_area_km2: float = 1.0, missing: str = "zero",
                 zero_deposit: str = "zero"):
        if unit_cell_km2 <= 0 or cell_area_km2 <= 0:
            raise ValueError("Cell areas must be positive.")
        self.unit_cell_km2 = float(unit_cell_km2)
        self.cell_area_km2 = float(cell_area_km2)
        self.missing = missing
        self.zero_deposit = zero_deposit

    @property
    def cell_factor(self) -> float:
        """Unit cells per raster cell."""
        return self.cell_area_km2 / self.unit_cell_km2

    # -------------------------------------------------------------- weights for one layer
    def layer_weights(self, ev: Evidence, dep_mask: np.ndarray, area_mask: np.ndarray) -> LayerWeights:
        arr = np.ma.asarray(ev.array)
        known = area_mask & ~np.ma.getmaskarray(arr)
        vals = np.ma.getdata(arr)[known]
        dep = dep_mask[known]
        classes, inv = np.unique(vals, return_inverse=True)
        inv = inv.ravel()
        if classes.size < 2:
            raise ValueError(f"Evidence '{ev.name}' has fewer than 2 classes inside the study area.")
        if ev.weight_type == "categorical" and classes.size > MAX_CATEGORICAL:
            raise ValueError(f"Evidence '{ev.name}' has {classes.size:,} distinct values — too many for categorical "
                             "weights (one weight per value). Use it as cumulative ascending / descending, or "
                             "reclassify it (Data ▸ Reclassify raster).")
        npix = np.bincount(inv, minlength=classes.size).astype(float)
        ndep = np.bincount(inv, weights=dep.astype(float), minlength=classes.size)
        area = npix * self.cell_factor
        n_t, n_d = area.sum(), ndep.sum()
        msgs = []
        if n_d < 1:
            raise ValueError(f"Evidence '{ev.name}': no training deposits fall on known evidence.")
        if np.any(ndep > area):
            msgs.append("some classes have more deposits than unit cells — use a smaller unit cell")
        labels = ev.labels or {}
        lab = [labels.get(float(c), labels.get(c, f"{c:g}")) for c in classes]
        zd = "haldane" if self.zero_deposit == "haldane" else "nan"

        if ev.weight_type == "categorical":
            wp, swp, wm, swm, c, sc = binary_weights(ndep, area, n_d, n_t, zd)
            w = np.where(np.isfinite(wp), wp, 0.0)
            s = np.where(np.isfinite(swp), swp, 0.0)
            gen_code = {float(k): int(i + 1) for i, k in enumerate(classes)}
            table = pd.DataFrame(dict(
                cls=classes, label=lab, cells=npix.astype(int), area=area, deposits=ndep.astype(int),
                **{"W+": wp, "s(W+)": swp, "W-": wm, "s(W-)": swm, "C": c, "s(C)": sc, "Stud(C)": c / sc},
                gen_class=[gen_code[float(k)] for k in classes], weight=w, s_weight=s))
            rule = "0 (zero-deposit rule)" if self.zero_deposit == "zero" else "from 0.5 deposits"
            empty = classes[ndep == 0]
            if 0 < empty.size <= 10:
                msgs += [f"class {k:g}: no deposits -> weight {rule}" for k in empty]
            elif empty.size:
                msgs.append(f"{empty.size} of {classes.size} classes have no deposits -> weight {rule}; consider "
                            "reclassifying into fewer classes or a cumulative type")
            lw = LayerWeights(ev.name, ev.weight_type, table,
                              {float(k): float(v) for k, v in zip(classes, w)},
                              {float(k): float(v) for k, v in zip(classes, s)}, gen_code,
                              table[["gen_class", "label", "area", "deposits", "weight", "s_weight"]].copy(),
                              n_deposits=int(n_d), area=float(n_t), messages=msgs)
            return lw

        # ---- cumulative
        order = np.arange(classes.size) if ev.weight_type == "ascending" else np.arange(classes.size)[::-1]
        cum_area = np.cumsum(area[order])
        cum_dep = np.cumsum(ndep[order])
        # Cumulative patterns that hold all (or no) deposits would leave W-/W+ undefined, yet the
        # "all deposits inside" pattern is often the best one -> always use the 0.5-deposit correction here.
        wp, swp, wm, swm, c, sc = binary_weights(cum_dep, cum_area, n_d, n_t, "haldane")
        stud = c / sc
        stud[-1] = c[-1] = np.nan                  # last threshold = whole area
        # a pattern without deposits is no evidence (its W+ only comes from the 0.5-deposit correction)
        valid = np.isfinite(c) & np.isfinite(sc) & (cum_dep > 0)
        if not valid.any():
            raise ValueError(f"Evidence '{ev.name}': no valid threshold (check deposits / classes).")
        if ev.criterion == "max_studentized":
            idx = int(np.nanargmax(np.where(valid, stud, np.nan)))
        else:
            ok = valid & (stud >= ev.min_studentized)
            if ok.any():
                idx = int(np.nanargmax(np.where(ok, c, np.nan)))
            else:
                idx = int(np.nanargmax(np.where(valid, stud, np.nan)))
                msgs.append(f"no threshold reaches Stud(C) >= {ev.min_studentized:g}; "
                            f"used maximum Stud(C) = {stud[idx]:.2f}")
        if cum_dep[idx] in (0, n_d):
            msgs.append("selected pattern contains all deposits: W- uses the 0.5-deposit correction")
        sel = np.zeros(classes.size, bool)
        sel[order[: idx + 1]] = True
        wplus, wminus = float(wp[idx]), float(wm[idx])
        s_plus, s_minus = float(swp[idx]), float(swm[idx])
        gen = {float(k): (2 if sel[i] else 1) for i, k in enumerate(classes)}
        cw = {float(k): (wplus if sel[i] else wminus) for i, k in enumerate(classes)}
        cs = {float(k): (s_plus if sel[i] else s_minus) for i, k in enumerate(classes)}
        table = pd.DataFrame(dict(
            cls=classes[order], label=np.asarray(lab, object)[order], cells=npix[order].astype(int),
            area=area[order], deposits=ndep[order].astype(int), cum_area=cum_area, cum_deposits=cum_dep.astype(int),
            **{"W+": wp, "s(W+)": swp, "W-": wm, "s(W-)": swm, "C": c, "s(C)": sc, "Stud(C)": stud},
            selected=[i == idx for i in range(classes.size)],
            gen_class=[gen[float(k)] for k in classes[order]],
            weight=[cw[float(k)] for k in classes[order]], s_weight=[cs[float(k)] for k in classes[order]]))
        op = "≤" if ev.weight_type == "ascending" else "≥"
        thr = float(classes[order[idx]])
        thr_lab = f"{thr:g}" + (f" ({labels[thr]})" if thr in labels and labels[thr] != f"{thr:g}" else "")
        gen_table = pd.DataFrame(dict(
            gen_class=[2, 1], label=[f"pattern: class {op} {thr_lab}", "outside pattern"],
            area=[cum_area[idx], n_t - cum_area[idx]], deposits=[int(cum_dep[idx]), int(n_d - cum_dep[idx])],
            weight=[wplus, wminus], s_weight=[s_plus, s_minus]))
        return LayerWeights(ev.name, ev.weight_type, table, cw, cs, gen, gen_table, threshold=thr,
                            contrast=float(c[idx]), studentized=float(stud[idx]), n_deposits=int(n_d),
                            area=float(n_t), messages=msgs)

    # -------------------------------------------------------------- full model
    def fit_predict(self, evidence: Sequence[Evidence], dep_rows, dep_cols,
                    study_mask: Optional[np.ndarray] = None) -> WofEResult:
        if not evidence:
            raise ValueError("At least one evidence layer is required.")
        shape = np.ma.asarray(evidence[0].array).shape
        for ev in evidence:
            if np.ma.asarray(ev.array).shape != shape:
                raise ValueError(f"Evidence '{ev.name}' is not on the same grid as '{evidence[0].name}'.")
        known = [~np.ma.getmaskarray(np.ma.asarray(ev.array)) for ev in evidence]
        base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool).copy()
        if study_mask is None:
            base &= np.logical_or.reduce(known)
        if self.missing == "exclude":
            base &= np.logical_and.reduce(known)

        dep = np.zeros(shape, bool)
        dep[np.asarray(dep_rows, int), np.asarray(dep_cols, int)] = True
        dep &= base
        n_d = int(dep.sum())
        n_t = float(base.sum()) * self.cell_factor
        if n_d < 2:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 2.")
        msgs = []
        if n_d < 10:
            msgs.append(f"only {n_d} training deposits: weights have large variances")

        layers: List[LayerWeights] = []
        logit = np.full(shape, np.log(n_d / (n_t - n_d)))
        var_w = np.zeros(shape)
        gen_stack = np.full((len(evidence),) + shape, -1, dtype=np.int16)
        weight_rasters = {}
        w_list = []
        for i, ev in enumerate(evidence):
            lw = self.layer_weights(ev, dep, base)
            layers.append(lw)
            data = np.ma.getdata(np.ma.asarray(ev.array))
            kn = known[i] & base
            w = np.zeros(shape)
            s = np.zeros(shape)
            g = np.full(shape, -1, np.int16)
            vals = data[kn]
            keys = np.array(list(lw.class_weight.keys()))
            idx = np.searchsorted(keys, vals.astype(float))
            idx = np.clip(idx, 0, len(keys) - 1)
            hit = keys[idx] == vals
            wv = np.array([lw.class_weight[k] for k in keys])[idx]
            sv = np.array([lw.class_std[k] for k in keys])[idx]
            gv = np.array([lw.class_gen[k] for k in keys])[idx]
            tmp_w, tmp_s, tmp_g = np.where(hit, wv, 0.0), np.where(hit, sv, 0.0), np.where(hit, gv, -1)
            w[kn], s[kn], g[kn] = tmp_w, tmp_s, tmp_g
            logit += w
            var_w += s ** 2
            gen_stack[i] = g
            w_list.append(w)
            weight_rasters[ev.name] = np.ma.array(w, mask=~kn)

        post = 1.0 / (1.0 + np.exp(-logit))
        post_std = np.sqrt(post ** 2 * (1.0 / n_d + var_w))
        md_var = None
        if self.missing == "zero":
            for i, ev in enumerate(evidence):
                miss = base & ~known[i]
                if miss.any():
                    v = missing_data_variance(w_list[i], post, miss, base)
                    md_var = v if md_var is None else md_var + v
                    msgs.append(f"{ev.name}: {int(miss.sum()):,} cells without evidence (weight 0, missing-data "
                                "variance added to the total std)")
        total_std = post_std if md_var is None else np.sqrt(post_std ** 2 + md_var)
        conf = np.divide(post, total_std, out=np.zeros(shape), where=total_std > 0)
        m = ~base
        result = WofEResult(
            layers=layers, prior=n_d / n_t, n_area=n_t, n_deposits=n_d,
            posterior=np.ma.array(post, mask=m), posterior_std=np.ma.array(post_std, mask=m),
            confidence=np.ma.array(conf, mask=m), weight_rasters=weight_rasters,
            ci=self._ci_tests(post, post_std, gen_stack, base, n_d),
            pairwise=self._pairwise(gen_stack, dep, [ev.name for ev in evidence]),
            study_mask=base, messages=msgs,
            params=dict(unit_cell_km2=self.unit_cell_km2, cell_area_km2=self.cell_area_km2,
                        missing=self.missing, zero_deposit=self.zero_deposit),
            md_variance=None if md_var is None else np.ma.array(md_var, mask=m),
            total_std=np.ma.array(total_std, mask=m))
        if result.ci["CI ratio N(D)/T"] < 0.85:
            result.messages.append("CI ratio < 0.85: conditional independence is likely violated "
                                   "(posterior probabilities are inflated)")
        return result

    # -------------------------------------------------------------- tests
    def _ci_tests(self, post, post_std, gen_stack, base, n_d) -> Dict[str, float]:
        """Overall CI ratio (Bonham-Carter 1994) and Agterberg & Cheng (2002) one-tailed test; s(T) is summed over
        unique conditions (cells of one condition share P). Also reported as in ArcSDM's CI tool."""
        f = self.cell_factor
        codes = gen_stack[:, base].T                       # cells x layers
        uc, inv = np.unique(codes, axis=0, return_inverse=True)
        inv = inv.ravel()
        a_k = np.bincount(inv).astype(float) * f
        p_k = np.bincount(inv, weights=post[base]) / np.bincount(inv)
        s_k = np.bincount(inv, weights=post_std[base]) / np.bincount(inv)
        t = float((a_k * p_k).sum())
        s_t = float(np.sqrt(((a_k * s_k) ** 2).sum()))
        z = (t - n_d) / s_t if s_t > 0 else np.nan
        phi = float(stats.norm.cdf(z)) if np.isfinite(z) else np.nan
        p_dep = 100.0 * phi                                  # ArcSDM: probability of conditional dependence
        overall = 100.0 * (100.0 - p_dep) / 50.0 if p_dep >= 50.0 else 2.0 * p_dep
        return {N_("Observed deposits N(D)"): float(n_d), N_("Predicted deposits T"): t, N_("CI ratio N(D)/T"): n_d / t,
                "s(T)": s_t, "Agterberg-Cheng z": float(z),
                N_("Agterberg-Cheng p (one-tailed, T > N(D))"): 1.0 - phi,
                N_("Probability of conditional dependence (%)"): p_dep,
                N_("Overall CI (%)"): overall,
                N_("Unique conditions"): float(len(uc))}

    @staticmethod
    def _pairwise(gen_stack, dep, names) -> pd.DataFrame:
        """Pairwise chi-square test of conditional independence on deposit cells (generalised classes)."""
        rows = []
        dcodes = gen_stack[:, dep]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = dcodes[i], dcodes[j]
                ok = (a >= 0) & (b >= 0)
                a, b = a[ok], b[ok]
                ua, ub = np.unique(a), np.unique(b)
                if ua.size < 2 or ub.size < 2:
                    rows.append(dict(layer_1=names[i], layer_2=names[j], chi2=np.nan, dof=0, p_value=np.nan,
                                     min_expected=np.nan, n=int(ok.sum()), note="single class among deposits"))
                    continue
                tab = np.array([[np.sum((a == x) & (b == y)) for y in ub] for x in ua])
                chi2, p, dof, exp = stats.chi2_contingency(tab, correction=False)
                rows.append(dict(layer_1=names[i], layer_2=names[j], chi2=float(chi2), dof=int(dof), p_value=float(p),
                                 min_expected=float(exp.min()), n=int(ok.sum()),
                                 note="dependent (p<0.05)" if p < 0.05 else
                                 ("low expected counts" if exp.min() < 5 else "independent")))
        return pd.DataFrame(rows, columns=["layer_1", "layer_2", "chi2", "dof", "p_value", "min_expected", "n", "note"])
