"""Data-calibrated fuzzy memberships (WofE / LR) combined by the expert inference network.

The data-driven fuzzy model of Porwal, Carranza & Hale (2003): the fuzzy membership of every evidence value is
derived from the known deposits instead of being assigned by the expert, and the memberships are combined by a
knowledge-based inference network (here the mineral-system network of the hybrid tools). The expert keeps the
structure — which layers, which process, which direction, how the processes combine — and the deposits set the
memberships.

Calibration of a layer (training deposit cells D among the n study cells where the layer has data, N of them):

* **WofE** — classes c (categorical classes or quantile classes of a continuous layer) with n_c cells and d_c deposit
  cells; class weight (multi-class weights of evidence, Bonham-Carter 1994; ½ for an empty count)

      W_c = ln[ (d_c / N) / ((n_c − d_c) / (n − N)) ],       s²(W_c) = 1/d_c + 1/(n_c − d_c)

  optionally made monotone along the expert direction (continuous layers) by weighted isotonic regression
  (pool-adjacent-violators, weights 1/s²; Ayer et al. 1955) — the expert fixes the shape, the data the size;
  a layer is reported as contradicting the expert when the weighted slope of its class weights is significantly
  against the expert direction (z < −1.96); a layer without a significant trend is flattened silently;
* **LR** — continuous layers: logistic regression of the deposit indicator on all study cells,
  logit P(x) = a + b·u(x) with u = d(x − median)/IQR the directed robust scaling of the knowledge network; the
  log-odds ratio against the prior W(x) = a + b·u(x) − logit(N/n) plays the role of the class weight; b < 0 (data
  against the expert direction) is set to 0 when the expert direction is kept. Categorical layers use the WofE
  class weights (a one-hot logistic regression gives the same class log-odds).

Membership: μ = σ(W) ("weight" scaling: W = 0 → 0.5 neutral, the strength of the layer is kept) or the weights of
the layer stretched logistically to 0.01 … 0.99 ("stretch", as the data memberships of fuzzy WofE). Optional
pooling with the expert memberships: μ = λ μ_data + (1 − λ) μ_expert, λ = N / (N + κ).

Combination: the expert inference network (OR-like power mean of the proxies of a process, AND-like weighted
geometric mean of the processes; missing proxies renormalised) or the fuzzy γ operator over all layers
(An, Moon & Rencz 1991; missing layers left out of the products).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from logic.hybrid.common.knowledge import KnowledgeNetwork
from logic.hybrid.fuzzy_wofe.model import _classes as layer_classes

CALIBRATIONS = ("wofe", "lr")
SCALINGS = ("weight", "stretch")
COMBINATIONS = ("network", "gamma")
MU_MIN, MU_MAX = 0.01, 0.99
Z_AGAINST = 1.96                  # a trend against the expert direction is reported when z < −1.96


@dataclass
class DCFuzzyResult:
    prob: np.ndarray                  # data-calibrated favourability (0 … 1)
    mu: np.ndarray                    # final memberships (n × m, NaN = missing)
    mu_data: np.ndarray
    expert: np.ndarray                # expert favourability K
    F: np.ndarray                     # process scores of the calibrated memberships
    memberships: pd.DataFrame
    layers: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


# ============================================================================== pieces
def class_weights(cls: np.ndarray, dep: np.ndarray, n_cls: int):
    """(cells, deposits, W, s(W)) per class; ``cls`` = class per observed cell, ``dep`` = deposit indicator."""
    cells = np.bincount(cls, minlength=n_cls).astype(float)
    deps = np.bincount(cls, weights=dep.astype(float), minlength=n_cls)
    N, n = deps.sum(), cells.sum()
    a = np.where(deps > 0, deps, 0.5)
    b = np.where(cells - deps > 0, cells - deps, 0.5)
    W = np.log((a / max(N, 0.5)) / (b / max(n - N, 0.5)))
    return cells, deps, W, np.sqrt(1.0 / a + 1.0 / b)


def isotonic(y: np.ndarray, w: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Weighted least-squares monotone fit (pool-adjacent-violators)."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if not increasing:
        return -isotonic(-y, w, True)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def trend_z(W: np.ndarray, sW: np.ndarray) -> float:
    """z of the weighted least-squares slope of the class weights against the class order (weights 1/s²)."""
    x = np.arange(W.size, dtype=float)
    w = 1.0 / sW ** 2
    xm = (w * x).sum() / w.sum()
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return 0.0
    slope = (w * (x - xm) * (W - (w * W).sum() / w.sum())).sum() / sxx
    return float(slope / np.sqrt(1.0 / sxx))


def logistic_1d(z: np.ndarray, y: np.ndarray, iters: int = 50):
    """Maximum-likelihood logit P = a + b z by Newton–Raphson; returns (a, b, s.e. of b)."""
    p0 = np.clip(y.mean(), 1e-9, 1 - 1e-9)
    th = np.array([np.log(p0 / (1 - p0)), 0.0])
    X = np.c_[np.ones(z.size), z]
    H = np.eye(2)
    for _ in range(iters):
        p = expit(X @ th)
        g = X.T @ (y - p)
        w = p * (1 - p)
        H = (X * w[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        th += step
        if np.abs(step).max() < 1e-8:
            break
    se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
    return float(th[0]), float(th[1]), se


def to_membership(W: np.ndarray, scaling: str, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """Weights (log-odds ratios) → memberships 0.01 … 0.99."""
    if scaling == "weight":
        return np.clip(expit(W), MU_MIN, MU_MAX)
    lo = np.nanmin(W) if lo is None else lo
    hi = np.nanmax(W) if hi is None else hi
    if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
        return np.full_like(W, 0.5, dtype=float)
    s = 2.0 * np.log(MU_MAX / MU_MIN) / (hi - lo)
    return np.clip(expit(s * (W - 0.5 * (lo + hi))), MU_MIN, MU_MAX)


def gamma_combine(mu: np.ndarray, gamma: float) -> np.ndarray:
    """Fuzzy γ over the observed layers of every cell: (1 − Π(1 − μ))^γ · (Π μ)^(1 − γ)."""
    if not 0 <= gamma <= 1:
        raise ValueError("γ must be between 0 and 1")
    obs = np.isfinite(mu)
    prod = np.prod(np.where(obs, mu, 1.0), 1)
    alg_sum = 1.0 - np.prod(np.where(obs, 1.0 - mu, 1.0), 1)
    out = alg_sum ** gamma * prod ** (1.0 - gamma)
    return np.where(obs.any(1), out, 0.5)


def _auc(score, y) -> float:
    from scipy.stats import rankdata
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ============================================================================== the method
def dc_fuzzy(net: KnowledgeNetwork, pos: np.ndarray, calibration: str = "wofe", scaling: str = "weight",
             monotone: bool = True, n_classes: int = 10, kappa: float = 0.0, combination: str = "network",
             gamma: float = 0.9) -> DCFuzzyResult:
    for name, v, ok in (("calibration", calibration, CALIBRATIONS), ("scaling", scaling, SCALINGS),
                        ("combination", combination, COMBINATIONS)):
        if v not in ok:
            raise ValueError(f"unknown {name} {v!r}")
    N_all = int(pos.size)
    if N_all < 3:
        raise ValueError(f"Only {N_all} training deposit cell(s) — need at least 3.")
    msgs: List[str] = []
    n, m = net.X.shape
    y = np.zeros(n)
    y[pos] = 1.0
    mu_exp, F_exp, K_exp = net.expert()
    mu_data = np.full((n, m), np.nan)
    lam = N_all / (N_all + kappa) if kappa > 0 else 1.0
    U = net.u()
    mem_rows, lay_rows = [], []
    for j, l in enumerate(net.table.layers):
        x = net.X[:, j]
        obs = np.isfinite(x)
        cls, labels = layer_classes(net, j, mu_exp[:, j], int(n_classes))
        k = len(labels)
        cells, deps, W, sW = class_weights(cls[obs], y[obs], k)
        N_j, n_j = deps.sum(), cells.sum()
        prior_lo = logit(np.clip(N_j / n_j, 1e-9, 1 - 1e-9))
        use_lr = calibration == "lr" and not net.categorical[j]
        Wm, info, b, se_b, z = W.copy(), "", np.nan, np.nan, np.nan
        if use_lr:
            a_, b, se_b = logistic_1d(U[obs, j], y[obs])
            z = b / se_b if se_b > 0 else 0.0
            if z < -Z_AGAINST:
                msgs.append(f"{l.name}: the deposits favour the opposite of the expert direction (slope {b:.3g}, "
                            f"z = {z:.1f})" + (" — the layer is set neutral (μ = 0.5)" if monotone else "") +
                            "; check its direction")
            b_use = 0.0 if (b < 0 and monotone) else b
            Wcell = a_ + b_use * U[obs, j] - prior_lo
            info = f"logistic regression, slope b = {b:.3g} ± {se_b:.2g} per IQR"
            lo, hi = np.nanmin(Wcell), np.nanmax(Wcell)
            mu_data[obs, j] = to_membership(Wcell, scaling, lo, hi)
            # class means for the table
            Wm = np.array([Wcell[cls[obs] == c].mean() if cells[c] else np.nan for c in range(k)])
        else:
            if not net.categorical[j] and k > 1:
                z = trend_z(W, sW) * net.direction[j]       # classes are in ascending raw value
                if z < -Z_AGAINST:
                    msgs.append(f"{l.name}: the class weights run against the expert direction (z = {z:.1f})" +
                                (" — the monotone fit flattens them" if monotone else "") + "; check its direction")
            if monotone and not net.categorical[j] and k > 1:
                Wm = isotonic(W, 1.0 / sW ** 2, increasing=net.direction[j] > 0)
            info = f"WofE, {k} classes" + (", monotone in the expert direction" if monotone and
                                            not net.categorical[j] and k > 1 else "")
            muc = to_membership(Wm, scaling)
            mu_data[obs, j] = muc[cls[obs]]
        mu_j = lam * mu_data[obs, j] + (1 - lam) * np.nan_to_num(mu_exp[obs, j], nan=0.5)
        auc = _auc(mu_data[obs, j], y[obs])
        for c in range(k):
            sel = cls[obs] == c
            mem_rows.append(dict(layer=l.name, cls=labels[c], cells=int(cells[c]), deposits=int(deps[c]),
                                 weight=float(W[c]), weight_sd=float(sW[c]),
                                 weight_used=float(Wm[c]) if np.isfinite(Wm[c]) else np.nan,
                                 mu_data=float(mu_data[obs, j][sel].mean()) if sel.any() else np.nan,
                                 mu_expert=float(np.nanmean(mu_exp[obs, j][sel])) if sel.any() else np.nan,
                                 mu=float(mu_j[sel].mean()) if sel.any() else np.nan))
        lay_rows.append(dict(layer=l.name, process=l.process, calibration=info,
                             contrast=float(np.nanmax(Wm) - np.nanmin(Wm)), slope=b, slope_se=se_b,
                             trend_z=float(z), agrees_with_expert=not (z < -Z_AGAINST), single_layer_auc=auc,
                             deposits=int(N_j)))
    mu = lam * mu_data + (1 - lam) * np.where(np.isfinite(mu_data), np.nan_to_num(mu_exp, nan=0.5), np.nan)
    F = net.process_scores(mu)                         # also reported for the γ operator (limiting process)
    prob = net.favourability(F) if combination == "network" else gamma_combine(mu, gamma)
    memberships = pd.DataFrame(mem_rows)
    layers = pd.DataFrame(lay_rows)
    st = dict(deposits=N_all, lam=lam, kappa=kappa, layers=m, mean_layer_auc=float(layers.single_layer_auc.mean()),
              against_expert=int((~layers.agrees_with_expert).sum()), training_auc=_auc(prob, y))
    comb = ("expert inference network (power mean p = 3 within a process, weighted geometric mean between)"
            if combination == "network" else f"fuzzy γ operator over all layers, γ = {gamma:g}")
    summary = pd.DataFrame([
        ("Calibration", "weights of evidence per class" if calibration == "wofe" else
         "logistic regression per continuous layer (categorical: class weights)"),
        ("Classes per continuous layer", str(int(n_classes)) + (" (table only)" if calibration == "lr" else "")),
        ("Monotone in the expert direction", "yes" if monotone else "no"),
        ("Membership scaling", "μ = σ(W) (layer strength kept)" if scaling == "weight"
         else "weights stretched to 0.01 … 0.99 per layer"),
        ("Pooling with the expert memberships", f"λ = N / (N + κ) = {lam:.3f} (κ = {kappa:g})" if kappa > 0
         else "none (data memberships only)"),
        ("Combination", comb),
        ("Training deposits (cells)", str(N_all)),
        ("Layers against the expert direction", str(st["against_expert"])),
        ("Training AUC (all cells)", f"{st['training_auc']:.4f}"),
    ], columns=["item", "value"])
    return DCFuzzyResult(prob, mu, mu_data, K_exp, F, memberships, layers, summary, st, msgs)
