"""M3 BS-WofE — Bayesian-shrinkage fuzzy weights of evidence (user's "Improved hybrid MPM methods", §2.3).

Every layer is cut into Q classes of its expert membership μ (quantiles; categorical layers: one class per class).
Data weights of class c (n_c cells, n_cD deposits; n cells and N deposits with data):

    W^d = ln{ [(n_cD + ½)/(N + Q/2)] / [(n_c − n_cD + ½)/(n − N + Q/2)] },   s² = 1/(n_cD + ½) + 1/(n_c − n_cD + ½)

are shrunk toward expert prior weights W⁰ = γ ω_j (μ̄_c − μ̄_j) by normal–normal precision weighting

    W = (W^d/s² + κ W⁰) / (1/s² + κ)            (κ = prior precision; κ = 4 → prior s.d. 0.5)

so classes with few deposits keep the expert weight. Dependence between layers is deflated:
logit P = b0 + ν Σ_j W_j,c(i),  ν = 1/(1 + (m − 1) ρ̄)  (ρ̄ = mean |correlation| of the weight columns), with b0 such
that Σ P = N over the cells with data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from logic.hybrid.common.knowledge import KnowledgeNetwork


@dataclass
class BSWofEResult:
    prob: np.ndarray
    eta: np.ndarray
    eta_sd: np.ndarray
    expert: np.ndarray
    F: np.ndarray
    nu: float
    rho: float
    b0: float
    weights: pd.DataFrame
    layers: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def bs_wofe(net: KnowledgeNetwork, pos: np.ndarray, n_classes: int = 5, kappa: float = 4.0, gamma: float = 2.0,
            known: Optional[np.ndarray] = None) -> BSWofEResult:
    """known: optional boolean mask over the study cells used for counting (e.g. a permissive tract)."""
    mu_all, F, K = net.expert()
    n_cells, m = mu_all.shape
    known = np.ones(n_cells, bool) if known is None else np.asarray(known, bool)
    is_dep = np.zeros(n_cells, bool)
    is_dep[pos] = True
    om = net.table.layer_weights()
    contrib = np.zeros((n_cells, m))
    var = np.zeros((n_cells, m))
    rows, lrows, msgs = [], [], []
    for j, l in enumerate(net.table.layers):
        mu = mu_all[:, j]
        obs = np.isfinite(mu)
        kn = obs & known
        if net.categorical[j]:
            vals = np.unique(net.X[kn, j])
            cls = np.full(n_cells, -1)
            for i, v in enumerate(vals):
                cls[obs & np.isclose(net.X[:, j], v)] = i
            labels = [f"class {v:g}" for v in vals]
        else:
            edges = np.unique(np.quantile(mu[kn], np.linspace(0, 1, n_classes + 1)[1:-1]))
            cls = np.where(obs, np.searchsorted(edges, np.nan_to_num(mu)), -1)
            bounds = np.r_[0.0, edges, 1.0]
            labels = [f"μ {bounds[i]:.2f}–{bounds[i + 1]:.2f}" for i in range(edges.size + 1)]
        Q = len(labels)
        n, N = int(kn.sum()), int((kn & is_dep).sum())
        if N == 0:
            msgs.append(f"{l.name}: no training deposit has data — the layer keeps its expert prior weights")
        mbar_c = np.array([mu[kn & (cls == c)].mean() if (kn & (cls == c)).any() else np.nan for c in range(Q)])
        n_c = np.array([(kn & (cls == c)).sum() for c in range(Q)], float)
        n_cd = np.array([(kn & (cls == c) & is_dep).sum() for c in range(Q)], float)
        mbar = np.nansum(n_c / max(n, 1) * mbar_c)
        Wd = np.log(((n_cd + 0.5) / (N + 0.5 * Q)) / ((n_c - n_cd + 0.5) / (n - N + 0.5 * Q)))
        s2 = 1.0 / (n_cd + 0.5) + 1.0 / (n_c - n_cd + 0.5)
        W0 = gamma * om[j] * (np.nan_to_num(mbar_c, nan=mbar) - mbar)
        W = (Wd / s2 + kappa * W0) / (1.0 / s2 + kappa)
        Wvar = 1.0 / (1.0 / s2 + kappa)
        ok = cls >= 0
        contrib[ok, j] = W[cls[ok]]
        var[ok, j] = Wvar[cls[ok]]
        for c in range(Q):
            rows.append(dict(layer=l.name, process=l.process, cls=labels[c], mean_membership=float(mbar_c[c]),
                             cells=int(n_c[c]), deposits=int(n_cd[c]), data_weight=float(Wd[c]),
                             data_sd=float(np.sqrt(s2[c])), prior_weight=float(W0[c]), posterior_weight=float(W[c]),
                             posterior_sd=float(np.sqrt(Wvar[c])),
                             shrinkage=float(kappa / (1.0 / s2[c] + kappa))))
        # do the data weights run with or against the expert ordering?
        agree = np.corrcoef(Wd, W0)[0, 1] if Q > 2 and np.std(Wd) > 0 and np.std(W0) > 0 else np.nan
        lrows.append(dict(layer=l.name, process=l.process, classes=Q, deposits_with_data=N, layer_weight_omega=om[j],
                          weight_contrast=float(W.max() - W.min()), data_prior_correlation=float(agree),
                          flag="review the direction — data weights run against the expert ordering"
                          if agree == agree and agree < -0.3 else ""))
        if agree == agree and agree < -0.3:
            msgs.append(f"{l.name}: the data weights run against the expert ordering (r = {agree:.2f}) — a geologist "
                        "should review the direction (it is not reversed automatically)")
    K_ok = known
    if m > 1:
        C = contrib[K_ok]
        sd = C.std(0)
        live = sd > 0
        if live.sum() > 1:
            R = np.corrcoef(C[:, live].T)
            off = np.abs(R[~np.eye(R.shape[0], dtype=bool)])
            rho = float(np.nanmean(off))
        else:
            rho = 0.0
    else:
        rho = 0.0
    nu = 1.0 / (1.0 + (m - 1) * rho)
    S = nu * contrib.sum(1)
    N_all = int((known & is_dep).sum())
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid + S[K_ok]).sum() > N_all:
            hi = mid
        else:
            lo = mid
    b0 = (lo + hi) / 2
    eta = b0 + S
    eta_sd = nu * np.sqrt(var.sum(1))
    weights = pd.DataFrame(rows)
    layers = pd.DataFrame(lrows).sort_values("weight_contrast", ascending=False).reset_index(drop=True)
    st = dict(deposits=N_all, nu=nu, rho=rho, b0=b0, mean_prob=float(expit(eta[K_ok]).mean()))
    summary = pd.DataFrame([
        ("Training deposits (cells)", str(N_all)), ("Layers / classes per continuous layer", f"{m} / {n_classes}"),
        ("Prior precision κ (prior s.d.)", f"{kappa:g} ({1 / np.sqrt(kappa):.2f})" if kappa > 0 else "0 (data only)"),
        ("Prior scale γ", f"{gamma:g}"), ("Mean |correlation| of the weight columns ρ̄", f"{rho:.3f}"),
        ("Dependence deflation ν = 1/(1 + (m − 1)ρ̄)", f"{nu:.3f}"),
        ("Intercept b0 (Σ P = N)", f"{b0:.3f}"),
        ("Counting area", "permissive tract" if known.sum() < n_cells else "study area"),
    ], columns=["item", "value"])
    return BSWofEResult(expit(eta), eta, eta_sd, K, F, nu, rho, b0, weights, layers, summary, st, msgs)
