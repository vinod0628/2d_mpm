"""Fuzzy weights of evidence (Cheng & Agterberg 1999; hybrid form of Porwal, Carranza & Hale 2006) and the modified
fuzzy weights of evidence (Agterberg 2011; Zhang, Agterberg, Cheng & Zuo 2014; Zhang & Agterberg 2018).

Every evidence layer is cut into ordered classes c; each class gets a fuzzy membership μ_c in the favourable set:

* **expert** (hybrid, knowledge-driven): categorical classes = expert class score; continuous classes = mean expert
  membership σ(1.5(u − 0.5)) of the class, u = d(x − median)/IQR (the knowledge network of the hybrid methods);
* **data** (data-driven): logistic of the class contrast C_c = W⁺_c − W⁻_c with range → 0.01 … 0.99.

The crisp favourable pattern B of a layer is the set of classes with the highest memberships, cut where the
studentized contrast of B is maximal (or at μ ≥ 0.5). The fuzzy weight of a class mixes the weights of B and B̄:

    W(μ) = ln{ [μ P(B|D) + (1 − μ) P(B̄|D)] / [μ P(B|D̄) + (1 − μ) P(B̄|D̄)] }       (μ = 1: W⁺, μ = 0: W⁻)

Classic posterior:  logit P = logit P(D) + Σ_j W_j(μ_j,c(i))       (conditional independence assumed)
Modified posterior: logit P = β0 + Σ_j β_j W_j(μ_j,c(i))         (β by logistic regression on all cells)

The modified model keeps the fuzzy weights as interpretable evidence scores but estimates their coefficients like a
logistic regression: Σ P = N exactly. β_j = 1 uses the fuzzy weights as they are; β_j > 1 when the memberships
understate the evidence (fuzzy weights mix W⁺ and W⁻, so memberships that do not span 0 … 1 compress the contrast);
a layer that repeats the information of another gets a smaller β than it would alone (dependence deflated).
Missing evidence: W = 0 (no information) in both models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from logic.hybrid.common.knowledge import KnowledgeNetwork

MEMBERSHIPS = ("expert", "data")
PATTERNS = ("max_contrast", "half")
MODELS = ("modified", "classic")


@dataclass
class FuzzyWofEResult:
    prob: np.ndarray                 # posterior of the selected model
    prob_classic: np.ndarray
    prob_modified: np.ndarray
    logit_sd: np.ndarray             # s.d. of the classic posterior logit (delta method)
    expert: np.ndarray
    F: np.ndarray
    model: str
    weights: pd.DataFrame            # per class
    layers: pd.DataFrame             # per layer: pattern, W+, W−, C, Stud(C), β
    ci_test: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def _classes(net: KnowledgeNetwork, j: int, mu: np.ndarray, n_classes: int):
    """Class index per cell (−1 = missing) and labels; continuous layers: quantiles of the raw values."""
    x = net.X[:, j]
    obs = np.isfinite(x)
    if net.categorical[j]:
        vals = np.unique(x[obs])
        cls = np.full(x.size, -1)
        for i, v in enumerate(vals):
            cls[obs & np.isclose(x, v)] = i
        return cls, [f"{v:g}" for v in vals]
    edges = np.unique(np.quantile(x[obs], np.linspace(0, 1, n_classes + 1)[1:-1]))
    cls = np.where(obs, np.searchsorted(edges, np.nan_to_num(x), side="right"), -1)
    b = np.r_[np.nanmin(x), edges, np.nanmax(x)]
    return cls, [f"{b[i]:.4g} – {b[i + 1]:.4g}" for i in range(edges.size + 1)]


def _cond(nBD, nB, N, n):
    """P(B|D), P(B̄|D), P(B|D̄), P(B̄|D̄) with ½ added to empty counts (Bayesian / Haldane correction)."""
    a, b = nBD, N - nBD
    c, d = nB - nBD, (n - nB) - (N - nBD)
    a, b, c, d = (np.where(v <= 0, 0.5, v).astype(float) for v in (a, b, c, d))
    return a / (a + b), b / (a + b), c / (c + d), d / (c + d), (a, b, c, d)


def fuzzy_weight(mu, pBD, pnBD, pBnD, pnBnD):
    return np.log((mu * pBD + (1 - mu) * pnBD) / (mu * pBnD + (1 - mu) * pnBnD))


def fuzzy_weight_var(mu, a, b, c, d):
    """Delta-method variance of W(μ) with Poisson counts a = n(B∩D), b = n(B̄∩D), c = n(B∩D̄), d = n(B̄∩D̄)
    (the assumption of the classical WofE variances, Bonham-Carter 1994):

        Var W(μ) = (μ²a + (1−μ)²b)/(μa + (1−μ)b)² + (μ²c + (1−μ)²d)/(μc + (1−μ)d)²

    μ = 1 gives Var(W⁺) = 1/a + 1/c, μ = 0 gives Var(W⁻) = 1/b + 1/d."""
    return ((mu ** 2 * a + (1 - mu) ** 2 * b) / (mu * a + (1 - mu) * b) ** 2
            + (mu ** 2 * c + (1 - mu) ** 2 * d) / (mu * c + (1 - mu) * d) ** 2)


def fit_modified(W: np.ndarray, y: np.ndarray, nonnegative: bool = True):
    """Modified fuzzy WofE: logit P = β0 + Σ β_j W_j by maximum likelihood over all cells (logistic regression).

    The fuzzy weights take one value per class, so the cells are grouped by their class combination (binomial
    likelihood of n cells with k deposits per group) — the same estimate as a logistic regression on every cell,
    much faster. nonnegative: β_j ≥ 0 (bounded L-BFGS-B), so no layer is used against its expert direction.
    Returns β0, β and the standard errors (observed information; for β_j = 0 on the bound: NaN)."""
    from scipy.optimize import minimize
    U, inv = np.unique(np.round(W, 10), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv).astype(float)
    k = np.bincount(inv, weights=y.astype(float))
    X = np.column_stack([np.ones(len(U)), U])
    p0 = k.sum() / n.sum()

    def nll(b):
        eta = X @ b
        mu = expit(eta)
        f = np.sum(n * np.logaddexp(0, eta) - k * eta)
        return f, X.T @ (n * mu - k)
    b0 = np.r_[np.log(p0 / (1 - p0)), np.ones(U.shape[1])]
    bounds = [(None, None)] + [((0.0, None) if nonnegative else (None, None))] * U.shape[1]
    res = minimize(nll, b0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "gtol": 1e-8})
    b = res.x
    mu = expit(X @ b)
    se = np.full(b.size, np.nan)
    free = np.r_[True, ~(nonnegative & (b[1:] <= 1e-8))]
    try:
        Xf = X[:, free]
        cov = np.linalg.inv((Xf * (n * mu * (1 - mu))[:, None]).T @ Xf)
        se[free] = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        pass
    return float(b[0]), b[1:].astype(float), se


def fuzzy_wofe(net: KnowledgeNetwork, pos: np.ndarray, membership: str = "expert", pattern: str = "max_contrast",
               model: str = "modified", n_classes: int = 10, min_deposits_pattern: int = 1,
               nonnegative: bool = True) -> FuzzyWofEResult:
    if membership not in MEMBERSHIPS or pattern not in PATTERNS or model not in MODELS:
        raise ValueError("unknown option")
    n_cells, m = net.cells.size, len(net.table.layers)
    N = int(pos.size)
    if N < 3:
        raise ValueError(f"Only {N} training deposit cell(s) — need at least 3.")
    is_dep = np.zeros(n_cells, bool)
    is_dep[pos] = True
    mu_cells, F, K = net.expert()
    Wcol = np.zeros((n_cells, m))
    Vcol = np.zeros((n_cells, m))
    rows, lrows, msgs = [], [], []
    for j, l in enumerate(net.table.layers):
        cls, labels = _classes(net, j, mu_cells[:, j], n_classes)
        obs = cls >= 0
        Q = len(labels)
        n = int(obs.sum())
        Nj = int((obs & is_dep).sum())
        n_c = np.bincount(cls[obs], minlength=Q).astype(float)
        d_c = np.bincount(cls[obs & is_dep], minlength=Q).astype(float)
        # class-wise crisp weights (class versus the rest) -> contrast, used by the data-driven membership
        pc = _cond(d_c, n_c, Nj, n)
        Wp, Wm = np.log(pc[0] / pc[2]), np.log(pc[1] / pc[3])
        Cc = Wp - Wm
        if membership == "expert":
            mu_c = np.array([np.nanmean(mu_cells[obs & (cls == c), j]) if n_c[c] else 0.5 for c in range(Q)])
        else:
            lo, hi = float(Cc.min()), float(Cc.max())
            mu_c = (expit(2 * np.log(99) / (hi - lo) * (Cc - (lo + hi) / 2)) if hi > lo else np.full(Q, 0.5))
        # crisp favourable pattern B: classes ordered by membership, cumulative from the most favourable
        order = np.argsort(-mu_c, kind="stable")
        best, best_s, cut = None, -np.inf, 1
        for k in range(1, Q):
            inB = order[:k]
            nB, nBD = n_c[inB].sum(), d_c[inB].sum()
            P = _cond(np.array([nBD]), np.array([nB]), Nj, n)
            a, b, c, d = (v[0] for v in P[4])
            C = np.log(P[0][0] / P[2][0]) - np.log(P[1][0] / P[3][0])
            s = C / np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            if pattern == "half":
                if mu_c[order[k - 1]] >= 0.5 and (k == Q - 1 or mu_c[order[k]] < 0.5):
                    best, best_s, cut = (P, C, s), s, k
            elif s > best_s and nBD >= min_deposits_pattern:
                best, best_s, cut = (P, C, s), s, k
        if best is None:                     # e.g. no class with μ ≥ 0.5: the single most favourable class
            k = 1
            nB, nBD = n_c[order[:1]].sum(), d_c[order[:1]].sum()
            P = _cond(np.array([nBD]), np.array([nB]), Nj, n)
            C = np.log(P[0][0] / P[2][0]) - np.log(P[1][0] / P[3][0])
            best, cut = (P, C, C / np.sqrt(sum(1 / v[0] for v in P[4]))), 1
        P, C, s = best
        pBD, pnBD, pBnD, pnBnD = (v[0] for v in P[:4])
        a, b, c, d = (v[0] for v in P[4])
        W = fuzzy_weight(mu_c, pBD, pnBD, pBnD, pnBnD)
        Wv = fuzzy_weight_var(mu_c, a, b, c, d)
        Wcol[obs, j] = W[cls[obs]]
        Vcol[obs, j] = Wv[cls[obs]]
        inB = set(order[:cut].tolist())
        for c_ in range(Q):
            rows.append(dict(layer=l.name, process=l.process, cls=labels[c_], membership=float(mu_c[c_]),
                             cells=int(n_c[c_]), deposits=int(d_c[c_]), in_pattern=c_ in inB,
                             class_contrast=float(Cc[c_]), fuzzy_weight=float(W[c_]),
                             fuzzy_weight_sd=float(np.sqrt(Wv[c_]))))
        Wplus, Wminus = float(np.log(pBD / pBnD)), float(np.log(pnBD / pnBnD))
        rho = np.corrcoef(mu_c, W)[0, 1] if Q > 2 and np.std(W) > 0 and np.std(mu_c) > 0 else np.nan
        lrows.append(dict(layer=l.name, process=l.process, classes=Q, pattern_classes=int(cut),
                          pattern_area_pct=float(100 * n_c[order[:cut]].sum() / max(n, 1)),
                          pattern_deposits=int(d_c[order[:cut]].sum()), w_plus=Wplus, w_minus=Wminus,
                          contrast=float(C), studentized_contrast=float(s),
                          membership_weight_corr=float(rho) if rho == rho else None))
        if C <= 0:
            msgs.append(f"{l.name}: the favourable pattern has no positive contrast (C = {C:.2f}) — the layer does "
                        "not separate deposits in the expert direction")
    # ---------------------------------------------------------------- classic posterior + CI test
    prior = N / n_cells
    eta_c = logit(prior) + Wcol.sum(1)
    p_c = expit(eta_c)
    var_eta = Vcol.sum(1)
    S = float(p_c.sum())
    var_S = float(np.sum((p_c * (1 - p_c)) ** 2 * var_eta))
    T = S - N
    z = T / np.sqrt(var_S) if var_S > 0 else np.nan
    # ---------------------------------------------------------------- modified posterior: β by logistic regression
    beta0, beta, se = fit_modified(Wcol, is_dep, nonnegative)
    eta_m = beta0 + Wcol @ beta
    p_m = expit(eta_m)
    for j, row in enumerate(lrows):
        row.update(beta=float(beta[j]), beta_se=float(se[j + 1]))
        if model != "modified":                 # the coefficients are only reported in the layer table
            continue
        if nonnegative and beta[j] <= 1e-6:
            msgs.append(f"{row['layer']}: modified coefficient β = 0 — the layer adds no information once the other "
                        "layers are accounted for (dependence), or its expert direction is not supported")
        elif beta[j] < 0:
            msgs.append(f"{row['layer']}: modified coefficient β < 0 — the layer's weights are used against the expert "
                        "direction once the other layers are accounted for (strong dependence or a wrong direction)")
    ratio = N / S if S > 0 else np.nan
    ci = pd.DataFrame([
        ("Observed training deposits N", f"{N}"),
        ("Predicted deposits S = Σ P (classic fuzzy WofE)", f"{S:.2f}"),
        ("Conditional-independence ratio N / S", f"{ratio:.3f}" + (
            "  (< 0.85: probabilities too high — dependent layers)" if ratio < 0.85 else
            "  (> 1.15: probabilities too low — memberships compress the evidence)" if ratio > 1.15 else "")),
        ("Agterberg–Cheng test: T = S − N, s.d.(T)", f"{T:.2f}, {np.sqrt(var_S):.2f}"),
        ("One-sided probability of T > 0 under independence", f"{norm.cdf(z):.4f}" if z == z else "—"),
        ("Predicted deposits Σ P (modified fuzzy WofE)", f"{p_m.sum():.2f}"),
    ], columns=["item", "value"])
    if ratio < 0.85 or ratio > 1.15:
        msgs.append(f"classic model: N/S = {ratio:.2f} — its probabilities are "
                    + ("too high (dependent layers)" if ratio < 0.85 else "too low (the memberships compress the evidence)")
                    + "; the modified model rescales the weights so that Σ P = N")
    sel = p_m if model == "modified" else p_c
    layers = pd.DataFrame(lrows)
    weights = pd.DataFrame(rows)
    st = dict(deposits=N, prior=prior, S_classic=S, ci_ratio=ratio, T=T, T_sd=float(np.sqrt(var_S)),
              ci_probability=float(norm.cdf(z)) if z == z else None, beta0=beta0,
              S_modified=float(p_m.sum()), mean_beta=float(np.mean(beta)))
    summary = pd.DataFrame([
        ("Training deposits (cells) / study cells", f"{N} / {n_cells:,}"),
        ("Prior probability per cell", f"{prior:.3g}"),
        ("Membership of the classes", "expert (knowledge table)" if membership == "expert" else
         "data (logistic of the class contrast)"),
        ("Favourable pattern B", "maximum studentized contrast" if pattern == "max_contrast" else "classes with μ ≥ 0.5"),
        ("Classes per continuous layer", str(n_classes)),
        ("Reported model", "modified fuzzy WofE (β by logistic regression)" if model == "modified" else
         "classic fuzzy WofE (Σ W, conditional independence)"),
        ("Modified model: β0 / mean β", f"{beta0:.3f} / {np.mean(beta):.3f}"
         + ("  (β ≥ 0)" if nonnegative else "  (β unconstrained)")),
        ("Conditional-independence ratio N / S", f"{ratio:.3f}"),
    ], columns=["item", "value"])
    return FuzzyWofEResult(sel, p_c, p_m, np.sqrt(var_eta), K, F, model, weights, layers, ci, summary, st, msgs)
