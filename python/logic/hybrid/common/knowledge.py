"""Mineral-system knowledge network shared by the hybrid methods (MS-KPRB, KP-RF, KW-SVM, BS-WofE, PT-Hurdle …).

Expert model of the user's hybrid documents ("Hybrid MPM — a proposed method" §5 and "Improved hybrid MPM methods"
§1): every evidence layer j is a proxy of one mineral-system process k(j) (source, pathway, trap, deposition,
preservation …) with an expert direction d_j (+1 high values favourable, −1 low values favourable), an expert weight
w_j inside its process and a weight v_k of its process.

    u_ij = d_j (x_ij − med_j) / IQR_j                       robust directed scaling (all study cells, no labels)
    μ_ij = σ(s (u_ij − c))                                    membership (shared s, c; categorical: expert score)
    F_ik = (Σ_{j∈k} w_j μ_ij^p)^{1/p}                        process score, OR-like power mean (p = 3)
    K_i  = Π_k F_ik^{v_k}                                     knowledge favourability, AND-like geometric mean

Missing proxies: w is renormalised over the proxies observed in the cell (cells missing a whole process get F = 0.5).
:class:`KnowledgeModel` (MS-KPRB) estimates per-layer s_j, c_j, w, v and a calibration α, β by MAP with priors
centred on the expert values (Dirichlet with pseudo-count κ for the weights, normal with s.d. τ for ln s and c).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit, logit, softmax

from logic.common.training import background_candidates

EPS = 1e-6


@dataclass
class ExpertLayer:
    """One evidence layer of the knowledge table."""
    name: str
    array: np.ma.MaskedArray
    process: str
    direction: int = 1                       # +1 / −1 (continuous)
    weight: float = 1.0                      # expert weight inside the process
    scores: Optional[Dict[float, float]] = None   # categorical layer: expert score (0–1) per class


@dataclass
class KnowledgeTable:
    layers: List[ExpertLayer]
    process_weights: Dict[str, float] = field(default_factory=dict)

    @property
    def processes(self) -> List[str]:
        return list(dict.fromkeys(l.process for l in self.layers))

    def index(self) -> np.ndarray:
        order = self.processes
        return np.array([order.index(l.process) for l in self.layers])

    def w0(self) -> np.ndarray:
        """Expert weights normalised within each process."""
        w = np.array([max(float(l.weight), 0.0) for l in self.layers])
        k = self.index()
        out = np.zeros_like(w)
        for p in np.unique(k):
            s = w[k == p].sum()
            out[k == p] = w[k == p] / s if s > 0 else 1.0 / (k == p).sum()
        return out

    def v0(self) -> np.ndarray:
        v = np.array([max(float(self.process_weights.get(p, 1.0)), 0.0) for p in self.processes])
        return v / v.sum() if v.sum() > 0 else np.full(len(v), 1.0 / len(v))

    def layer_weights(self) -> np.ndarray:
        """ω_j = w_j v_k(j), rescaled to mean 1 (importance of every layer in the expert model)."""
        om = self.w0() * self.v0()[self.index()]
        return om / om.mean()


def study_cells(table: KnowledgeTable, mask: Optional[np.ndarray]) -> np.ndarray:
    """Study area: the mask, or cells where at least one layer has data."""
    known = [~np.ma.getmaskarray(np.ma.asarray(l.array)) for l in table.layers]
    base = np.logical_or.reduce(known) if mask is None else np.asarray(mask, bool).copy()
    if not base.any():
        raise ValueError("The study area is empty.")
    return base


class KnowledgeNetwork:
    """Fixed expert network (no labels): matrices of the study cells, memberships, process scores, favourability."""

    def __init__(self, table: KnowledgeTable, study: np.ndarray, s: float = 1.5, c: float = 0.5, p: float = 3.0):
        if not table.layers:
            raise ValueError("At least one evidence layer is required.")
        self.table, self.study = table, np.asarray(study, bool)
        self.s, self.c, self.p = float(s), float(c), float(p)
        self.cells = np.flatnonzero(self.study)
        self.shape = self.study.shape
        n, m = self.cells.size, len(table.layers)
        self.X = np.full((n, m), np.nan)                     # raw values (NaN = missing)
        for j, l in enumerate(table.layers):
            a = np.ma.asarray(l.array)
            if a.shape != self.shape:
                raise ValueError(f"Evidence '{l.name}' is not on the same grid as the first layer.")
            v = np.ma.getdata(a).ravel()[self.cells].astype(float)
            ok = ~np.ma.getmaskarray(a).ravel()[self.cells] & np.isfinite(v)
            self.X[ok, j] = v[ok]
            if not ok.any():
                raise ValueError(f"Evidence '{l.name}' has no data inside the study area.")
        self.categorical = np.array([l.scores is not None for l in table.layers])
        med = np.nanmedian(self.X, 0)
        q75, q25 = np.nanpercentile(self.X, 75, 0), np.nanpercentile(self.X, 25, 0)
        iqr = q75 - q25
        self.med = np.where(self.categorical, 0.0, med)
        self.iqr = np.where((iqr > 0) & ~self.categorical, iqr, 1.0)
        self.direction = np.array([float(np.sign(l.direction) or 1.0) for l in table.layers])
        self.k = table.index()
        self.K = len(table.processes)
        self.w0, self.v0 = table.w0(), table.v0()

    # ------------------------------------------------------------------ pieces
    def u(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        X = self.X if X is None else X
        return self.direction * (X - self.med) / self.iqr

    def memberships(self, X=None, s=None, c=None) -> np.ndarray:
        """μ (NaN where missing); categorical layers use the expert class scores."""
        X = self.X if X is None else X
        s = self.s if s is None else s
        c = self.c if c is None else c
        mu = expit(np.asarray(s) * (self.u(X) - np.asarray(c)))
        for j, l in enumerate(self.table.layers):
            if l.scores is not None:
                col = np.full(X.shape[0], np.nan)
                ok = np.isfinite(X[:, j])
                col[ok] = 0.05                              # classes without an expert score: low favourability
                for cls, sc in l.scores.items():
                    col[ok & np.isclose(X[:, j], cls)] = sc
                mu[:, j] = col
        mu[~np.isfinite(X)] = np.nan
        return mu

    def process_scores(self, mu: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """F (n × K), power mean with weights renormalised over the observed proxies."""
        w = self.w0 if w is None else w
        F = np.empty((mu.shape[0], self.K))
        for kk in range(self.K):
            cols = self.k == kk
            m = mu[:, cols]
            ww = np.where(np.isfinite(m), w[cols], 0.0)
            tot = ww.sum(1)
            val = (np.nansum(ww * np.nan_to_num(m) ** self.p, 1) / np.where(tot > 0, tot, 1.0)) ** (1.0 / self.p)
            F[:, kk] = np.where(tot > 0, val, 0.5)
        return F

    def favourability(self, F: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
        v = self.v0 if v is None else v
        return np.exp(np.log(np.clip(F, EPS, 1.0)) @ v)

    def expert(self):
        """(μ, F, K) at the expert parameters — no deposits used."""
        mu = self.memberships()
        F = self.process_scores(mu)
        return mu, F, self.favourability(F)

    def features(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Robust-scaled directed values (categorical layers: expert class score) with missing = median (0) /
        score 0.5 — inputs for trees and kernels."""
        X = self.X if X is None else X
        U = np.nan_to_num(self.u(X), nan=0.0)
        if self.categorical.any():
            mu = self.memberships(X)
            for j in np.flatnonzero(self.categorical):
                U[:, j] = np.nan_to_num(mu[:, j], nan=0.5)
        return U

    def grid(self, v, dtype=float) -> np.ma.MaskedArray:
        out = np.zeros(self.shape, dtype)
        out.ravel()[self.cells] = v
        return np.ma.array(out, mask=~self.study)

    def positions(self, rows, cols) -> np.ndarray:
        flat = np.asarray(rows, np.int64) * self.shape[1] + np.asarray(cols, np.int64)
        pos = np.clip(np.searchsorted(self.cells, flat), 0, max(self.cells.size - 1, 0))
        return np.unique(pos[self.cells[pos] == flat])


# ============================================================================== MS-KPRB knowledge model
class KnowledgeModel:
    """Calibrated knowledge network with expert priors (MAP), after the user's ``kprb.py``.

    Parameters θ = (a_j = ln s_j, c_j, ω (softmax → w within processes), ν (softmax → v), α, b = ln β):
        η = α + β logit(K(x; s, c, w, v))
        J(θ) = Σ sw_i [ln(1 + e^η_i) − y_i η_i] + Σ_j [(ln s_j − ln s0)² + (c_j − c0)²] / (2τ²)
               − κ Σ_j w0_j ln w_j − κ Σ_k v0_k ln v_k
    """

    def __init__(self, net: KnowledgeNetwork, s0: float = 1.5, c0: float = 0.5, kappa: float = 10.0,
                 tau: float = 0.75):
        self.net, self.s0, self.c0, self.kappa, self.tau = net, float(s0), float(c0), float(kappa), float(tau)
        self.m, self.K = len(net.table.layers), net.K
        self.theta = self.theta0()

    def theta0(self) -> np.ndarray:
        return np.concatenate([np.full(self.m, np.log(self.s0)), np.full(self.m, self.c0),
                               np.log(self.net.w0 + EPS), np.log(self.net.v0 + EPS), [0.0, 0.0]])

    def unpack(self, th):
        m, K = self.m, self.K
        a, c, om, nu = th[:m], th[m:2 * m], th[2 * m:3 * m], th[3 * m:3 * m + K]
        w = np.empty(m)
        for kk in range(K):
            idx = self.net.k == kk
            w[idx] = softmax(om[idx])
        return np.exp(np.clip(a, -12, 12)), c, w, softmax(nu), th[-2], np.exp(np.clip(th[-1], -12, 12))

    def parts(self, X, th=None):
        s, c, w, v, alpha, beta = self.unpack(self.theta if th is None else th)
        mu = self.net.memberships(X, s, c)
        F = self.net.process_scores(mu, w)
        Kf = self.net.favourability(F, v)
        eta = alpha + beta * logit(np.clip(Kf, EPS, 1 - EPS))
        return mu, F, Kf, eta

    def objective(self, th, X, y, sw):
        eta = self.parts(X, th)[3]
        nll = np.sum(sw * (np.logaddexp(0, eta) - y * eta))
        s, c, w, v, _, _ = self.unpack(th)
        pen = (np.sum((np.log(s) - np.log(self.s0)) ** 2) + np.sum((c - self.c0) ** 2)) / (2 * self.tau ** 2)
        pen -= self.kappa * (np.sum(self.net.w0 * np.log(w + EPS)) + np.sum(self.net.v0 * np.log(v + EPS)))
        return nll + pen

    def value_grad(self, th, U, obs, catmu, y, sw):
        """J(θ) and its analytic gradient (user's document §5.12, extended to missing proxies: the weights are
        renormalised over the observed proxies of a cell, S_k = Σ_obs w_j μ_j^p / Σ_obs w_j).

        U: directed robust-scaled values (n × m, NaN-free), obs: observed mask, catmu: expert scores of the
        categorical columns (other columns ignored)."""
        net, m, K, p = self.net, self.m, self.K, self.net.p
        s, c, w, v, alpha, beta = self.unpack(th)
        cont = ~net.categorical
        mu = np.where(cont, expit(s * (U - c)), catmu)
        mup = np.where(obs, mu ** p, 0.0)
        n = U.shape[0]
        lnF = np.empty((n, K))
        Sk = np.empty((n, K))
        Wk = np.empty((n, K))
        for kk in range(K):
            cols = net.k == kk
            ww = np.where(obs[:, cols], w[cols], 0.0)
            tot = ww.sum(1)
            A = (ww * mup[:, cols]).sum(1)
            S = np.where(tot > 0, A / np.where(tot > 0, tot, 1.0), 0.5 ** p)
            Sk[:, kk], Wk[:, kk] = S, np.where(tot > 0, tot, np.inf)
            lnF[:, kk] = np.log(np.clip(S, EPS ** p, 1.0)) / p
        lnK = lnF @ v
        Kf = np.exp(lnK)
        Kc = np.clip(Kf, EPS, 1 - EPS)
        L = np.log(Kc) - np.log1p(-Kc)
        eta = alpha + beta * L
        nll = np.sum(sw * (np.logaddexp(0, eta) - y * eta))
        pen = (np.sum((np.log(s) - np.log(self.s0)) ** 2) + np.sum((c - self.c0) ** 2)) / (2 * self.tau ** 2)
        pen -= self.kappa * (np.sum(net.w0 * np.log(w + EPS)) + np.sum(net.v0 * np.log(v + EPS)))
        g = sw * (expit(eta) - y)
        h = np.where((Kf > EPS) & (Kf < 1 - EPS), g * beta / (1 - Kc), 0.0)        # dJ/d ln K
        grad = np.zeros_like(th)
        ga, gc, gom = np.zeros(m), np.zeros(m), np.zeros(m)
        for kk in range(K):
            cols = np.flatnonzero(net.k == kk)
            live = (Sk[:, kk] > EPS ** p) & np.isfinite(Wk[:, kk])
            hk = np.where(live, h * v[kk], 0.0)                                       # dJ/d ln F_k
            dS = hk / (p * np.where(live, Sk[:, kk], 1.0))                             # dJ/d S_k
            Gw = np.zeros(cols.size)
            for t, j in enumerate(cols):
                o = obs[:, j]
                dmu = np.where(o, dS * w[j] * p * mu[:, j] ** (p - 1) / Wk[:, kk], 0.0)  # dJ/d μ_j
                if cont[j]:
                    dm = mu[:, j] * (1 - mu[:, j])
                    ga[j] = np.sum(dmu * dm * s[j] * (U[:, j] - c[j]))
                    gc[j] = -np.sum(dmu * dm * s[j])
                Gw[t] = np.sum(np.where(o, dS * (mup[:, j] - Sk[:, kk]) / Wk[:, kk], 0.0))
            wk = w[cols]
            gom[cols] = wk * (Gw - np.sum(wk * Gw)) - self.kappa * (net.w0[cols] - wk)
        ga += (np.log(s) - np.log(self.s0)) / self.tau ** 2
        gc += (c - self.c0) / self.tau ** 2
        D = np.array([np.sum(h * lnF[:, kk]) for kk in range(K)])
        gnu = v * (D - np.sum(v * D)) - self.kappa * (net.v0 - v)
        grad[:m], grad[m:2 * m], grad[2 * m:3 * m], grad[3 * m:3 * m + K] = ga, gc, gom, gnu
        grad[-2] = np.sum(g)
        grad[-1] = beta * np.sum(g * L)
        return nll + pen, grad

    def _prepare(self, X):
        U = self.net.u(X)
        obs = np.isfinite(U)
        catmu = np.nan_to_num(self.net.memberships(X), nan=0.5)
        return np.nan_to_num(U), obs, catmu

    def fit(self, X, y, sample_weight=None, maxiter: int = 200):
        from scipy.optimize import minimize
        y = np.asarray(y, float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        th0 = self.theta0()
        th0[-2] = logit(np.clip(np.average(y, weights=sw), EPS, 1 - EPS))
        U, obs, catmu = self._prepare(X)
        res = minimize(self.value_grad, th0, args=(U, obs, catmu, y, sw), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        self.theta = res.x
        self.converged = bool(res.success)
        return self

    def table(self) -> List[dict]:
        s, c, w, v, alpha, beta = self.unpack(self.theta)
        rows = []
        for j, l in enumerate(self.net.table.layers):
            rows.append(dict(layer=l.name, process=l.process, direction=int(self.net.direction[j]),
                             expert_weight=float(self.net.w0[j]), fitted_weight=float(w[j]),
                             slope=float(s[j]) if not self.net.categorical[j] else None,
                             inflection=float(c[j]) if not self.net.categorical[j] else None))
        procs = [dict(process=p, expert_weight=float(self.net.v0[i]), fitted_weight=float(v[i]))
                 for i, p in enumerate(self.net.table.processes)]
        return rows, procs, float(alpha), float(beta)


# ============================================================================== residual boosting
def boost_residual(X, y, offset, n_trees=150, lr=0.05, depth=2, min_leaf=5, seed=0):
    """Logistic gradient boosting that starts from a fixed offset (the knowledge log-odds); Newton leaf values
    clipped to [−4, 4] (kprb.py)."""
    from sklearn.tree import DecisionTreeRegressor
    F = np.asarray(offset, float).copy()
    y = np.asarray(y, float)
    trees = []
    for t in range(n_trees):
        p = expit(F)
        r = y - p
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed + t).fit(X, r)
        leaf = tree.apply(X)
        num = np.bincount(leaf, weights=r, minlength=tree.tree_.node_count)
        den = np.bincount(leaf, weights=p * (1 - p), minlength=tree.tree_.node_count)
        gam = np.clip(num / np.maximum(den, 1e-3), -4, 4)
        F += lr * gam[leaf]
        trees.append((tree, gam))
    return trees


def boost_predict(trees, X, offset, lr=0.05, n_use=None):
    F = np.asarray(offset, float).copy()
    for tree, gam in trees[:n_use]:
        F += lr * gam[tree.apply(X)]
    return F


def grid_blocks(shape: Tuple[int, int], cells: np.ndarray, nb: int = 4) -> np.ndarray:
    """Spatial block id (nb × nb blocks) of study cells (flat indices)."""
    r, c = cells // shape[1], cells % shape[1]
    return (r * nb // shape[0]) * nb + c * nb // shape[1]


def draw_negatives(rng: np.random.Generator, cand: np.ndarray, n: int, K: Optional[np.ndarray] = None,
                   a: float = 1.0) -> np.ndarray:
    """n candidate positions without replacement; probability ∝ (1 − K)^a (knowledge-guided) or uniform."""
    n = min(n, cand.size)
    if K is None or a == 0:
        return rng.choice(cand, n, replace=False)
    pr = (1.0 - np.clip(K[cand], 0, 1)) ** a + 1e-9
    return rng.choice(cand, n, replace=False, p=pr / pr.sum())


def negative_candidates(net: KnowledgeNetwork, dep_rows, dep_cols, buffer_m: float, res) -> np.ndarray:
    """Study-cell positions at least ``buffer_m`` from every known deposit."""
    cand = background_candidates(net.study, dep_rows, dep_cols, buffer_m, res)
    return np.searchsorted(net.cells, np.flatnonzero(cand))


# ============================================================================== helpers of the improved methods M1–M5
def clogit(p) -> np.ndarray:
    return logit(np.clip(p, 1e-4, 1 - 1e-4))


def platt(score, y, sw=None) -> Tuple[float, float]:
    """Weighted one-feature logistic regression: logit P = a·score + b (LogisticRegression, C = 1000)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e3, max_iter=1000)
    lr.fit(np.asarray(score, float).reshape(-1, 1), np.asarray(y, int), sample_weight=sw)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def membership_matrix(net: KnowledgeNetwork) -> np.ndarray:
    """Expert memberships of the study cells, missing = 0.5 (neutral)."""
    return np.nan_to_num(net.expert()[0], nan=0.5)
