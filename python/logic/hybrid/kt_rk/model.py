"""M5 KT-RK — knowledge-trend regression-kriging of a continuous target (user's "Improved hybrid MPM methods", §2.5).

Target y at points (drill-hole accumulation / thickness / grade, or a geochemical sample value), log-transformed
z = ln(y + c), c = 0.1 × median of the positive values:

    z(x) = b0 + α b1 k(x) + s f_RF(x) + ε(x)

k = expert favourability (fixed knowledge network), b1 = max(cov(k, z)/var(k), 0) (sign-constrained), f_RF = random
forest on the trend residuals, α, s ∈ {0, ½, 1} chosen by cross-validation with the one-standard-error rule (the
simplest model whose RMSE is within one standard error of the best). The residual r = z − trend (out-of-bag forest
predictions) is kriged with an anisotropic Gaussian process: coordinates rotated to the expert strike and divided by
the expert ranges, C(h) = σ² exp(−‖h'‖²/(2ℓ²)) + τ² δ(h), with ℓ ∈ [0.3, 3] and σ², τ² fitted by marginal likelihood.
Prediction ŷ = exp(ẑ) − c (a median-type estimate, for ranking).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from logic.hybrid.common.knowledge import KnowledgeNetwork

CHUNK = 2_000        # kriging rows per block: 2 000 × n_obs × 8 B (≈ 55 MB for 3 500 observations)
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class KTRKResult:
    pred: np.ndarray            # ŷ per study cell
    log_pred: np.ndarray        # ẑ
    log_sd: np.ndarray          # GP s.d. of the residual (log scale)
    trend: np.ndarray           # b0 + α b1 k + s f_RF (log scale)
    expert: np.ndarray
    F: np.ndarray
    alpha: float
    shrink: float
    cv: pd.DataFrame
    observations: pd.DataFrame
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def rotate(coords: np.ndarray, strike_deg: float, range_along: float, range_across: float) -> np.ndarray:
    """Map coordinates -> (along strike, across strike) / ranges. Strike: azimuth in degrees clockwise from north."""
    t = np.deg2rad(strike_deg)
    along = np.array([np.sin(t), np.cos(t)])        # unit vector of the strike (x = east, y = north)
    across = np.array([np.cos(t), -np.sin(t)])
    return np.column_stack([coords @ along / range_along, coords @ across / range_across])


class Kriging:
    """Simple kriging of zero-mean residuals with the fitted covariance σ² exp(−d²/(2ℓ²)) + τ² δ (memory-lean:
    one cross-covariance block of at most CHUNK rows at a time; same result as sklearn's GP predict)."""

    def __init__(self, kernel, U_train: np.ndarray, r: np.ndarray):
        from scipy.linalg import cho_factor, cho_solve
        self.s2 = float(kernel.k1.k1.constant_value)
        self.ell = float(kernel.k1.k2.length_scale)
        self.tau2 = float(kernel.k2.noise_level)
        self.U = np.ascontiguousarray(U_train, float)
        Ktt = self._cov(self.U, self.U)
        Ktt[np.diag_indices_from(Ktt)] += self.tau2
        self.cf = cho_factor(Ktt, lower=True)
        self.w = cho_solve(self.cf, np.asarray(r, float))

    def _cov(self, A, B):
        d2 = np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * (A @ B.T)
        np.maximum(d2, 0.0, out=d2)
        d2 *= -0.5 / self.ell ** 2
        np.exp(d2, out=d2)
        d2 *= self.s2
        return d2

    def predict(self, U: np.ndarray, return_std: bool = False):
        from scipy.linalg import solve_triangular
        n = U.shape[0]
        mean = np.empty(n)
        sd = np.empty(n) if return_std else None
        for s in range(0, n, CHUNK):
            ch = slice(s, min(n, s + CHUNK))
            Ks = self._cov(U[ch], self.U)
            mean[ch] = Ks @ self.w
            if return_std:
                V = solve_triangular(self.cf[0], Ks.T, lower=True, check_finite=False)
                sd[ch] = np.sqrt(np.maximum(self.s2 + self.tau2 - np.einsum("ij,ij->j", V, V), 0.0))
        return (mean, sd) if return_std else mean


def _gp(scale_bounds, var):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    kern = (ConstantKernel(max(var, 1e-6), (1e-6, 1e3)) * RBF(1.0, scale_bounds)
            + WhiteKernel(max(0.3 * var, 1e-8), (1e-8, 1e2)))
    return GaussianProcessRegressor(kern, normalize_y=False)


def kt_rk(net: KnowledgeNetwork, obs_pos: np.ndarray, y_obs: np.ndarray, coords: np.ndarray, strike_deg: float = 0.0,
          range_along: float = 5000.0, range_across: float = 1500.0, c: Optional[float] = None,
          alpha_grid: Sequence[float] = (0, 0.5, 1), shrink_grid: Sequence[float] = (0, 0.5, 1),
          scale_bounds=(0.3, 3.0), n_folds: int = 10, n_trees: int = 200, min_leaf: int = 5,
          max_gp_fit: int = 1500, seed: int = 0, progress: ProgressFn = None) -> KTRKResult:
    """obs_pos: positions (in net.cells) of the observations; coords: map x, y of every study cell (n × 2)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    say = progress or (lambda f, m: None)
    y_obs = np.asarray(y_obs, float)
    ok = np.isfinite(y_obs) & (y_obs >= 0)
    msgs = []
    if (~ok).any():
        msgs.append(f"{int((~ok).sum())} observation(s) without a valid value (negative or missing) were skipped")
    obs_pos, y_obs = np.asarray(obs_pos)[ok], y_obs[ok]
    n_h = obs_pos.size
    if n_h < 10:
        raise ValueError(f"Only {n_h} observations inside the study area — need at least 10 (about 25–30 or more).")
    pos_vals = y_obs[y_obs > 0]
    c = float(0.1 * np.median(pos_vals)) if c is None or c <= 0 else float(c)
    if not c > 0:
        c = 1e-3
    z = np.log(y_obs + c)
    _, F, K = net.expert()
    k = K
    Xf = net.features()
    U_all = rotate(coords, strike_deg, range_along, range_across)
    rng = np.random.default_rng(seed)

    # one GP kernel for the cross-validation (fitted on the residuals of the full trend model)
    def trend_fit(idx, alpha, s, z_tr):
        kk = k[obs_pos[idx]]
        b1 = max(np.cov(kk, z_tr)[0, 1] / max(np.var(kk, ddof=1), 1e-12), 0.0) * alpha
        b0 = float(z_tr.mean() - b1 * kk.mean())
        res = z_tr - b0 - b1 * kk
        rf, oob = None, np.zeros(idx.size)
        if s > 0:
            rf = RandomForestRegressor(n_trees, min_samples_leaf=min_leaf, oob_score=True, n_jobs=-1,
                                       random_state=seed).fit(Xf[obs_pos[idx]], res)
            oob = rf.oob_prediction_
        return b0, b1, rf, res - s * oob

    say(0.05, "fitting the residual covariance")
    b0, b1, rf, r_full = trend_fit(np.arange(n_h), 1.0, 1.0, z)
    sub = rng.choice(n_h, min(n_h, max_gp_fit), replace=False)
    gp = _gp(scale_bounds, float(np.var(r_full))).fit(U_all[obs_pos[sub]], r_full[sub])
    kernel = gp.kernel_

    def krige(idx_tr, r_tr, U_te):
        return Kriging(kernel, U_all[obs_pos[idx_tr]], r_tr).predict(U_te)

    # cross-validation of (α, s)
    folds = KFold(min(n_folds, n_h), shuffle=True, random_state=seed)
    grid = [(a, s) for a in alpha_grid for s in shrink_grid]
    err = {g_: [] for g_ in grid}
    for f_i, (tr, te) in enumerate(folds.split(np.arange(n_h))):
        say(0.1 + 0.6 * f_i / folds.get_n_splits(), f"cross-validation fold {f_i + 1}")
        for a, s in grid:
            b0_, b1_, rf_, r_tr = trend_fit(tr, a, s, z[tr])
            zt = b0_ + b1_ * k[obs_pos[te]] + (s * rf_.predict(Xf[obs_pos[te]]) if rf_ is not None else 0.0)
            zt = zt + krige(tr, r_tr, U_all[obs_pos[te]])
            err[(a, s)].append(float(np.sqrt(np.mean((zt - z[te]) ** 2))))
    rows = [dict(alpha=a, shrink=s, cv_rmse=float(np.mean(e)), cv_rmse_sd=float(np.std(e, ddof=1)))
            for (a, s), e in err.items()]
    cv = pd.DataFrame(rows)
    best = cv.cv_rmse.idxmin()
    limit = cv.cv_rmse[best] + cv.cv_rmse_sd[best] / np.sqrt(folds.get_n_splits())
    cand = cv[cv.cv_rmse <= limit].assign(tot=lambda d: d.alpha + d.shrink).sort_values(["tot", "cv_rmse"])
    alpha, shrink = float(cand.alpha.iloc[0]), float(cand.shrink.iloc[0])
    cv["selected"] = (cv.alpha == alpha) & (cv.shrink == shrink)
    cv["within_one_se"] = cv.cv_rmse <= limit

    say(0.75, "final model and kriging")
    b0, b1, rf, r = trend_fit(np.arange(n_h), alpha, shrink, z)
    n = net.cells.size
    trend = b0 + b1 * k
    if rf is not None:
        for s in range(0, n, 200_000):
            ch = slice(s, min(n, s + 200_000))
            trend[ch] += shrink * rf.predict(Xf[ch])
    rk, sd = Kriging(kernel, U_all[obs_pos], r).predict(U_all, return_std=True)
    zhat = trend + rk
    pred = np.exp(zhat) - c
    loo = zhat[obs_pos]
    obs = pd.DataFrame(dict(value=y_obs, log_value=z, trend=trend[obs_pos], prediction_at_cell=pred[obs_pos],
                            residual_log=z - loo, expert_K=k[obs_pos]))
    kp = kernel.get_params()
    ell = float(kernel.k1.k2.length_scale)
    st = dict(observations=n_h, c=c, alpha=alpha, shrink=shrink, b0=b0, b1=b1, length_scale=ell,
              sill=float(kernel.k1.k1.constant_value), nugget=float(kernel.k2.noise_level),
              cv_rmse=float(cv.cv_rmse[cv.selected].iloc[0]),
              cv_rmse_mean_only=float(cv[(cv.alpha == 0) & (cv.shrink == 0)].cv_rmse.iloc[0])
              if ((cv.alpha == 0) & (cv.shrink == 0)).any() else float("nan"),
              corr_k_z=float(np.corrcoef(k[obs_pos], z)[0, 1]))
    del kp
    summary = pd.DataFrame([
        ("Observations", str(n_h)), ("Log offset c", f"{c:.4g}"),
        ("Correlation of expert K with ln(y + c)", f"{st['corr_k_z']:.3f}"),
        ("Selected α (knowledge trend) / s (forest)", f"{alpha:g} / {shrink:g} (one-SE rule)"),
        ("Trend b0 / α·b1", f"{b0:.3f} / {b1:.3f}"),
        ("CV RMSE (log) selected / mean + kriging only", f"{st['cv_rmse']:.3f} / {st['cv_rmse_mean_only']:.3f}"),
        ("Anisotropy: strike / ranges along × across", f"{strike_deg:g}° / {range_along:g} × {range_across:g}"),
        ("Kernel σ² / ℓ / nugget τ²", f"{st['sill']:.3g} / {ell:.3g} / {st['nugget']:.3g}"),
        ("Folds", f"{folds.get_n_splits()} (random over the observations)"),
    ], columns=["item", "value"])
    say(1.0, "done")
    return KTRKResult(pred, zhat, sd, trend, K, F, alpha, shrink, cv, obs, summary, st, msgs)
