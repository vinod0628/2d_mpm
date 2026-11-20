"""Relevance vector machine (RVM) for mineral prospectivity mapping — own implementation, no GUI.

Tipping (2001): sparse Bayesian learning. The deposit probability is a logistic function of a weighted sum of basis
functions, y(x) = σ(Σ_m w_m φ_m(x)), with φ = [1, K(x, x_1), …, K(x, x_N)] (kernel on the training samples) or
φ = [1, x_1 … x_p] (linear basis = sparse Bayesian logistic regression with automatic relevance determination).
Every weight has its own zero-mean Gaussian prior with precision α_m (hyper-parameter). Type-II maximum likelihood
(evidence maximisation) drives most α_m to infinity — the corresponding basis functions are pruned; the remaining
training samples are the *relevance vectors* (RVs), typically far fewer than the support vectors of an SVM.

Classification (Tipping 2001, §4): for fixed α, the posterior mode w_MP and covariance Σ are found by the Laplace
approximation (IRLS / Newton on the penalised log-likelihood)

    Σ = (Φᵀ B Φ + A)⁻¹,   B = diag(y_n (1 − y_n)),   A = diag(α)
    γ_m = 1 − α_m Σ_mm,   α_m ← γ_m / μ_m²                                   (MacKay 1992 re-estimation)

repeated until log α converges; α_m > 10⁶ → basis function m removed. Prediction (MacKay 1992):

    a = φ(x)ᵀ μ,   s² = φ(x)ᵀ Σ φ(x),   P(x) = σ(a / √(1 + π s² / 8))   (moderated output)

Outputs: score = mean moderated probability over R random non-deposit samples (depends on the deposit / non-deposit
ratio — not a probability per unit cell), score_std over the samples, predictive s.d. (Bayesian model uncertainty,
delta method σ(a)(1 − σ(a)) s), relevance vectors (points).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

KERNELS = ("rbf", "linear")
KERNEL_LABELS = {"rbf": N_("RBF (Gaussian) on the training samples"),
                 "linear": N_("linear (evidence columns, relevance determination)")}
GAMMA_FACTORS = tuple(2.0 ** k for k in range(-6, 5, 2))      # × γ0 = 1 / (p · mean column variance)
ALPHA_ACTIVE = 1e4        # convergence is judged on the weights with α below this (prior s.d. > 0.01)
ALPHA_MAX = 1e6           # weight s.d. ≤ 0.001 (scaled units): pruned — same model as 1e9, 3× faster
CHUNK = 100_000
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class RVMFit:
    """A fitted sparse Bayesian logistic model on a basis of ``keep`` columns."""
    keep: np.ndarray            # indices of the retained basis functions (0 = bias)
    mu: np.ndarray              # posterior mean of the retained weights
    sigma: np.ndarray           # posterior covariance
    alpha: np.ndarray           # hyper-parameters of the retained weights
    iterations: int
    converged: bool
    log_evidence: float


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _posterior(Phi, t, alpha, w, newton_iter=25):
    """Laplace approximation: posterior mode (Newton / IRLS with step halving) and covariance."""
    def objective(v):
        a = Phi @ v
        return float(np.sum(t * a - np.logaddexp(0.0, a)) - 0.5 * np.sum(alpha * v * v))
    f = objective(w)
    for _ in range(newton_iter):
        y = expit(Phi @ w)
        g = Phi.T @ (t - y) - alpha * w
        H = (Phi.T * (y * (1 - y))) @ Phi + np.diag(alpha)
        try:
            L = np.linalg.cholesky(H)
            step = np.linalg.solve(L.T, np.linalg.solve(L, g))
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        lam = 1.0
        while lam > 1e-6:
            w_new = w + lam * step
            f_new = objective(w_new)
            if f_new >= f - 1e-12:
                break
            lam /= 2
        w, f_old, f = w_new, f, f_new
        if abs(f - f_old) < 1e-8 * (1 + abs(f)) or np.max(np.abs(lam * step)) < 1e-8:
            break
    y = expit(Phi @ w)
    H = (Phi.T * (y * (1 - y))) @ Phi + np.diag(alpha)
    try:
        L = np.linalg.cholesky(H)
        Linv = np.linalg.inv(L)
        sigma = Linv.T @ Linv
        logdet_h = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        sigma = np.linalg.pinv(H)
        logdet_h = float(np.linalg.slogdet(H)[1])
    # Laplace log evidence (up to a constant): log p(t|w) − ½wᵀAw + ½ log|A| − ½ log|H|
    evidence = f + 0.5 * np.sum(np.log(alpha)) - 0.5 * logdet_h
    return w, sigma, float(evidence)


def fit_rvm(Phi: np.ndarray, t: np.ndarray, max_iter: int = 2000, tol: float = 1e-6, prune_bias: bool = False,
            alpha_init: float = 1.0) -> RVMFit:
    """Sparse Bayesian logistic regression on the basis matrix ``Phi`` (column 0 = bias): Tipping (2001) model,
    fast sequential marginal-likelihood maximisation of Tipping & Faul (2003).

    At the current posterior mode (Laplace approximation: B = diag(y(1 − y)), pseudo-targets t̂ = Φμ + B⁻¹(t − y)),
    for every candidate basis function m

        S_m = φ_mᵀBφ_m − φ_mᵀBΦΣΦᵀBφ_m,   Q_m = φ_mᵀBt̂ − φ_mᵀBΦΣΦᵀBt̂
        s_m, q_m = S_m, Q_m (not in the model)   or   α_m S_m/(α_m − S_m), α_m Q_m/(α_m − S_m) (in the model)
        θ_m = q_m² − s_m;   optimal α_m = s_m²/θ_m if θ_m > 0, else ∞

    and the change of the log marginal likelihood of adding (θ > 0, not in), re-estimating (θ > 0, in) or deleting
    (θ ≤ 0, in) it (Tipping & Faul 2003, eqs 27–29):

        add:        ½[(Q² − S)/S + ln(S/Q²)]
        re-estimate: ½[Q²/(S + 1/(1/α̃ − 1/α)) − ln(1 + S(1/α̃ − 1/α))]
        delete:     ½[Q²/(S − α) − ln(1 − S/α)]

    The single best action is applied, the posterior is updated, and the loop stops when no action increases the
    marginal likelihood by more than ``tol``. Starts from the bias alone; functions may re-enter after deletion.
    """
    t = np.asarray(t, float)
    n, m = Phi.shape
    keep = np.array([0])
    alpha = np.array([float(alpha_init)])
    w = np.zeros(1)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        P = Phi[:, keep]
        w, sigma, evidence = _posterior(P, t, alpha, w)
        y = expit(P @ w)
        b = np.maximum(y * (1 - y), 1e-10)
        that = P @ w + (t - y) / b
        BPhi = Phi * b[:, None]                          # N × M
        PtB = P.T @ BPhi                                 # |A| × M   (Φ_AᵀBΦ)
        S = np.einsum("ij,ij->j", Phi, BPhi) - np.einsum("ij,ik,kj->j", PtB, sigma, PtB)
        q0 = BPhi.T @ that                               # ΦᵀBt̂
        Q = q0 - PtB.T @ (sigma @ (P.T @ (b * that)))
        in_model = np.zeros(m, bool)
        in_model[keep] = True
        a_full = np.full(m, np.inf)
        a_full[keep] = alpha
        s_, q_ = S.copy(), Q.copy()
        k = in_model
        denom = np.maximum(a_full[k] - S[k], 1e-12)
        s_[k], q_[k] = a_full[k] * S[k] / denom, a_full[k] * Q[k] / denom
        theta = q_ * q_ - s_
        dL = np.full(m, -np.inf)
        new_alpha = np.full(m, np.inf)
        pos = theta > 1e-12
        new_alpha[pos] = s_[pos] ** 2 / theta[pos]
        add = pos & ~in_model
        dL[add] = 0.5 * ((Q[add] ** 2 - S[add]) / S[add] + np.log(S[add] / Q[add] ** 2))
        re = pos & in_model
        if re.any():
            d = 1.0 / new_alpha[re] - 1.0 / a_full[re]
            with np.errstate(divide="ignore", invalid="ignore"):
                dL[re] = 0.5 * (Q[re] ** 2 / (S[re] + 1.0 / np.where(d == 0, 1e-300, d)) - np.log1p(S[re] * d))
            dL[re] = np.where(np.isfinite(dL[re]), dL[re], 0.0)
        dele = ~pos & in_model
        if not prune_bias:
            dele[0] = False
        if dele.any() and in_model.sum() > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                dL[dele] = 0.5 * (Q[dele] ** 2 / (S[dele] - a_full[dele]) - np.log1p(-S[dele] / a_full[dele]))
            dL[dele] = np.where(np.isfinite(dL[dele]), dL[dele], -np.inf)
        best = int(np.argmax(dL))
        if not np.isfinite(dL[best]) or dL[best] < tol:
            converged = True
            break
        if add[best]:
            keep, alpha, w = np.r_[keep, best], np.r_[alpha, new_alpha[best]], np.r_[w, 0.0]
        elif re[best]:
            alpha[np.flatnonzero(keep == best)[0]] = new_alpha[best]
        else:
            i = np.flatnonzero(keep == best)[0]
            keep, alpha, w = np.delete(keep, i), np.delete(alpha, i), np.delete(w, i)
    P = Phi[:, keep]
    w, sigma, evidence = _posterior(P, t, alpha, w)
    order = np.argsort(keep)
    keep, w, alpha = keep[order], w[order], alpha[order]
    sigma = sigma[np.ix_(order, order)]
    return RVMFit(keep, w, sigma, alpha, it, converged, evidence)


def predict(fit: RVMFit, Phi_keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(moderated probability, linear predictor a, its s.d. s) for basis rows already restricted to ``fit.keep``."""
    a = Phi_keep @ fit.mu
    s2 = np.einsum("ij,jk,ik->i", Phi_keep, fit.sigma, Phi_keep)
    s2 = np.maximum(s2, 0.0)
    return expit(a / np.sqrt(1.0 + np.pi * s2 / 8.0)), a, np.sqrt(s2)


@dataclass
class RVMResult:
    score: np.ma.MaskedArray                  # mean moderated probability over the samples
    score_std: Optional[np.ma.MaskedArray]    # s.d. over the samples (repeats ≥ 2)
    pred_sd: np.ma.MaskedArray                # mean predictive (Bayesian) s.d. of the probability
    relevance: pd.DataFrame                   # relevance vectors of sample 1 (cell, class, weight, evidence)
    importance: pd.DataFrame
    tuning: pd.DataFrame
    sample_validation: pd.DataFrame
    summary: pd.DataFrame
    repeats: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    n_negatives: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


class RVMModel:
    """RVM prospectivity model (see module doc).

    kernel ("rbf" | "linear"), gamma (≤ 0 = auto γ0), tune (cross-validated choice of γ for the RBF kernel),
    scaling as the SVM (z-score with study-area statistics); training samples as the random forest / SVM;
    ``cv_folds`` ≥ 2 (stratified) for the cross-validated AUC, the confusion matrix and the permutation importance.
    """

    def __init__(self, kernel: str = "rbf", gamma: float = 0.0, tune: bool = True, max_iter: int = 2000,
                 neg_ratio: float = 1.0, min_distance: float = 0.0, min_spacing: float = -1.0,
                 res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, cv_folds: int = 5, missing: str = "exclude",
                 seed: int = 0, n_jobs: int = -1):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2")
        self.kernel, self.gamma, self.tune, self.max_iter = kernel, float(gamma), bool(tune), int(max_iter)
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed, self.n_jobs = missing, int(seed), int(n_jobs)

    # -------------------------------------------------------------- basis
    def basis(self, Z: np.ndarray, centres: Optional[np.ndarray], gamma: float) -> np.ndarray:
        """Basis rows [1, K(z, c_1..c_N)] (RBF) or [1, z] (linear)."""
        if self.kernel == "linear":
            return np.c_[np.ones(len(Z)), Z]
        d2 = (np.sum(Z * Z, 1)[:, None] + np.sum(centres * centres, 1)[None, :] - 2.0 * Z @ centres.T)
        return np.c_[np.ones(len(Z)), np.exp(-gamma * np.maximum(d2, 0.0))]

    def train(self, Z: np.ndarray, y: np.ndarray, gamma: float) -> RVMFit:
        return fit_rvm(self.basis(Z, Z, gamma), y, self.max_iter)

    def predict_rows(self, fit: RVMFit, Z: np.ndarray, centres: np.ndarray, gamma: float):
        if self.kernel == "linear":
            Phi = np.c_[np.ones(len(Z)), Z][:, fit.keep]
        else:
            rv = fit.keep[fit.keep > 0] - 1
            Phi_rv = self.basis(Z, centres[rv], gamma)                 # [1, K(z, rv)]
            cols = [0] + list(range(1, rv.size + 1)) if 0 in fit.keep else list(range(1, rv.size + 1))
            Phi = Phi_rv[:, cols]
        return predict(fit, Phi)

    def folds(self, y: np.ndarray, k: int, seed: int) -> np.ndarray:
        k = min(k, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(y.size, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        return fold

    def cross_validate(self, Z, y, gamma, fold):
        held = np.empty(y.size)
        models = []
        for f in np.unique(fold):
            te = fold == f
            fit = self.train(Z[~te], y[~te], gamma)
            held[te] = self.predict_rows(fit, Z[te], Z[~te], gamma)[0]
            models.append((te, fit, Z[~te]))
        return held, models

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> RVMResult:
        say = progress or (lambda f, m: None)
        say(0.02, N_("encoding evidence"))
        base = study_area(features, study_mask, self.missing)
        stack = FeatureStack(features, base, self.missing)
        msgs = list(stack.notes)
        dep_pos = np.unique(stack.positions(dep_rows, dep_cols))
        n_d = int(dep_pos.size)
        if n_d < 3:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 3.")
        m = int(round(self.neg_ratio * n_d))
        if m < 2:
            raise ValueError("The number of non-deposit samples must be at least 2.")
        if n_d + m > 3000 and self.kernel == "rbf":
            msgs.append(f"{n_d + m} training samples: the RBF basis has one function per sample — training is slow")
        kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
        cand = background_candidates(base, kr, kc, self.min_distance, self.res)
        cand_pos = np.searchsorted(stack.cells, np.flatnonzero(cand))
        spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
        cand_cells = stack.cells[cand_pos]
        width = base.shape[1]
        # scaling: z-scores with study-area statistics (identical for every random sample)
        p = stack.n_columns
        shift = np.array([float(stack.column(j).mean()) for j in range(p)])
        scale = np.array([float(stack.column(j).std()) or 1.0 for j in range(p)])
        gamma0 = 1.0 / p
        gamma = self.gamma if self.gamma > 0 else gamma0

        n = stack.n_cells
        p_sum, p_sq, sd_sum = np.zeros(n), np.zeros(n), np.zeros(n)
        rep_rows, tuning, first = [], pd.DataFrame(), None
        for r in range(self.repeats):
            frac = r / self.repeats
            step = 0.85 / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            Z = (stack.matrix(idx) - shift) / scale
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            fold = self.folds(y, self.cv_folds, self.seed + 7 + r)
            if r == 0 and self.tune and self.kernel == "rbf":
                say(0.05, N_("tuning the kernel width (cross-validation)"))
                rows = []
                for gf in GAMMA_FACTORS:
                    h, _ = self.cross_validate(Z, y, gamma0 * gf, fold)
                    rows.append(dict(gamma=gamma0 * gf, gamma_factor=gf, log2_gamma=np.log2(gamma0 * gf),
                                     cv_auc=_auc(h[y == 1], h[y == 0])))
                tuning = pd.DataFrame(rows)
                top = tuning.cv_auc.round(4).max()
                gamma = float(tuning[tuning.cv_auc.round(4) >= top].gamma.min())     # ties -> smoother kernel
                tuning["selected"] = tuning.gamma == gamma
            say(0.05 + 0.85 * frac + 0.3 * step, N_("cross-validating"))
            held, fold_models = self.cross_validate(Z, y, gamma, fold)
            say(0.05 + 0.85 * frac + 0.6 * step, N_("training the RVM"))
            fit = self.train(Z, y, gamma)
            say(0.05 + 0.85 * frac + 0.75 * step, N_("predicting the study area"))
            for s in range(0, n, CHUNK):
                ch = np.arange(s, min(n, s + CHUNK))
                prob, a, sd = self.predict_rows(fit, (stack.matrix(ch) - shift) / scale, Z, gamma)
                p_sum[ch] += prob
                p_sq[ch] += prob * prob
                sd_sum[ch] += expit(a) * (1 - expit(a)) * sd
            p_tr = self.predict_rows(fit, Z, Z, gamma)[0]
            n_rv = int((fit.keep > 0).sum())
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 relevance_vectors=n_rv,
                                 rv_deposits=int(y[fit.keep[fit.keep > 0] - 1].sum()) if self.kernel == "rbf" else None,
                                 iterations=fit.iterations, converged=fit.converged,
                                 training_auc=_auc(p_tr[y == 1], p_tr[y == 0]),
                                 cv_auc=_auc(held[y == 1], held[y == 0])))
            if first is None:
                first = dict(fit=fit, Z=Z, y=y, idx=idx, held=held, p_tr=p_tr, fold=fold, fold_models=fold_models)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        R = self.repeats
        if R > 1:
            msgs.append(f"{R} random non-deposit samples: one RVM each, scores averaged; tuning, importance and "
                        "relevance vectors refer to sample 1")
        if not first["fit"].converged:
            msgs.append(f"the evidence maximisation stopped after {self.max_iter} iterations without full convergence")
        score = p_sum / R
        std = np.sqrt(np.maximum(p_sq / R - score ** 2, 0.0)) if R > 1 else None
        pred_sd = sd_sum / R

        say(0.92, N_("variable importance"))
        imp = self._importance(first, stack, gamma)
        val = self._validation(first)
        rel = self._relevance_table(first, stack, base.shape)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        rep = pd.DataFrame(rep_rows)
        fit = first["fit"]
        st = dict(cv_auc=float(rep.cv_auc.iloc[0]), training_auc=float(rep.training_auc.iloc[0]),
                  relevance_vectors=int(rep.relevance_vectors.iloc[0]), gamma=gamma, gamma0=gamma0,
                  iterations=int(fit.iterations), log_evidence=fit.log_evidence, spacing=spacing,
                  samples=int(first["y"].size))
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws (repeats)", str(R)), ("Basis", self.kernel),
            ("γ (kernel width)", "—" if self.kernel == "linear" else
             f"{gamma:.4g} = {gamma / gamma0:.4g} × γ0" + (" (tuned)" if self.tune else "")),
            ("Relevance vectors (sample 1)", f"{st['relevance_vectors']} of {st['samples']} samples"
             + (f" ({int(rep.rv_deposits.iloc[0])} deposits)" if self.kernel == "rbf" else " (evidence columns)")),
            ("Bias retained", "yes" if 0 in fit.keep else "no"),
            ("Iterations / converged", f"{fit.iterations} / {'yes' if fit.converged else 'no'}"),
            ("Log evidence (Laplace, sample 1)", f"{fit.log_evidence:.3f}"),
            ("Evidence columns", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"),
            ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(kernel=self.kernel, gamma=gamma, tune=self.tune, max_iter=self.max_iter, neg_ratio=self.neg_ratio,
                      min_distance=self.min_distance, min_spacing=spacing, repeats=R, cv_folds=self.cv_folds,
                      missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return RVMResult(grid(score), None if std is None else grid(std), grid(pred_sd), rel, imp, tuning, val,
                         summary, rep, base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- tables
    def _relevance_table(self, first, stack: FeatureStack, shape) -> pd.DataFrame:
        fit, Z, y, idx = first["fit"], first["Z"], first["y"], first["idx"]
        if self.kernel == "linear":
            rows = [dict(term="(bias)" if k == 0 else stack.names[k - 1], weight=float(w), sd=float(np.sqrt(s)),
                         alpha=float(a)) for k, w, s, a in zip(fit.keep, fit.mu, np.diag(fit.sigma), fit.alpha)]
            return pd.DataFrame(rows)
        rows = []
        for k, w, a in zip(fit.keep, fit.mu, fit.alpha):
            if k == 0:
                continue
            cell = int(stack.cells[idx[k - 1]])
            row = dict(row=cell // shape[1], col=cell % shape[1], cls="deposit" if y[k - 1] == 1 else "non-deposit",
                       weight=float(w), alpha=float(a))
            raw = stack.matrix(np.array([idx[k - 1]]))[0]
            for j, name in enumerate(stack.names):
                row[name] = float(raw[j])
            rows.append(row)
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True) if rows else \
            pd.DataFrame(columns=["row", "col", "cls", "weight", "alpha"])

    def _importance(self, first, stack: FeatureStack, gamma) -> pd.DataFrame:
        """Permutation importance on the held-out folds (drop of the CV AUC, 5 shuffles); linear basis: also the
        posterior weight of every column (z-score units) and whether ARD kept it."""
        Z, y, held = first["Z"], first["y"], first["held"]
        base_auc = _auc(held[y == 1], held[y == 0])
        rng = np.random.default_rng(self.seed + 17)
        fit = first["fit"]
        rows = []
        for layer, cols in stack.groups.items():
            drops = []
            for _ in range(5):
                h = np.empty(y.size)
                for te, mdl, Ztr in first["fold_models"]:
                    Zt = Z[te].copy()
                    Zt[:, cols] = Zt[rng.permutation(Zt.shape[0])][:, cols]
                    h[te] = self.predict_rows(mdl, Zt, Ztr, gamma)[0]
                drops.append(base_auc - _auc(h[y == 1], h[y == 0]))
            row = dict(layer=layer, columns=len(cols), permutation_auc_drop=float(np.mean(drops)),
                       permutation_sd=float(np.std(drops)))
            if self.kernel == "linear":
                kept = [c for c in cols if c + 1 in fit.keep]
                pos = {k: i for i, k in enumerate(fit.keep)}
                row["relevant_columns"] = f"{len(kept)} / {len(cols)}"
                row["weights"] = "; ".join(f"{stack.names[c]} {fit.mu[pos[c + 1]]:+.3f}" for c in kept)[:200] or "—"
            rows.append(row)
        tab = pd.DataFrame(rows).sort_values("permutation_auc_drop", ascending=False).reset_index(drop=True)
        tab.insert(0, "rank", np.arange(1, len(tab) + 1))
        return tab

    @staticmethod
    def _validation(first) -> pd.DataFrame:
        y, held, p_tr = first["y"], first["held"], first["p_tr"]
        pos, neg = y == 1, y == 0

        def conf(p):
            pr = p >= 0.5
            return (int((neg & ~pr).sum()), int((neg & pr).sum()), int((pos & ~pr).sum()), int((pos & pr).sum()))
        tn, fp, fn, tp = conf(p_tr)
        ctn, cfp, cfn, ctp = conf(held)
        fold = first["fold"]
        accs = [float(((held[fold == f] >= 0.5) == pos[fold == f]).mean()) for f in np.unique(fold)]
        items = [("Samples (deposits / non-deposits)", f"{int(pos.sum())} / {int(neg.sum())}"),
                 ("Training confusion: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("Training confusion: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("Training accuracy", f"{(tn + tp) / y.size:.4f}"),
                 ("Training AUC", f"{_auc(p_tr[pos], p_tr[neg]):.4f}"),
                 ("Cross-validation", f"{int(np.unique(fold).size)}-fold (stratified)"),
                 ("CV confusion: true non-deposit → predicted non-deposit / deposit", f"{ctn} / {cfp}"),
                 ("CV confusion: true deposit → predicted non-deposit / deposit", f"{cfn} / {ctp}"),
                 ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                 ("Cross-validated AUC", f"{_auc(held[pos], held[neg]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])
