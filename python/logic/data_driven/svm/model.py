"""Support vector machine (SVM) for mineral prospectivity mapping — scikit-learn SVC (LIBSVM), no GUI.

Theory (Vapnik 1995; Cortes & Vapnik 1995): the soft-margin SVM finds the hyperplane f(x) = Σ_i α_i y_i K(x_i, x) + b
in the feature space of a kernel K that separates deposit (y = +1) and non-deposit (y = −1) samples with the largest
margin; C trades margin width against training errors. In MPM (Zuo & Carranza 2011; Abedi et al. 2012;
Rodriguez-Galiano et al. 2015) the training data are deposit cells and random non-deposit cells away from known
deposits (Carranza et al. 2008); continuous evidence is scaled (SVMs are not scale-invariant), categorical evidence
enters as one-hot columns. ArcSDM's SVM tool (GTK) uses the same SVC with z-scores fitted on the training points,
C = 1, RBF kernel, γ = 1/p, leave-one-out accuracy and the decision value f(x) as output.

Workflow of :meth:`SVMModel.fit_predict` (per random non-deposit sample r = 1 … R):
    1. scale the evidence columns with study-area statistics (z-score or min–max), identical for every sample;
    2. (sample 1, optional) grid search of C and γ by stratified k-fold cross-validated AUC (Hsu et al. 2003);
    3. k-fold cross-validation with the chosen parameters → held-out decision values f_cv: CV AUC and confusion,
       Platt (1999) sigmoid P = 1 / (1 + exp(A f + B)) fitted to f_cv (algorithm of Lin, Lin & Weng 2007), and
       permutation importance (drop of the held-out AUC when one layer is shuffled, Breiman 2001);
    4. final SVM on all samples; f(x) and P(x) for every study cell.
Outputs: score = mean of P over the R samples (a favourability in [0, 1] that depends on the deposit / non-deposit
ratio — not a probability per unit cell), score_std = its s.d. over the samples (R ≥ 2), decision = mean f(x).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_
from logic.common.features import Feature, FeatureStack, study_area
from logic.common.training import auto_spacing, background_candidates, sample_positions

KERNELS = ("rbf", "linear", "poly", "sigmoid")
KERNEL_LABELS = {"rbf": N_("RBF (Gaussian)"), "linear": N_("linear"), "poly": N_("polynomial"),
                 "sigmoid": N_("sigmoid")}
SCALINGS = ("standard", "minmax", "none")
SCALING_LABELS = {"standard": N_("z-score (mean 0, s.d. 1)"), "minmax": N_("min–max [0, 1]"), "none": N_("none")}
CLASS_WEIGHTS = ("none", "balanced")
CLASS_WEIGHT_LABELS = {"none": N_("none (sample ratio)"), "balanced": N_("balanced (deposits weighted up)")}
C_GRID = tuple(2.0 ** k for k in range(-3, 10, 2))            # 0.125 … 512 (Hsu et al. 2003: exponential grid)
GAMMA_FACTORS = tuple(2.0 ** k for k in range(-6, 5, 2))      # × γ0 = 1 / (p · mean column variance)
TUNE_FOLDS = 5
LOO_MAX = 300             # leave-one-out = one SVM per sample: above this, 10-fold
CHUNK = 200_000
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class SVMResult:
    score: np.ma.MaskedArray                 # mean Platt probability over the samples
    score_std: Optional[np.ma.MaskedArray]   # s.d. over the samples (repeats ≥ 2)
    decision: np.ma.MaskedArray              # mean decision value f(x)
    tuning: pd.DataFrame                     # grid search (empty if not tuned)
    importance: pd.DataFrame                 # per evidence layer
    sample_validation: pd.DataFrame
    summary: pd.DataFrame
    repeats: pd.DataFrame
    study_mask: np.ndarray
    n_deposits: int
    n_negatives: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def platt_fit(f: np.ndarray, y: np.ndarray, max_iter: int = 100) -> Tuple[float, float]:
    """Platt (1999) sigmoid P(y=1|f) = 1 / (1 + exp(A·f + B)) — Newton method with backtracking of Lin, Lin & Weng
    (2007, Algorithm 1), with Platt's regularised targets t₊ = (N₊+1)/(N₊+2), t₋ = 1/(N₋+2)."""
    f = np.asarray(f, float)
    y = np.asarray(y) == 1
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    t = np.where(y, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    a, b = 0.0, float(np.log((n_neg + 1.0) / (n_pos + 1.0)))
    sigma, eps = 1e-12, 1e-5

    def nll(a_, b_):
        z = f * a_ + b_
        z = np.clip(z, -700, 700)
        # Σ t·z + log(1 + e^{−z}) for z ≥ 0, (t − 1)·z + log(1 + e^{z}) otherwise (stable form)
        return float(np.sum(np.where(z >= 0, t * z + np.log1p(np.exp(-z)), (t - 1) * z + np.log1p(np.exp(z)))))
    fval = nll(a, b)
    for _ in range(max_iter):
        p = platt_prob(f, a, b)                                                      # P = 1/(1+e^z)
        q = 1 - p
        d2 = p * q
        h11, h22, h21 = sigma + np.sum(f * f * d2), sigma + np.sum(d2), np.sum(f * d2)
        d1 = t - p
        g1, g2 = np.sum(f * d1), np.sum(d1)
        if abs(g1) < eps and abs(g2) < eps:
            break
        det = h11 * h22 - h21 * h21
        da, db = -(h22 * g1 - h21 * g2) / det, -(-h21 * g1 + h11 * g2) / det
        gd = g1 * da + g2 * db
        step = 1.0
        while step >= 1e-10:
            na, nb = a + step * da, b + step * db
            nf = nll(na, nb)
            if nf < fval + 1e-4 * step * gd:
                a, b, fval = na, nb, nf
                break
            step /= 2
        else:
            break
    return float(a), float(b)


def platt_prob(f: np.ndarray, a: float, b: float) -> np.ndarray:
    from scipy.special import expit
    return expit(-(np.asarray(f, float) * a + b))


class SVMModel:
    """SVM prospectivity model (see module doc).

    kernel ("rbf" | "linear" | "poly" | "sigmoid"), C (> 0), gamma (≤ 0 = auto γ0 = 1 / (p · mean column variance)),
    degree / coef0 (poly, sigmoid), class_weight ("none" | "balanced"), scaling ("standard" | "minmax" | "none"),
    tune (grid search of C and γ on sample 1); training samples as in the random forest: ``neg_ratio`` non-deposit
    cells per deposit at least ``min_distance`` map units from every known deposit and ``min_spacing`` apart
    (< 0 = auto, 0 = none), ``repeats`` random draws; ``cv_folds``: k ≥ 2 = stratified k-fold, -1 = leave-one-out
    (≤ 300 samples, else 10-fold) — used for the Platt calibration, the CV AUC and the importances.
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, gamma: float = 0.0, degree: int = 3, coef0: float = 0.0,
                 class_weight: str = "none", scaling: str = "standard", tune: bool = True, neg_ratio: float = 1.0,
                 min_distance: float = 0.0, min_spacing: float = -1.0, res: Tuple[float, float] = (1.0, 1.0),
                 repeats: int = 1, cv_folds: int = 5, missing: str = "exclude", seed: int = 0, n_jobs: int = -1):
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}")
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be one of {SCALINGS}")
        if class_weight not in CLASS_WEIGHTS:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTS}")
        if C <= 0:
            raise ValueError("C must be positive")
        if cv_folds in (0, 1) or cv_folds < -1:
            raise ValueError("cv_folds must be ≥ 2 or -1 (leave-one-out)")
        self.kernel, self.C, self.gamma = kernel, float(C), float(gamma)
        self.degree, self.coef0 = max(1, int(degree)), float(coef0)
        self.class_weight, self.scaling, self.tune = class_weight, scaling, bool(tune)
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed, self.n_jobs = missing, int(seed), int(n_jobs)

    # -------------------------------------------------------------- building blocks
    def svc(self, C: float, gamma: float):
        from sklearn.svm import SVC
        return SVC(C=C, kernel=self.kernel, gamma=gamma, degree=self.degree, coef0=self.coef0,
                   class_weight="balanced" if self.class_weight == "balanced" else None,
                   shrinking=True, tol=1e-3, cache_size=500, max_iter=-1)

    @staticmethod
    def scaler(stack: FeatureStack, how: str) -> Tuple[np.ndarray, np.ndarray]:
        """(shift, scale) per column from the study-area cells: x' = (x − shift) / scale."""
        p = stack.n_columns
        shift, scale = np.zeros(p), np.ones(p)
        for j in range(p):
            col = stack.column(j).astype(float)
            if how == "standard":
                sd = col.std()
                shift[j], scale[j] = col.mean(), sd if sd > 0 else 1.0
            elif how == "minmax":
                lo, hi = col.min(), col.max()
                shift[j], scale[j] = lo, (hi - lo) if hi > lo else 1.0
        return shift, scale

    def folds(self, y: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, str]:
        """Stratified fold id per sample; k < 0 → leave-one-out (≤ LOO_MAX samples, else 10-fold)."""
        n = y.size
        if k < 0 and n > LOO_MAX:
            k = 10
        if k < 0 or k >= n:
            return np.arange(n), "leave-one-out"
        k = min(k, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(n, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        name = f"{k}-fold (stratified)" + (f"; leave-one-out needs ≤ {LOO_MAX} samples" if self.cv_folds < 0 else "")
        return fold, name

    def cross_decision(self, X, y, fold, C, gamma, keep_models=False):
        """Held-out decision values of every sample (one SVM per fold)."""
        held = np.empty(y.size)
        models = []
        for f in np.unique(fold):
            te = fold == f
            m = self.svc(C, gamma).fit(X[~te], y[~te])
            held[te] = m.decision_function(X[te])
            if keep_models:
                models.append((te, m))
        return held, models

    def _grid(self, X, y, gamma0, seed):
        fold, _ = self.folds(y, TUNE_FOLDS, seed)
        gammas = [gamma0] if self.kernel == "linear" else [gamma0 * g for g in GAMMA_FACTORS]
        from joblib import Parallel, delayed

        def one(C, g):                                 # LIBSVM releases the GIL while fitting -> threads
            d, _ = self.cross_decision(X, y, fold, C, g)
            return dict(C=C, gamma=g, log2_C=np.log2(C), log2_gamma=np.log2(g), cv_auc=_auc(d[y == 1], d[y == 0]))
        jobs = self.n_jobs if self.n_jobs > 0 else -1
        rows = Parallel(n_jobs=jobs, prefer="threads")(delayed(one)(C, g) for C in C_GRID for g in gammas)
        tab = pd.DataFrame(rows)
        # best CV AUC (Hsu et al. 2003); ties at 4 decimals → the smoother model (smaller C, then smaller γ)
        top = tab.cv_auc.round(4).max()
        best = tab[tab.cv_auc.round(4) >= top].sort_values(["C", "gamma"]).iloc[0]
        tab["selected"] = (tab.C == best.C) & (tab.gamma == best.gamma)
        return float(best.C), float(best.gamma), tab

    def _decision_map(self, svc, stack: FeatureStack, shift, scale, progress=None):
        """f(x) for every study cell, in chunks (threads)."""
        from joblib import Parallel, delayed
        n = stack.n_cells
        starts = list(range(0, n, CHUNK))

        def run(s):
            idx = np.arange(s, min(n, s + CHUNK))
            return s, svc.decision_function((stack.matrix(idx) - shift) / scale)
        jobs = self.n_jobs if self.n_jobs > 0 else 4
        out = np.empty(n)
        for s, v in Parallel(n_jobs=min(jobs, max(1, len(starts))), prefer="threads")(delayed(run)(s) for s in starts):
            out[s: s + v.size] = v
        return out

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> SVMResult:
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
        kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
        cand = background_candidates(base, kr, kc, self.min_distance, self.res)
        cand_pos = np.searchsorted(stack.cells, np.flatnonzero(cand))
        spacing = auto_spacing(cand_pos.size, m, self.res) if self.min_spacing < 0 else self.min_spacing
        cand_cells = stack.cells[cand_pos]
        width = base.shape[1]
        if n_d < 20:
            msgs.append(f"only {n_d} training deposits: the decision boundary and the calibration are uncertain — "
                        "use several repeats and look at the cross-validated AUC")
        shift, scale = self.scaler(stack, self.scaling)
        p = stack.n_columns
        col_var = np.array([((stack.column(j) - shift[j]) / scale[j]).var() for j in range(p)])
        gamma0 = 1.0 / (p * max(float(col_var.mean()), 1e-12))
        C, gamma = self.C, (self.gamma if self.gamma > 0 else gamma0)

        n = stack.n_cells
        p_sum, p_sq, f_sum = np.zeros(n), np.zeros(n), np.zeros(n)
        rep_rows, tuning, first = [], pd.DataFrame(), None
        for r in range(self.repeats):
            frac = r / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            X = (stack.matrix(idx) - shift) / scale
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            if r == 0 and self.tune:
                say(0.05, N_("tuning C and γ (cross-validation)"))
                C, gamma, tuning = self._grid(X, y, gamma0, self.seed + 11)
            say(0.05 + 0.8 * (frac + 0.3 / self.repeats), N_("cross-validating"))
            fold, cv_name = self.folds(y, self.cv_folds, self.seed + 7 + r)
            d_cv, fold_models = self.cross_decision(X, y, fold, C, gamma, keep_models=(r == 0))
            a, b = platt_fit(d_cv, y)
            cv_auc = _auc(d_cv[y == 1], d_cv[y == 0])
            say(0.05 + 0.8 * (frac + 0.6 / self.repeats), N_("training the SVM"))
            svc = self.svc(C, gamma).fit(X, y)
            d_tr = svc.decision_function(X)
            if a >= 0:            # held-out f carries no information -> the sigmoid would reverse the ranking of f
                a, b = platt_fit(d_tr, y)
                a = min(a, -1e-6)
                if r == 0 or not any("Platt" in x for x in msgs):
                    msgs.append(f"sample {r + 1}: the cross-validated decision values do not separate deposits "
                                f"(CV AUC {cv_auc:.2f}); the Platt sigmoid was fitted to the training decision values "
                                "— the score keeps the ranking of f(x) but is over-confident")
            say(0.05 + 0.8 * (frac + 0.7 / self.repeats), N_("predicting the study area"))
            f = self._decision_map(svc, stack, shift, scale)
            prob = platt_prob(f, a, b)
            p_sum += prob
            p_sq += prob * prob
            f_sum += f
            n_sv = svc.n_support_                                  # per class: [non-deposit, deposit]
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 support_vectors=int(n_sv.sum()), sv_deposits=int(n_sv[1]),
                                 training_auc=_auc(d_tr[y == 1], d_tr[y == 0]), cv_auc=cv_auc, platt_A=a, platt_B=b))
            if first is None:
                first = dict(svc=svc, X=X, y=y, d_cv=d_cv, d_tr=d_tr, fold=fold, cv_name=cv_name,
                             fold_models=fold_models, a=a, b=b)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        if self.repeats > 1:
            msgs.append(f"{self.repeats} random non-deposit samples: one SVM each, scores averaged; tuning, "
                        "importance and validation refer to sample 1")
        R = self.repeats
        score = p_sum / R
        std = np.sqrt(np.maximum(p_sq / R - score ** 2, 0.0)) if R > 1 else None
        dec = f_sum / R

        say(0.88, N_("variable importance"))
        imp = self._importance(first, stack)
        val = self._validation(first, C)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        rep = pd.DataFrame(rep_rows)
        st = dict(cv_auc=float(rep.cv_auc.iloc[0]), training_auc=float(rep.training_auc.iloc[0]),
                  support_vectors=int(rep.support_vectors.iloc[0]), C=C, gamma=gamma, gamma0=gamma0,
                  platt_A=first["a"], platt_B=first["b"], spacing=spacing)
        bounded = val.set_index("item")["value"].get("Bounded support vectors (α = C)")
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws (repeats)", str(R)), ("Kernel", self.kernel),
            ("C (penalty)", f"{C:g}" + (" (tuned)" if self.tune else "")),
            ("γ (kernel width)", "—" if self.kernel == "linear" else
             f"{gamma:.4g} = {gamma / gamma0:.4g} × γ0" + (" (tuned)" if self.tune else "")),
            ("γ0 = 1 / (p · mean column variance)", f"{gamma0:.4g}"),
            ("Degree / coef0", f"{self.degree} / {self.coef0:g}" if self.kernel in ("poly", "sigmoid") else "—"),
            ("Class weight", self.class_weight), ("Scaling (study-area statistics)", self.scaling),
            ("Evidence columns", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Support vectors (sample 1)", f"{st['support_vectors']} ({int(rep.sv_deposits.iloc[0])} deposits)"
             + (f"; bounded {bounded}" if bounded else "")),
            ("Platt sigmoid A / B (sample 1)", f"{first['a']:.4g} / {first['b']:.4g}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"), ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(kernel=self.kernel, C=C, gamma=gamma, degree=self.degree, coef0=self.coef0,
                      class_weight=self.class_weight, scaling=self.scaling, tune=self.tune, neg_ratio=self.neg_ratio,
                      min_distance=self.min_distance, min_spacing=spacing, repeats=R, cv_folds=self.cv_folds,
                      missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return SVMResult(grid(score), None if std is None else grid(std), grid(dec), tuning, imp, val, summary, rep,
                         base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- importance / validation
    def _importance(self, first, stack: FeatureStack) -> pd.DataFrame:
        """Permutation importance on the held-out folds: drop of the cross-validated AUC when the values of one
        evidence layer (all its one-hot columns) are shuffled; 5 shuffles, mean ± s.d."""
        X, y, models = first["X"], first["y"], first["fold_models"]
        base_auc = _auc(first["d_cv"][y == 1], first["d_cv"][y == 0])
        rng = np.random.default_rng(self.seed + 17)
        rows = []
        lin = first["svc"].coef_.ravel() if self.kernel == "linear" else None
        for layer, cols in stack.groups.items():
            drops = []
            for _ in range(5):
                held = np.empty(y.size)
                for te, mdl in models:
                    Xt = X[te].copy()
                    Xt[:, cols] = Xt[rng.permutation(Xt.shape[0])][:, cols]
                    held[te] = mdl.decision_function(Xt)
                drops.append(base_auc - _auc(held[y == 1], held[y == 0]))
            row = dict(layer=layer, columns=len(cols), permutation_auc_drop=float(np.mean(drops)),
                       permutation_sd=float(np.std(drops)))
            if lin is not None:
                row["linear_weight"] = "; ".join(f"{stack.names[c]} {lin[c]:+.3f}" for c in cols)[:200]
            rows.append(row)
        tab = pd.DataFrame(rows).sort_values("permutation_auc_drop", ascending=False).reset_index(drop=True)
        tab.insert(0, "rank", np.arange(1, len(tab) + 1))
        return tab

    def _validation(self, first, C) -> pd.DataFrame:
        y, d_cv, d_tr, svc = first["y"], first["d_cv"], first["d_tr"], first["svc"]
        pos, neg = y == 1, y == 0

        def conf(d):
            pr = d >= 0
            return (int((neg & ~pr).sum()), int((neg & pr).sum()), int((pos & ~pr).sum()), int((pos & pr).sum()))
        tn, fp, fn, tp = conf(d_tr)
        ctn, cfp, cfn, ctp = conf(d_cv)
        dual = np.abs(svc.dual_coef_.ravel())
        cw = svc.class_weight_                                   # weight per class index (0 = non-deposit)
        c_eff = C * cw[y[svc.support_]]
        bounded = int(np.sum(dual >= c_eff * (1 - 1e-6)))
        k_acc = []
        for f in np.unique(first["fold"]):
            te = first["fold"] == f
            k_acc.append(float(((d_cv[te] >= 0) == pos[te]).mean()))
        items = [("Samples (deposits / non-deposits)", f"{int(pos.sum())} / {int(neg.sum())}"),
                 ("Training confusion: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("Training confusion: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("Training accuracy", f"{(tn + tp) / y.size:.4f}"),
                 ("Training AUC", f"{_auc(d_tr[pos], d_tr[neg]):.4f}"),
                 ("Cross-validation", first["cv_name"]),
                 ("CV confusion: true non-deposit → predicted non-deposit / deposit", f"{ctn} / {cfp}"),
                 ("CV confusion: true deposit → predicted non-deposit / deposit", f"{cfn} / {ctp}"),
                 ("Cross-validated accuracy (mean of folds)", f"{np.mean(k_acc):.4f}"),
                 ("Cross-validated AUC", f"{_auc(d_cv[pos], d_cv[neg]):.4f}"),
                 ("Support vectors (non-deposit / deposit)", f"{int(svc.n_support_[0])} / {int(svc.n_support_[1])}"),
                 ("Bounded support vectors (α = C)", f"{bounded} of {dual.size}"),
                 ("Platt sigmoid P = 1 / (1 + exp(A·f + B))", f"A = {first['a']:.4g}, B = {first['b']:.4g}")]
        return pd.DataFrame(items, columns=["item", "value"])
