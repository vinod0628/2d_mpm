"""Logistic regression (LR) for mineral prospectivity mapping — pure numpy / scipy (no GUI).

Model (Agterberg 1974; Agterberg et al. 1993; Bonham-Carter 1994, ch. 9)::

    logit P(D | x) = ln[P / (1 − P)] = β0 + Σ_j β_j x_j

x_j is a continuous evidence value (optionally log10-transformed) or a 0/1 dummy variable of one class of a
categorical evidence layer (reference class = the class with the largest area, coefficient 0). Unlike weights of
evidence, the coefficients are estimated jointly, so correlated evidence layers are not double-counted
(no conditional-independence assumption; Agterberg et al. 1993).

Training designs
    ``"all"``      every study-area cell is an observation (y = 1 for training-deposit cells). With categorical
                   evidence only, cells are pooled into unique conditions weighted by their number of cells — the
                   weighted logistic regression of Agterberg et al. (1993) / ArcSDM (same likelihood). Above
                   ``max_rows`` cells, all deposit cells and a uniform random fraction s of the other cells are
                   used and the intercept is corrected exactly by ln s (case-control sampling).
    ``"sampled"``  training deposits (y = 1) + random non-deposit cells (y = 0) at least ``min_distance`` from any
                   known deposit (Carranza & Hale 2001; Carranza et al. 2008). Case-control sampling changes only
                   the intercept; the prior correction of King & Zeng (2001)::

                       β0* = β0 − ln[ (1 − τ)/τ · ȳ/(1 − ȳ) ]    τ = deposit fraction of the study area,
                                                                 ȳ = deposit fraction of the sample
                   turns the output back into probabilities. Several random draws (Zuo & Wang 2020) are pooled
                   with Rubin's (1987) rules: β̄ = mean β_r, T = W̄ + (1 + 1/R)·B.

Estimation — Newton–Raphson / IRLS with step halving on the binomial log-likelihood
(a = deposits, n = cells of an observation row, W = n·p(1 − p))::

    ML     U(β)  = Xᵀ(a − n·p)                        I(β) = Xᵀ W X
    Firth  U*(β) = Xᵀ(a − n·p + h(½ − p))             h = diag(W^½ X I⁻¹ Xᵀ W^½); maximises ℓ + ½ ln|I(β)|
           (Firth 1993; Heinze & Schemper 2002: finite estimates under separation, e.g. a class without deposits)
    Ridge  U(β) − λβ,  I(β) + λ  (intercept not penalised; continuous columns are z-scores)

Inference: Cov(β) = I⁻¹; Wald z = β / s(β); likelihood-ratio (penalised LR for Firth) tests per evidence layer;
s²(η) = xᵀ Cov x, s(P) ≈ P(1 − P)·s(η) (delta method), t = P / s(P) (Agterberg et al. 1993).
Fit: −2 ln L, model χ², AIC, BIC, McFadden R², predicted vs observed deposits, Hosmer–Lemeshow (2013) test,
Kolmogorov–Smirnov check of unique conditions (Agterberg et al. 1993), GVIF collinearity (Fox & Monette 1992).

Observations are raster cells; probabilities are reported per *unit cell* of area A_u (as in WofE / ArcSDM)::

    P_unit = 1 − (1 − P_cell)^(A_u / A_cell)        (≈ P_cell · A_u / A_cell for rare deposits)

Correspondence with ArcSDM (GTK, github.com/gtkfi/ArcSDM; logic only, no arcpy):
  * classic *Logistic Regression* (Spatial Data Modeller; Kemp et al. 2001, Sawatzky et al. 2004) — unique
    conditions weighted by area, ordered evidence as numeric values (here: "continuous"), free evidence expanded to
    0/1 dummies (here: "categorical"), missing data replaced by the area-weighted mean of the known evidence
    (class area fractions for dummies), outputs LRPostProb / LR_Std_Dev / LRTValue (here: post / post_std /
    tvalue). Difference: sdmlr.exe counts points as successes and the area (unit cells) as failures, i.e.
    trials = area + points, stops at |score| < 0.01 and reports the covariance of the previous iteration; here the
    trials are raster cells (exact binomial) and P is converted to the unit cell (same result for rare deposits).
    Our ML engine reproduces sdmlr.exe on its own case data (tests/test_logistic.py).
  * experimental *Create Random Points* → *Logistic Regression Train* — random non-deposit points outside a
    buffer around the deposits, inside cells where every evidence raster has data; report = confusion matrix,
    training AUC, leave-one-out (default) or k-fold cross-validation.

References: see docs/python/mpm_studio.md (Logistic regression).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from i18n import N_
from logic.common.training import auto_spacing, background_candidates, sample_positions

VAR_KINDS = ("continuous", "categorical")
VAR_KIND_LABELS = {"continuous": N_("Continuous"), "categorical": N_("Categorical (dummy variables)")}
TRANSFORMS = ("none", "log10")
TRANSFORM_LABELS = {"none": N_("none"), "log10": N_("log10(x)")}
ESTIMATORS = ("firth", "ml", "ridge")
ESTIMATOR_LABELS = {"firth": N_("Firth bias-reduced (recommended)"), "ml": N_("Maximum likelihood"),
                    "ridge": N_("Ridge (L2-penalised)")}
DESIGNS = ("all", "sampled")
DESIGN_LABELS = {"all": N_("All study-area cells (weighted LR; Agterberg et al. 1993)"),
                 "sampled": N_("Deposits + random non-deposit cells (Carranza & Hale 2001)")}
SELECTIONS = ("none", "backward_aic", "backward_lr")
SELECTION_LABELS = {"none": N_("none (use all ticked layers)"), "backward_aic": N_("backward elimination (AIC)"),
                    "backward_lr": N_("backward elimination (LR test)")}
MISSING = ("exclude", "impute")
MISSING_LABELS = {"exclude": N_("exclude cells from study area"),
                  "impute": N_("area-weighted mean of the known evidence")}
MAX_CLASSES = 255
EPV_MIN = 10              # events per variable (Peduzzi et al. 1996)
LOO_MAX = 300             # leave-one-out cross-validation up to this many samples, else 10-fold
INTERCEPT = "(intercept)"

ProgressFn = Optional[Callable[[float, str], None]]


# ============================================================================== data classes
@dataclass
class Predictor:
    """One evidence layer. ``array`` masked = missing evidence."""

    name: str
    array: np.ma.MaskedArray
    kind: str = "continuous"
    transform: str = "none"
    labels: Optional[Dict[float, str]] = None

    def __post_init__(self):
        if self.kind not in VAR_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {VAR_KINDS}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"{self.name}: transform must be one of {TRANSFORMS}")


@dataclass
class Term:
    """One column of the design matrix."""

    name: str
    variable: str
    kind: str                     # intercept | continuous | categorical
    level: Optional[float] = None  # categorical class value (NaN = missing-data dummy)
    label: str = ""


@dataclass
class Fit:
    beta: np.ndarray              # coefficients of ``columns`` (design scale: z-scores / dummies)
    cov: np.ndarray
    columns: np.ndarray           # design-matrix column indices that were estimated
    loglik: float                 # unpenalised log-likelihood
    objective: float              # penalised log-likelihood (= loglik for ML)
    n_iter: int
    converged: bool
    dropped: List[int] = field(default_factory=list)   # columns without variation in the training rows
    separation: bool = False

    @property
    def k(self) -> int:
        return int(self.beta.size)

    def aic(self) -> float:
        return -2.0 * self.loglik + 2.0 * self.k


@dataclass
class TrainingRows:
    """Observation rows: design matrix, deposits (a) and non-deposit cells (b) per row, intercept offset."""

    X: np.ndarray
    a: np.ndarray
    b: np.ndarray
    offset: float = 0.0           # added to the intercept after fitting (sampling correction)
    aggregated: bool = False      # rows are unique conditions
    n_neg: int = 0                # sampled non-deposit cells (case-control rows)
    note: str = ""


@dataclass
class LRResult:
    probability: np.ma.MaskedArray          # per unit cell
    probability_std: np.ma.MaskedArray
    tvalue: np.ma.MaskedArray               # P / s(P)
    logit: np.ma.MaskedArray                # linear predictor (per raster cell)
    coefficients: pd.DataFrame
    variables: pd.DataFrame
    fit_table: pd.DataFrame
    hosmer_lemeshow: pd.DataFrame
    selection: pd.DataFrame
    repeats: pd.DataFrame
    correlation: pd.DataFrame
    sample_validation: pd.DataFrame          # sampled design: cross-validation + confusion matrix (ArcSDM style)
    study_mask: np.ndarray
    n_deposits: int
    n_cells: int
    stats: Dict[str, float]
    params: Dict[str, object]
    messages: List[str] = field(default_factory=list)

    def summary_table(self) -> pd.DataFrame:
        return self.fit_table


# ============================================================================== estimation
def _inv(m: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m)


def _loglik(eta, a, b) -> float:
    """Σ a·ln p + b·ln(1 − p), p = expit(eta), without overflow."""
    return _ll_p(eta, a, b)[0]


def _ll_p(eta, a, b):
    """(log-likelihood, p) from one exp/log1p pass: −ln(1−p) = softplus(η), −ln p = softplus(η) − η."""
    e = np.exp(-np.abs(eta))
    sp = np.maximum(eta, 0.0) + np.log1p(e)
    ll = float(-(a @ (sp - eta)) - (b @ sp))
    p = np.where(eta >= 0, 1.0, e) / (1.0 + e)
    return ll, p


def fit_binomial(X: np.ndarray, a: np.ndarray, b: np.ndarray, method: str = "firth", ridge: float = 1.0,
                 max_iter: int = 100, tol: float = 1e-8, max_step: float = 5.0,
                 free: Optional[np.ndarray] = None) -> Fit:
    """Binomial logistic regression of ``a`` successes in ``a + b`` trials on ``X`` (first column = intercept).

    method: "ml" (maximum likelihood), "firth" (Firth 1993 bias-reduced) or "ridge" (L2 penalty ``ridge``).
    Newton–Raphson with step halving on the (penalised) log-likelihood; steps are capped at ``max_step``.
    free: boolean mask of estimated columns; the others are fixed at 0 while the Firth penalty ½ ln|I| still uses
    every column — the restricted fit of the penalised likelihood-ratio test (Heinze & Schemper 2002).
    """
    if method not in ESTIMATORS:
        raise ValueError(f"method must be one of {ESTIMATORS}")
    X = np.asarray(X, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a + b
    k = X.shape[1]
    free = np.ones(k, bool) if free is None else np.asarray(free, bool)
    pen = np.zeros(k)
    if method == "ridge":
        pen[1:] = float(ridge)
    beta = np.zeros(k)
    pbar = (a.sum() + 0.5) / (n.sum() + 1.0)
    beta[0] = np.log(pbar / (1.0 - pbar))

    def state(bt):
        ll, p = _ll_p(X @ bt, a, b)
        w = n * p * (1.0 - p)
        info = (X * w[:, None]).T @ X + np.diag(pen)
        obj = ll - 0.5 * float((pen * bt ** 2).sum())
        if method == "firth":
            sign, logdet = np.linalg.slogdet(info)
            obj = obj + 0.5 * logdet if sign > 0 else -np.inf
        return ll, obj, p, w, info

    ll, obj, p, w, info = state(beta)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        resid = a - n * p
        if method == "firth":
            h = w * ((X @ _inv(info)) * X).sum(axis=1)          # leverages of the weighted hat matrix
            resid = resid + h * (0.5 - p)
        score = X.T @ resid - pen * beta
        step = np.zeros(k)
        sub = info[np.ix_(free, free)]
        try:
            step[free] = np.linalg.solve(sub, score[free])
        except np.linalg.LinAlgError:
            step[free] = np.linalg.lstsq(sub, score[free], rcond=None)[0]
        big = np.max(np.abs(step))
        if big > max_step:
            step *= max_step / big
        t = 1.0
        for _ in range(40):                                       # step halving
            cand = state(beta + t * step)
            if np.isfinite(cand[1]) and cand[1] >= obj - 1e-12 * max(1.0, abs(obj)):
                break
            t *= 0.5
        beta = beta + t * step
        ll, obj, p, w, info = cand
        if np.max(np.abs(t * step)) < tol:
            converged = True
            break
    cov = _inv(info)
    separation = method == "ml" and (not converged or bool(np.any(np.abs(beta[1:]) > 15)))
    return Fit(beta, cov, np.arange(k), ll, obj, it, converged, separation=separation)


# ============================================================================== design matrix
def _fmt(v: float) -> str:
    return f"{v:g}"


class Design:
    """Encodes the evidence of the study-area cells as a design matrix (intercept + z-scores + dummies).

    Scaling (mean, s.d.), classes and reference classes are learned once from the whole study area so every
    training sample and the prediction use the same encoding.
    """

    def __init__(self, predictors: Sequence[Predictor], study: np.ndarray, missing: str = "exclude"):
        if missing not in MISSING:
            raise ValueError(f"missing must be one of {MISSING}")
        self.shape = study.shape
        self.cells = np.flatnonzero(study)                 # flat grid index of each study cell
        self.missing = missing
        self.terms: List[Term] = [Term(INTERCEPT, INTERCEPT, "intercept")]
        self.groups: Dict[str, List[int]] = {}
        self.vars: List[dict] = []
        for pr in predictors:
            arr = np.ma.asarray(pr.array)
            data = np.ma.getdata(arr).ravel()[self.cells].astype(float)
            known = ~np.ma.getmaskarray(arr).ravel()[self.cells] & np.isfinite(data)
            if not known.any():
                raise ValueError(f"Evidence '{pr.name}' has no data inside the study area.")
            info = dict(name=pr.name, kind=pr.kind, transform=pr.transform, known=known,
                        n_missing=int((~known).sum()))
            start = len(self.terms)
            if pr.kind == "continuous":
                if pr.transform == "log10":
                    if np.any(data[known] <= 0):
                        raise ValueError(f"Evidence '{pr.name}': log10 needs values > 0 "
                                         f"(minimum {data[known].min():g}).")
                    data = np.log10(np.where(known, data, 1.0))
                mu, sd = float(data[known].mean()), float(data[known].std())
                if not sd > 0:
                    raise ValueError(f"Evidence '{pr.name}' is constant inside the study area.")
                info.update(values=np.where(known, data, mu), mean=mu, sd=sd)
                unit = "log10(x)" if pr.transform == "log10" else "x"
                self.terms.append(Term(pr.name, pr.name, "continuous", label=unit))
            else:
                classes, counts = np.unique(data[known], return_counts=True)
                if classes.size < 2:
                    raise ValueError(f"Evidence '{pr.name}' has fewer than 2 classes inside the study area.")
                if classes.size > MAX_CLASSES:
                    raise ValueError(f"Evidence '{pr.name}' has {classes.size} distinct values — use it as "
                                     "continuous or reclassify it (Data ▸ Reclassify raster).")
                ref = int(np.argmax(counts))
                codes = np.full(data.size, -1, np.int32)
                codes[known] = np.searchsorted(classes, data[known])
                colmap = np.full(classes.size, -1, np.int64)
                labels = pr.labels or {}
                for i, c in enumerate(classes):
                    if i == ref:
                        continue
                    colmap[i] = len(self.terms)
                    lab = str(labels.get(float(c), _fmt(c)))
                    self.terms.append(Term(f"{pr.name}[{_fmt(c)}]", pr.name, "categorical", float(c), lab))
                info.update(values=codes, classes=classes, counts=counts, fractions=counts / counts.sum(),
                            reference=float(classes[ref]),
                            reference_label=str(labels.get(float(classes[ref]), _fmt(classes[ref]))),
                            colmap=colmap)
            self.groups[pr.name] = list(range(start, len(self.terms)))
            self.vars.append(info)

    @property
    def n_cells(self) -> int:
        return int(self.cells.size)

    @property
    def n_columns(self) -> int:
        return len(self.terms)

    def var(self, name: str) -> dict:
        return next(v for v in self.vars if v["name"] == name)

    def matrix(self, idx: np.ndarray) -> np.ndarray:
        """Design matrix for study-cell positions ``idx`` (0 … n_cells − 1)."""
        idx = np.asarray(idx, np.int64)
        X = np.zeros((idx.size, self.n_columns))
        X[:, 0] = 1.0
        for v, cols in zip(self.vars, self.groups.values()):
            kn = v["known"][idx]
            if v["kind"] == "continuous":
                X[:, cols[0]] = (v["values"][idx] - v["mean"]) / v["sd"]     # missing -> mean -> 0
            else:
                code = v["values"][idx]
                col = np.where(kn, v["colmap"][np.clip(code, 0, None)], -1)
                hit = np.nonzero(col >= 0)[0]
                X[hit, col[hit]] = 1.0
                miss = np.nonzero(~kn)[0]
                if miss.size:                                  # ArcSDM: dummy = area fraction of the class
                    for i, c in enumerate(v["colmap"]):
                        if c >= 0:
                            X[miss, c] = v["fractions"][i]
        return X

    def back_transform(self, cols: np.ndarray) -> np.ndarray:
        """Matrix T with β_original = T β_design for the columns ``cols`` (continuous: per unit of x / log10 x)."""
        cols = list(cols)
        T = np.eye(len(cols))
        for v in self.vars:
            if v["kind"] != "continuous":
                continue
            c = self.groups[v["name"]][0]
            if c in cols:
                j = cols.index(c)
                T[j, j] = 1.0 / v["sd"]
                T[0, j] = -v["mean"] / v["sd"]
        return T


# ============================================================================== statistics
def weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    m = (X * w[:, None]).sum(0) / w.sum()
    d = X - m
    c = (d * w[:, None]).T @ d / w.sum()
    s = np.sqrt(np.diag(c))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = c / np.outer(s, s)
    return r


def gvif(R: np.ndarray, groups: Dict[str, List[int]]) -> Dict[str, float]:
    """Generalised VIF (Fox & Monette 1992): det(R_gg)·det(R_rest)/det(R); = VIF for one column."""
    out = {}
    k = R.shape[0]
    det_all = np.linalg.det(R) if k else 1.0
    for g, idx in groups.items():
        rest = [i for i in range(k) if i not in idx]
        d_g = np.linalg.det(R[np.ix_(idx, idx)]) if idx else 1.0
        d_r = np.linalg.det(R[np.ix_(rest, rest)]) if rest else 1.0
        out[g] = float(d_g * d_r / det_all) if det_all > 1e-300 else float("inf")
    return out


def hosmer_lemeshow(p: np.ndarray, a: np.ndarray, n: np.ndarray, groups: int = 10):
    """Hosmer–Lemeshow C statistic on rows sorted by fitted probability (groups of ~equal cell count)."""
    order = np.argsort(p, kind="mergesort")
    p, a, n = p[order], a[order], n[order]
    cum = np.cumsum(n)
    gid = np.minimum((groups * (cum - 0.5 * n) / cum[-1]).astype(int), groups - 1)
    rows = []
    for g in np.unique(gid):
        s = gid == g
        ng, og, eg = n[s].sum(), a[s].sum(), (n[s] * p[s]).sum()
        rows.append(dict(group=int(len(rows) + 1), cells=float(ng), observed=float(og), expected=float(eg),
                         p_min=float(p[s].min()), p_max=float(p[s].max())))
    tab = pd.DataFrame(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (tab.observed - tab.expected) ** 2 / (tab.expected * (1 - tab.expected / tab.cells))
    c = float(np.nansum(terms))
    dof = len(tab) - 2
    pv = float(stats.chi2.sf(c, dof)) if dof > 0 else float("nan")
    return tab, c, dof, pv


def ks_unique_conditions(p: np.ndarray, a: np.ndarray, n: np.ndarray) -> Tuple[float, float]:
    """Agterberg et al. (1993): max |observed − expected| cumulative relative deposit frequency over unique
    conditions sorted by probability, and the 95 % two-sided K-S critical value for that many conditions."""
    order = np.argsort(p, kind="mergesort")
    d = a.sum()
    obs = np.cumsum(a[order]) / d
    exp = np.cumsum(n[order] * p[order]) / d
    return float(np.max(np.abs(obs - exp))), float(stats.kstwo.ppf(0.95, p.size))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney AUC (ties count ½)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = stats.rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# ============================================================================== model
class LogisticModel:
    """Logistic-regression prospectivity model.

    design: "all" | "sampled";  estimator: "firth" | "ml" | "ridge";  selection: "none" | "backward_aic" |
    "backward_lr" (remove the layer with the largest LR-test p while p > ``alpha``);  missing: "exclude" | "impute".
    unit_cell_km2 / cell_area_km2: output probability per unit cell.  Sampled design: ``neg_ratio`` non-deposit
    cells per training deposit, ``min_distance`` (map units) from any known deposit, ``repeats`` random draws,
    ``prior_correction`` (King & Zeng 2001).  ``max_rows``: cell limit of the "all" design before case-control
    subsampling of the background.
    """

    def __init__(self, design: str = "all", estimator: str = "firth", ridge: float = 1.0, selection: str = "none",
                 alpha: float = 0.10, missing: str = "exclude", unit_cell_km2: float = 1.0,
                 cell_area_km2: float = 1.0, neg_ratio: float = 1.0, min_distance: float = 0.0,
                 res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, prior_correction: bool = True,
                 cv_folds: int = -1, max_rows: int = 1_000_000, seed: int = 0, min_spacing: float = 0.0):
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}")
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if unit_cell_km2 <= 0 or cell_area_km2 <= 0:
            raise ValueError("Cell areas must be positive.")
        self.design, self.estimator, self.ridge = design, estimator, float(ridge)
        self.selection, self.alpha, self.missing = selection, float(alpha), missing
        self.unit_cell_km2, self.cell_area_km2 = float(unit_cell_km2), float(cell_area_km2)
        self.neg_ratio, self.min_distance, self.res = float(neg_ratio), float(min_distance), tuple(res)
        self.repeats = max(1, int(repeats)) if design == "sampled" else 1
        self.prior_correction = bool(prior_correction)
        self.cv_folds = int(cv_folds)          # sampled design: -1 = leave-one-out, 0 = none, k >= 2 = k-fold
        self.min_spacing = float(min_spacing)  # between non-deposit samples: < 0 = auto, 0 = none (ArcSDM)
        self.spacing = 0.0
        self.max_rows, self.seed = int(max_rows), int(seed)

    @property
    def relative(self) -> bool:
        """Sampled design without prior correction: output = favourability in the sample, not a probability."""
        return self.design == "sampled" and not self.prior_correction

    @property
    def unit_factor(self) -> float:
        """Unit cells per raster cell (A_cell / A_unit)."""
        return self.cell_area_km2 / self.unit_cell_km2

    def to_unit(self, p_cell: np.ndarray) -> np.ndarray:
        """P_unit = 1 − (1 − P_cell)^(A_unit / A_cell)."""
        return -np.expm1(np.log1p(-np.clip(p_cell, 0.0, 1.0 - 1e-15)) / self.unit_factor)

    def _fit(self, rows: TrainingRows, cols) -> Fit:
        cols = np.asarray(sorted(set(cols) | {0}), np.int64)
        used = (rows.a + rows.b) > 0
        sub = rows.X[used][:, cols]
        varies = np.r_[True, np.ptp(sub[:, 1:], axis=0) > 0] if sub.shape[0] else np.ones(cols.size, bool)
        keep = cols[varies]
        fit = fit_binomial(rows.X[:, keep], rows.a, rows.b, self.estimator, self.ridge)
        fit.columns = keep
        fit.dropped = [int(c) for c in cols[~varies]]
        return fit

    # -------------------------------------------------------------- training rows
    def _rows_all(self, design: Design, dep_pos: np.ndarray, rng) -> TrainingRows:
        n = design.n_cells
        y = np.zeros(n)
        y[dep_pos] = 1.0
        if all(v["kind"] == "categorical" for v in design.vars):
            X = design.matrix(np.arange(n))
            uc, inv = np.unique(X, axis=0, return_inverse=True)
            inv = inv.ravel()
            a = np.bincount(inv, weights=y, minlength=uc.shape[0])
            cnt = np.bincount(inv, minlength=uc.shape[0]).astype(float)
            return TrainingRows(uc, a, cnt - a, aggregated=True)
        if n <= self.max_rows:
            return TrainingRows(design.matrix(np.arange(n)), y, 1.0 - y)
        neg = np.flatnonzero(y == 0)
        m = max(self.max_rows - dep_pos.size, 1)
        pick = np.sort(rng.choice(neg, size=m, replace=False))
        idx = np.r_[dep_pos, pick]
        s = m / neg.size
        return TrainingRows(design.matrix(idx), np.r_[np.ones(dep_pos.size), np.zeros(m)],
                            np.r_[np.zeros(dep_pos.size), np.ones(m)], offset=float(np.log(s)), n_neg=m,
                            note=f"{n:,} study cells > {self.max_rows:,}: all deposit cells + random {s:.1%} of the "
                                 "other cells were used; the intercept was corrected by ln(s) (case-control sampling)")

    def _n_negatives(self, n_dep: int) -> int:
        m = int(round(self.neg_ratio * n_dep))
        if m < 1:
            raise ValueError("The number of non-deposit samples must be at least 1.")
        return m

    def _rows_sampled(self, design: Design, dep_pos: np.ndarray, cand_pos: np.ndarray, rng) -> TrainingRows:
        m = self._n_negatives(dep_pos.size)
        cells = design.cells[cand_pos]
        width = design.shape[1]
        neg = sample_positions(cand_pos, cells // width, cells % width, m, rng, self.spacing, self.res)
        idx = np.r_[dep_pos, neg]
        offset = 0.0
        if self.prior_correction:
            tau = dep_pos.size / design.n_cells
            ybar = dep_pos.size / (dep_pos.size + m)
            offset = -float(np.log((1 - tau) / tau * ybar / (1 - ybar)))
        return TrainingRows(design.matrix(idx), np.r_[np.ones(dep_pos.size), np.zeros(m)],
                            np.r_[np.zeros(dep_pos.size), np.ones(m)], offset=offset, n_neg=m)

    # -------------------------------------------------------------- tests / variable selection
    def _lr_test(self, rows: TrainingRows, fit: Fit, group_cols) -> Tuple[float, int, float]:
        """Likelihood-ratio test of β_group = 0. Firth: penalised LR test — restricted fit with the group fixed at
        0 but the penalty of the full model (Heinze & Schemper 2002); a smaller refit would change the penalty."""
        cols = np.asarray(fit.columns)
        drop = np.isin(cols, list(group_cols))
        df = int(drop.sum())
        if df == 0:
            return float("nan"), 0, float("nan")
        if self.estimator == "firth":
            r = fit_binomial(rows.X[:, cols], rows.a, rows.b, "firth", free=~drop)
        else:
            r = fit_binomial(rows.X[:, cols[~drop]], rows.a, rows.b, self.estimator, self.ridge)
        chi = max(0.0, 2.0 * (fit.objective - r.objective))
        return chi, df, float(stats.chi2.sf(chi, df))

    def _select(self, rows: TrainingRows, groups: Dict[str, List[int]]):
        active = list(groups)

        def cols(act):
            return [0] + [c for g in act for c in groups[g]]

        cur = self._fit(rows, cols(active))
        steps = [dict(step=0, action="start", variable="", layers=len(active), aic=cur.aic(),
                      lr_chi2=np.nan, df=np.nan, p_value=np.nan)]
        if self.selection == "none":
            return active, cur, pd.DataFrame()
        while len(active) > 1:
            tests = {g: self._lr_test(rows, cur, groups[g]) for g in active}
            if self.selection == "backward_aic":
                trials = {g: self._fit(rows, cols([h for h in active if h != g])) for g in active}
                g = min(trials, key=lambda x: trials[x].aic())
                if trials[g].aic() >= cur.aic():
                    break
                new = trials[g]
            else:
                g = max(tests, key=lambda x: -1.0 if np.isnan(tests[x][2]) else tests[x][2])
                if not tests[g][2] > self.alpha:
                    break
                new = self._fit(rows, cols([h for h in active if h != g]))
            active.remove(g)
            cur = new
            steps.append(dict(step=len(steps), action="removed", variable=g, layers=len(active), aic=cur.aic(),
                              lr_chi2=tests[g][0], df=tests[g][1], p_value=tests[g][2]))
        return active, cur, pd.DataFrame(steps)

    # -------------------------------------------------------------- prediction
    @staticmethod
    def _predict(design: Design, cols: np.ndarray, beta: np.ndarray, cov: np.ndarray, offset: float,
                 chunk: int = 200_000):
        n = design.n_cells
        eta, var = np.empty(n), np.empty(n)
        for s in range(0, n, chunk):
            idx = np.arange(s, min(n, s + chunk))
            X = design.matrix(idx)[:, cols]
            eta[idx] = X @ beta + offset
            var[idx] = np.einsum("ij,jk,ik->i", X, cov, X)
        return eta, np.maximum(var, 0.0)

    # -------------------------------------------------------------- full model
    def fit_predict(self, predictors: Sequence[Predictor], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> LRResult:
        """Fit on the training deposits and predict the study area.

        known_deposits: (rows, cols) of *all* known deposits (training + test); non-deposit samples keep
        ``min_distance`` from every one of them.  progress(fraction 0–1, message).
        """
        say = progress or (lambda f, m: None)
        if not predictors:
            raise ValueError("At least one evidence layer is required.")
        shape = np.ma.asarray(predictors[0].array).shape
        for pr in predictors:
            if np.ma.asarray(pr.array).shape != shape:
                raise ValueError(f"Evidence '{pr.name}' is not on the same grid as '{predictors[0].name}'.")
        names = [p.name for p in predictors]
        if len(set(names)) != len(names):
            raise ValueError("Evidence layer names must be unique.")
        known = [~np.ma.getmaskarray(np.ma.asarray(p.array)) for p in predictors]
        base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool).copy()
        if study_mask is None:
            base &= np.logical_or.reduce(known)
        if self.missing == "exclude":
            base &= np.logical_and.reduce(known)
        if not base.any():
            raise ValueError("The study area is empty (no cell has all evidence).")
        msgs: List[str] = []

        say(0.02, N_("encoding evidence"))
        design = Design(predictors, base, self.missing)
        dep = np.zeros(shape, bool)
        dep[np.asarray(dep_rows, int), np.asarray(dep_cols, int)] = True
        dep &= base
        dep_pos = np.searchsorted(design.cells, np.flatnonzero(dep))
        n_d = int(dep_pos.size)
        if n_d < 2:
            raise ValueError(f"Only {n_d} training deposit cell(s) inside the study area — need at least 2.")
        k_all = design.n_columns - 1
        if n_d < EPV_MIN * k_all:
            msgs.append(f"{n_d} deposits for {k_all} coefficients (< {EPV_MIN} per coefficient, Peduzzi et al. 1996): "
                        "estimates are unstable — prefer Firth, fewer layers or variable selection")
        for v in design.vars:
            if v["n_missing"] and self.missing == "impute":
                msgs.append(f"{v['name']}: {v['n_missing']:,} cells without data were set to the area-weighted mean "
                            + ("value" if v["kind"] == "continuous" else "of the class dummies (class area fractions)"))

        rng = np.random.default_rng(self.seed)
        cand_pos = None
        if self.design == "sampled":
            kr, kc = known_deposits if known_deposits is not None else (dep_rows, dep_cols)
            cand = background_candidates(base, kr, kc, self.min_distance, self.res)
            cand_pos = np.searchsorted(design.cells, np.flatnonzero(cand))
            self.spacing = (auto_spacing(cand_pos.size, self._n_negatives(n_d), self.res)
                            if self.min_spacing < 0 else self.min_spacing)
            rows = self._rows_sampled(design, dep_pos, cand_pos, rng)
            if self.spacing > 0:
                msgs.append(f"non-deposit samples are at least {self.spacing:,.0f} map units apart"
                            + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        else:
            rows = self._rows_all(design, dep_pos, rng)
        if rows.note:
            msgs.append(rows.note)

        say(0.15, N_("fitting the model"))
        active, fit, selection = self._select(rows, design.groups)
        if self.selection != "none":
            dropped_layers = [g for g in design.groups if g not in active]
            msgs.append("variable selection kept " + ", ".join(active)
                        + (f"; removed {', '.join(dropped_layers)}" if dropped_layers else ""))
        if fit.separation:
            msgs.append("maximum likelihood did not converge or gave very large coefficients (separation, e.g. a "
                        "class without deposits) — use the Firth estimator")
        for c in fit.dropped:
            msgs.append(f"{design.terms[c].name}: no variation in the training samples — coefficient fixed at 0")

        say(0.45, N_("testing variables"))
        variables = self._variable_table(design, rows, fit, active)

        # ---- repeated random non-deposit samples (sampled design)
        fits = [(fit, rows)]
        for r in range(1, self.repeats):
            say(0.5 + 0.2 * r / self.repeats, N_("fitting repeated samples"))
            rr = self._rows_sampled(design, dep_pos, cand_pos, np.random.default_rng(self.seed + 1000 + r))
            fits.append((self._fit(rr, fit.columns), rr))
        cols = fit.columns
        betas = []
        for f, rr in fits:
            full = np.zeros(cols.size)
            pos = np.searchsorted(cols, f.columns)
            full[pos] = f.beta
            betas.append(full)

        say(0.72, N_("predicting the study area"))
        p_sum = np.zeros(design.n_cells)
        p_sq = np.zeros(design.n_cells)
        v_sum = np.zeros(design.n_cells)
        eta_sum = np.zeros(design.n_cells)
        rep_rows = []
        dep_mask_pos = np.zeros(design.n_cells, bool)
        dep_mask_pos[dep_pos] = True
        for r, ((f, rr), bfull) in enumerate(zip(fits, betas)):
            covf = np.zeros((cols.size, cols.size))
            pos = np.searchsorted(cols, f.columns)
            covf[np.ix_(pos, pos)] = f.cov
            eta, var = self._predict(design, cols, bfull, covf, rr.offset)
            pc = expit(eta)
            p_sum += pc
            p_sq += pc ** 2
            v_sum += (pc * (1 - pc)) ** 2 * var
            eta_sum += eta
            if self.repeats > 1:
                bg = pc[~dep_mask_pos]
                if bg.size > 200_000:
                    bg = bg[:: bg.size // 200_000]
                rep_rows.append(dict(repeat=r + 1, seed=self.seed + (1000 + r if r else 0), non_deposits=rr.n_neg,
                                     auc_training=_auc(pc[dep_pos], bg), aic=f.aic(), converged=f.converged,
                                     **{design.terms[c].name: b for c, b in zip(cols, bfull)}))
        R = len(fits)
        p_cell = p_sum / R
        var_cell = v_sum / R
        if R > 1:                                                   # Rubin (1987): T = W + (1 + 1/R) B
            var_cell = var_cell + (1 + 1 / R) * np.maximum(p_sq / R - p_cell ** 2, 0) * R / (R - 1)
        eta_cell = eta_sum / R

        # pooled coefficients (Rubin's rules) for the table
        beta_pool = np.mean(betas, axis=0)
        cov_pool = np.zeros((cols.size, cols.size))
        for f, _ in fits:
            pos = np.searchsorted(cols, f.columns)
            cov_pool[np.ix_(pos, pos)] += f.cov / R
        if R > 1:
            B = np.cov(np.array(betas).T, ddof=1).reshape(cols.size, cols.size)
            cov_pool = cov_pool + (1 + 1 / R) * B
            msgs.append(f"{R} random non-deposit samples pooled (Rubin's rules); fit statistics and tests refer to "
                        "sample 1")
        offset = float(np.mean([rr.offset for _, rr in fits]))

        say(0.9, N_("summarising"))
        coef = self._coef_table(design, cols, beta_pool, cov_pool, offset, set(fit.dropped))
        if self.relative:
            msgs.append("no prior correction: the output is a relative favourability (deposit share of the training "
                        "sample), not a probability per unit cell")
            p_unit, s_unit = p_cell, np.sqrt(var_cell)
        else:
            p_unit = self.to_unit(p_cell)
            deriv = (1 - p_cell) ** (1 / self.unit_factor - 1) / self.unit_factor      # dP_unit / dP_cell
            s_unit = deriv * np.sqrt(var_cell)
        t_val = np.divide(p_unit, s_unit, out=np.zeros_like(p_unit), where=s_unit > 0)

        fit_table, hl, extra = self._fit_stats(design, rows, fit, active, p_cell, dep_pos, n_d)
        corr = self._correlation(design, rows, cols)
        sample_val = pd.DataFrame()
        if self.design == "sampled":
            say(0.95, N_("cross-validating"))
            sample_val = self._sample_validation(rows, fit)
            if R > 1:
                sample_val.loc[len(sample_val)] = ("Note", "computed on sample 1")
        stats_ = dict(extra, n_deposits=n_d, n_cells=design.n_cells, predicted=float(p_cell.sum()))

        def grid(v):
            out = np.zeros(shape)
            out.ravel()[design.cells] = v
            return np.ma.array(out, mask=~base)

        params = dict(design=self.design, estimator=self.estimator, ridge=self.ridge, selection=self.selection,
                      alpha=self.alpha, missing=self.missing, unit_cell_km2=self.unit_cell_km2,
                      cell_area_km2=self.cell_area_km2, neg_ratio=self.neg_ratio, min_distance=self.min_distance,
                      repeats=self.repeats, prior_correction=self.prior_correction, cv_folds=self.cv_folds,
                      seed=self.seed, active=active, offset=offset, min_spacing=self.spacing)
        say(1.0, N_("done"))
        return LRResult(grid(p_unit), grid(s_unit), grid(t_val), grid(eta_cell), coef, variables, fit_table, hl,
                        selection, pd.DataFrame(rep_rows), corr, sample_val, base, n_d, design.n_cells, stats_,
                        params, msgs)

    # -------------------------------------------------------------- tables
    def _coef_table(self, design: Design, cols, beta, cov, offset, dropped) -> pd.DataFrame:
        T = design.back_transform(cols)
        b_orig = T @ beta
        b_orig[0] += offset
        cov_orig = T @ cov @ T.T
        se = np.sqrt(np.maximum(np.diag(cov_orig), 0))
        se_z = np.sqrt(np.maximum(np.diag(cov), 0))
        z = 1.959963984540054
        rows = []
        for j, c in enumerate(cols):
            t = design.terms[c]
            fixed = int(c) in dropped
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                wald = b_orig[j] / se[j] if se[j] > 0 else np.nan
            rows.append(dict(term=t.name, layer=t.variable, cls="" if t.level is None else
                             ("missing" if np.isnan(t.level) else _fmt(t.level)),
                             label=t.label, B=b_orig[j], s_B=se[j], z=wald,
                             p_value=float(2 * stats.norm.sf(abs(wald))) if np.isfinite(wald) else np.nan,
                             ci_low=b_orig[j] - z * se[j], ci_high=b_orig[j] + z * se[j],
                             odds_ratio=float(np.exp(np.clip(b_orig[j], -700, 700))) if j else np.nan,
                             B_std=beta[j] if t.kind == "continuous" else np.nan,
                             note="fixed at 0 (no variation)" if fixed else ""))
            if t.kind == "categorical" and (j + 1 == len(cols) or design.terms[cols[j + 1]].variable != t.variable):
                v = design.var(t.variable)
                rows.append(dict(term=f"{t.variable}[{_fmt(v['reference'])}]", layer=t.variable,
                                 cls=_fmt(v["reference"]), label=v["reference_label"], B=0.0, s_B=np.nan, z=np.nan,
                                 p_value=np.nan, ci_low=np.nan, ci_high=np.nan, odds_ratio=1.0, B_std=np.nan,
                                 note="reference class (largest area)"))
        out = pd.DataFrame(rows)
        if offset:
            out.loc[0, "note"] = "includes sampling correction {:+.4f}".format(offset)
        return out

    def _variable_table(self, design: Design, rows: TrainingRows, fit: Fit, active) -> pd.DataFrame:
        cols = list(fit.columns)
        groups = {g: [cols.index(c) - 1 for c in design.groups[g] if c in cols] for g in active}
        n = rows.a + rows.b
        R = weighted_corr(rows.X[:, cols[1:]], n) if len(cols) > 2 else np.ones((1, 1))
        gv = gvif(R, {g: idx for g, idx in groups.items() if idx}) if len(cols) > 2 else {g: 1.0 for g in active}
        out = []
        for g in design.groups:
            v = design.var(g)
            if g not in active:
                out.append(dict(layer=g, type=v["kind"], transform=v.get("transform", ""), status="removed",
                                df=np.nan, wald_chi2=np.nan, wald_p=np.nan, lr_chi2=np.nan, lr_p=np.nan,
                                gvif=np.nan, gvif_adj=np.nan))
                continue
            idx = [cols.index(c) for c in design.groups[g] if c in cols]
            df = len(idx)
            if df:
                bsub = fit.beta[idx]
                wald = float(bsub @ np.linalg.lstsq(fit.cov[np.ix_(idx, idx)], bsub, rcond=None)[0])
                lr = self._lr_test(rows, fit, design.groups[g])[0]
            else:
                wald = lr = np.nan
            gg = gv.get(g, np.nan)
            out.append(dict(layer=g, type=v["kind"], transform=v.get("transform", "") if v["kind"] == "continuous"
                            else "", status="in model", df=df, wald_chi2=wald,
                            wald_p=float(stats.chi2.sf(wald, df)) if df else np.nan, lr_chi2=lr,
                            lr_p=float(stats.chi2.sf(lr, df)) if df else np.nan, gvif=gg,
                            gvif_adj=gg ** (1 / (2 * df)) if df and np.isfinite(gg) else np.nan))
        return pd.DataFrame(out)

    def _fit_stats(self, design, rows: TrainingRows, fit: Fit, active, p_cell, dep_pos, n_d):
        n = rows.a + rows.b
        null = fit_binomial(rows.X[:, :1], rows.a, rows.b, "ml")
        chi = 2.0 * (fit.loglik - null.loglik)
        df = fit.k - 1
        p_fit = expit(rows.X[:, fit.columns] @ fit.beta)
        hl_tab, hl_c, hl_df, hl_p = hosmer_lemeshow(p_fit, rows.a, n)
        ks, ks_crit = ks_unique_conditions(p_fit, rows.a, n) if rows.aggregated else (np.nan, np.nan)
        n_obs = float(n.sum())
        mcf = 1.0 - fit.loglik / null.loglik if null.loglik else np.nan
        predicted = float(p_cell.sum())
        items = [
            ("Training design", {"all": "all study-area cells", "sampled": "deposits + random non-deposit cells"}
             [self.design] + (" (unique conditions)" if rows.aggregated else "")),
            ("Estimator", self.estimator + (f" (λ = {self.ridge:g})" if self.estimator == "ridge" else "")),
            ("Training deposits (cells)", n_d),
            ("Study area (cells)", f"{design.n_cells:,}"),
            ("Observation rows", f"{rows.X.shape[0]:,}"),
            ("Non-deposit samples", rows.n_neg if rows.n_neg else "—"),
            ("Layers in model", ", ".join(active)),
            ("Coefficients (incl. intercept)", fit.k),
            ("Iterations / converged", f"{fit.n_iter} / {'yes' if fit.converged else 'NO'}"),
            ("−2 ln L (null)", -2 * null.loglik),
            ("−2 ln L (model)", -2 * fit.loglik),
            ("Model χ² (LR) / df", f"{chi:.3f} / {df}"),
            ("Model χ² p-value", float(stats.chi2.sf(chi, df)) if df > 0 else np.nan),
            ("AIC", fit.aic()),
            ("BIC", -2 * fit.loglik + fit.k * np.log(n_obs)),
            ("McFadden pseudo-R²", mcf),
            ("Hosmer–Lemeshow C / df", f"{hl_c:.3f} / {hl_df}"),
            ("Hosmer–Lemeshow p-value", hl_p),
            ("Observed deposits N(D)", n_d),
            ("Predicted deposits Σ P(cell)", predicted),
            ("Prior probability (per unit cell)", float(self.to_unit(np.array([n_d / design.n_cells]))[0])),
            ("Unit cell / raster cell (km²)", f"{self.unit_cell_km2:g} / {self.cell_area_km2:.6g}"),
        ]
        if rows.aggregated:
            items += [("Unique conditions", rows.X.shape[0]), ("K-S max |obs − exp| (Agterberg et al. 1993)", ks),
                      ("K-S 95 % critical value", ks_crit)]
        if rows.offset:
            items.append(("Intercept correction (sampling)", rows.offset))
        tab = pd.DataFrame([(k, f"{v:.6g}" if isinstance(v, float) else str(v)) for k, v in items],
                           columns=["item", "value"])
        extra = dict(mcfadden=mcf, aic=fit.aic(), hl_p=hl_p, ks=ks, ks_crit=ks_crit, model_chi2=chi, model_df=df)
        return tab, hl_tab, extra

    def _sample_validation(self, rows: TrainingRows, fit: Fit) -> pd.DataFrame:
        """ArcSDM *Logistic Regression Train* report for presence/absence samples: confusion matrix (p ≥ 0.5 on
        the sample scale), training AUC and leave-one-out / stratified k-fold cross-validation (accuracy, AUC of
        the pooled held-out scores)."""
        y = rows.a > 0
        cols = fit.columns
        score = rows.X[:, cols] @ fit.beta
        pred = score >= 0
        tn, fp = int((~y & ~pred).sum()), int((~y & pred).sum())
        fn, tp = int((y & ~pred).sum()), int((y & pred).sum())
        items = [("Samples (deposits / non-deposits)", f"{int(y.sum())} / {int((~y).sum())}"),
                 ("Confusion matrix: true non-deposit → predicted non-deposit / deposit", f"{tn} / {fp}"),
                 ("Confusion matrix: true deposit → predicted non-deposit / deposit", f"{fn} / {tp}"),
                 ("Training accuracy", f"{(tn + tp) / y.size:.4f}"),
                 ("Training AUC (samples)", f"{_auc(score[y], score[~y]):.4f}")]
        k = self.cv_folds
        n = y.size
        if k < 0 and n > LOO_MAX:                                   # leave-one-out = n fits: cap the cost
            k = 10
        if k != 0:
            if k < 0 or k >= n:
                fold = np.arange(n)
                name = "leave-one-out"
            else:
                fold = np.empty(n, int)
                rng = np.random.default_rng(self.seed + 7)
                for cls in (True, False):                          # stratified folds
                    idx = rng.permutation(np.flatnonzero(y == cls))
                    fold[idx] = np.arange(idx.size) % k
                name = f"{k}-fold (stratified)" + (f"; leave-one-out needs ≤ {LOO_MAX} samples"
                                                     if self.cv_folds < 0 else "")
            held = np.empty(n)
            accs = []
            for f in np.unique(fold):
                te = fold == f
                trn = TrainingRows(rows.X[~te], rows.a[~te], rows.b[~te])
                ff = self._fit(trn, cols)
                held[te] = rows.X[te][:, ff.columns] @ ff.beta
                accs.append(float(((held[te] >= 0) == y[te]).mean()))
            items += [("Cross-validation", name),
                      ("Cross-validated accuracy (mean of folds)", f"{np.mean(accs):.4f}"),
                      ("Cross-validated AUC (pooled held-out scores)", f"{_auc(held[y], held[~y]):.4f}")]
        return pd.DataFrame(items, columns=["item", "value"])

    @staticmethod
    def _correlation(design: Design, rows: TrainingRows, cols) -> pd.DataFrame:
        cc = list(cols)[1:]
        if not cc:
            return pd.DataFrame()
        R = weighted_corr(rows.X[:, cc], rows.a + rows.b)
        names = [design.terms[c].name for c in cc]
        out = pd.DataFrame(R, columns=names)
        out.insert(0, "term", names)
        return out
