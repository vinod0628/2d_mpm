"""Extreme learning machine (ELM) for mineral prospectivity mapping — own NumPy implementation, no GUI.

Huang et al. (2006, 2012): a single-hidden-layer feed-forward network whose input weights a_j and biases b_j are drawn
at random and never trained; only the output weights β are learned, in closed form, by regularised least squares:

    H_ij = g(a_jᵀ z_i + b_j)              (N × L hidden-layer output matrix)
    β = (HᵀH + I/C)⁻¹ Hᵀ t   (N ≥ L)      or   β = Hᵀ (HHᵀ + I/C)⁻¹ t   (N < L)
    f(z) = h(z)ᵀ β

with targets t ∈ {0, 1}. Training is a single linear solve — orders of magnitude faster than back-propagation — and
the ridge term 1/C controls over-fitting (C tuned by cross-validated AUC). Because a single ELM depends on its random
hidden layer, ``networks`` ELMs with different random weights are averaged (ensemble; Liu & Wang 2010). The averaged
output is mapped to [0, 1] by a Platt sigmoid fitted to the cross-validated outputs (monotone — the ranking of f is
kept). In MPM: Chen & Wu (2017) (ELM regression of deposit / non-deposit targets).

Outputs: score (calibrated ensemble output, averaged over R random non-deposit samples — a favourability that depends
on the sample ratio, not a probability per unit cell), score_std (s.d. of the calibrated outputs of the ensemble
members = model uncertainty).
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
from logic.data_driven.svm.model import platt_fit, platt_prob

ACTIVATIONS = ("sigmoid", "tanh", "relu", "rbf")
ACTIVATION_LABELS = {"sigmoid": N_("sigmoid"), "tanh": N_("tanh"), "relu": N_("ReLU"),
                     "rbf": N_("Gaussian RBF (random centres)")}
C_GRID = tuple(2.0 ** k for k in range(-6, 13, 2))      # 1/64 … 4096
CHUNK = 100_000
ProgressFn = Optional[Callable[[float, str], None]]


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


@dataclass
class HiddenLayer:
    """Random hidden layer: weights W (p × L) and biases b (L), or RBF centres / widths."""
    activation: str
    W: np.ndarray
    b: np.ndarray

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "rbf":                         # W = centres (L × p), b = widths
            d2 = (np.sum(Z * Z, 1)[:, None] + np.sum(self.W * self.W, 1)[None, :] - 2.0 * Z @ self.W.T)
            return np.exp(-np.maximum(d2, 0.0) / (self.b[None, :] ** 2))
        U = Z @ self.W + self.b
        if self.activation == "sigmoid":
            return expit(U)
        if self.activation == "tanh":
            return np.tanh(U)
        return np.maximum(U, 0.0)


def random_hidden(p: int, L: int, activation: str, rng: np.random.Generator, Z_train: Optional[np.ndarray] = None
                  ) -> HiddenLayer:
    """Input weights ~ U(−1, 1), biases ~ U(−1, 1) (Huang et al. 2006); RBF: centres drawn from the training samples,
    widths = median distance between centres."""
    if activation == "rbf":
        if Z_train is None:
            raise ValueError("RBF hidden layer needs training samples")
        idx = rng.choice(len(Z_train), size=min(L, len(Z_train)), replace=L > len(Z_train))
        centres = Z_train[idx] + rng.normal(scale=1e-3, size=(len(idx), p))
        d = np.sqrt(np.maximum(np.sum((centres[:, None] - centres[None]) ** 2, -1), 0))
        width = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return HiddenLayer("rbf", centres, np.full(len(idx), width))
    return HiddenLayer(activation, rng.uniform(-1, 1, size=(p, L)), rng.uniform(-1, 1, size=L))


def solve_output(H: np.ndarray, t: np.ndarray, C: float) -> np.ndarray:
    """Regularised least-squares output weights (Huang et al. 2012, eqs 38 / 32)."""
    n, L = H.shape
    if n >= L:
        return np.linalg.solve(H.T @ H + np.eye(L) / C, H.T @ t)
    return H.T @ np.linalg.solve(H @ H.T + np.eye(n) / C, t)


@dataclass
class ELMResult:
    score: np.ma.MaskedArray
    score_std: np.ma.MaskedArray
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


class ELMModel:
    """ELM prospectivity model (see module doc).

    hidden (neurons L), activation ("sigmoid" | "tanh" | "relu" | "rbf"), C (> 0) or tune (cross-validated C from
    C_GRID), networks (ensemble size per random sample); training samples as the other machine-learning methods.
    """

    def __init__(self, hidden: int = 100, activation: str = "sigmoid", C: float = 1.0, tune: bool = True,
                 networks: int = 10, neg_ratio: float = 1.0, min_distance: float = 0.0, min_spacing: float = -1.0,
                 res: Tuple[float, float] = (1.0, 1.0), repeats: int = 1, cv_folds: int = 5, missing: str = "exclude",
                 seed: int = 0):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if C <= 0:
            raise ValueError("C must be positive")
        if cv_folds < 2:
            raise ValueError("cv_folds must be ≥ 2")
        self.hidden, self.activation, self.C, self.tune = max(1, int(hidden)), activation, float(C), bool(tune)
        self.networks = max(1, int(networks))
        self.neg_ratio, self.min_distance, self.min_spacing = float(neg_ratio), float(min_distance), float(min_spacing)
        self.res, self.repeats, self.cv_folds = tuple(res), max(1, int(repeats)), int(cv_folds)
        self.missing, self.seed = missing, int(seed)

    # -------------------------------------------------------------- ensemble
    def fit_ensemble(self, Z, y, C, seed):
        rng = np.random.default_rng(seed)
        members = []
        for _ in range(self.networks):
            hl = random_hidden(Z.shape[1], self.hidden, self.activation, rng, Z)
            members.append((hl, solve_output(hl(Z), y.astype(float), C)))
        return members

    @staticmethod
    def member_outputs(members, Z):
        return np.array([hl(Z) @ beta for hl, beta in members])          # networks × N

    def folds(self, y, seed):
        k = min(self.cv_folds, int(min((y == 1).sum(), (y == 0).sum())))
        fold = np.empty(y.size, int)
        rng = np.random.default_rng(seed)
        for cls in (1, 0):
            ids = rng.permutation(np.flatnonzero(y == cls))
            fold[ids] = np.arange(ids.size) % k
        return fold

    def cross_validate(self, Z, y, C, fold, seed):
        held = np.empty(y.size)
        models = []
        for f in np.unique(fold):
            te = fold == f
            mem = self.fit_ensemble(Z[~te], y[~te], C, seed + 13 * int(f))
            held[te] = self.member_outputs(mem, Z[te]).mean(0)
            models.append((te, mem))
        return held, models

    # -------------------------------------------------------------- full model
    def fit_predict(self, features: Sequence[Feature], dep_rows, dep_cols, study_mask: Optional[np.ndarray] = None,
                    known_deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    progress: ProgressFn = None) -> ELMResult:
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
        p = stack.n_columns
        shift = np.array([float(stack.column(j).mean()) for j in range(p)])
        scale = np.array([float(stack.column(j).std()) or 1.0 for j in range(p)])
        C = self.C

        n = stack.n_cells
        s1, s2, cnt = np.zeros(n), np.zeros(n), 0
        rep_rows, tuning, first = [], pd.DataFrame(), None
        for r in range(self.repeats):
            frac = r / self.repeats
            step = 0.85 / self.repeats
            rng = np.random.default_rng(self.seed + 1000 * r)
            neg = sample_positions(cand_pos, cand_cells // width, cand_cells % width, m, rng, spacing, self.res)
            idx = np.r_[dep_pos, neg]
            Z = (stack.matrix(idx) - shift) / scale
            y = np.r_[np.ones(n_d, int), np.zeros(neg.size, int)]
            fold = self.folds(y, self.seed + 7 + r)
            if r == 0 and self.tune:
                say(0.05, N_("tuning C (cross-validation)"))
                rows = []
                for c in C_GRID:
                    h, _ = self.cross_validate(Z, y, c, fold, self.seed + 3)
                    rows.append(dict(C=c, log2_C=np.log2(c), cv_auc=_auc(h[y == 1], h[y == 0])))
                tuning = pd.DataFrame(rows)
                top = tuning.cv_auc.round(4).max()
                C = float(tuning[tuning.cv_auc.round(4) >= top].C.min())      # ties -> stronger regularisation
                tuning["selected"] = tuning.C == C
            say(0.05 + 0.85 * frac + 0.3 * step, N_("cross-validating"))
            held, fold_models = self.cross_validate(Z, y, C, fold, self.seed + 3 + r)
            a, b = platt_fit(held, y)
            if a >= 0:                     # uninformative held-out outputs: keep the ranking of f (see SVM)
                members_tmp = self.fit_ensemble(Z, y, C, self.seed + 1000 * r + 5)
                a, b = platt_fit(self.member_outputs(members_tmp, Z).mean(0), y)
                a = min(a, -1e-6)
                msgs.append(f"sample {r + 1}: cross-validated outputs do not separate deposits; the sigmoid was "
                            "fitted to the training outputs")
            say(0.05 + 0.85 * frac + 0.6 * step, N_("training the ELM ensemble"))
            members = self.fit_ensemble(Z, y, C, self.seed + 1000 * r + 5)
            say(0.05 + 0.85 * frac + 0.75 * step, N_("predicting the study area"))
            for s in range(0, n, CHUNK):
                ch = np.arange(s, min(n, s + CHUNK))
                out = self.member_outputs(members, (stack.matrix(ch) - shift) / scale)
                pr = platt_prob(out, a, b)
                s1[ch] += pr.sum(0)
                s2[ch] += (pr * pr).sum(0)
            cnt += len(members)
            f_tr = self.member_outputs(members, Z).mean(0)
            rep_rows.append(dict(repeat=r + 1, seed=self.seed + 1000 * r, deposits=n_d, non_deposits=int(neg.size),
                                 C=C, networks=len(members), training_auc=_auc(f_tr[y == 1], f_tr[y == 0]),
                                 cv_auc=_auc(held[y == 1], held[y == 0]), platt_A=a, platt_B=b))
            if first is None:
                first = dict(Z=Z, y=y, held=held, f_tr=f_tr, fold=fold, fold_models=fold_models, a=a, b=b)
        if spacing > 0:
            msgs.append(f"non-deposit samples are at least {spacing:,.0f} map units apart"
                        + (" (auto: ½·√(area / n))" if self.min_spacing < 0 else ""))
        R = self.repeats
        if R > 1:
            msgs.append(f"{R} random non-deposit samples × {self.networks} ELMs averaged; tuning, importance and "
                        "validation refer to sample 1")
        score = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - score ** 2, 0.0))

        say(0.92, N_("variable importance"))
        imp = self._importance(first, stack)
        val = self._validation(first)

        def grid(v):
            out = np.zeros(base.shape)
            out.ravel()[stack.cells] = v
            return np.ma.array(out, mask=~base)

        rep = pd.DataFrame(rep_rows)
        st = dict(cv_auc=float(rep.cv_auc.iloc[0]), training_auc=float(rep.training_auc.iloc[0]), C=C,
                  networks=int(cnt), hidden=self.hidden, platt_A=first["a"], platt_B=first["b"], spacing=spacing,
                  samples=int(first["y"].size))
        summary = pd.DataFrame([
            ("Training deposits (cells)", str(n_d)), ("Non-deposit samples per draw", str(m)),
            ("Random draws × ELMs", f"{R} × {self.networks} = {cnt}"),
            ("Hidden neurons L / activation", f"{self.hidden} / {self.activation}"),
            ("C (regularisation)", f"{C:g}" + (" (tuned)" if self.tune else "")),
            ("Platt sigmoid A / B (sample 1)", f"{first['a']:.4g} / {first['b']:.4g}"),
            ("Evidence columns", ", ".join(stack.names)[:300]), ("Study area (cells)", f"{n:,}"),
            ("Min. distance of non-deposits from deposits", f"{self.min_distance:g}"),
            ("Min. spacing between non-deposits", f"{spacing:,.0f}"),
            ("Training AUC (samples)", f"{st['training_auc']:.4f}"),
            ("Cross-validated AUC (samples)", f"{st['cv_auc']:.4f}"),
            ("Mean score in the study area", f"{score.mean():.4f}"),
        ], columns=["item", "value"])
        params = dict(hidden=self.hidden, activation=self.activation, C=C, tune=self.tune, networks=self.networks,
                      neg_ratio=self.neg_ratio, min_distance=self.min_distance, min_spacing=spacing, repeats=R,
                      cv_folds=self.cv_folds, missing=self.missing, seed=self.seed)
        say(1.0, N_("done"))
        return ELMResult(grid(score), grid(std), imp, tuning, val, summary, rep, base, n_d, m, st, params, msgs)

    # -------------------------------------------------------------- importance / validation
    def _importance(self, first, stack: FeatureStack) -> pd.DataFrame:
        Z, y, held = first["Z"], first["y"], first["held"]
        base_auc = _auc(held[y == 1], held[y == 0])
        rng = np.random.default_rng(self.seed + 17)
        rows = []
        for layer, cols in stack.groups.items():
            drops = []
            for _ in range(5):
                h = np.empty(y.size)
                for te, mem in first["fold_models"]:
                    Zt = Z[te].copy()
                    Zt[:, cols] = Zt[rng.permutation(Zt.shape[0])][:, cols]
                    h[te] = self.member_outputs(mem, Zt).mean(0)
                drops.append(base_auc - _auc(h[y == 1], h[y == 0]))
            rows.append(dict(layer=layer, columns=len(cols), permutation_auc_drop=float(np.mean(drops)),
                             permutation_sd=float(np.std(drops))))
        tab = pd.DataFrame(rows).sort_values("permutation_auc_drop", ascending=False).reset_index(drop=True)
        tab.insert(0, "rank", np.arange(1, len(tab) + 1))
        return tab

    @staticmethod
    def _validation(first) -> pd.DataFrame:
        y, a, b = first["y"], first["a"], first["b"]
        held, p_tr = platt_prob(first["held"], a, b), platt_prob(first["f_tr"], a, b)
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
