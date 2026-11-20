"""MS-KPRB — Mineral-System Knowledge-Prior Residual Boosting (the user's proposed hybrid method; kprb.py).

    logit P(D | x) = η^K(x) + Δ(x)

η^K = α + β logit K(x; s, c, w, v) is the calibrated two-level knowledge network (OR-like power mean of the proxies
inside each mineral-system process, AND-like weighted geometric mean across processes; expert weights are the modes
of Dirichlet priors with confidence κ, membership slopes / inflections have normal priors with s.d. τ; MAP fit).
Δ = ν Σ_t γ^(t) is a logistic gradient boosting of the residuals that starts from the knowledge offset (depth-2 trees,
Newton leaf steps); its number of trees T is chosen by held-out spatial-block deviance and may be 0 — then the map
is the calibrated expert model. Everything is repeated for B random non-deposit samples; mean and s.d. are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from logic.hybrid.common.knowledge import (KnowledgeModel, KnowledgeNetwork, boost_predict, boost_residual,
                                           draw_negatives, grid_blocks)
from logic.hybrid.common.pipeline import binomial_deviance

T_GRID = (0, 10, 25, 50, 100, 150)
CHUNK = 200_000
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class KPRBConfig:
    kappa: float = 10.0            # confidence in the expert weights (Dirichlet pseudo-count); 3 low, 30 high
    tau: float = 0.75              # prior s.d. of ln(slope) and inflection
    s0: float = 1.5
    c0: float = 0.5
    learning_rate: float = 0.05
    depth: int = 2
    min_leaf: int = 5
    neg_ratio: float = 3.0
    draws: int = 20                # B
    folds: int = 4                 # spatial folds (blocks of a 4 × 4 tiling grouped into folds)
    t_grid: Sequence[int] = T_GRID
    guided: float = 0.0            # exponent a of knowledge-guided negatives (0 = uniform)
    prior_correction: bool = False
    seed: int = 0


@dataclass
class KPRBResult:
    prob: np.ndarray               # per study cell
    prob_sd: np.ndarray
    knowledge_prob: np.ndarray     # calibrated π_K (T = 0)
    delta: np.ndarray              # mean data correction Δ (log-odds)
    expert: np.ndarray             # K at the expert parameters
    F_expert: np.ndarray
    F_fit: np.ndarray              # mean calibrated process scores
    layers: pd.DataFrame
    processes: pd.DataFrame
    draws: pd.DataFrame
    t_selection: pd.DataFrame
    summary: pd.DataFrame
    importance: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def spatial_folds(net: KnowledgeNetwork, n_folds: int, seed: int, nb: int = 4) -> np.ndarray:
    """Fold id of every study cell: the nb × nb blocks are shuffled and dealt to n_folds folds."""
    blocks = grid_blocks(net.shape, net.cells, nb)
    order = np.random.default_rng(seed).permutation(nb * nb)
    fold_of_block = np.empty(nb * nb, int)
    fold_of_block[order] = np.arange(nb * nb) % max(2, n_folds)
    return fold_of_block[blocks]


def _fit_pair(net, cfg, X, F_tr, y, n_trees):
    km = KnowledgeModel(net, cfg.s0, cfg.c0, cfg.kappa, cfg.tau).fit(X, y)
    off = km.parts(X)[3]
    trees = boost_residual(F_tr, y, off, n_trees, cfg.learning_rate, cfg.depth, cfg.min_leaf, cfg.seed) \
        if n_trees > 0 else []
    return km, trees


def choose_trees(net, cfg, X, Z, y, fold) -> np.ndarray:
    """Held-out deviance (summed over the usable spatial folds) for every T of the grid."""
    grid = np.asarray(cfg.t_grid, int)
    dev = np.zeros(grid.size)
    used = 0
    for g in np.unique(fold):
        te = fold == g
        if (y[~te] == 1).sum() < 3 or (y[te] == 1).sum() < 1 or te.sum() < 5:
            continue
        km, trees = _fit_pair(net, cfg, X[~te], Z[~te], y[~te], int(grid.max()))
        off = km.parts(X[te])[3]
        for i, t in enumerate(grid):
            dev[i] += binomial_deviance(y[te], boost_predict(trees, Z[te], off, cfg.learning_rate, int(t)))
        used += 1
    return dev if used >= 2 else None


def fit_kprb(net: KnowledgeNetwork, pos: np.ndarray, neg_cand: np.ndarray, cfg: KPRBConfig,
             progress: ProgressFn = None) -> KPRBResult:
    say = progress or (lambda f, m: None)
    n, m, Kp = net.cells.size, len(net.table.layers), net.K
    N = int(pos.size)
    if N < 3:
        raise ValueError(f"Only {N} training deposit cell(s) inside the study area — need at least 3.")
    mu0, F0, K0 = net.expert()
    Z_all = net.features()
    fold_cell = spatial_folds(net, cfg.folds, cfg.seed + 11)
    n_neg = int(round(cfg.neg_ratio * N))
    msgs = []
    s1, s2, sk, sd_, sF = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros((n, Kp))
    W, V, S, C, rows, devs = [], [], [], [], [], []
    for b in range(cfg.draws):
        say(0.03 + 0.9 * b / cfg.draws, f"draw {b + 1} / {cfg.draws}")
        rng = np.random.default_rng(cfg.seed + 1000 * b)
        neg = draw_negatives(rng, neg_cand, n_neg, K0 if cfg.guided > 0 else None, cfg.guided)
        idx = np.r_[pos, neg]
        y = np.r_[np.ones(N), np.zeros(neg.size)]
        X, Z = net.X[idx], Z_all[idx]
        dev = choose_trees(net, cfg, X, Z, y, fold_cell[idx])
        if dev is None:
            T = 0
            if b == 0:
                msgs.append("fewer than two spatial folds contain deposits — no residual trees (T = 0)")
        else:
            T = int(np.asarray(cfg.t_grid)[np.argmin(dev)])
            devs.append(dev)
        km, trees = _fit_pair(net, cfg, X, Z, y, T)
        shift = np.log(max(n - N, 1) / max(neg.size, 1)) if cfg.prior_correction else 0.0
        for s in range(0, n, CHUNK):
            ch = slice(s, min(n, s + CHUNK))
            _, Fc, _, etaK = km.parts(net.X[ch])
            eta = boost_predict(trees, Z_all[ch], etaK, cfg.learning_rate, T) - shift
            p = expit(eta)
            s1[ch] += p
            s2[ch] += p * p
            sk[ch] += expit(etaK - shift)
            sd_[ch] += eta + shift - etaK
            sF[ch] += Fc
        s_, c_, w_, v_, a_, be_ = km.unpack(km.theta)
        W.append(w_)
        V.append(v_)
        S.append(s_)
        C.append(c_)
        eta_tr = boost_predict(trees, Z, km.parts(X)[3], cfg.learning_rate, T)
        rows.append(dict(draw=b + 1, non_deposits=int(neg.size), trees=T, alpha=float(a_), beta=float(be_),
                         training_auc=_auc(eta_tr[y == 1], eta_tr[y == 0]),
                         knowledge_auc=_auc(km.parts(X)[3][y == 1], km.parts(X)[3][y == 0]),
                         converged=bool(getattr(km, "converged", True))))
    B = cfg.draws
    prob = s1 / B
    sd = np.sqrt(np.maximum(s2 / B - prob ** 2, 0.0))
    W, V, S, C = map(np.array, (W, V, S, C))
    lay = []
    for j, l in enumerate(net.table.layers):
        cat = bool(net.categorical[j])
        lay.append(dict(layer=l.name, process=l.process, direction="+" if net.direction[j] > 0 else "−",
                        type="categorical" if cat else "continuous", expert_weight=float(net.w0[j]),
                        fitted_weight=float(W[:, j].mean()), fitted_weight_sd=float(W[:, j].std()),
                        slope=None if cat else float(S[:, j].mean()),
                        inflection=None if cat else float(C[:, j].mean()),
                        median=None if cat else float(net.med[j]), iqr=None if cat else float(net.iqr[j])))
    proc = pd.DataFrame([dict(code=i + 1, process=p, layers=int((net.k == i).sum()), expert_weight=float(net.v0[i]),
                              fitted_weight=float(V[:, i].mean()), fitted_weight_sd=float(V[:, i].std()),
                              limiting_area_pct=float(100 * np.mean(np.argmin(sF / B, 1) == i)) if Kp > 1 else 100.0)
                         for i, p in enumerate(net.table.processes)])
    draws = pd.DataFrame(rows)
    tsel = pd.DataFrame(dict(trees=list(cfg.t_grid),
                             mean_heldout_deviance=np.mean(devs, 0) if devs else np.full(len(cfg.t_grid), np.nan),
                             times_selected=[int((draws.trees == t).sum()) for t in cfg.t_grid]))
    imp = _importance(net, pos, neg_cand, cfg, n_neg)
    st = dict(draws=B, deposits=N, non_deposits=n_neg, median_trees=float(draws.trees.median()),
              share_T0=float((draws.trees == 0).mean()), expert_auc_samples=float(draws.knowledge_auc.mean()),
              training_auc=float(draws.training_auc.mean()), mean_prob=float(prob.mean()),
              mean_abs_delta=float(np.mean(np.abs(sd_ / B))))
    summary = pd.DataFrame([
        ("Training deposits (cells)", str(N)), ("Non-deposit samples per draw", str(n_neg)),
        ("Random draws B", str(B)), ("Processes / layers", f"{Kp} / {m}"),
        ("Expert confidence κ / tolerance τ", f"{cfg.kappa:g} / {cfg.tau:g}"),
        ("Residual trees T (median over draws; grid)", f"{st['median_trees']:g}; {list(cfg.t_grid)}"),
        ("Draws with T = 0 (knowledge model kept)", f"{100 * st['share_T0']:.0f} %"),
        ("Learning rate / depth / min. leaf", f"{cfg.learning_rate:g} / {cfg.depth} / {cfg.min_leaf}"),
        ("Spatial folds", f"{cfg.folds} (blocks of a 4 × 4 tiling)"),
        ("Non-deposit sampling", "uniform" if cfg.guided == 0 else f"knowledge-guided, ∝ (1 − K)^{cfg.guided:g}"),
        ("Probability scale", "corrected for the sampling ratio" if cfg.prior_correction else
         "relative (balanced samples; ranks only)"),
        ("Training AUC (samples, mean)", f"{st['training_auc']:.4f}"),
        ("Calibrated knowledge AUC (samples, mean)", f"{st['expert_auc_samples']:.4f}"),
        ("Mean |Δ| (log-odds)", f"{st['mean_abs_delta']:.3f}"),
    ], columns=["item", "value"])
    say(1.0, "done")
    return KPRBResult(prob, sd, sk / B, sd_ / B, K0, F0, sF / B, pd.DataFrame(lay), proc, draws, tsel, summary,
                      imp, st, msgs)


def _importance(net, pos, neg_cand, cfg, n_neg, repeats: int = 3) -> pd.DataFrame:
    """Permutation importance on one draw: drop of the held-out (spatial-fold) AUC when a layer is shuffled."""
    rng = np.random.default_rng(cfg.seed + 5)
    neg = draw_negatives(rng, neg_cand, n_neg)
    idx = np.r_[pos, neg]
    y = np.r_[np.ones(pos.size), np.zeros(neg.size)]
    X, Z = net.X[idx], net.features()[idx]
    fold = spatial_folds(net, cfg.folds, cfg.seed + 11)[idx]
    T = int(np.median([t for t in cfg.t_grid]))
    models = []
    for g in np.unique(fold):
        te = fold == g
        if (y[~te] == 1).sum() < 3 or (y[te] == 1).sum() < 1:
            continue
        models.append((te, *_fit_pair(net, cfg, X[~te], Z[~te], y[~te], T)))
    if len(models) < 2:
        return pd.DataFrame(columns=["rank", "layer", "process", "heldout_auc_drop", "sd"])

    def held(Xp, Zp):
        h = np.full(y.size, np.nan)
        for te, km, trees in models:
            h[te] = boost_predict(trees, Zp[te], km.parts(Xp[te])[3], cfg.learning_rate, T)
        ok = np.isfinite(h)
        return _auc(h[ok & (y == 1)], h[ok & (y == 0)])
    base = held(X, Z)
    rows = []
    for j, l in enumerate(net.table.layers):
        drops = []
        for _ in range(repeats):
            perm = rng.permutation(y.size)
            Xp, Zp = X.copy(), Z.copy()
            Xp[:, j], Zp[:, j] = X[perm, j], Z[perm, j]
            drops.append(base - held(Xp, Zp))
        rows.append(dict(layer=l.name, process=l.process, heldout_auc_drop=float(np.mean(drops)),
                         sd=float(np.std(drops))))
    tab = pd.DataFrame(rows).sort_values("heldout_auc_drop", ascending=False).reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    return tab


def _auc(pos, neg) -> float:
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos), np.asarray(neg)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.r_[pos, neg])
    return float((r[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
