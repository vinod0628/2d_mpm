"""TOPSIS and VIKOR multi-criteria decision making for prospectivity mapping — knowledge-driven, no GUI.

Every grid cell is an *alternative*, every evidence layer a *criterion*. The criteria values are the expert scores
of the index overlay (0 unfavourable … 1 favourable: a score per class, a linear rescaling in the favourable
direction, or a binary condition — see :mod:`logic.knowledge_driven.index_overlay.model`), so every criterion is a
benefit criterion and categorical maps can be used.

**TOPSIS** (Hwang & Yoon 1981; MPM: Pazand et al. 2011, Pazand & Hezarkhani 2015, Abedi et al. 2012):

    r_ij = x_ij / sqrt(Σ_i x_ij²)          (vector normalisation; "linear": r_ij = x_ij, scores already 0 … 1)
    v_ij = w_j r_ij
    A⁺_j = max_i v_ij,   A⁻_j = min_i v_ij  (positive / negative ideal solution over the study area)
    D⁺_i = sqrt(Σ_j (v_ij − A⁺_j)²),  D⁻_i = sqrt(Σ_j (v_ij − A⁻_j)²)
    C_i  = D⁻_i / (D⁺_i + D⁻_i)             (relative closeness to the ideal, 0 … 1 = prospectivity)

**VIKOR** (Opricovic 1998; Opricovic & Tzeng 2004; MPM: Abedi et al. 2016):

    S_i = Σ_j w_j (f*_j − f_ij) / (f*_j − f⁻_j)          group utility (weighted L1 distance to the ideal)
    R_i = max_j w_j (f*_j − f_ij) / (f*_j − f⁻_j)        individual regret (L∞ distance)
    Q_i = v (S_i − S*)/(S⁻ − S*) + (1 − v)(R_i − R*)/(R⁻ − R*)
    prospectivity = 1 − Q_i                               (Q = 0 best; v = 0.5 "by consensus")

Weights w_j: equal, custom, AHP, fuzzy AHP (as the index overlay) or **Shannon entropy** (objective; Shannon
1948, Zeleny 1982): p_ij = x_ij / Σ_i x_ij, E_j = −Σ_i p_ij ln p_ij / ln m, w_j ∝ 1 − E_j — a layer whose scores
are spread unevenly over the cells (more information) gets a larger weight.

Missing evidence ("renormalise"): the distances / utilities of a cell use the criteria with data, with the weights
renormalised over them; "exclude" keeps only cells with every criterion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_

from ..index_overlay.model import AHPResult, IOLayer, ahp_weights, fuzzy_ahp_weights, layer_scores

METHODS = ("topsis", "vikor")
METHOD_LABELS = {"topsis": N_("TOPSIS — relative closeness to the ideal"),
                 "vikor": N_("VIKOR — compromise ranking (group utility and regret)")}
WEIGHTINGS = ("equal", "custom", "ahp", "fuzzy_ahp", "entropy")
WEIGHTING_LABELS = {"equal": N_("equal"), "custom": N_("custom weight per layer"),
                    "ahp": N_("AHP pairwise comparison (Saaty)"),
                    "fuzzy_ahp": N_("fuzzy AHP (triangular judgements, Buckley)"),
                    "entropy": N_("Shannon entropy (objective, from the evidence)")}
NORMALISATIONS = ("vector", "linear")
NORMALISATION_LABELS = {"vector": N_("vector  x / √Σx² (classic TOPSIS)"),
                        "linear": N_("none — scores 0 … 1")}


@dataclass
class MCDMResult:
    index: np.ma.MaskedArray                 # prospectivity 0 … 1 (TOPSIS closeness C or VIKOR 1 − Q)
    scores: Dict[str, np.ma.MaskedArray]
    parts: Dict[str, np.ma.MaskedArray]      # TOPSIS: d_plus, d_minus; VIKOR: S, R, Q
    table: pd.DataFrame
    ahp: Optional[AHPResult]
    ahp_table: Optional[pd.DataFrame]
    study_mask: np.ndarray
    weights: np.ndarray
    method: str
    messages: List[str] = field(default_factory=list)


def entropy_weights(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Shannon-entropy weights of criteria (rows of S, cells in columns; M = missing)."""
    n = S.shape[0]
    d = np.zeros(n)
    for j in range(n):
        x = S[j][~M[j]]
        m = x.size
        tot = x.sum()
        if m < 2 or tot <= 0:
            continue
        p = x / tot
        p = p[p > 0]
        e = float(-(p * np.log(p)).sum() / np.log(m))
        d[j] = max(1.0 - e, 0.0)
    if d.sum() <= 0:
        return np.ones(n) / n
    return d / d.sum()


def _weights(layers, weighting, pairwise, delta, S, M, msgs):
    n = len(layers)
    ahp = None
    if weighting == "equal":
        w = np.ones(n)
    elif weighting == "custom":
        w = np.array([max(float(l.weight), 0.0) for l in layers])
    elif weighting == "entropy":
        w = entropy_weights(S, M)
    else:
        if pairwise is None:
            raise ValueError("AHP weighting needs a pairwise-comparison matrix")
        P = np.asarray(pairwise, float)
        if P.shape != (n, n):
            raise ValueError(f"the pairwise matrix must be {n} × {n} (one row per evidence layer)")
        ahp = ahp_weights(P) if weighting == "ahp" else fuzzy_ahp_weights(P, delta)
        w = ahp.weights
        if n > 2 and ahp.cr >= 0.1:
            msgs.append(f"AHP consistency ratio CR = {ahp.cr:.3f} ≥ 0.10 — revise the pairwise judgements")
    if w.sum() <= 0:
        raise ValueError("at least one layer needs a positive weight")
    return w / w.sum(), ahp


def _range01(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    lo, hi = x[valid].min(), x[valid].max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def mcdm(layers: Sequence[IOLayer], method: str = "topsis", weighting: str = "equal",
         pairwise: Optional[np.ndarray] = None, delta: float = 1.0, normalisation: str = "vector", v: float = 0.5,
         missing: str = "renormalise", study_mask: Optional[np.ndarray] = None,
         deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> MCDMResult:
    if not layers:
        raise ValueError("at least one evidence layer is required")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if normalisation not in NORMALISATIONS:
        raise ValueError(f"normalisation must be one of {NORMALISATIONS}")
    if not 0.0 <= v <= 1.0:
        raise ValueError("the VIKOR strategy weight v must lie in [0, 1]")
    msgs: List[str] = []
    n = len(layers)
    scores = {l.name: layer_scores(l) for l in layers}
    shape = next(iter(scores.values())).shape
    S = np.stack([np.ma.getdata(s) for s in scores.values()]).reshape(n, -1)
    M = np.stack([np.ma.getmaskarray(s) for s in scores.values()]).reshape(n, -1)
    base = (np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool)).ravel()
    study = base & (~M.any(0) if missing == "exclude" else ~M.all(0))
    if not study.any():
        raise ValueError("no cell has evidence in every layer — use missing = renormalise or check the layers")
    excluded = int((base & ~study).sum())
    if excluded:
        msgs.append(f"{excluded:,} study cells without evidence in some layer were "
                    + ("excluded" if missing == "exclude" else "left without any evidence"))
    Ms = M | ~study[None, :]                        # missing or outside the study area
    w, ahp = _weights(layers, weighting, pairwise, delta, S, Ms, msgs)
    obs = (~Ms).astype(float)
    W = obs * w[:, None]                            # per-cell weights, renormalised over observed criteria
    tot = W.sum(0)
    W = W / np.where(tot > 0, tot, 1.0)
    X = np.where(Ms, 0.0, S)
    for j, l in enumerate(layers):
        x = X[j][~Ms[j]]
        if x.size and np.ptp(x) == 0:
            msgs.append(f"{l.name}: the score is constant in the study area — it cannot discriminate")
    parts = {}
    if method == "topsis":
        if normalisation == "vector":
            norm = np.sqrt((X ** 2).sum(1))
            R = X / np.where(norm > 0, norm, 1.0)[:, None]
        else:
            R = X
        hi = np.array([R[j][~Ms[j]].max() if (~Ms[j]).any() else 0.0 for j in range(n)])
        lo = np.array([R[j][~Ms[j]].min() if (~Ms[j]).any() else 0.0 for j in range(n)])
        # v_ij = w_ij r_ij with the per-cell renormalised weights; the ideal points follow the same weights
        dp = np.sqrt((W ** 2 * (R - hi[:, None]) ** 2 * obs).sum(0))
        dm = np.sqrt((W ** 2 * (R - lo[:, None]) ** 2 * obs).sum(0))
        den = dp + dm
        idx = np.where(den > 0, dm / np.where(den > 0, den, 1.0), 0.5)
        parts["d_plus"], parts["d_minus"] = dp, dm
    else:
        hi = np.array([X[j][~Ms[j]].max() if (~Ms[j]).any() else 1.0 for j in range(n)])
        lo = np.array([X[j][~Ms[j]].min() if (~Ms[j]).any() else 0.0 for j in range(n)])
        span = np.where(hi > lo, hi - lo, 1.0)
        G = W * (hi[:, None] - X) / span[:, None] * obs           # weighted normalised regret per criterion
        Sv, Rv = G.sum(0), G.max(0)
        Q = v * _range01(Sv, study) + (1.0 - v) * _range01(Rv, study)
        idx = 1.0 - Q
        parts["S"], parts["R"], parts["Q"] = Sv, Rv, Q
    as_map = lambda a: np.ma.array(np.where(study, a, 0.0).reshape(shape), mask=~study.reshape(shape))  # noqa: E731
    index = as_map(idx)
    parts = {k: as_map(a) for k, a in parts.items()}
    st2 = study.reshape(shape)
    rows = []
    for i, l in enumerate(layers):
        s = scores[l.name]
        val = np.ma.getdata(s)[st2 & ~np.ma.getmaskarray(s)]
        spec = ("; ".join(f"{k:g}={v_:g}" for k, v_ in l.spec.items()) if l.rule == "classes" else
                f"{l.spec[0]:.6g} … {l.spec[1]:.6g}" if l.rule in ("positive", "negative") else f"{float(l.spec):.6g}")
        row = dict(layer=l.name, rule=l.rule, scores=spec, weight=float(w[i]),
                   mean_score=float(val.mean()) if val.size else np.nan,
                   favourable_area_pct=float(100 * np.mean(val >= 0.5)) if val.size else np.nan)
        if deposits is not None and len(deposits[0]):
            r, c = deposits
            ok = st2[r, c]
            row["mean_score_at_deposits"] = float(np.ma.getdata(s)[r[ok], c[ok]].mean()) if ok.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    ahp_table = None
    if ahp is not None:
        names = [l.name for l in layers]
        ahp_table = pd.DataFrame(ahp.matrix, columns=names)
        ahp_table.insert(0, "layer", names)
        ahp_table["weight"] = ahp.weights
        if ahp.fuzzy is not None:
            ahp_table["fuzzy_l"], ahp_table["fuzzy_m"], ahp_table["fuzzy_u"] = ahp.fuzzy.T
        ahp_table["lambda_max / CI / CR"] = [f"{ahp.lambda_max:.4f} / {ahp.ci:.4f} / {ahp.cr:.4f}"] + [""] * (n - 1)
    return MCDMResult(index, scores, parts, table, ahp, ahp_table, st2, w, method, msgs)
