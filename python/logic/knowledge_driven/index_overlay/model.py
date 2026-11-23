"""Index overlay with expert, AHP or fuzzy-AHP layer weights — knowledge-driven, no GUI.

Bonham-Carter (1994, ch. 9): every evidence map i is converted to scores s_ij (class j) by the expert and the maps
are combined by their weighted mean

    S = Σ_i W_i s_i / Σ_i W_i

* **binary index overlay** — each map is a 0/1 pattern (condition ≥ / ≤ a threshold, or a set of classes);
* **multi-class index overlay** — each class has a score (categorical maps), continuous maps are rescaled linearly
  between two values in the expert direction (0 … 1).

Layer weights W_i: equal, custom, from an **AHP** pairwise-comparison matrix (Saaty 1980: principal eigenvector,
consistency ratio CR = CI/RI, CI = (λ_max − n)/(n − 1); CR < 0.1 acceptable) or from a **fuzzy AHP** matrix in which
every judgement a is the triangular fuzzy number (a − δ, a, a + δ) on Saaty's 1–9 scale (reciprocals inverted), with
weights by Buckley's (1985) geometric-mean method and centroid defuzzification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_

RULES = ("classes", "positive", "negative", "ge", "le")
RULE_LABELS = {"classes": N_("score per class"), "positive": N_("linear, high values favourable"),
               "negative": N_("linear, low values favourable"), "ge": N_("binary: value ≥ threshold"),
               "le": N_("binary: value ≤ threshold")}
WEIGHTINGS = ("equal", "custom", "ahp", "fuzzy_ahp")
WEIGHTING_LABELS = {"equal": N_("equal"), "custom": N_("custom weight per layer"),
                    "ahp": N_("AHP pairwise comparison (Saaty)"),
                    "fuzzy_ahp": N_("fuzzy AHP (triangular judgements, Buckley)")}
MISSING = ("exclude", "renormalise")
# Saaty's random consistency index RI(n), n = 1 … 15 (Saaty 1980)
RANDOM_INDEX = (0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59)


@dataclass
class IOLayer:
    name: str
    array: np.ma.MaskedArray
    rule: str
    spec: object                   # classes: {class: score}; positive / negative: (lo, hi); ge / le: threshold
    weight: float = 1.0            # custom weight


@dataclass
class AHPResult:
    weights: np.ndarray
    lambda_max: float
    ci: float
    cr: float
    matrix: np.ndarray
    fuzzy: Optional[np.ndarray] = None      # n × 3 fuzzy weights (l, m, u), fuzzy AHP only


@dataclass
class IOResult:
    index: np.ma.MaskedArray
    scores: Dict[str, np.ma.MaskedArray]
    table: pd.DataFrame
    ahp: Optional[AHPResult]
    ahp_table: Optional[pd.DataFrame]
    study_mask: np.ndarray
    weights: np.ndarray
    messages: List[str] = field(default_factory=list)


# ------------------------------------------------------------------ parsing
def parse_scores(text: str) -> Dict[float, float]:
    out = {}
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("class scores must look like '2=1; 1=0'")
        k, v = part.split("=", 1)
        out[float(k)] = float(v)
    if not out:
        raise ValueError("give a score for at least one class")
    return out


def parse_range(text: str) -> Tuple[float, float]:
    vals = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    if len(vals) != 2 or vals[1] == vals[0]:
        raise ValueError("give the value range as 'low, high' with two different values")
    return min(vals), max(vals)


def parse_threshold(text: str) -> float:
    try:
        return float(str(text).strip())
    except ValueError:
        raise ValueError("give one threshold value") from None


def parse_judgement(text) -> float:
    """'3', '1/5', '0.2' -> positive number (Saaty scale)."""
    t = str(text).strip()
    if "/" in t:
        a, b = t.split("/", 1)
        v = float(a) / float(b)
    else:
        v = float(t)
    if not v > 0:
        raise ValueError("pairwise judgements must be positive")
    return v


# ------------------------------------------------------------------ scores
def layer_scores(layer: IOLayer) -> np.ma.MaskedArray:
    a = np.ma.asarray(layer.array).astype(float)
    x = np.ma.getdata(a)
    m = np.ma.getmaskarray(a) | ~np.isfinite(x)
    if layer.rule == "classes":
        s = np.full(x.shape, np.nan)
        for c, v in layer.spec.items():
            s[np.isclose(x, c)] = v
        m = m | np.isnan(s)                     # classes without a score: no data
    elif layer.rule in ("positive", "negative"):
        lo, hi = layer.spec
        s = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        if layer.rule == "negative":
            s = 1.0 - s
    elif layer.rule == "ge":
        s = (x >= float(layer.spec)).astype(float)
    elif layer.rule == "le":
        s = (x <= float(layer.spec)).astype(float)
    else:
        raise ValueError(f"unknown rule {layer.rule}")
    return np.ma.array(np.nan_to_num(s), mask=m)


# ------------------------------------------------------------------ AHP
def ahp_weights(A: np.ndarray) -> AHPResult:
    """Principal right eigenvector of a positive reciprocal matrix, λ_max, CI and CR (Saaty 1980)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    if A.shape != (n, n) or np.any(A <= 0):
        raise ValueError("the pairwise matrix must be square and positive")
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX[min(n, len(RANDOM_INDEX)) - 1] if n >= 1 else 0.0
    cr = ci / ri if ri > 0 else 0.0
    return AHPResult(w, lam, float(ci), float(cr), A)


def reciprocal_matrix(upper: Dict[Tuple[int, int], float], n: int) -> np.ndarray:
    """Full reciprocal matrix from the judgements a_ij (i < j)."""
    A = np.ones((n, n))
    for (i, j), v in upper.items():
        A[i, j] = v
        A[j, i] = 1.0 / v
    return A


def fuzzy_ahp_weights(A: np.ndarray, delta: float = 1.0) -> AHPResult:
    """Fuzzy AHP with triangular fuzzy numbers and Buckley's (1985) geometric-mean method.

    Judgement a ≥ 1 -> (max(1, a − δ), a, min(9, a + δ)); a < 1 -> reciprocal of the fuzzy number of 1/a;
    a = 1 on the diagonal stays (1, 1, 1). Fuzzy geometric mean of row i: r_i = (Π_j l_ij, Π m_ij, Π u_ij)^(1/n);
    fuzzy weight w_i = r_i ⊗ (Σ r)^-1 = (l_i/Σu, m_i/Σm, u_i/Σl); crisp weight = centroid (l + m + u)/3,
    normalised. Consistency is reported for the crisp (middle) matrix."""
    A = np.asarray(A, float)
    n = A.shape[0]
    L, M, U = np.ones((n, n)), np.ones((n, n)), np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = A[i, j]
            if a >= 1:
                l, m, u = max(1.0, a - delta), a, min(9.0, a + delta)
            else:
                b = 1.0 / a
                lb, mb, ub = max(1.0, b - delta), b, min(9.0, b + delta)
                l, m, u = 1.0 / ub, 1.0 / mb, 1.0 / lb
            L[i, j], M[i, j], U[i, j] = l, m, u
    rl, rm, ru = (np.prod(X, axis=1) ** (1.0 / n) for X in (L, M, U))
    fw = np.column_stack([rl / ru.sum(), rm / rm.sum(), ru / rl.sum()])
    crisp = fw.mean(1)
    crisp = crisp / crisp.sum()
    base = ahp_weights(A)
    return AHPResult(crisp, base.lambda_max, base.ci, base.cr, A, fw)


# ------------------------------------------------------------------ overlay
def index_overlay(layers: Sequence[IOLayer], weighting: str = "equal", pairwise: Optional[np.ndarray] = None,
                  delta: float = 1.0, missing: str = "renormalise", study_mask: Optional[np.ndarray] = None,
                  deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> IOResult:
    if not layers:
        raise ValueError("at least one evidence layer is required")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    msgs: List[str] = []
    n = len(layers)
    ahp = None
    if weighting == "equal":
        w = np.ones(n)
    elif weighting == "custom":
        w = np.array([max(float(l.weight), 0.0) for l in layers])
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
    w = w / w.sum()
    scores = {l.name: layer_scores(l) for l in layers}
    shape = next(iter(scores.values())).shape
    S = np.stack([np.ma.getdata(s) for s in scores.values()])
    M = np.stack([np.ma.getmaskarray(s) for s in scores.values()])
    base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool)
    if missing == "exclude":
        study = base & ~M.any(0)
        idx = np.tensordot(w, S, axes=1)
    else:
        study = base & ~M.all(0)
        ww = np.where(M, 0.0, w[:, None, None])
        tot = ww.sum(0)
        idx = (ww * S).sum(0) / np.where(tot > 0, tot, 1.0)
    if not study.any():
        raise ValueError("no cell has evidence in every layer — use missing = renormalise or check the layers")
    excluded = int((base & ~study).sum())
    if excluded:
        msgs.append(f"{excluded:,} study cells without evidence in some layer were "
                    + ("excluded" if missing == "exclude" else "left without any evidence"))
    index = np.ma.array(np.where(study, idx, 0.0), mask=~study)
    rows = []
    for i, l in enumerate(layers):
        s = scores[l.name]
        v = np.ma.getdata(s)[study & ~np.ma.getmaskarray(s)]
        spec = ("; ".join(f"{k:g}={v_:g}" for k, v_ in l.spec.items()) if l.rule == "classes" else
                f"{l.spec[0]:.6g} … {l.spec[1]:.6g}" if l.rule in ("positive", "negative") else f"{float(l.spec):.6g}")
        row = dict(layer=l.name, rule=l.rule, scores=spec, weight=float(w[i]), mean_score=float(v.mean()) if v.size else np.nan,
                   favourable_area_pct=float(100 * np.mean(v >= 0.5)) if v.size else np.nan)
        if deposits is not None and len(deposits[0]):
            r, c = deposits
            ok = study[r, c]
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
    return IOResult(index, scores, table, ahp, ahp_table, study, w, msgs)
