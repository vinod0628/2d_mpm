"""Logistic-function fuzzification + geometric average (Yousefi & Carranza 2015) — knowledge-driven, no GUI.

Yousefi & Carranza (2015a, Comput. Geosci. 74): continuous evidence values are transformed into fuzzy scores by a
logistic function without expert thresholds — the inflection point i and slope s are set from the range of the data
so that its minimum and maximum map to ≈ 0.01 and ≈ 0.99:

    F(x) = 1 / (1 + e^(−s (x − i))),     i = (min + max) / 2,     s = ± 9.2 / (max − min)

(+ for positive evidence — high values favourable, − for negative evidence; 9.2 = 2 ln 99). Categorical evidence gets
expert scores per class. Yousefi & Carranza (2015b, Comput. Geosci. 83): the fuzzified layers are combined by their
geometric average — a multi-criteria decision-making operator that, unlike the product, does not shrink with the
number of layers and, unlike the minimum, uses every layer:

    G = (Π F_k^{w_k})^(1 / Σ w_k)          (w_k = 1: plain geometric average)

Optional weights from the prediction–area (P–A) plot of every fuzzified layer (Yousefi & Carranza 2015c, Comput.
Geosci. 79): at the intersection of the prediction-rate and occupied-area curves, Nd = Pr / Oa (normalised density)
and w = ln(Nd) — layers that capture more deposits in less area weigh more (uses the deposits → data-guided).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_

DIRECTIONS = ("positive", "negative", "classes")
DIRECTION_LABELS = {"positive": N_("positive (high values favourable)"),
                    "negative": N_("negative (low values favourable)"), "classes": N_("score per class")}
WEIGHTINGS = ("equal", "pa", "custom")
WEIGHTING_LABELS = {"equal": N_("equal (geometric average)"), "pa": N_("from the P–A plot of each layer (ln Nd)"),
                    "custom": N_("custom weight per layer")}


def logistic_parameters(lo: float, hi: float, direction: str) -> Tuple[float, float]:
    """(i, s) of Yousefi & Carranza (2015a) for the value range [lo, hi]."""
    if not hi > lo:
        raise ValueError("the value range must have max > min")
    s = 9.2 / (hi - lo)
    return (lo + hi) / 2.0, (s if direction == "positive" else -s)


def logistic(x, i, s):
    from scipy.special import expit
    return expit(s * (np.asarray(x, float) - i))


def parse_range(text: str) -> Tuple[float, float]:
    vals = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    if len(vals) != 2 or not vals[1] > vals[0]:
        raise ValueError("give the value range as 'min, max' with max > min")
    return vals[0], vals[1]


def parse_scores(text: str) -> Dict[float, float]:
    out = {}
    for part in str(text).replace(",", ";").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[float(k)] = float(v)
    if not out or any(not 0 < v <= 1 for v in out.values()):
        raise ValueError("give class scores as 'class=score; …' with 0 < score ≤ 1")
    return out


@dataclass
class LGLayer:
    name: str
    array: np.ma.MaskedArray
    direction: str               # positive | negative | classes
    spec: object                 # (min, max) or {class: score}
    weight: float = 1.0


@dataclass
class LGResult:
    favourability: np.ma.MaskedArray
    fuzzified: Dict[str, np.ma.MaskedArray]
    table: pd.DataFrame
    study_mask: np.ndarray
    messages: List[str] = field(default_factory=list)


def fuzzify(layer: LGLayer, x: np.ndarray) -> Tuple[np.ndarray, str]:
    if layer.direction == "classes":
        f = np.full(x.shape, np.nan)
        ok = np.isfinite(x)
        f[ok] = 0.01                                     # classes not scored: lowest score, never exactly 0
        for c, v in layer.spec.items():
            f[ok & np.isclose(x, c)] = v
        return f, "; ".join(f"{c:g}={v:g}" for c, v in layer.spec.items())
    i, s = logistic_parameters(*layer.spec, layer.direction)
    return logistic(x, i, s), f"i = {i:.6g}, s = {s:.6g}"


def logistic_geometric(layers: Sequence[LGLayer], weighting: str = "equal", study_mask: Optional[np.ndarray] = None,
                       deposits: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> LGResult:
    if not layers:
        raise ValueError("At least one evidence layer is required.")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    shape = np.ma.asarray(layers[0].array).shape
    known = [~np.ma.getmaskarray(np.ma.asarray(l.array)) for l in layers]
    base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool).copy()
    base &= np.logical_and.reduce(known)
    if not base.any():
        raise ValueError("The study area is empty (no cell has all evidence).")
    msgs, fz, rows = [], {}, []
    for l in layers:
        if np.ma.asarray(l.array).shape != shape:
            raise ValueError(f"Evidence '{l.name}' is not on the same grid as '{layers[0].name}'.")
        x = np.where(base, np.ma.getdata(np.ma.asarray(l.array)).astype(float), np.nan)
        f, desc = fuzzify(l, x)
        fz[l.name] = np.clip(f, 1e-6, 1.0)
        rows.append(dict(layer=l.name, direction=l.direction, function=desc, mean=float(np.nanmean(f[base]))))
    weights = {l.name: float(l.weight) for l in layers}
    if weighting == "equal":
        weights = {k: 1.0 for k in weights}
    elif weighting == "pa":
        if deposits is None or not len(deposits[0]):
            raise ValueError("P–A weights need deposits.")
        from evaluation.metrics import evaluate
        for r, l in zip(rows, layers):
            ev = evaluate(np.ma.array(fz[l.name], mask=~base), deposits[0], deposits[1], valid_mask=base)
            pa = ev.curves["training"].pa_intersection
            pr, oa = pa.get("prediction", np.nan) / 100.0, pa.get("area", np.nan) / 100.0
            nd = pr / oa if oa and oa > 0 else np.nan
            w = float(np.log(nd)) if np.isfinite(nd) and nd > 0 else 0.0
            if w <= 0:
                msgs.append(f"{l.name}: P–A normalised density ≤ 1 (no better than random) — weight 0")
                w = 0.0
            weights[l.name] = w
            r.update(pa_prediction_pct=100 * pr, pa_area_pct=100 * oa, normalised_density=nd)
        if sum(weights.values()) <= 0:
            raise ValueError("No layer is better than random on the P–A plot — P–A weights are undefined.")
    for r in rows:
        r["weight"] = weights[r["layer"]]
    wsum = sum(weights.values())
    if wsum <= 0:
        raise ValueError("The weights must sum to a positive number.")
    logg = sum(weights[k] * np.log(v) for k, v in fz.items()) / wsum
    fav = np.exp(logg)
    if len(layers) == 1:
        msgs.append("one evidence layer: the favourability is its fuzzified map")

    def ma(a):
        return np.ma.array(np.nan_to_num(a, nan=0.0), mask=~base)
    return LGResult(ma(fav), {k: ma(v) for k, v in fz.items()}, pd.DataFrame(rows), base, msgs)
