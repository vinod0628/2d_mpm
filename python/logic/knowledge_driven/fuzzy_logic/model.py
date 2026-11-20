"""Fuzzy logic overlay for mineral prospectivity mapping (knowledge-driven) — NumPy + scikit-fuzzy, no GUI.

Bonham-Carter (1994, ch. 9), An et al. (1991): every evidence layer is transformed into a fuzzy membership μ ∈ [0, 1]
("how favourable is this value for the deposit model?") chosen by the geologist; memberships are combined by fuzzy
operators in an **inference network** — layers are grouped into the components of the mineral system (e.g. source,
pathway, trap, footprint; Porwal et al. 2003; McCuaig et al. 2010), each group is combined with its own operator, and
the groups are combined by a final operator. Deposits are not used to build the model — only to validate it.

Membership functions (x = evidence value; parameters a, b in evidence units):
    linear      μ = (x − a)/(b − a) clipped to [0, 1]    (a > b gives a decreasing function)
    s_shape     smooth S-curve from a (μ=0) to b (μ=1)    — skfuzzy.smf   (z_shape: decreasing, skfuzzy.zmf)
    logistic    μ = 1/(1 + exp(−s (x − i)))               — Yousefi & Carranza (2015); i = inflection, s = slope
                (skfuzzy.sigmf); default i = (p₂ + p₉₈)/2, s = 9.2/(p₉₈ − p₂) maps the 2–98 % range to 0.01–0.99
    gaussian    μ = exp(−(x − c)² / (2σ²))                — "near": favourable around c (skfuzzy.gaussmf)
    large       μ = 1/(1 + (x/m)^−f)                     — ArcGIS Fuzzy Large (m = midpoint, f = spread)
    small       μ = 1/(1 + (x/m)^f)                      — ArcGIS Fuzzy Small
    classes     a membership per class of a categorical layer
Operators (memberships μ₁ … μₙ):
    and = min μᵢ     or = max μᵢ     product = Π μᵢ     sum = 1 − Π (1 − μᵢ)
    gamma = sum^γ · product^(1 − γ)                          (Zimmermann & Zysno 1980)
    geometric mean = (Π μᵢ)^(1/n)                           (Yousefi & Carranza 2015, with logistic memberships)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from i18n import N_

MEMBERSHIPS = ("linear", "s_shape", "z_shape", "logistic", "gaussian", "large", "small", "classes")
MEMBERSHIP_LABELS = {"linear": N_("linear (a → b)"), "s_shape": N_("S-shaped (increasing)"),
                     "z_shape": N_("Z-shaped (decreasing)"), "logistic": N_("logistic (inflection, slope)"),
                     "gaussian": N_("Gaussian (near a value)"), "large": N_("large (ArcGIS)"),
                     "small": N_("small (ArcGIS)"), "classes": N_("membership per class")}
PARAM_HINTS = {"linear": "a, b", "s_shape": "a, b", "z_shape": "a, b", "logistic": "inflection, slope",
               "gaussian": "centre, sigma", "large": "midpoint, spread", "small": "midpoint, spread",
               "classes": "class=membership; …"}
OPERATORS = ("and", "or", "product", "sum", "gamma", "geomean")
OPERATOR_LABELS = {"and": N_("fuzzy AND (minimum)"), "or": N_("fuzzy OR (maximum)"),
                   "product": N_("fuzzy algebraic product"), "sum": N_("fuzzy algebraic sum"),
                   "gamma": N_("fuzzy gamma"), "geomean": N_("geometric average")}


def default_params(kind: str, values: np.ndarray, classes: Optional[Sequence[float]] = None) -> str:
    """Data-based starting parameters (the geologist is expected to edit them)."""
    v = values[np.isfinite(values)]
    if kind == "classes":
        return "; ".join(f"{c:g}=0.5" for c in (classes if classes is not None else np.unique(v)))
    if v.size == 0:
        return "0, 1"
    p2, p50, p98 = np.percentile(v, [2, 50, 98])
    if p98 <= p2:
        p2, p98 = float(v.min()), float(v.max()) + 1e-9
    if kind in ("linear", "s_shape"):
        return f"{p2:.6g}, {p98:.6g}"
    if kind == "z_shape":
        return f"{p2:.6g}, {p98:.6g}"
    if kind == "logistic":
        return f"{(p2 + p98) / 2:.6g}, {9.2 / (p98 - p2):.6g}"
    if kind == "gaussian":
        return f"{p50:.6g}, {(p98 - p2) / 4:.6g}"
    if kind in ("large", "small"):
        return f"{p50:.6g}, 5"
    return ""


def parse_params(kind: str, text: str):
    if kind == "classes":
        out = {}
        for part in str(text).replace(",", ";").split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[float(k.strip())] = float(v.strip())
        if not out:
            raise ValueError("give class memberships as 'class=membership; …'")
        if any(not 0 <= v <= 1 for v in out.values()):
            raise ValueError("class memberships must be between 0 and 1")
        return out
    vals = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    if len(vals) != 2:
        raise ValueError(f"{kind}: give two parameters ({PARAM_HINTS[kind]})")
    a, b = vals
    if kind in ("linear", "s_shape", "z_shape") and a == b:
        raise ValueError(f"{kind}: a and b must differ")
    if kind in ("gaussian",) and b <= 0:
        raise ValueError("gaussian: sigma must be positive")
    if kind in ("large", "small") and a <= 0:
        raise ValueError(f"{kind}: midpoint must be positive")
    return a, b


def membership(x: np.ndarray, kind: str, params) -> np.ndarray:
    """Fuzzy membership of the values ``x`` (NaN stays NaN)."""
    import skfuzzy as fuzz
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    v = x[ok]
    if kind == "classes":
        mu = np.full(v.shape, np.nan)
        for c, m in params.items():
            mu[np.isclose(v, c)] = m
        mu = np.where(np.isnan(mu), 0.0, mu)                 # classes not listed: not favourable
    elif kind == "linear":
        a, b = params
        mu = np.clip((v - a) / (b - a), 0.0, 1.0)
    elif kind == "s_shape":
        a, b = sorted(params)
        mu = fuzz.smf(v, a, b)
    elif kind == "z_shape":
        a, b = sorted(params)
        mu = fuzz.zmf(v, a, b)
    elif kind == "logistic":
        i, s = params
        mu = fuzz.sigmf(v, i, s)
    elif kind == "gaussian":
        c, sd = params
        mu = fuzz.gaussmf(v, c, sd)
    elif kind in ("large", "small"):
        m, f = params
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            r = np.where(v > 0, v / m, 0.0)
            mu = 1.0 / (1.0 + np.power(np.maximum(r, 1e-300), -f if kind == "large" else f))
        mu = np.where(v <= 0, 0.0 if kind == "large" else 1.0, mu)
    else:
        raise ValueError(f"unknown membership function {kind}")
    out[ok] = np.clip(mu, 0.0, 1.0)
    return out


def combine(stack: np.ndarray, operator: str, gamma: float = 0.9) -> np.ndarray:
    """Fuzzy operator over axis 0 of (n, …) memberships."""
    if operator == "and":
        return np.min(stack, 0)
    if operator == "or":
        return np.max(stack, 0)
    if operator == "product":
        return np.prod(stack, 0)
    if operator == "sum":
        return 1.0 - np.prod(1.0 - stack, 0)
    if operator == "gamma":
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        return (1.0 - np.prod(1.0 - stack, 0)) ** gamma * np.prod(stack, 0) ** (1.0 - gamma)
    if operator == "geomean":
        return np.prod(stack, 0) ** (1.0 / stack.shape[0])
    raise ValueError(f"unknown operator {operator}")


@dataclass
class FuzzyLayer:
    name: str
    array: np.ma.MaskedArray
    kind: str                     # membership function
    params: object                # parsed parameters
    group: str = "1"


@dataclass
class FuzzyResult:
    favourability: np.ma.MaskedArray
    memberships: Dict[str, np.ma.MaskedArray]
    groups: Dict[str, np.ma.MaskedArray]
    network: pd.DataFrame
    layer_table: pd.DataFrame
    study_mask: np.ndarray
    messages: List[str] = field(default_factory=list)


def fuzzy_overlay(layers: Sequence[FuzzyLayer], group_ops: Dict[str, Tuple[str, float]], final_op: Tuple[str, float],
                  study_mask: Optional[np.ndarray] = None, missing: str = "exclude",
                  missing_value: float = 0.5) -> FuzzyResult:
    """Inference network: memberships → group operators → final operator. ``missing``: "exclude" (cells with
    missing evidence leave the study area) or "value" (missing membership = ``missing_value``)."""
    if not layers:
        raise ValueError("At least one evidence layer is required.")
    shape = np.ma.asarray(layers[0].array).shape
    known = [~np.ma.getmaskarray(np.ma.asarray(l.array)) for l in layers]
    base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool).copy()
    if study_mask is None:
        base &= np.logical_or.reduce(known)
    msgs = []
    if missing == "exclude":
        base &= np.logical_and.reduce(known)
    mus, rows = {}, []
    for l, kn in zip(layers, known):
        if np.ma.asarray(l.array).shape != shape:
            raise ValueError(f"Evidence '{l.name}' is not on the same grid as '{layers[0].name}'.")
        x = np.where(kn, np.ma.getdata(np.ma.asarray(l.array)).astype(float), np.nan)
        mu = membership(x, l.kind, l.params)
        if missing == "value":
            mu = np.where(np.isnan(mu), missing_value, mu)
        mus[l.name] = mu
        m = mu[base]
        rows.append(dict(layer=l.name, group=l.group, membership=l.kind, parameters=str(l.params)[:120],
                         mean=float(np.nanmean(m)), min=float(np.nanmin(m)), max=float(np.nanmax(m)),
                         area_above_0_5=float(np.nanmean(m >= 0.5))))
    groups, net = {}, []
    order = list(dict.fromkeys(l.group for l in layers))
    for gname in order:
        members = [l.name for l in layers if l.group == gname]
        op, gam = group_ops.get(gname, ("and", 0.9))
        g = combine(np.stack([mus[n] for n in members]), op, gam) if len(members) > 1 else mus[members[0]].copy()
        groups[gname] = g
        net.append(dict(node=f"group {gname}", inputs=", ".join(members), operator=op if len(members) > 1 else "—",
                        gamma=gam if op == "gamma" and len(members) > 1 else None,
                        mean=float(np.nanmean(g[base]))))
    fop, fgam = final_op
    fav = combine(np.stack([groups[g] for g in order]), fop, fgam) if len(order) > 1 else groups[order[0]]
    net.append(dict(node="favourability", inputs=", ".join(f"group {g}" for g in order),
                    operator=fop if len(order) > 1 else "—", gamma=fgam if fop == "gamma" and len(order) > 1 else None,
                    mean=float(np.nanmean(fav[base]))))
    if fop in ("product",) and len(order) > 2:
        msgs.append("the product of several groups becomes very small — consider gamma or the geometric average")

    def ma(a):
        return np.ma.array(np.nan_to_num(a, nan=0.0), mask=~base | np.isnan(a))
    return FuzzyResult(ma(fav), {k: ma(v) for k, v in mus.items()}, {k: ma(v) for k, v in groups.items()},
                       pd.DataFrame(net), pd.DataFrame(rows), base, msgs)
