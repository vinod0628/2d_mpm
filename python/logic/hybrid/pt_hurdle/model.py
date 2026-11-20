"""M4 PT-Hurdle — permissive-tract hurdle model (user's "Improved hybrid MPM methods", §2.4).

A necessary criterion (e.g. the host unit of a stratabound deposit) is a hurdle, not one evidence layer among others:

    G_i = σ(−(d_i − d0)/s_g)          d_i = distance (cells) outside the permissive area, 0 inside
    tract T = {i : G_i > g0}
    P_i = G_i · P^c_i                 P^c: conditional model trained with deposits and non-deposits inside T only

Training inside the tract avoids the "trivial negative" problem (a model that only learns host versus non-host).
G is normalised so that it is 1 inside the permissive area (the document's σ(d0/s_g) = 0.88 would scale every
permissive cell); d0 ≈ the positional error of the map, s_g ≈ d0/2. Optional masked smoothing of P^c along the tract:
P̃^c = g * (P^c 1_T) / g * 1_T.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.special import expit

from logic.hybrid.bs_wofe.model import bs_wofe
from logic.hybrid.common.knowledge import KnowledgeNetwork
from logic.hybrid.kp_rf.model import kp_rf

ENGINES = ("kp_rf", "bs_wofe", "rf")
ProgressFn = Optional[Callable[[float, str], None]]


@dataclass
class PTHurdleResult:
    prob: np.ndarray
    gate: np.ndarray
    cond: np.ndarray
    tract: np.ndarray
    expert: np.ndarray
    F: np.ndarray
    inner: object
    summary: pd.DataFrame
    stats: Dict[str, float]
    messages: List[str] = field(default_factory=list)


def permissive_mask(data: np.ma.MaskedArray, rule: str, values: str) -> np.ndarray:
    """rule: "in" (class list '2; 3'), "ge" / "le" (threshold)."""
    a = np.ma.getdata(data).astype(float)
    ok = ~np.ma.getmaskarray(data)
    vals = [float(v) for v in str(values).replace(",", ";").split(";") if v.strip()]
    if not vals:
        raise ValueError("the permissive criterion needs a value")
    if rule == "in":
        hit = np.zeros(a.shape, bool)
        for v in vals:
            hit |= np.isclose(a, v)
    elif rule == "ge":
        hit = a >= vals[0]
    elif rule == "le":
        hit = a <= vals[0]
    else:
        raise ValueError(f"unknown rule {rule}")
    return hit & ok


def host_gate(host: np.ndarray, d0: float = 2.0, s: float = 1.0, normalise: bool = True) -> np.ndarray:
    """Gate on the grid from a boolean permissive mask (distance in cells outside it)."""
    d = ndimage.distance_transform_edt(~host)
    g = expit(-(d - d0) / s)
    return np.minimum(g / expit(d0 / s), 1.0) if normalise else g


def pt_hurdle(net: KnowledgeNetwork, pos: np.ndarray, neg_cand: np.ndarray, host: np.ndarray, d0: float = 2.0,
              s_gate: float = 1.0, g0: float = 0.2, engine: str = "kp_rf", smooth: float = 0.0,
              normalise: bool = True, n_bags: int = 10, ratio: float = 3.0, a: float = 1.0, kappa: float = 20.0,
              n_trees: int = 200, n_classes: int = 5, kappa_w: float = 4.0, gamma: float = 2.0, seed: int = 0,
              progress: ProgressFn = None) -> PTHurdleResult:
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    say = progress or (lambda f, m: None)
    msgs = []
    G = host_gate(host, d0, s_gate, normalise).ravel()[net.cells]
    tract = G > g0
    if tract.sum() < 20:
        raise ValueError("The permissive tract is almost empty — check the criterion.")
    inside = pos[tract[pos]]
    out = int(pos.size - inside.size)
    if out:
        msgs.append(f"{out} of {pos.size} training deposits lie outside the permissive tract — they are not used for "
                    "training and get P ≈ 0; a necessary criterion should contain (nearly) every deposit")
    if inside.size < 3:
        raise ValueError("Fewer than 3 training deposits lie inside the permissive tract.")
    if engine == "bs_wofe":
        inner = bs_wofe(net, inside, n_classes, kappa_w, gamma, known=tract)
        pc = inner.prob / max(float(inner.prob[tract].max()), 1e-12)
    else:
        inner = kp_rf(net, inside, neg_cand, n_bags=n_bags, ratio=ratio, a=a if engine == "kp_rf" else 0.0,
                      lam=None if engine == "kp_rf" else 0.0, kappa=kappa, n_trees=n_trees, region=tract,
                      use_process_scores=engine == "kp_rf", seed=seed,
                      progress=lambda f, m: say(0.05 + 0.85 * f, m))
        pc = inner.prob
    msgs += inner.messages
    if smooth > 0:
        num = np.zeros(net.shape)
        den = np.zeros(net.shape)
        num.ravel()[net.cells] = pc * tract
        den.ravel()[net.cells] = tract
        num = ndimage.gaussian_filter(num, smooth)
        den = ndimage.gaussian_filter(den, smooth)
        sm = num.ravel()[net.cells] / np.maximum(den.ravel()[net.cells], 1e-9)
        pc = np.where(tract, sm, pc)
    prob = G * pc
    st = dict(deposits=int(pos.size), deposits_in_tract=int(inside.size), tract_pct=float(100 * tract.mean()),
              engine=engine, mean_prob=float(prob.mean()),
              deposits_tract_pct=float(100 * inside.size / max(pos.size, 1)))
    summary = pd.DataFrame([
        ("Training deposits / inside the tract", f"{pos.size} / {inside.size}"),
        ("Permissive tract (G > g0)", f"{st['tract_pct']:.1f} % of the study area (g0 = {g0:g})"),
        ("Gate", f"σ(−(d − {d0:g})/{s_gate:g}) with d = distance in cells outside the permissive area"
                 + (", normalised to 1 inside" if normalise else "")),
        ("Conditional model", {"kp_rf": "KP-RF (knowledge-pooled random forest)", "bs_wofe": "BS-WofE (÷ max in "
                               "the tract)", "rf": "random forest"}[engine]),
        ("Smoothing of P^c along the tract", f"Gaussian σ = {smooth:g} cells" if smooth > 0 else "none"),
    ], columns=["item", "value"])
    return PTHurdleResult(prob, G, pc, tract, inner.expert, inner.F, inner, summary, st, msgs)
