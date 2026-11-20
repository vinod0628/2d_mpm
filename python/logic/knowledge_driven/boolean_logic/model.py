"""Boolean logic overlay for mineral prospectivity mapping (knowledge-driven) — NumPy, no GUI.

Bonham-Carter (1994, ch. 9): every evidence layer is turned into a binary map by a condition chosen from the deposit
model (e.g. "within 2 km of a fault", "favourable host rock", "As ≥ anomaly threshold"); the binary maps are combined
with Boolean AND / OR. As for the fuzzy overlay, conditions can be grouped into mineral-system components (inference
network): each group is combined with AND or OR, the groups with a final AND or OR. Deposits only validate the map.

Conditions:  ≥ t,  ≤ t,  between a and b (inclusive),  outside a … b,  in classes {c₁, c₂, …} (categorical).
Outputs: the favourable area (1 / 0), the number of satisfied conditions (0 … n — a simple index overlay that ranks
cells and can be validated by ROC), per-condition statistics (area, deposits inside, ratio of deposit share to area
share — Bonham-Carter's "how much better than random").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from i18n import N_

CONDITIONS = ("ge", "le", "between", "outside", "in")
CONDITION_LABELS = {"ge": N_("≥ value"), "le": N_("≤ value"), "between": N_("between a and b"),
                    "outside": N_("outside a … b"), "in": N_("in classes")}
VALUE_HINTS = {"ge": "t", "le": "t", "between": "a, b", "outside": "a, b", "in": "c1, c2, …"}
BOOL_OPS = ("and", "or")
BOOL_OP_LABELS = {"and": N_("AND (all true)"), "or": N_("OR (any true)")}


def parse_values(cond: str, text: str):
    vals = [float(v) for v in str(text).replace(";", ",").split(",") if v.strip()]
    if cond in ("ge", "le"):
        if len(vals) != 1:
            raise ValueError("give one threshold")
        return vals[0]
    if cond in ("between", "outside"):
        if len(vals) != 2:
            raise ValueError("give two values a, b")
        return tuple(sorted(vals))
    if not vals:
        raise ValueError("give one or more classes")
    return tuple(vals)


def condition(x: np.ndarray, cond: str, values) -> np.ndarray:
    if cond == "ge":
        return x >= values
    if cond == "le":
        return x <= values
    if cond == "between":
        return (x >= values[0]) & (x <= values[1])
    if cond == "outside":
        return (x < values[0]) | (x > values[1])
    if cond == "in":
        return np.isin(x, np.asarray(values, float))
    raise ValueError(f"unknown condition {cond}")


@dataclass
class BoolLayer:
    name: str
    array: np.ma.MaskedArray
    cond: str
    values: object
    group: str = "1"


@dataclass
class BooleanResult:
    favourable: np.ma.MaskedArray          # 1 / 0
    count: np.ma.MaskedArray               # number of true conditions
    conditions: Dict[str, np.ma.MaskedArray]
    table: pd.DataFrame
    network: pd.DataFrame
    study_mask: np.ndarray
    messages: List[str] = field(default_factory=list)


def boolean_overlay(layers: Sequence[BoolLayer], group_ops: Dict[str, str], final_op: str,
                    study_mask: Optional[np.ndarray] = None, deposits: Optional[Sequence[np.ndarray]] = None
                    ) -> BooleanResult:
    """Combine the binary conditions; ``deposits`` = (rows, cols) for the per-condition statistics (optional)."""
    if not layers:
        raise ValueError("At least one evidence layer is required.")
    shape = np.ma.asarray(layers[0].array).shape
    known = [~np.ma.getmaskarray(np.ma.asarray(l.array)) for l in layers]
    base = np.ones(shape, bool) if study_mask is None else np.asarray(study_mask, bool).copy()
    base &= np.logical_and.reduce(known)                 # a condition on missing evidence is undefined
    if not base.any():
        raise ValueError("The study area is empty (no cell has all evidence).")
    dep = np.zeros(shape, bool)
    if deposits is not None and len(deposits[0]):
        dep[np.asarray(deposits[0], int), np.asarray(deposits[1], int)] = True
    dep &= base
    n_area, n_dep = base.sum(), dep.sum()
    conds, rows = {}, []
    for l in layers:
        if np.ma.asarray(l.array).shape != shape:
            raise ValueError(f"Evidence '{l.name}' is not on the same grid as '{layers[0].name}'.")
        b = condition(np.ma.getdata(np.ma.asarray(l.array)).astype(float), l.cond, l.values) & base
        conds[l.name] = b
        area = b.sum() / n_area
        dshare = (b & dep).sum() / n_dep if n_dep else np.nan
        rows.append(dict(layer=l.name, group=l.group, condition=l.cond, values=str(l.values), area_pct=100 * area,
                         deposits_inside=int((b & dep).sum()), deposits_pct=100 * dshare if n_dep else None,
                         ratio=dshare / area if n_dep and area > 0 else None))
    order = list(dict.fromkeys(l.group for l in layers))
    groups, net = {}, []
    for g in order:
        members = [l.name for l in layers if l.group == g]
        op = group_ops.get(g, "and")
        st = np.stack([conds[n] for n in members])
        groups[g] = st.all(0) if op == "and" else st.any(0)
        net.append(dict(node=f"group {g}", inputs=", ".join(members), operator=op if len(members) > 1 else "—",
                        area_pct=100 * (groups[g] & base).sum() / n_area))
    st = np.stack([groups[g] for g in order])
    fav = (st.all(0) if final_op == "and" else st.any(0)) & base
    net.append(dict(node="favourable", inputs=", ".join(f"group {g}" for g in order),
                    operator=final_op if len(order) > 1 else "—", area_pct=100 * fav.sum() / n_area))
    count = np.sum(np.stack(list(conds.values())), 0)
    msgs = []
    if n_dep:
        inside = (fav & dep).sum()
        msgs.append(f"favourable area {100 * fav.sum() / n_area:.1f} % of the study area contains {inside} of {n_dep} "
                    f"deposits ({100 * inside / n_dep:.0f} %)")
        rows.append(dict(layer="FAVOURABLE (combined)", group="", condition=final_op, values="",
                         area_pct=100 * fav.sum() / n_area, deposits_inside=int(inside),
                         deposits_pct=100 * inside / n_dep,
                         ratio=(inside / n_dep) / (fav.sum() / n_area) if fav.sum() else None))
    if not fav.any():
        msgs.append("no cell satisfies the combined conditions — relax a threshold or use OR")

    def ma(a, dtype=float):
        return np.ma.array(a.astype(dtype), mask=~base)
    return BooleanResult(ma(fav, np.uint8), ma(count, np.int16), {k: ma(v, np.uint8) for k, v in conds.items()},
                         pd.DataFrame(rows), pd.DataFrame(net), base, msgs)
