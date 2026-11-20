"""Layer symbology (renderers) without Qt: classification, per-class / per-feature styles, raster colour indexes.

Renderers (ArcMap terms):

* ``single``       — every feature / every valid raster cell with one symbol;
* ``unique``       — vectors: every feature its own symbol; rasters: every cell value its own colour;
* ``stretch``      — rasters only: a colour ramp stretched between min and max;
* ``categorized``  — classes from an attribute (vectors: a field of the attribute table; rasters: the cell value or a
  column of the class table) by unique values or by ranges (equal interval, quantile, natural breaks).

A category is a dict: ``value`` (unique values) or ``lo`` / ``hi`` (ranges), ``label``, ``count`` and the symbol
(``color`` — rasters / points / lines / polygon fill; polygons use ``facecolor``; plus ``edgecolor``, ``marker``,
``markersize``, ``linewidth`` for vectors).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import colors as mcolors

RENDERERS_RASTER = ("single", "unique", "stretch", "categorized")
RENDERERS_VECTOR = ("single", "unique", "categorized")
METHODS = ("unique", "equal", "quantile", "jenks")
VALUE_FIELD = "__value__"            # raster categorized by the cell value itself
MAX_UNIQUE = 1024                    # unique values renderer / classification limit
DEFAULT_RASTER_COLOR = "#4f8fd6"

QUALITATIVE = ("tab20", "tab10", "Set3", "Paired", "Set2", "Set1", "Accent", "Pastel1", "Dark2")


def get_cmap(name: str):
    try:
        return mpl.colormaps[name]
    except KeyError:
        return mpl.colormaps["viridis"]


def ramp_colors(cmap: str, n: int) -> List[str]:
    """``n`` colours from a colour map: qualitative maps cycle, continuous ones are sampled evenly."""
    cm = get_cmap(cmap)
    if n <= 0:
        return []
    if getattr(cm, "N", 256) <= 20:
        rgba = cm(np.arange(n) % cm.N)
    else:
        rgba = cm(np.arange(n) / max(1, n - 1))
    rgb = np.round(np.asarray(rgba)[:, :3] * 255).astype(int)     # one call instead of one per colour
    return ["#%02x%02x%02x" % (r, g, b) for r, g, b in rgb]


def to_py(v):
    """numpy scalar -> plain Python (JSON)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def is_numeric(values) -> bool:
    try:
        return np.issubdtype(np.asarray(values).dtype, np.number) and np.asarray(values).dtype != bool
    except TypeError:
        return False


# ------------------------------------------------------------------------------------------- classification
def jenks_breaks(values: np.ndarray, k: int) -> List[float]:
    """Natural breaks (Jenks / Fisher optimal 1-D classification) on at most 1000 representative values."""
    x = np.sort(np.asarray(values, float))
    if x.size > 1000:
        x = np.quantile(x, np.linspace(0, 1, 1000))
    n = x.size
    k = max(1, min(k, np.unique(x).size))
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def ssd(i, m):                               # squared deviations of x[i:m] (i vector, m scalar)
        cnt = m - i
        s = s1[m] - s1[i]
        return (s2[m] - s2[i]) - s * s / np.maximum(cnt, 1)

    cost = np.full((k + 1, n + 1), np.inf)
    arg = np.zeros((k + 1, n + 1), int)
    cost[1, 1:] = ssd(np.zeros(n, int), np.arange(1, n + 1))
    for c in range(2, k + 1):
        for m in range(c, n + 1):
            i = np.arange(c - 1, m)
            tot = cost[c - 1, i] + ssd(i, m)
            j = int(np.argmin(tot))
            cost[c, m], arg[c, m] = tot[j], i[j]
    cuts, m = [], n
    for c in range(k, 1, -1):
        m = arg[c, m]
        cuts.append(x[m - 1])
    return [float(x[0])] + sorted(float(v) for v in cuts) + [float(x[-1])]


def class_breaks(values, method: str, n: int) -> List[float]:
    """Break values (n + 1 edges, min ... max) for ``equal`` / ``quantile`` / ``jenks``."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [0.0, 1.0]
    lo, hi = float(v.min()), float(v.max())
    n = max(1, int(n))
    if hi <= lo:
        return [lo, hi]
    if method == "quantile":
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
        return [float(e) for e in edges]
    if method == "jenks":
        return jenks_breaks(v, n)
    return [float(e) for e in np.linspace(lo, hi, n + 1)]


def fmt_number(v) -> str:
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}"
    return f"{v:.6g}"


def _representative(values: np.ndarray, weights: Optional[np.ndarray], k: int = 2000) -> np.ndarray:
    """Values standing for a weighted sample (weighted quantiles) — for breaks of raster value / count pairs."""
    if weights is None:
        return values
    o = np.argsort(values)
    v, w = values[o], np.asarray(weights, float)[o]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return v
    q = np.linspace(0, cw[-1], min(k, max(2, int(cw[-1]))))
    return v[np.clip(np.searchsorted(cw, q, side="left"), 0, v.size - 1)]


def build_categories(values, method: str = "unique", n: int = 5, cmap: str = "tab20",
                     labels: Optional[Dict] = None, weights=None) -> List[dict]:
    """Categories for ``values`` (NaN / None are ignored).

    ``values`` is one entry per feature, or — with ``weights`` — distinct values and how often each occurs (raster
    cell counts). ``labels`` maps a unique value to its label (e.g. raster class names)."""
    arr = np.asarray(values, dtype=object)
    w = np.ones(arr.size) if weights is None else np.asarray(weights, float)
    keep = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in arr], bool)
    arr, w = arr[keep], w[keep]
    if arr.size == 0:
        return []
    plain = np.asarray(arr.tolist())
    numeric = is_numeric(plain)
    if method == "unique" or not numeric:
        vals, inv = np.unique(plain, return_inverse=True)
        if vals.size > MAX_UNIQUE:
            raise ValueError(f"{vals.size} unique values — more than {MAX_UNIQUE}; classify by ranges instead.")
        counts = np.bincount(inv.ravel(), weights=w, minlength=vals.size)
        cols = ramp_colors(cmap, vals.size)
        labels = labels or {}
        return [dict(value=to_py(v), label=str(labels.get(_key(v), labels.get(to_py(v), fmt_number(v) if numeric else v))),
                     count=int(round(c)), color=col) for v, c, col in zip(vals, counts, cols)]
    num = plain.astype(float)
    edges = class_breaks(_representative(num, None if weights is None else w), method, n)
    if len(edges) < 2:
        edges = [edges[0], edges[0]]
    cols = ramp_colors(cmap, len(edges) - 1)
    cats = [dict(lo=float(a), hi=float(b), label=f"{fmt_number(a)} – {fmt_number(b)}", count=0, color=cols[i])
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
    idx = assign_categories(num, cats)
    for i, c in enumerate(cats):
        c["count"] = int(round(w[idx == i].sum()))
    return cats


def assign_categories(values, cats: Sequence[dict]) -> np.ndarray:
    """Category index per value (-1 = no category). Ranges: first class [lo, hi], the others (lo, hi]."""
    arr = np.asarray(values, dtype=object) if not isinstance(values, np.ndarray) else values
    out = np.full(len(arr), -1, int)
    if not cats:
        return out
    if "lo" in cats[0]:
        try:
            num = np.asarray(arr, float)
        except (TypeError, ValueError):
            return out
        for i, c in enumerate(cats):
            lo, hi = float(c["lo"]), float(c["hi"])
            sel = (num <= hi) & ((num >= lo) if i == 0 else (num > lo))
            out[(out < 0) & sel] = i
        return out
    lut = {}
    for i, c in enumerate(cats):
        v = c["value"]
        lut.setdefault(_key(v), i)
    for j, v in enumerate(arr):
        out[j] = lut.get(_key(v), -1)
    return out


def _key(v):
    v = to_py(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return str(v)


# ------------------------------------------------------------------------------------------- vectors
def feature_styles(layer) -> List[dict]:
    """Resolved symbol of every feature of a vector layer (``None`` = not drawn: no category matched)."""
    base = dict(layer.style)
    n = len(layer.gdf)
    mode = getattr(layer, "renderer", "single")
    fill_key = "facecolor" if layer.geom_type == "polygon" else "color"
    if mode == "unique":
        u = layer.unique or {}
        cols = ramp_colors(u.get("cmap", "tab20"), n)
        over = u.get("styles", {})
        out = []
        for i in range(n):
            st = dict(base)
            st[fill_key] = cols[i]
            st.update(over.get(str(i), {}))
            out.append(st)
        return out
    if mode == "categorized":
        c = layer.categorized or {}
        cats = c.get("classes") or []
        field = c.get("field")
        if not cats or field not in layer.gdf.columns:
            return [dict(base) for _ in range(n)]
        idx = assign_categories(layer.gdf[field].to_numpy(), cats)
        out = []
        for i in idx:
            if i < 0:
                out.append(None)
                continue
            st = dict(base)
            cat = cats[i]
            for k in ("color", "facecolor", "edgecolor", "marker", "markersize", "linewidth"):
                if k in cat:
                    st[k] = cat[k]
            if layer.geom_type == "polygon" and "facecolor" not in cat and "color" in cat:
                st["facecolor"] = cat["color"]
            out.append(st)
        return out
    return [dict(base) for _ in range(n)]


def category_style(layer, cat: dict) -> dict:
    """Symbol of one category of a vector layer (for legends and swatches)."""
    st = dict(layer.style)
    for k in ("color", "facecolor", "edgecolor", "marker", "markersize", "linewidth"):
        if k in cat:
            st[k] = cat[k]
    if layer.geom_type == "polygon" and "facecolor" not in cat and "color" in cat:
        st["facecolor"] = cat["color"]
    return st


# ------------------------------------------------------------------------------------------- rasters
def raster_unique_colors(layer, classes: np.ndarray) -> List[str]:
    """Colours for the unique values renderer: user colours > class table 'color' column > colour map."""
    u = getattr(layer, "unique", None) or {}
    over = {float(k): v for k, v in (u.get("colors") or {}).items()}
    table = {}
    if layer.class_table is not None and "color" in layer.class_table.columns:
        table = {float(v): c for v, c in zip(layer.class_table["value"], layer.class_table["color"])
                 if isinstance(c, str) and c}
    base = ramp_colors(u.get("cmap") or layer.cmap, len(classes))
    out = []
    for i, c in enumerate(classes):
        f = float(c)
        out.append(over.get(f) or table.get(f) or base[i])
    return out


def raster_unique_labels(layer) -> Dict[float, str]:
    u = getattr(layer, "unique", None) or {}
    labels = dict(layer.labels()) if hasattr(layer, "labels") else {}
    labels.update({float(k): str(v) for k, v in (u.get("labels") or {}).items()})
    return labels


def raster_field_values(layer, field: str, values: np.ndarray):
    """Values of ``field`` for raster cell values (``VALUE_FIELD`` = the values themselves)."""
    if field in (None, VALUE_FIELD) or layer.class_table is None or field not in layer.class_table.columns:
        return values
    lut = {float(v): f for v, f in zip(layer.class_table["value"], layer.class_table[field])}
    return np.array([lut.get(float(v)) for v in values], dtype=object)


def raster_category_index(layer, data: np.ndarray) -> np.ma.MaskedArray:
    """Category index per cell for the categorized renderer (masked = NoData or no category)."""
    c = layer.categorized or {}
    cats = c.get("classes") or []
    field = c.get("field", VALUE_FIELD)
    mask = np.ma.getmaskarray(data)
    raw = np.ma.getdata(data)
    if field in (None, VALUE_FIELD) or layer.class_table is None or field not in layer.class_table.columns:
        if cats and "lo" in cats[0]:
            idx = np.full(raw.shape, -1, int)
            vals = raw.astype(float)
            for i, cat in enumerate(cats):
                lo, hi = float(cat["lo"]), float(cat["hi"])
                sel = (vals <= hi) & ((vals >= lo) if i == 0 else (vals > lo))
                idx[(idx < 0) & sel] = i
        else:
            uniq, inv = np.unique(raw, return_inverse=True)
            lut = assign_categories(uniq, cats)
            idx = lut[inv].reshape(raw.shape)
    else:
        uniq, inv = np.unique(raw, return_inverse=True)
        lut = assign_categories(raster_field_values(layer, field, uniq), cats)
        idx = lut[inv].reshape(raw.shape)
    return np.ma.array(idx, mask=mask | (idx < 0))
