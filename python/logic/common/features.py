"""Evidence rasters -> feature matrix for the machine-learning methods (random forest, SVM, MLP, boosting …).

A :class:`FeatureStack` encodes the study-area cells once: continuous layers as their values, categorical layers as
0/1 one-hot columns (one per class — tree models and neural networks need no reference class). Missing evidence is
either excluded from the study area or imputed (median of a continuous layer, most frequent class of a categorical
layer). ``matrix(idx)`` returns the rows of any study cells (training samples or prediction chunks).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from i18n import N_

FEATURE_KINDS = ("continuous", "categorical")
FEATURE_KIND_LABELS = {"continuous": N_("Continuous"), "categorical": N_("Categorical (one-hot)")}
MISSING_RULES = ("exclude", "impute", "keep")      # keep = NaN (only for models with native missing-value support)
MISSING_RULE_LABELS = {"exclude": N_("exclude cells from study area"),
                       "impute": N_("median (continuous) / most frequent class (categorical)")}
MAX_CLASSES = 255


@dataclass
class Feature:
    """One evidence layer. ``array`` masked = missing evidence."""

    name: str
    array: np.ma.MaskedArray
    kind: str = "continuous"
    labels: Optional[Dict[float, str]] = None

    def __post_init__(self):
        if self.kind not in FEATURE_KINDS:
            raise ValueError(f"{self.name}: kind must be one of {FEATURE_KINDS}")


def study_area(features: Sequence[Feature], study_mask: Optional[np.ndarray], missing: str) -> np.ndarray:
    """Study cells: the mask (or cells where any layer has data); with ``missing="exclude"`` only cells where every
    layer has data."""
    known = [~np.ma.getmaskarray(np.ma.asarray(f.array)) for f in features]
    base = np.logical_or.reduce(known) if study_mask is None else np.asarray(study_mask, bool).copy()
    if missing == "exclude":
        base &= np.logical_and.reduce(known)
    if not base.any():
        raise ValueError("The study area is empty (no cell has all evidence).")
    return base


class FeatureStack:
    """Encoded evidence of the study cells (``cells`` = flat grid index of every study cell)."""

    def __init__(self, features: Sequence[Feature], study: np.ndarray, missing: str = "exclude"):
        if missing not in MISSING_RULES:
            raise ValueError(f"missing must be one of {MISSING_RULES}")
        if not features:
            raise ValueError("At least one evidence layer is required.")
        self.shape = study.shape
        self.cells = np.flatnonzero(study)
        self.names: List[str] = []                       # column names
        self.layer_of: List[str] = []                    # evidence layer of each column
        self.groups: Dict[str, List[int]] = {}
        self._cols: List[np.ndarray] = []                # float32 column per feature column (study cells)
        self.notes: List[str] = []
        for f in features:
            arr = np.ma.asarray(f.array)
            if arr.shape != self.shape:
                raise ValueError(f"Evidence '{f.name}' is not on the same grid as the first layer.")
            data = np.ma.getdata(arr).ravel()[self.cells].astype(float)
            known = ~np.ma.getmaskarray(arr).ravel()[self.cells] & np.isfinite(data)
            if not known.any():
                raise ValueError(f"Evidence '{f.name}' has no data inside the study area.")
            n_miss = int((~known).sum())
            start = len(self.names)
            if f.kind == "continuous":
                if np.ptp(data[known]) == 0:
                    raise ValueError(f"Evidence '{f.name}' is constant inside the study area.")
                if missing == "keep":
                    col = np.where(known, data, np.nan).astype(np.float32)
                    if n_miss:
                        self.notes.append(f"{f.name}: {n_miss:,} cells without data kept as missing values")
                else:
                    fill = float(np.median(data[known]))
                    col = np.where(known, data, fill).astype(np.float32)
                    if n_miss:
                        self.notes.append(f"{f.name}: {n_miss:,} cells without data set to the median {fill:g}")
                self._add(f.name, f.name, col)
            else:
                classes, counts = np.unique(data[known], return_counts=True)
                if classes.size < 2:
                    raise ValueError(f"Evidence '{f.name}' has fewer than 2 classes inside the study area.")
                if classes.size > MAX_CLASSES:
                    raise ValueError(f"Evidence '{f.name}' has {classes.size} distinct values — use it as "
                                     "continuous or reclassify it (Data ▸ Reclassify raster).")
                mode = classes[np.argmax(counts)]
                vals = np.where(known, data, mode)
                labels = f.labels or {}
                for c in classes:
                    col = (vals == c).astype(np.float32)
                    if missing == "keep":
                        col[~known] = np.nan
                    self._add(f"{f.name}[{labels.get(float(c), f'{c:g}')}]", f.name, col)
                if n_miss and missing == "keep":
                    self.notes.append(f"{f.name}: {n_miss:,} cells without data kept as missing values")
                elif n_miss:
                    self.notes.append(f"{f.name}: {n_miss:,} cells without data set to the most frequent class "
                                      f"{mode:g}")
            self.groups[f.name] = list(range(start, len(self.names)))

    def _add(self, name, layer, col):
        self.names.append(name)
        self.layer_of.append(layer)
        self._cols.append(col)

    @property
    def n_cells(self) -> int:
        return int(self.cells.size)

    @property
    def n_columns(self) -> int:
        return len(self.names)

    def column(self, j: int) -> np.ndarray:
        """Values of feature column ``j`` for all study cells (float32, read-only view)."""
        v = self._cols[j].view()
        v.flags.writeable = False
        return v

    def matrix(self, idx) -> np.ndarray:
        """Feature rows (float32, C-contiguous) of study-cell positions ``idx`` (0 … n_cells − 1)."""
        idx = np.asarray(idx, np.int64)
        X = np.empty((idx.size, self.n_columns), np.float32)
        for j, col in enumerate(self._cols):
            X[:, j] = col[idx]
        return X

    def positions(self, rows, cols) -> np.ndarray:
        """Study-cell positions of grid cells (rows, cols) that are inside the study area."""
        flat = np.asarray(rows, np.int64) * self.shape[1] + np.asarray(cols, np.int64)
        pos = np.searchsorted(self.cells, flat)
        pos = np.clip(pos, 0, max(self.n_cells - 1, 0))
        return pos[self.cells[pos] == flat] if self.n_cells else pos[:0]
