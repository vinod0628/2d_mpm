"""Validation metrics for prospectivity maps, shared by every MPM method.

Cell-based convention: positives = cells containing known deposits, negatives = every other valid
cell of the study area (background) — or, as in ArcSDM's ROC tool, cells of known non-deposit (absence) points.
Scores can be any "higher = more prospective" raster.

ROC AUC = Mann–Whitney probability that a deposit cell scores higher than a negative cell (ties count ½), with
the DeLong et al. (1988) standard error and a 95 % confidence interval. Compared with ArcSDM's ROC tool
(github.com/gtkfi/ArcSDM, roctool.py): ArcSDM draws as many random background points as deposits 10,000 times and
reports the AUC of the median curve with a percentile interval, and orders tied scores arbitrarily; using every
background cell gives the deterministic limit of that procedure, and ties are counted ½.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from i18n import N_, tr
from i18n import tr as tr_                       # ``tr`` is shadowed inside evaluate()


@dataclass
class CurveSet:
    name: str
    n_pos: int
    n_neg: int
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    area_frac: np.ndarray            # success-rate x: fraction of study area (highest scores first)
    capture_frac: np.ndarray         # success-rate y: fraction of deposits captured
    sr_auc: float                    # area under success-rate curve
    pa_threshold: np.ndarray         # P-A: score thresholds
    pa_prediction: np.ndarray        # P-A: % deposits with score >= t
    pa_area: np.ndarray              # P-A: % area with score >= t
    pa_intersection: Dict[str, float] = field(default_factory=dict)
    auc_se: float = float("nan")     # DeLong standard error
    auc_ci: tuple = (float("nan"), float("nan"))
    negatives: str = "background"    # "background" cells or "absence" points


@dataclass
class Evaluation:
    score_name: str
    curves: Dict[str, CurveSet]

    def summary_rows(self):
        rows = []
        for k, c in self.curves.items():
            rows.append(dict(set=k, deposits=c.n_pos, background_cells=c.n_neg, negatives=c.negatives, roc_auc=c.auc,
                             roc_auc_se=c.auc_se, roc_auc_ci_low=c.auc_ci[0], roc_auc_ci_high=c.auc_ci[1],
                             success_rate_auc=c.sr_auc,
                             pa_threshold=c.pa_intersection.get("threshold", np.nan),
                             pa_prediction_rate=c.pa_intersection.get("prediction", np.nan),
                             pa_area=c.pa_intersection.get("area", np.nan)))
        return rows


def delong_auc(pos: np.ndarray, neg: np.ndarray):
    """AUC (ties ½), DeLong et al. (1988) standard error and 95 % confidence interval.

    Placement values V10_i = P(neg < pos_i), V01_j = P(pos > neg_j); Var(AUC) = s²(V10)/m + s²(V01)/n.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m, n = pos.size, neg.size
    sn, sp = np.sort(neg), np.sort(pos)
    v10 = (np.searchsorted(sn, pos, "left") + np.searchsorted(sn, pos, "right")) / (2.0 * n)
    v01 = (2 * m - np.searchsorted(sp, neg, "left") - np.searchsorted(sp, neg, "right")) / (2.0 * m)
    auc = float(v10.mean())
    var = (v10.var(ddof=1) / m if m > 1 else 0.0) + (v01.var(ddof=1) / n if n > 1 else 0.0)
    se = float(np.sqrt(var))
    return auc, se, (max(0.0, auc - 1.959963984540054 * se), min(1.0, auc + 1.959963984540054 * se))


def negative_cells(gdf, grid, valid_mask: Optional[np.ndarray] = None):
    """Cells (rows, cols) of absence (non-deposit) points for the ROC, or None."""
    if gdf is None or len(gdf) == 0:
        return None
    from utils.raster_tools import points_to_cells
    r, c = points_to_cells(gdf, grid, valid_mask=valid_mask)
    return (r, c) if len(r) else None


def _curves(name, scores_valid, pos_scores, neg_scores, n_thresholds=400, negatives="background") -> CurveSet:
    y = np.r_[np.ones(pos_scores.size), np.zeros(neg_scores.size)]
    s = np.r_[pos_scores, neg_scores]
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError(tr("{name}: need both deposit and background cells.", name=name))
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=True)
    auc = float(roc_auc_score(y, s))
    _, se, ci = delong_auc(pos_scores, neg_scores)
    if fpr.size > 5000:                                  # keep plots light
        keep = np.unique(np.r_[0, np.linspace(0, fpr.size - 1, 5000).astype(int), fpr.size - 1])
        fpr, tpr = fpr[keep], tpr[keep]

    # thresholds: quantiles of all scores + every deposit score
    qs = np.quantile(scores_valid, np.linspace(0, 1, n_thresholds))
    thr = np.unique(np.r_[qs, pos_scores])[::-1]
    sorted_all = np.sort(scores_valid)
    sorted_pos = np.sort(pos_scores)
    area_ge = 1.0 - np.searchsorted(sorted_all, thr, side="left") / sorted_all.size
    cap_ge = 1.0 - np.searchsorted(sorted_pos, thr, side="left") / sorted_pos.size
    area_frac = np.r_[0.0, area_ge, 1.0]
    capture = np.r_[0.0, cap_ge, 1.0]
    sr_auc = float(np.trapz(capture, area_frac))

    pa_thr = thr[::-1]
    pa_pred, pa_area = cap_ge[::-1] * 100, area_ge[::-1] * 100
    inter = {}
    diff = pa_pred - (100 - pa_area)                     # prediction rate vs (100 - occupied area)
    sign = np.where(np.diff(np.sign(diff)) != 0)[0]
    if sign.size:
        i = sign[0]
        f = diff[i] / (diff[i] - diff[i + 1]) if diff[i] != diff[i + 1] else 0.0
        inter = dict(threshold=float(pa_thr[i] + f * (pa_thr[i + 1] - pa_thr[i])),
                     prediction=float(pa_pred[i] + f * (pa_pred[i + 1] - pa_pred[i])),
                     area=float(pa_area[i] + f * (pa_area[i + 1] - pa_area[i])))
    return CurveSet(name, int(pos_scores.size), int(neg_scores.size), auc, fpr, tpr, area_frac, capture, sr_auc,
                    pa_thr, pa_pred, pa_area, inter, auc_se=se, auc_ci=ci, negatives=negatives)


def evaluate(scores, train_rows, train_cols, test_rows=None, test_cols=None,
             valid_mask: Optional[np.ndarray] = None, score_name: str = "prospectivity",
             negatives=None) -> Evaluation:
    """ROC/AUC, success-rate and P-A curves for training (and optional test) deposits.

    negatives: optional (rows, cols) of absence-point cells used as ROC negatives instead of every background
    cell (ArcSDM ROC tool with "absence locations"); success-rate and P-A curves always use the whole area.
    """
    s = np.ma.asarray(scores)
    valid = ~np.ma.getmaskarray(s) & np.isfinite(np.ma.getdata(s))
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, bool)
    data = np.ma.getdata(s).astype(float)
    pos_all = np.zeros(s.shape, bool)

    def cells(r, c):
        m = np.zeros(s.shape, bool)
        if r is not None and len(r):
            m[np.asarray(r, int), np.asarray(c, int)] = True
        return m & valid

    tr = cells(train_rows, train_cols)
    te = cells(test_rows, test_cols)
    pos_all = tr | te
    kind = "background"
    if negatives is not None:
        neg = cells(*negatives) & ~pos_all
        kind = "absence"
        if not neg.any():
            raise ValueError(tr_("No absence point falls on a valid cell outside the deposits."))
    else:
        neg = valid & ~pos_all
    scores_valid = data[valid]
    curves = {N_("training"): _curves("training", scores_valid, data[tr], data[neg], negatives=kind)}
    if te.any():
        curves[N_("test")] = _curves("test", scores_valid, data[te], data[neg], negatives=kind)
    return Evaluation(score_name, curves)
