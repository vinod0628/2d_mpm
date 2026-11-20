"""M5 KT-RK plug-in: knowledge-trend regression-kriging of a continuous target (dialog -> worker -> results)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from i18n import N_, tr
from logic.hybrid.common.pipeline import finish, knowledge_outputs, prepare

from .model import kt_rk

TITLE = N_("Knowledge-Trend Regression-Kriging (KT-RK)")


def sample_cells(gdf, field: str, grid, study: np.ndarray, cells: np.ndarray):
    """Positions (in the study cells) and per-cell mean values of the samples (composite: one value per cell)."""
    if gdf.crs is not None and grid.crs is not None and gdf.crs != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    g = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    vals = pd.to_numeric(g[field], errors="coerce").to_numpy(float)
    xs = np.array([p.centroid.x for p in g.geometry])
    ys = np.array([p.centroid.y for p in g.geometry])
    rows, cols = grid.xy_to_rowcol(xs, ys)
    ok = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width) & np.isfinite(vals)
    rows, cols, vals = rows[ok], cols[ok], vals[ok]
    ok = study[rows, cols]
    flat = rows[ok] * grid.width + cols[ok]
    vals = vals[ok]
    u, inv = np.unique(flat, return_inverse=True)
    mean = np.bincount(inv, weights=vals) / np.bincount(inv)
    pos = np.searchsorted(cells, u)
    return pos, mean, int(ok.sum())


@register
class KTRKMethod(MPMMethod):
    key = "kt_rk"
    name = TITLE
    category = N_("Hybrid")
    group = N_("Knowledge prior + data correction")
    description = N_("Regression of a continuous target at points (drill-hole grade, thickness, accumulation or a "
                     "geochemical value): sign-constrained trend on the expert favourability, random forest on the "
                     "trend residuals (both shrunk by the one-standard-error rule) and anisotropic kriging of the "
                     "remaining residual along the expert strike.")
    icon = "mdi6.chart-scatter-plot"

    def create_dialog(self, parent, store):
        from .dialogs import KTRKDialog
        return KTRKDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        p = dict(params, test_fraction=0.0)            # deposits (optional) only validate the map
        ctx = prepare(p, progress, need_deposits=False)
        net, g = ctx.net, ctx.grid
        progress(8, tr("locating the samples"))
        pos, vals, n_raw = sample_cells(params["samples"], params["field"], g, net.study, net.cells)
        xs, ys = g.cell_centers()
        coords = np.column_stack([xs.ravel()[net.cells], ys.ravel()[net.cells]])
        res = kt_rk(net, pos, vals, coords, strike_deg=params.get("strike", 0.0),
                    range_along=params.get("range_along", 10 * g.res[0]),
                    range_across=params.get("range_across", 3 * g.res[0]), c=params.get("offset", 0.0) or None,
                    n_folds=int(params.get("folds", 10)), max_gp_fit=int(params.get("max_gp_fit", 1500)),
                    seed=params.get("seed", 0), progress=lambda f, msg: progress(10 + int(82 * f), tr(msg)))
        msgs = list(res.messages)
        if n_raw > len(pos):
            msgs.append(tr("{n} samples fall in {c} cells — values averaged per cell", n=n_raw, c=len(pos)))
        fld = params["field"]
        outs = [OutputRaster("prediction", net.grid(res.pred).astype("float32"), g, cmap="inferno",
                             description=tr("KT-RK prediction of {field} (exp(ẑ) − c)", field=fld)),
                OutputRaster("log_prediction", net.grid(res.log_pred).astype("float32"), g, cmap="inferno",
                             visible=False, description=tr("ln({field} + c): trend + kriged residual", field=fld)),
                OutputRaster("log_sd", net.grid(res.log_sd).astype("float32"), g, cmap="viridis", visible=False,
                             description=tr("Kriging s.d. of the residual (log scale)")),
                OutputRaster("trend", net.grid(res.trend).astype("float32"), g, cmap="inferno", visible=False,
                             description=tr("Knowledge + forest trend (log scale)"))]
        outs += knowledge_outputs(ctx, res.F, res.expert, with_processes=params.get("out_processes", False))
        tables = dict(summary=res.summary, cv=res.cv, observations=res.observations)
        st = res.stats
        extra = dict(ktrk=res, auc_caption=N_("ROC AUC — deposits (validation)"), cards=[(f"{st['observations']}", N_("observations (cells)")),
                                      (f"α {st['alpha']:g} / s {st['shrink']:g}", N_("knowledge trend / forest")),
                                      (f"{st['cv_rmse']:.3f}", N_("CV RMSE (log)"))],
                     tabs=[("summary", N_("Summary"), N_("<b>Model</b> — log transform, knowledge trend, forest "
                                                         "shrinkage, anisotropic covariance.")),
                           ("cv", N_("Cross-validation"), N_(
                               "<b>Cross-validation</b> — RMSE of ln(y + c) for every α (weight of the knowledge "
                               "trend) and s (weight of the forest); the simplest setting within one standard error "
                               "of the best is selected.")),
                           ("observations", N_("Observations"), N_("<b>Observations</b> — value, trend and "
                                                                   "prediction at every sampled cell."))])
        return finish(self.key, ctx, p, outs, tables, msgs, extra, progress, "ktrk")

    def show_result(self, window, result: MethodResult):
        from logic.hybrid.common.dialogs import show_hybrid_result
        show_hybrid_result(window, result, tr(TITLE))
