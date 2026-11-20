"""Logistic regression plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from utils.raster_tools import align_to_grid, points_to_cells

from .model import LogisticModel, Predictor


@register
class LogisticRegressionMethod(MPMMethod):
    key = "logistic_regression"
    name = N_("Logistic Regression")
    category = N_("Data-driven")
    group = N_("Statistical")
    description = N_("Logistic regression on all study-area cells (weighted LR) or on deposit / non-deposit samples, "
                     "with Firth estimation, significance tests and probability uncertainty.")
    icon = "mdi6.chart-bell-curve"

    def create_dialog(self, parent, store):
        from .dialogs import LogisticRegressionDialog
        return LogisticRegressionDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(3, tr("preparing evidence"))
        preds = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"], ref=evs[0]["name"]))
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["kind"] == "categorical" else "bilinear")
            preds.append(Predictor(e["name"], data, e["kind"], e.get("transform", "none"), labels=e.get("labels") or None))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        progress(8, tr("locating deposits"))
        valid_any = mask if mask is not None else np.logical_or.reduce(
            [~np.ma.getmaskarray(np.ma.asarray(p.array)) for p in preds])
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=valid_any)
        all_rows, all_cols = rows, cols                         # non-deposit samples keep away from every deposit
        rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
        if len(rows) < 2:
            raise ValueError(tr("Fewer than 2 deposits fall inside the study area / evidence grid."))
        tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                                params.get("split_method", "spatial"))
        model = LogisticModel(
            design=params.get("design", "all"), estimator=params.get("estimator", "firth"),
            ridge=params.get("ridge", 1.0), selection=params.get("selection", "none"), alpha=params.get("alpha", 0.10),
            missing=params.get("missing", "exclude"), unit_cell_km2=params.get("unit_cell_km2", 1.0),
            cell_area_km2=grid.cell_area / 1e6, neg_ratio=params.get("neg_ratio", 1.0),
            min_distance=params.get("min_distance", 0.0), res=grid.res, repeats=params.get("repeats", 1),
            prior_correction=params.get("prior_correction", True), cv_folds=params.get("cv_folds", -1),
            seed=params.get("seed", 0), min_spacing=params.get("min_spacing", -1.0))
        res = model.fit_predict(preds, tr_r, tr_c, mask, known_deposits=(all_rows, all_cols),
                                progress=lambda f, msg: progress(10 + int(78 * f), tr(msg)))
        progress(90, tr("validating"))
        ev = evaluate(res.probability, tr_r, tr_c, te_r, te_c, valid_mask=res.study_mask, score_name="post",
                      negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(res.params["active"])
        what = tr("relative favourability") if model.relative else tr("posterior probability")
        outs = [OutputRaster("post", res.probability.astype("float32"), grid, cmap="inferno",
                             description=tr("LR {what} ({names})", what=what, names=names))]
        if params.get("out_std", True):
            outs.append(OutputRaster("post_std", res.probability_std.astype("float32"), grid, cmap="viridis",
                                     description=tr("Std. deviation of the LR probability"), visible=False))
        if params.get("out_t", True):
            outs.append(OutputRaster("tvalue", res.tvalue.astype("float32"), grid, cmap="RdYlBu_r",
                                     description=tr("LR t-value P / s(P); > ~1.645 = significant"), visible=False))
        if params.get("out_logit", False):
            outs.append(OutputRaster("logit", res.logit.astype("float32"), grid, cmap="RdBu_r",
                                     description=tr("LR linear predictor (logit per raster cell)"), visible=False))
        progress(100, tr("done"))
        vectors = deposit_point_outputs(params["deposits"], grid, tr_r, tr_c, te_r, te_c)
        tables = {"coefficients": res.coefficients, "model_fit": res.fit_table, "variables": res.variables,
                  "hosmer_lemeshow": res.hosmer_lemeshow, "correlation": res.correlation}
        for name, df in (("selection", res.selection), ("repeats", res.repeats),
                         ("sample_validation", res.sample_validation)):
            if len(df):
                tables[name] = df
        return MethodResult(self.key, outs, vectors=vectors, label=params.get("prefix", "lr"), tables=tables,
                            evaluation=ev,
                            messages=thin_msg + res.messages,
                            extra=dict(lr=res, deposits_name=params.get("deposits_name"),
                                       train_cells=(tr_r, tr_c), test_cells=(te_r, te_c)))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)
        from .dialogs import LogisticRegressionResultsDialog
        dlg = LogisticRegressionResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
