"""XGBoost plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from logic.common.features import Feature
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from utils.raster_tools import align_to_grid, points_to_cells

from .model import XGBoostModel


@register
class XGBoostMethod(MPMMethod):
    key = "xgboost"
    name = N_("XGBoost")
    category = N_("Data-driven")
    group = N_("Machine learning")
    description = N_("Extreme gradient boosting (Chen & Guestrin 2016) trained on deposits and random non-deposit "
                     "cells: number of trees by cross-validated early stopping, monotone constraints, SHAP "
                     "importance and contribution maps; uses an NVIDIA GPU automatically when available.")
    icon = "mdi6.file-tree"

    def create_dialog(self, parent, store):
        from .dialogs import XGBoostDialog
        return XGBoostDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(3, tr("preparing evidence"))
        feats = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"], ref=evs[0]["name"]))
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["kind"] == "categorical" else "bilinear")
            feats.append(Feature(e["name"], data, e["kind"], labels=e.get("labels") or None))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        progress(8, tr("locating deposits"))
        valid_any = mask if mask is not None else np.logical_or.reduce(
            [~np.ma.getmaskarray(np.ma.asarray(f.array)) for f in feats])
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=valid_any)
        all_rows, all_cols = rows, cols                         # non-deposit samples keep away from every deposit
        rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
        if len(rows) < 3:
            raise ValueError(tr("Fewer than 3 deposits fall inside the study area / evidence grid."))
        tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                                params.get("split_method", "spatial"))
        model = XGBoostModel(
            n_rounds=params.get("n_rounds", 500), learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3), min_child_weight=params.get("min_child_weight", 1.0),
            subsample=params.get("subsample", 0.8), colsample=params.get("colsample", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0), reg_alpha=params.get("reg_alpha", 0.0),
            min_split_loss=params.get("min_split_loss", 0.0), early_stopping=params.get("early_stopping", 50),
            class_weight=params.get("class_weight", "none"),
            monotone={e["name"]: e.get("monotone", 0) for e in evs if e["kind"] == "continuous"},
            neg_ratio=params.get("neg_ratio", 1.0), min_distance=params.get("min_distance", 0.0),
            min_spacing=params.get("min_spacing", -1.0), res=grid.res, repeats=params.get("repeats", 1),
            cv_folds=params.get("cv_folds", 5), missing=params.get("missing", "exclude"),
            device=params.get("device", "auto"), shap=params.get("out_shap", False), seed=params.get("seed", 0))
        res = model.fit_predict(feats, tr_r, tr_c, mask, known_deposits=(all_rows, all_cols),
                                progress=lambda f, msg: progress(10 + int(80 * f), tr(msg)))
        progress(92, tr("validating"))
        ev = evaluate(res.score, tr_r, tr_c, te_r, te_c, valid_mask=res.study_mask, score_name="score",
                      negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(f.name for f in feats)
        outs = [OutputRaster("score", res.score.astype("float32"), grid, cmap="inferno",
                             description=tr("XGBoost prospectivity score — predicted probability of the boosted "
                                            "trees ({names})", names=names))]
        if params.get("out_std", True) and res.score_std is not None:
            outs.append(OutputRaster("score_std", res.score_std.astype("float32"), grid, cmap="viridis",
                                     description=tr("Std. deviation of the score over the random non-deposit "
                                                    "samples"), visible=False))
        for layer, arr in res.shap_maps.items():
            outs.append(OutputRaster(f"shap_{layer}", arr.astype("float32"), grid, cmap="RdBu_r",
                                     description=tr("SHAP contribution of {layer} to the log-odds (> 0 raises the "
                                                    "score)", layer=layer), visible=False))
        vectors = deposit_point_outputs(params["deposits"], grid, tr_r, tr_c, te_r, te_c)
        progress(100, tr("done"))
        tables = {"summary": res.summary, "importance": res.importance, "boosting_curve": res.curve,
                  "sample_validation": res.sample_validation, "repeats": res.repeats}
        return MethodResult(self.key, outs, vectors=vectors, label=params.get("prefix", "xgb"), tables=tables,
                            evaluation=ev, messages=thin_msg + res.messages,
                            extra=dict(xgb=res, deposits_name=params.get("deposits_name"),
                                       train_cells=(tr_r, tr_c), test_cells=(te_r, te_c)))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)
        from .dialogs import XGBoostResultsDialog
        dlg = XGBoostResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
