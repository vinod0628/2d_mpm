"""Weights of Evidence plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from logic.common.training import deposit_point_outputs, split_deposits, thin_deposits
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import Evidence, WofEModel


@register
class WofEMethod(MPMMethod):
    key = "wofe"
    name = N_("Weights of Evidence")
    category = N_("Data-driven")
    group = N_("Statistical")
    description = N_("Bayesian weights of evidence with generalisation, CI tests and posterior uncertainty.")
    icon = "mdi6.scale-balance"

    def create_dialog(self, parent, store):
        from .dialogs import WofEDialog
        return WofEDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("preparing evidence"))
        evidence = []
        for i, e in enumerate(evs):
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e['name'], ref=evs[0]['name']))
                data = align_to_grid(data, e["grid"], grid, "nearest")
            evidence.append(Evidence(e["name"], data, e["type"], labels=e.get("labels") or None,
                                     criterion=e.get("criterion", "max_contrast"),
                                     min_studentized=float(e.get("min_stud", 1.96))))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        progress(15, tr("locating deposits"))
        valid_any = mask if mask is not None else np.logical_or.reduce([~np.ma.getmaskarray(e.array) for e in evidence])
        rows, cols = points_to_cells(params["deposits"], grid, valid_mask=valid_any)
        rows, cols, thin_msg = thin_deposits(rows, cols, params.get("thin_distance", 0.0), grid.res)
        if len(rows) < 2:
            raise ValueError(tr("Fewer than 2 deposits fall inside the study area / evidence grid."))
        tr_r, tr_c, te_r, te_c = split_deposits(rows, cols, params.get("test_fraction", 0.0), params.get("seed", 0),
                                                params.get("split_method", "spatial"))
        progress(30, tr("computing weights"))
        model = WofEModel(unit_cell_km2=params.get("unit_cell_km2", 1.0), cell_area_km2=grid.cell_area / 1e6,
                          missing=params.get("missing", "zero"), zero_deposit=params.get("zero_deposit", "zero"))
        res = model.fit_predict(evidence, tr_r, tr_c, mask)
        progress(80, tr("validating"))
        neg = negative_cells(params.get("negatives"), grid, res.study_mask)
        ev = evaluate(res.posterior, tr_r, tr_c, te_r, te_c, valid_mask=res.study_mask, score_name="post",
                      negatives=neg)
        pre = params.get("prefix", "wofe")
        names = ", ".join(e.name for e in evidence)
        outs = [OutputRaster("post", res.posterior.astype("float32"), grid, cmap="inferno",
                             description=tr("WofE posterior probability ({names})", names=names))]
        if params.get("out_std", True):
            outs.append(OutputRaster("post_std", res.posterior_std.astype("float32"), grid, cmap="viridis",
                                     description=tr("Std. deviation of the posterior probability due to the weights"),
                                     visible=False))
            if res.md_variance is not None:
                outs.append(OutputRaster("md_var", res.md_variance.astype("float32"), grid, cmap="viridis",
                                         description=tr("Variance of the posterior probability due to missing data"),
                                         visible=False))
                outs.append(OutputRaster("total_std", res.total_std.astype("float32"), grid, cmap="viridis",
                                         description=tr("Total std. deviation of the posterior probability"),
                                         visible=False))
        if params.get("out_conf", True):
            outs.append(OutputRaster("conf", res.confidence.astype("float32"), grid, cmap="RdYlBu_r",
                                     description=tr("Confidence P / total s(P); > ~1.5–2 = significant"), visible=False))
        if params.get("out_weights", False):
            for k, w in res.weight_rasters.items():
                outs.append(OutputRaster(f"W_{k}", w.astype("float32"), grid, cmap="RdBu_r",
                                         description=tr("Generalised weight of {name}", name=k), visible=False))
        progress(100, tr("done"))
        vectors = deposit_point_outputs(params["deposits"], grid, tr_r, tr_c, te_r, te_c)
        tables = {f"weights_{l.name}": l.table for l in res.layers}
        tables["summary"] = res.summary_table()
        tables["pairwise"] = res.pairwise
        return MethodResult(self.key, outs, vectors=vectors, label=pre, tables=tables, evaluation=ev,
                            messages=thin_msg + res.messages,
                            extra=dict(wofe=res, deposits_name=params.get("deposits_name"),
                                       train_cells=(tr_r, tr_c), test_cells=(te_r, te_c)))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        for vec in result.vectors:
            window.add_output_vector(vec)
        from .dialogs import WofEResultsDialog
        dlg = WofEResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
