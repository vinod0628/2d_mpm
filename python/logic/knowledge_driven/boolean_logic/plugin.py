"""Boolean logic overlay plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import BoolLayer, boolean_overlay, parse_values


@register
class BooleanLogicMethod(MPMMethod):
    key = "boolean_logic"
    name = N_("Boolean Logic Overlay")
    category = N_("Knowledge-driven")
    group = N_("Overlay")
    description = N_("Boolean logic overlay (Bonham-Carter 1994): expert conditions per evidence layer (≥, ≤, "
                     "between, outside, classes) combined with AND / OR within and between mineral-system groups; "
                     "favourable area, number of satisfied conditions and deposits captured.")
    icon = "mdi6.gate-and"

    def create_dialog(self, parent, store):
        from .dialogs import BooleanLogicDialog
        return BooleanLogicDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("evaluating conditions"))
        layers = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["cond"] == "in" else "bilinear")
            layers.append(BoolLayer(e["name"], data, e["cond"], parse_values(e["cond"], e["values"]),
                                    e.get("group", "1") or "1"))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        dep_cells = None
        if params.get("deposits") is not None and len(params["deposits"]):
            dep_cells = points_to_cells(params["deposits"], grid, valid_mask=mask)
        res = boolean_overlay(layers, params.get("group_ops", {}), params.get("final_op", "and"), mask, dep_cells)
        progress(70, tr("validating"))
        ev = None
        if dep_cells is not None and len(dep_cells[0]):
            ev = evaluate(res.count.astype(float), dep_cells[0], dep_cells[1], valid_mask=res.study_mask,
                          score_name="conditions met", negatives=negative_cells(params.get("negatives"), grid,
                                                                                 res.study_mask))
        names = ", ".join(l.name for l in layers)
        outs = [OutputRaster("favourable", res.favourable, grid, cmap="RdYlGn",
                             description=tr("Boolean favourable area, 1 = favourable ({names})", names=names)),
                OutputRaster("conditions_met", res.count, grid, cmap="viridis",
                             description=tr("Number of satisfied conditions"), visible=False)]
        if params.get("out_conditions", False):
            for n, arr in res.conditions.items():
                outs.append(OutputRaster(f"cond_{n}", arr, grid, cmap="Greys",
                                         description=tr("Condition on {layer}", layer=n), visible=False))
        progress(100, tr("done"))
        return MethodResult(self.key, outs, label=params.get("prefix", "boolean"),
                            tables={"conditions": res.table, "network": res.network}, evaluation=ev,
                            messages=res.messages, extra=dict(boolean=res, deposits_name=params.get("deposits_name")))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        from .dialogs import BooleanLogicResultsDialog
        dlg = BooleanLogicResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
