"""Logistic fuzzification + geometric average plug-in (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import LGLayer, logistic_geometric, parse_range, parse_scores


@register
class LogisticGeometricMethod(MPMMethod):
    key = "logistic_geometric"
    name = N_("Logistic Fuzzification + Geometric Average")
    category = N_("Knowledge-driven")
    group = N_("Fuzzy")
    description = N_("Yousefi & Carranza (2015): continuous evidence fuzzified by a logistic function set from the "
                     "data range (no expert thresholds), categorical evidence by class scores, combined by the "
                     "(optionally P–A-weighted) geometric average; evaluated with the prediction–area plot.")
    icon = "mdi6.sine-wave"

    def create_dialog(self, parent, store):
        from .dialogs import LogisticGeometricDialog
        return LogisticGeometricDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("fuzzifying evidence"))
        layers = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["direction"] == "classes" else "bilinear")
            spec = parse_scores(e["spec"]) if e["direction"] == "classes" else parse_range(e["spec"])
            layers.append(LGLayer(e["name"], data, e["direction"], spec, float(e.get("weight", 1.0))))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        dep = None
        if params.get("deposits") is not None and len(params["deposits"]):
            dep = points_to_cells(params["deposits"], grid, valid_mask=mask)
        res = logistic_geometric(layers, params.get("weighting", "equal"), mask, dep)
        progress(70, tr("validating"))
        ev = None
        if dep is not None and len(dep[0]):
            ev = evaluate(res.favourability, dep[0], dep[1], valid_mask=res.study_mask, score_name="favourability",
                          negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(l.name for l in layers)
        outs = [OutputRaster("favourability", res.favourability.astype("float32"), grid, cmap="inferno",
                             description=tr("Geometric average of the fuzzified evidence ({names})", names=names))]
        if params.get("out_fuzzy", False):
            for n, arr in res.fuzzified.items():
                outs.append(OutputRaster(f"fz_{n}", arr.astype("float32"), grid, cmap="viridis",
                                         description=tr("Fuzzified {layer}", layer=n), visible=False))
        progress(100, tr("done"))
        return MethodResult(self.key, outs, label=params.get("prefix", "lgfa"), tables={"evidence": res.table},
                            evaluation=ev, messages=res.messages,
                            extra=dict(lg=res, layers=layers, deposits_name=params.get("deposits_name")))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        from .dialogs import LogisticGeometricResultsDialog
        dlg = LogisticGeometricResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
