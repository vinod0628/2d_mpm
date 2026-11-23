"""Index overlay plug-in (binary / multi-class; equal, custom, AHP or fuzzy-AHP weights)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import IOLayer, index_overlay, parse_range, parse_scores, parse_threshold

TITLE = N_("Index Overlay (AHP / Fuzzy AHP weights)")


def parse_spec(rule: str, text: str):
    if rule == "classes":
        return parse_scores(text)
    if rule in ("positive", "negative"):
        return parse_range(text)
    return parse_threshold(text)


@register
class IndexOverlayMethod(MPMMethod):
    key = "index_overlay"
    name = TITLE
    category = N_("Knowledge-driven")
    group = N_("Overlay")
    description = N_("Binary or multi-class index overlay (Bonham-Carter 1994): expert scores per class or linear "
                     "rescaling of continuous layers, combined by the weighted mean; layer weights equal, custom, "
                     "from an AHP pairwise-comparison matrix (Saaty) or from fuzzy AHP (Buckley).")
    icon = "mdi6.layers-triple-outline"

    def create_dialog(self, parent, store):
        from .dialogs import IndexOverlayDialog
        return IndexOverlayDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("scoring evidence"))
        layers = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["rule"] in ("classes", "ge", "le")
                                     else "bilinear")
            layers.append(IOLayer(e["name"], data, e["rule"], parse_spec(e["rule"], e["spec"]),
                                  float(e.get("weight", 1.0))))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        dep = None
        if params.get("deposits") is not None and len(params["deposits"]):
            dep = points_to_cells(params["deposits"], grid, valid_mask=mask)
        res = index_overlay(layers, params.get("weighting", "equal"), params.get("pairwise"),
                            params.get("delta", 1.0), params.get("missing", "renormalise"), mask, dep)
        progress(70, tr("validating"))
        ev = None
        if dep is not None and len(dep[0]):
            ev = evaluate(res.index, dep[0], dep[1], valid_mask=res.study_mask, score_name="index",
                          negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(l.name for l in layers)
        outs = [OutputRaster("index", res.index.astype("float32"), grid, cmap="inferno",
                             description=tr("Index overlay: weighted mean of the evidence scores ({names})",
                                            names=names))]
        if params.get("out_scores", False):
            for n, arr in res.scores.items():
                outs.append(OutputRaster(f"score_{n}", arr.astype("float32"), grid, cmap="viridis",
                                         description=tr("Score of {layer}", layer=n), visible=False))
        tables = {"evidence": res.table}
        if res.ahp_table is not None:
            tables["ahp"] = res.ahp_table
        progress(100, tr("done"))
        return MethodResult(self.key, outs, label=params.get("prefix", "index"), tables=tables, evaluation=ev,
                            messages=res.messages, extra=dict(io=res, layers=layers,
                                                              deposits_name=params.get("deposits_name"),
                                                              weighting=params.get("weighting", "equal")))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        from .dialogs import IndexOverlayResultsDialog
        dlg = IndexOverlayResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
