"""Fuzzy logic overlay plug-in: registers the method (dialog -> run in worker -> result display)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import FuzzyLayer, fuzzy_overlay, parse_params


@register
class FuzzyLogicMethod(MPMMethod):
    key = "fuzzy_logic"
    name = N_("Fuzzy Logic Overlay")
    category = N_("Knowledge-driven")
    group = N_("Fuzzy")
    description = N_("Fuzzy logic overlay (Bonham-Carter 1994): expert membership functions per evidence layer "
                     "(linear, S/Z, logistic, Gaussian, large / small, per class) combined in an inference network "
                     "of mineral-system groups with fuzzy AND, OR, product, sum, gamma or geometric average.")
    icon = "mdi6.blur"

    def create_dialog(self, parent, store):
        from .dialogs import FuzzyLogicDialog
        return FuzzyLogicDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:
        evs = params["evidence"]
        grid = evs[0]["grid"]
        progress(5, tr("computing memberships"))
        layers = []
        for e in evs:
            data = e["data"]
            if not e["grid"].same_as(grid):
                if not params.get("auto_align", True):
                    raise ValueError(tr("{layer} is not on the grid of {ref}", layer=e["name"], ref=evs[0]["name"]))
                data = align_to_grid(data, e["grid"], grid, "nearest" if e["function"] == "classes" else "bilinear")
            layers.append(FuzzyLayer(e["name"], data, e["function"], parse_params(e["function"], e["params"]),
                                     e.get("group", "1") or "1"))
        mask = None
        if params.get("mask") is not None:
            m, mg = params["mask"]
            if not mg.same_as(grid):
                m = align_to_grid(m, mg, grid, "nearest")
            mask = ~np.ma.getmaskarray(m)
        res = fuzzy_overlay(layers, params.get("group_ops", {}), params.get("final_op", ("gamma", 0.9)), mask,
                            params.get("missing", "exclude"), params.get("missing_value", 0.5))
        progress(70, tr("validating"))
        ev = None
        if params.get("deposits") is not None and len(params["deposits"]):
            rows, cols = points_to_cells(params["deposits"], grid, valid_mask=res.study_mask)
            if len(rows):
                ev = evaluate(res.favourability, rows, cols, valid_mask=res.study_mask, score_name="favourability",
                              negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(l.name for l in layers)
        outs = [OutputRaster("favourability", res.favourability.astype("float32"), grid, cmap="inferno",
                             description=tr("Fuzzy favourability ({names})", names=names))]
        if params.get("out_groups", True) and len(res.groups) > 1:
            for g, arr in res.groups.items():
                outs.append(OutputRaster(f"group_{g}", arr.astype("float32"), grid, cmap="viridis",
                                         description=tr("Fuzzy group {g}", g=g), visible=False))
        if params.get("out_memberships", False):
            for n, arr in res.memberships.items():
                outs.append(OutputRaster(f"mu_{n}", arr.astype("float32"), grid, cmap="viridis",
                                         description=tr("Fuzzy membership of {layer}", layer=n), visible=False))
        progress(100, tr("done"))
        return MethodResult(self.key, outs, label=params.get("prefix", "fuzzy"),
                            tables={"inference_network": res.network, "memberships": res.layer_table},
                            evaluation=ev, messages=res.messages,
                            extra=dict(fuzzy=res, layers=layers, deposits_name=params.get("deposits_name")))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        from .dialogs import FuzzyLogicResultsDialog
        dlg = FuzzyLogicResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
