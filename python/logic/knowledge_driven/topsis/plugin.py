"""TOPSIS / VIKOR plug-in (criteria = expert evidence scores; equal, custom, AHP, fuzzy-AHP or entropy weights)."""
from __future__ import annotations

import numpy as np

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate, negative_cells
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from ..index_overlay.model import IOLayer
from ..index_overlay.plugin import parse_spec
from .model import mcdm

TITLE = N_("TOPSIS / VIKOR (AHP / Fuzzy AHP / entropy weights)")
PART_DESCRIPTIONS = {"d_plus": N_("TOPSIS distance to the positive ideal solution D⁺"),
                     "d_minus": N_("TOPSIS distance to the negative ideal solution D⁻"),
                     "S": N_("VIKOR group utility S (0 = best)"),
                     "R": N_("VIKOR individual regret R (0 = best)"),
                     "Q": N_("VIKOR compromise index Q (0 = best)")}


@register
class TopsisMethod(MPMMethod):
    key = "topsis"
    name = TITLE
    category = N_("Knowledge-driven")
    group = N_("MCDM")
    description = N_("Multi-criteria decision making: every cell is an alternative, every evidence layer a criterion "
                     "scored 0 … 1 by the expert. TOPSIS ranks cells by their relative closeness to the ideal "
                     "solution (Hwang & Yoon 1981), VIKOR by the compromise of group utility and individual regret "
                     "(Opricovic 1998); weights equal, custom, AHP, fuzzy AHP or Shannon entropy.")
    icon = "mdi6.target-variant"

    def create_dialog(self, parent, store):
        from .dialogs import TopsisDialog
        return TopsisDialog(store, parent)

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
        method = params.get("method", "topsis")
        v = float(params.get("v", 0.5))
        progress(25, tr("ranking cells ({method})", method=method.upper()))
        res = mcdm(layers, method, params.get("weighting", "equal"), params.get("pairwise"),
                   params.get("delta", 1.0), params.get("normalisation", "vector"), v,
                   params.get("missing", "renormalise"), mask, dep)
        progress(70, tr("validating"))
        ev = None
        if dep is not None and len(dep[0]):
            ev = evaluate(res.index, dep[0], dep[1], valid_mask=res.study_mask, score_name=method,
                          negatives=negative_cells(params.get("negatives"), grid, res.study_mask))
        names = ", ".join(l.name for l in layers)
        desc = (tr("TOPSIS relative closeness to the ideal solution ({names})", names=names) if method == "topsis"
                else tr("VIKOR prospectivity 1 − Q (v = {v}; {names})", v=f"{v:g}", names=names))
        outs = [OutputRaster(method, res.index.astype("float32"), grid, cmap="inferno", description=desc)]
        if params.get("out_parts", False):
            for k, arr in res.parts.items():
                outs.append(OutputRaster(k, arr.astype("float32"), grid, cmap="viridis",
                                         description=tr(PART_DESCRIPTIONS[k]), visible=False))
        if params.get("out_scores", False):
            for n, arr in res.scores.items():
                outs.append(OutputRaster(f"score_{n}", arr.astype("float32"), grid, cmap="viridis",
                                         description=tr("Score of {layer}", layer=n), visible=False))
        tables = {"evidence": res.table}
        if res.ahp_table is not None:
            tables["ahp"] = res.ahp_table
        progress(100, tr("done"))
        return MethodResult(self.key, outs, label=params.get("prefix", method), tables=tables, evaluation=ev,
                            messages=res.messages, extra=dict(mcdm=res, layers=layers, method=method,
                                                              deposits_name=params.get("deposits_name"),
                                                              weighting=params.get("weighting", "equal"),
                                                              normalisation=params.get("normalisation", "vector"),
                                                              v=v))

    def show_result(self, window, result: MethodResult):
        for out in reversed(result.outputs):
            window.add_output_raster(out)
        from .dialogs import TopsisResultsDialog
        dlg = TopsisResultsDialog(result, window, window)
        dlg.show()
        window.keep_dialog(dlg)
