"""Plug-in of the method: registration + run (worker thread) + result display."""
from __future__ import annotations

from core.methods import MethodResult, MPMMethod, OutputRaster, register
from evaluation.metrics import evaluate
from i18n import N_, tr
from utils.raster_tools import align_to_grid, points_to_cells

from .model import weighted_mean_score


@register
class TemplateMethod(MPMMethod):
    key = "template_weighted_mean"            # unique id
    # N_() marks texts for translation (python tools/i18n_update.py); the GUI shows them with tr()
    name = N_("Template: weighted mean")       # menu text
    category = N_("Knowledge-driven")          # "Knowledge-driven" | "Data-driven" | "Hybrid"
    group = N_("Overlay")                      # sub-menu
    description = N_("Weighted mean of min-max normalised evidence layers.")
    icon = "mdi6.function-variant"

    def create_dialog(self, parent, store):    # GUI thread
        from .dialogs import TemplateDialog
        return TemplateDialog(store, parent)

    def run(self, params: dict, progress) -> MethodResult:     # worker thread: no Qt calls here
        layers = params["evidence"]
        grid = layers[0]["grid"]
        arrays = [l["data"] if l["grid"].same_as(grid) else align_to_grid(l["data"], l["grid"], grid, "bilinear")
                  for l in layers]
        progress(50, tr("computing score"))
        score = weighted_mean_score(arrays, [l["weight"] for l in layers])
        ev = None
        if params.get("deposits") is not None:                 # optional validation (shared ROC/AUC dialog)
            r, c = points_to_cells(params["deposits"], grid, valid_mask=~score.mask)
            if len(r):
                ev = evaluate(score, r, c, score_name="score")
        progress(100, tr("done"))
        return MethodResult(self.key, [OutputRaster("score", score.astype("float32"), grid,
                                                    cmap="inferno", description=tr(self.description))], evaluation=ev)

    def show_result(self, window, result: MethodResult):     # GUI thread
        super().show_result(window, result)                    # adds the output rasters to the map
        if result.evaluation is not None:
            from gui.dialogs.evaluation_dialogs import EvaluationDialog
            dlg = EvaluationDialog(result.evaluation, tr(self.name), window)
            dlg.show()
            window.keep_dialog(dlg)
